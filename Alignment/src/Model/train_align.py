"""
train_align.py -- Stage 2. Teach the projector to translate.

Everything is frozen except the projector. The vision encoder and the LLM
are both downloaded and good already; the only thing that doesn't exist yet
is the bridge between them.

Three modes, run them in this order:

    python train_align.py --mode smoke     # 2 min. does anything run at all
    python train_align.py --mode overfit   # 20 min. do gradients reach the projector
    python train_align.py --mode train     # the real thing

Do not skip overfit. It is the single most informative 20 minutes in this
whole project.
"""

import argparse
import torch
from transformers import (
    AutoTokenizer, AutoProcessor, Trainer, TrainingArguments,
)

from model import MathVLM, VLMConfig
from data import AlignDataset, Collator, verify

IMAGE_TOKEN = "<|image_pad|>"     # already in Qwen's vocab, no resize needed


def build(args):
    tok = AutoTokenizer.from_pretrained(args.llm)
    proc = AutoProcessor.from_pretrained(args.vision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = VLMConfig(
        llm_path=args.llm,
        vision_path=args.vision,
        image_token_id=tok.convert_tokens_to_ids(IMAGE_TOKEN),
        n_visual_tokens=args.n_visual_tokens,
    )

    model = MathVLM(cfg)
    model.freeze_vision()
    model.freeze_llm()
    for p in model.projector.parameters():
        p.requires_grad = True

    # Should print ~15M, NOT 3000M. If it prints billions, a freeze failed
    # and you are about to train the whole LLM at projector learning rates.
    model.trainable_report()

    return model, tok, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "overfit", "train"], default="smoke")
    ap.add_argument("--data", default="data/align_train.json")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vision", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--out", default="ckpt/stage2")
    ap.add_argument("--n_visual_tokens", type=int, default=196)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16","fp32"])
    ap.add_argument("--attn", default="flash_attention_2",
                    help="use 'sdpa' on a T4 -- Turing has no flash-attn")
    ap.add_argument("--smoke_n", type=int,default =100)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[args.dtype]
    else:
        model_dtype = torch.float32
        if args.dtype != "fp32":
            print(f"CUDA is unavailable; using fp32 on {device} instead of {args.dtype}.")

    model, tok, proc = build(args)
    ds = AlignDataset(args.data)
    coll = Collator(tok, proc, args.n_visual_tokens, IMAGE_TOKEN, args.max_len)

    # ---- always verify the batch before training ----
    print("\n" + "=" * 60)
    verify(coll, ds, tok)
    print("=" * 60 + "\n")


    if args.mode == "smoke":
        import collections
        model = model.to(device=device, dtype=model_dtype)

        def batch_loss(rows, black=False):
            b = coll(rows)
            expected_tokens = len(rows) * args.n_visual_tokens
            actual_tokens = int((b["input_ids"] == coll.image_id).sum())
            assert actual_tokens == expected_tokens, (
                f"batch has {actual_tokens} image tokens, "
                f"expected {expected_tokens}"
            )
            b = {k: v.to(device) for k, v in b.items()}
            b["pixel_values"] = (torch.zeros_like(b["pixel_values"]) if black
                                 else b["pixel_values"]).to(model_dtype)
            with torch.no_grad():
                loss = model(**b).loss
            assert loss.ndim == 0 and torch.isfinite(loss), (
                f"smoke loss is invalid: {loss.item()}"
            )
            return loss.item()

        N, BS = min(args.smoke_n, len(ds)), 1
        if N <= 0:
            raise ValueError("--smoke_n must be greater than zero and the dataset cannot be empty")
        real, black, per_src = [], [], collections.defaultdict(list)

        for i in range(0, N, BS):
            rows = [ds[j] for j in range(i, min(i + BS, N))]
            lr, lb = batch_loss(rows), batch_loss(rows, black=True)
            real.append(lr); black.append(lb)
            src = ds.rows[i].get("align_source", "?").split(":")[0]
            per_src[src].append(lr)

        assert len(real) == len(black), "smoke test produced mismatched batch results"

        m = lambda x: sum(x) / len(x)
        print(f"\nn={N}")
        print(f"  loss, real image   {m(real):.3f}")
        print(f"  loss, black image  {m(black):.3f}   delta {m(black)-m(real):+.3f}")
        print("  (delta near zero is EXPECTED here -- projector is random)")
        print("\n  headroom by source (how much a projector could earn):")
        for s, v in per_src.items():
            print(f"    {s:12s} n={len(v):3d}  loss {m(v):.3f}")
        return

    if args.mode == "overfit":
        # Ten examples, 200 steps. The model should MEMORISE them: loss near
        # zero and it recites those exact captions. That proves gradients flow
        # from the LLM's loss all the way back into your projector. It has
        # learned nothing general, and that is fine -- this is a wiring test.
        ds.rows = ds.rows[:10]
        steps, accum, lr, save = 200, 1, 1e-3, 1_000_000
    else:
        steps, accum, lr, save = -1, args.accum, args.lr, 500

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        max_steps=steps,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=save,
        bf16=(device.type == "cuda" and args.dtype == "bf16"),
        fp16=(device.type == "cuda" and args.dtype == "fp16"),
        gradient_checkpointing=True,
        remove_unused_columns=False,      # our collator needs the raw columns
        report_to="none",
        dataloader_num_workers=4,
    )

    Trainer(model=model, args=targs, train_dataset=ds, data_collator=coll).train()

    # Save ONLY the projector. The other two parts are unchanged downloads;
    # saving them would waste 7GB per checkpoint.
    torch.save(model.projector.state_dict(), f"{args.out}/projector.pt")
    print(f"saved projector -> {args.out}/projector.pt")


if __name__ == "__main__":
    main()