#!/usr/bin/env python3
"""
preflight.py -- run this FIRST on any rented GPU box, before any long job.

Every check here is something that, if broken, wastes hours of paid GPU time
before you notice. Most of them fail in ways that look like something else:
a dtype mismatch surfaces as a shape error deep in attention, a missing
flash-attn build surfaces as "no kernel image available", a silently failed
freeze surfaces as a model that trains fine and learns nothing.

Model- and data-agnostic. Point it at whatever you are using:

    python preflight.py
    python preflight.py --llm Qwen/Qwen2.5-3B-Instruct --data data/align_train.json
    python preflight.py --skip-model          # env + disk only, no downloads

Exit code 0 = safe to launch. 1 = something will bite you.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

FAILS, WARNS = [], []
W = 62


def hdr(t):
    print(f"\n{'=' * W}\n{t}\n{'=' * W}")


def ok(t, detail=""):
    print(f"  [ OK ] {t}" + (f"  -- {detail}" if detail else ""))


def warn(t, detail=""):
    WARNS.append(t)
    print(f"  [WARN] {t}" + (f"  -- {detail}" if detail else ""))


def fail(t, detail=""):
    FAILS.append(t)
    print(f"  [FAIL] {t}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ 1. env

def check_env():
    hdr("1. ENVIRONMENT")

    v = sys.version_info
    (ok if (3, 9) <= (v.major, v.minor) < (3, 13) else warn)(
        f"python {v.major}.{v.minor}.{v.micro}")

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                ok("gpu", line.strip())
        else:
            fail("nvidia-smi failed", out.stderr.strip()[:120])
    except FileNotFoundError:
        fail("nvidia-smi not found", "no NVIDIA driver on this box")
    except Exception as e:
        warn("nvidia-smi", str(e)[:120])


# ---------------------------------------------------------------- 2. torch

def check_torch():
    hdr("2. TORCH / CUDA")
    try:
        import torch
    except ImportError:
        fail("torch not installed")
        return None

    ok("torch", torch.__version__)

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False",
             "wrong wheel (cpu build?) or driver mismatch")
        return torch

    ok("cuda runtime", torch.version.cuda or "unknown")

    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]
    ok("compute capability", f"sm_{sm}")

    # The classic silent killer: a wheel compiled without your arch. The GPU
    # is visible, tensors allocate, and the first real kernel dies with
    # "no kernel image is available for execution on this device".
    try:
        archs = torch.cuda.get_arch_list()
        if f"sm_{sm}" in archs:
            ok("arch in wheel", f"sm_{sm} present")
        else:
            fail(f"wheel has no sm_{sm} kernels", f"built for: {archs}")
    except Exception:
        warn("could not read arch list")

    if sm >= 120:
        print("       note: Blackwell (sm_120). Needs CUDA 12.8+ wheels.")
        print("       flash-attn must be built with TORCH_CUDA_ARCH_LIST=12.0")
        print("       vLLM: export VLLM_FLASH_ATTN_VERSION=2 (FA3 unsupported)")
    elif sm < 80:
        print("       note: pre-Ampere. No bf16 tensor cores, no flash-attn.")
        print("       Use --dtype fp16 --attn sdpa")

    # bf16 needs Ampere+. Silently falling back to fp16 on Qwen causes NaN.
    if torch.cuda.is_bf16_supported():
        ok("bf16 supported")
    else:
        warn("bf16 NOT supported", "use fp16 and expect possible NaN on Qwen")

    # Actually run a kernel. is_available() lies more often than you'd think.
    try:
        a = torch.randn(512, 512, device="cuda")
        t0 = time.time()
        for _ in range(20):
            a = a @ a.T
            a = a / a.norm()
        torch.cuda.synchronize()
        ok("matmul on gpu", f"{time.time()-t0:.2f}s for 20 iters")
    except Exception as e:
        fail("gpu matmul failed", str(e)[:150])

    free, total = torch.cuda.mem_get_info()
    ok("vram", f"{free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    if free / total < 0.85:
        warn("vram already in use", "another process may be running")

    return torch


# -------------------------------------------------------------- 3. attention

def check_attn(torch):
    hdr("3. ATTENTION BACKENDS")
    if torch is None:
        return "eager"

    best = "eager"

    try:
        import torch.nn.functional as F
        q = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
        F.scaled_dot_product_attention(q, q, q)
        ok("sdpa", "always safe fallback")
        best = "sdpa"
    except Exception as e:
        warn("sdpa failed", str(e)[:100])

    try:
        import flash_attn
        from flash_attn import flash_attn_func
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16)
        flash_attn_func(q, q, q)
        ok("flash_attention_2", getattr(flash_attn, "__version__", "?"))
        best = "flash_attention_2"
    except ImportError:
        warn("flash-attn not installed", "use --attn sdpa (~15% slower)")
    except Exception as e:
        fail("flash-attn installed but BROKEN", str(e)[:120])
        print("       This is the arch mismatch. Use --attn sdpa and move on;")
        print("       do not burn paid hours rebuilding it mid-session.")

    print(f"\n  --> use --attn {best}")
    return best


# ---------------------------------------------------------------- 4. packages

def check_packages():
    hdr("4. PACKAGES")
    import importlib
    need = ["transformers", "accelerate", "datasets", "PIL", "numpy"]
    opt = ["peft", "trl", "vllm", "bitsandbytes", "sympy", "wandb"]

    for m in need:
        try:
            mod = importlib.import_module(m)
            ok(m, getattr(mod, "__version__", ""))
        except ImportError:
            fail(f"{m} missing")

    present = []
    for m in opt:
        try:
            mod = importlib.import_module(m)
            present.append(f"{m}={getattr(mod, '__version__', '?')}")
        except ImportError:
            pass
    print(f"  optional present: {', '.join(present) or 'none'}")


# ------------------------------------------------------------------ 5. disk

def check_disk(args):
    hdr("5. DISK & DATA")

    for label, p in [("cwd", os.getcwd()), ("output", args.out)]:
        try:
            os.makedirs(p, exist_ok=True)
            u = shutil.disk_usage(p)
            f = u.free / 1e9
            (ok if f > args.min_disk_gb else fail)(
                f"{label} free space", f"{f:.1f} GB at {p}")
        except Exception as e:
            fail(f"{label} unusable", str(e)[:100])

    # Writability. Read-only mounts and full disks both fail here, early.
    try:
        t = os.path.join(args.out, ".preflight")
        with open(t, "wb") as f:
            f.write(os.urandom(1 << 20))
        os.remove(t)
        ok("output dir writable")
    except Exception as e:
        fail("cannot write to output dir", str(e)[:100])

    if not args.data:
        return
    if not os.path.exists(args.data):
        fail("data file missing", args.data)
        return

    import json
    import random
    try:
        rows = json.load(open(args.data, encoding="utf-8"))
    except Exception as e:
        fail("data file unreadable", str(e)[:100])
        return

    ok("data rows", f"{len(rows):,}")
    if rows:
        print(f"       fields: {list(rows[0].keys())}")

    # Sample image paths. A missing file crashes training at step 40,000,
    # which is the worst possible moment to find out.
    if rows and "image" in rows[0]:
        sample = random.sample(rows, min(300, len(rows)))
        miss = [r["image"] for r in sample if not os.path.exists(r["image"])]
        if miss:
            fail(f"{len(miss)}/{len(sample)} sampled images missing",
                 miss[0][:80])
            print("       paths are probably absolute from another machine")
        else:
            ok("image paths resolve", f"{len(sample)} sampled")


# ----------------------------------------------------------------- 6. model

def check_model(args, torch, attn):
    hdr("6. MODEL LOAD + FORWARD + BACKWARD")
    if torch is None or not torch.cuda.is_available():
        warn("skipped, no gpu")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        fail("transformers missing")
        return

    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        tok = AutoTokenizer.from_pretrained(args.llm)
        ok("tokenizer", args.llm)
    except Exception as e:
        fail("tokenizer load failed", str(e)[:150])
        print("       gated repo? run: huggingface-cli login")
        return

    # The single most important check for a VLM: does this LLM accept
    # inputs_embeds instead of input_ids? Splicing image vectors into the
    # embedding stream is the ENTIRE mechanism. Custom architectures
    # (trust_remote_code) sometimes do not wire this up.
    try:
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(
            args.llm, torch_dtype=dt, attn_implementation=attn,
            trust_remote_code=args.trust_remote_code).cuda()
        ok("model load", f"{time.time()-t0:.0f}s, hidden={m.config.hidden_size}")
    except Exception as e:
        fail("model load failed", str(e)[:200])
        return

    try:
        ids = tok("hello world", return_tensors="pt").input_ids.cuda()
        emb = m.get_input_embeddings()(ids)
        out = m(inputs_embeds=emb, labels=ids)
        assert torch.isfinite(out.loss), "loss is not finite"
        ok("inputs_embeds forward", f"loss={out.loss.item():.3f}")
    except Exception as e:
        fail("inputs_embeds NOT supported", str(e)[:200])
        print("       Your VLM design depends on this. Pick another LLM.")
        return

    # A frozen base must still pass gradient THROUGH to a trainable head,
    # or the projector can never learn.
    try:
        for p in m.parameters():
            p.requires_grad = False
        head = torch.nn.Linear(m.config.hidden_size, m.config.hidden_size,
                               dtype=dt).cuda()
        emb2 = head(m.get_input_embeddings()(ids))
        m(inputs_embeds=emb2, labels=ids).loss.backward()
        g = head.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        ok("grad flows through frozen model", f"norm={g.norm().item():.2e}")
    except Exception as e:
        fail("gradient does not reach the head", str(e)[:200])

    try:
        m.gradient_checkpointing_enable()
        ok("gradient checkpointing")
    except Exception as e:
        warn("gradient checkpointing unavailable", str(e)[:100])

    peak = torch.cuda.max_memory_allocated() / 1e9
    ok("peak vram this test", f"{peak:.1f} GB")
    del m
    torch.cuda.empty_cache()


# ------------------------------------------------------------- 7. long jobs

def check_longrun():
    hdr("7. LONG-RUN HYGIENE")

    if os.environ.get("TMUX") or os.environ.get("STY"):
        ok("inside tmux/screen")
    else:
        warn("NOT in tmux/screen",
             "a dropped ssh connection kills your run")
        print("       tmux new -s train   (Ctrl-B then D to detach)")

    if shutil.which("tmux") or shutil.which("screen"):
        ok("tmux/screen available")
    else:
        warn("no tmux or screen installed", "apt install tmux")

    for var in ["HF_HOME", "HF_TOKEN", "WANDB_API_KEY"]:
        v = os.environ.get(var)
        if v:
            ok(var, "set" if "TOKEN" in var or "KEY" in var else v)

    if not os.environ.get("HF_HOME"):
        warn("HF_HOME not set",
             "downloads go to ~/.cache, often a small root disk")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="small default: this is a check, not a benchmark")
    ap.add_argument("--data", default=None, help="normalized json to validate")
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--min-disk-gb", type=float, default=30.0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    print("=" * W)
    print("PREFLIGHT".center(W))
    print("=" * W)

    check_env()
    torch = check_torch()
    attn = check_attn(torch)
    check_packages()
    check_disk(args)
    if not args.skip_model:
        check_model(args, torch, attn)
    check_longrun()

    hdr("SUMMARY")
    if FAILS:
        print(f"  {len(FAILS)} FAILURE(S) -- do not start a long run:")
        for f in FAILS:
            print(f"    - {f}")
    if WARNS:
        print(f"  {len(WARNS)} warning(s):")
        for w in WARNS:
            print(f"    - {w}")
    if not FAILS and not WARNS:
        print("  all clear")
    elif not FAILS:
        print("\n  no blockers. safe to launch.")

    print(f"\n  suggested: --attn {attn}")
    print("=" * W)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()