'''
1. Main Goal is to change the normalized Json into Tenesors 

2. Make sure that each embeddings of the vision gets its own <vision> token and then questions and answer

3. Then we get pixel values then -> image tenors -> Labels -> the [-100] This helps us not allow prompts to contibute to the trianing loss


How to answer it 

First it Inserts the correct number of image tokens into teh text sequence. The vision encoder prodiced a fixed numbner of N of visual mebeddings, so we need N corresponding psoitoins in the language models input where those emebddings can be inserted 

Second, it creates the trianing labels and makss the prompt tokens with -100 In pytorch -100 tells the cross entropy loss to ignore that position this means the model dosetn get penalized for preiccing the users question or the prompt , the loss is clcualted only on the assitant answers 

So the collator makes sure that the vision embeddings align correctly with text sequwnce and ensures that training focuses on generating the answer rather than trying to reproduce the prompt 

Necassary because the Modal may start predicting the proempt than be able to get the deseired output 

'''

import json 
import torch
from PIL import Image
from torch.utils.data import Dataset

class AlignDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as file:
            self.rows = json.load(file)
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self,i):
        r = self.rows[i]
        return {
            "image": Image.open(r["image"]).convert("RGB"),
            "prompt": r.get("instruction", r["instruction"]),
            "target": r["output"],
        }

class Collator:
    """
    1. Why need this the simplest way to explain this is that , lets say the Vision Encoder example converts the image into 196 x Vsion_Hidden_dims , the projection layer converts thse unto the dimensionaloty expedcterd by the LLm
    now the LLm needs 196 dims wher ethe visula embeddings can be inserted , that why we have <|image_pad|>
    """
    # For every image we expect 196 tokens 
    # Assert statement is tp make sure that its <image_roken> than rather a <unk> from the otkenizer

    def __init__(
        self,
        tokenizer,
        processor,
        n_visual_tokens=196,
        image_token="<|image_pad|>",
        max_len=1024,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.n = n_visual_tokens
        self.image_token = image_token
        self.max_len = max_len
        self.image_id = tokenizer.convert_tokens_to_ids(image_token)
    
        assert self.image_id is not None and self.image_id != tokenizer.unk_token_id, \
            f"{image_token} is not in the tokenizer's vocabulary"
    
    def __call__(self,batch):
        """
        batch = [
    {
        "image": image1,
        "prompt": "What is this?",
        "target": "A dog."
    },
    {
        "image": image2,
        "prompt": "What is this?",
        "target": "A cat."
    }
]
        """
        ids_list = []
        lab_list = []
        for ex in batch:
            # Creates the image slots for N copies 
            """
            Looks like this <|image_pad|><|image_pad|><|image_pad|><|image_pad|>
            What is this image?

            because the Modal expects lets say self.n = 4 then expects 4 image token paddings
            the actual image is made into pixels and passed into the Langaude head as pixels 
            """
            prompt = self.image_token * self.n + "\n" + ex["prompt"]
            chat = self.tokenizer.apply_chat_template(
                # HF template user and assitant type 
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                # Add generation prompt tells the Modal that we need to generate the assitant reposnse 
            )

            # Tokenzize the prompt 
            p_ids = self.tokenizer(chat, add_special_tokens=False).input_ids
            # becomes "<image> What is this?" -> [151655, 151655, 151655, 151655, 389, 374, 420, 30] -> These are the input ids

            a_ids = self.tokenizer(
                ex["target"] + self.tokenizer.eos_token,
                add_special_tokens=False,
            ).input_ids
            # Suppose target is "A dog." -> becomes [32, 5821, 13, EOS] -> henceboth the result and the answer are tokenized 

            room = self.max_len - len(p_ids) # We want the maximum space present after this that would be 1024 - 800 = 224
            if room < 16:
                raise ValueError(
                    f"prompt is {len(p_ids)} tokens, max_len {self.max_len} "
                    "leaves no room for an answer. Raise max_len."
                )
            a_ids = a_ids[:room] # Keep atmost 225 tokens why not truncate them completly we dont do that to prompts becuse theu might eat of fthe visual embeddings
            ids_list.append(p_ids + a_ids)
            # Now append the mask as well for p_ids
            lab_list.append([-100] * len(p_ids) + a_ids)

            # Padding the batch 
        m = max(len(x) for x in ids_list)
        pad = self.tokenizer.pad_token_id

        batch_out = {
                # we pad the shorter sequences 
                "input_ids" : torch.tensor(
                    [x + [pad] * (m - len(x)) for x in ids_list]
                ),
                # For real token make the mask = 1 and then for PAD make it 0
                """
                input_ids:

                [A][dog][is][here][PAD][PAD]

                attention_mask:

                [1] [1] [1] [1]  [0]  [0]
                """
                "attention_mask": torch.tensor(
                    [[1] * len(x) + [0] * (m - len(x)) for x in ids_list]),

                "labels": torch.tensor([
                    x + [-100] * (m - len(x)) for x in lab_list]),
                
                "pixel_values": torch.stack([
                    self.processor(images=ex["image"], return_tensors="pt").pixel_values[0]
                    for ex in batch
                ]),
        }

        got = int((batch_out["input_ids"] == self.image_id).sum())
        want = len(batch) * self.n
        assert got == want, f"{got} image tokens, expected {want}"
 
        return batch_out

def verify(collator, dataset, tokenizer):
    """Run this before every training run. Ten seconds, saves days."""
    b = collator([dataset[0], dataset[1]])
 
    print("input_ids   ", tuple(b["input_ids"].shape))
    print("pixel_values", tuple(b["pixel_values"].shape))
    print("image tokens", int((b["input_ids"] == collator.image_id).sum()),
          "(expect", 2 * collator.n, ")")
 
    print("\n--- what the model reads (image tokens collapsed) ---")
    text = tokenizer.decode(b["input_ids"][0])
    print(text.replace(collator.image_token * collator.n,
                       f"[{collator.n} x IMAGE]")[:600])
 
    print("\n--- what the model is GRADED on (must be answer only) ---")
    scored = [t for t in b["labels"][0].tolist() if t != -100]
    print(tokenizer.decode(scored)[:400])
 
    print(f"\ngraded on {len(scored)} of {b['labels'].shape[1]} positions")


    """
                     RAW DATA
                    │
          ┌─────────┴─────────┐
          │                   │
        IMAGE              TEXT DATA
          │              prompt + target
          │                   │
          ▼                   ▼
   Image Processor       Tokenizer
          │                   │
          ▼                   ▼
   pixel_values          input_ids
          │                   │
          ▼                   │
   Vision Encoder             │
          │                   │
          ▼                   │
   N visual features          │
          │                   │
          ▼                   │
   Projection Layer           │
          │                   │
          ▼                   │
   N visual embeddings        │
          │                   │
          └─────────┬─────────┘
                    ▼
             ┌─────────────┐
             │     LLM     │
             │             │
             │ [V1...VN]   │
             │ [prompt]    │
             │ [answer]    │
             └──────┬──────┘
                    │
                    ▼
               LOSS ONLY ON
                  ANSWER
    """