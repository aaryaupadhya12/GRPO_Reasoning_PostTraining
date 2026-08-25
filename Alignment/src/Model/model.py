"""
model.py -- the VLM itself.
 
The whole idea in one sentence:
    An LLM only ever eats a list of vectors. Normally those vectors come from
    looking up words in a table. We make some vectors from an image instead,
    and paste them into the list. The LLM cannot tell the difference.
 
Three parts:
    vision     frozen, downloaded. image -> 729 vectors of size 1152
    projector  OURS, trained from scratch. 729x1152 -> 196x2048
    llm        frozen (stage 2), LoRA (stage 3). eats vectors, emits text
"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    SiglipVisionModel,
)


class VLMConfig(PretrainedConfig):
    # The own custom configuration class that contructs our VLM 
    # Vision Encoer sdigLip and teh Qwen VL as the pretriand LLm backbone 
    model_type = "mathvlm"

    def __init__(
        self,
        llm_path="Qwen/Qwen2.5-3B-Instruct",
        vision_path="google/siglip2-so400m-patch14-384",
        image_token_id=None,
        n_visual_tokens=196,
        vision_layer=-2,
        **kwargs,
    ):
        self.llm_path = llm_path
        self.vision_path = vision_path
        self.image_token_id = image_token_id
        self.n_visual_tokens = n_visual_tokens
        # Which toekns in input_ids represent image positions 
        self.vision_layer = vision_layer
        # Only use the vision layers the last 2 
        super().__init__(**kwargs)


class Projector(nn.Module):
    """
    Two-layer MLP, per LLaVA-1.5 and MAVIS. Deliberately boring.
 
    The pixel-unshuffle in front cuts 729 tokens to 196 by folding each 2x2
    block of patches into the channel dimension. Nothing is thrown away --
    it moves from the length axis to the channel axis. That is why we can
    afford 196 tokens without a learned bottleneck deciding what to discard.

    IMAGE
  │
  ▼
SigLIP2 Vision Encoder
  │
  │  [B, 729, 1152]
  │
  │  729 visual tokens
  │  each has 1152 features
  ▼
Pixel Unshuffle
  │
  │  [B, 196, 4608]
  │
  │  196 visual tokens
  │  each has 4608 features
  ▼
2-Layer MLP Projector
  │
  │  [B, 196, d_llm]
  ▼
Qwen LLM


    """
    def __init__(self, d_vision, d_llm, n_out):
        super().__init__()
        # d_vision the dimensions comign from teh vision encoder 
        # d_llm -> The embedding dimesnions expected by qwen 
        # n_out -> how many visiola tokens we need - 196 
        d_in = d_vision * 4 # pixel unshuffle where we take A , B , c , d EACH one of there tokens and multiple them together as 2x2 rather than take each one induviosual as its a 2x32 Grid its 4
        # so 4 x 11052 dims becomes 1 x 4608 dims 

        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )
        self.n_out = n_out

    def unshuffle(self, x):
        batch_size, n_patches, hidden_size = x.shape
        height = width = int(round(n_patches ** 0.5))
        assert height * width == n_patches, f"{n_patches} patches must form a square grid"

        x = x.view(batch_size, height, width, hidden_size)
        if height % 2:
            x = F.pad(x, (0, 0, 0, 1, 0, 1))
            height = width = height + 1

        x = x.view(batch_size, height // 2, 2, width // 2, 2, hidden_size)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, (height // 2) * (width // 2), hidden_size * 4)

    def forward(self, x):
        out = self.net(self.unshuffle(x))
        assert out.shape[1] == self.n_out, (
            f"Projector emitted {out.shape[1]} tokens, "
            f"config specifies {self.n_out}"
        )
        return out

        """
        Vision features -> Unshuffle -> [B, 196, 4608] -> MLP -> [B, 196 , llm_dim]
        """

class MathVLM(PreTrainedModel):
    config_class = VLMConfig
    supports_gradient_checkpointing = True

    def __init__(self,config):
        super().__init__(config)
        self.vision = SiglipVisionModel.from_pretrained(config.vision_path)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_path)
        self.projector = Projector(
            self.vision.config.hidden_size,      # 1152
            self.llm.config.hidden_size,         # 2048 for Qwen2.5-3B
            config.n_visual_tokens,
        )

    def freeze_vision(self):
        for parameter in self.vision.parameters():
            parameter.requires_grad = False
        self.vision.eval()

    def freeze_llm(self):
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

    def trainable_report(self):
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        print(f"Trainable {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")
        return trainable

    def encode_images(self, pixel_values):
        with torch.no_grad():
            output = self.vision(pixel_values, output_hidden_states=True)
            features = output.hidden_states[self.config.vision_layer]
        return self.projector(features)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None, **kwargs):
            # Convert the tokens into embeddings 
            embeds = self.llm.get_input_embeddings()(input_ids)
            # Image features -> 2048 and now Qwen text Embeddings re 2048 as well 
            if pixel_values is not None:
                feats = self.encode_images(pixel_values)
            
                mask = input_ids == self.config.image_token_id
                # suppsoe input_ids are 999 and the list is input_ids:[999, 999, 999, 999, 50, 60, 70]
                # The boolean mask results would be [True, True, True, True, False, False, False]
                # Tells pytorch that these are image positons 

                #count the number of image slots 
                n_slots = int(mask.sum())
                n_feats = feats.shape[0] * feats.shape[1]
                assert n_slots == n_feats # coolator and projector should not diaagreee

                embeds = embeds.clone()
                # Puts teh 196 tokens in the sequnce 
                embeds[mask] = feats.reshape(-1, feats.shape[-1]).to(embeds.dtype)
            return self.llm(
                inputs_embeds=embeds,
                attention_mask = attention_mask,
                labels = labels,
                **kwargs,
            )

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)


        """
                             IMAGE
                       │
                       ▼
                 pixel_values
                       │
                       ▼
                 Vision Encoder
                       │
                       ▼
                   Projector
                       │
                       ▼
              [B, 196, 2048]
                       │
                       │
                       │
TEXT                  │
 │                    │
 ▼                    │
input_ids             │
 │                    │
 ▼                    │
Qwen Embedding        │
 │                    │
 ▼                    │
[B, seq_len, 2048]    │
 │                    │
 │  Find <IMG>        │
 │  positions         │
 ▼                    │
[IMG][IMG]...[IMG]    │
 │                    │
 └─────────┬──────────┘
           │
           ▼
   Replace <IMG> embeddings
   with visual embeddings
           │
           ▼
[B, seq_len, 2048]
           │
           ▼
        QWEN
           │
           ▼
        ANSWER

        
        COLLATOR

input_ids:
[IMG][IMG][IMG][What][is][this]
   ↓
"Reserve these positions for image information"


         ↓


QWEN EMBEDDING

[Qwen_IMG][Qwen_IMG][Qwen_IMG][Qwen_What][Qwen_is][Qwen_this]


         ↓


VISION ENCODER + PROJECTOR

[V1][V2][V3]


         ↓


REPLACEMENT

[ V1 ][ V2 ][ V3 ][Qwen_What][Qwen_is][Qwen_this]
   ↑     ↑     ↑
actual image information
        """
            


            




            











