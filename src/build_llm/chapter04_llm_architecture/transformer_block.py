"""
Now it's time to connect attention and linear layers in a transformer block
[ ----- LayerNorm1 -> Masked multi-head attention -> Dropout - + -----> LayerNorm2 -> Feed forward -> Dropout - + -----> ... ]
[   |                                                          |   |                                            |
[   |----------------------------------------------------------|   |--------------------------------------------|
"""

import torch
from torch import nn

from build_llm.chapter03_self_attention_mechanism.self_attention import (
    MultiHeadAttention,
)
from build_llm.chapter04_llm_architecture.config import GPT_CONFIG_124M
from build_llm.chapter04_llm_architecture.layer_norm import LayerNorm

from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.attn = MultiHeadAttention(
            dim_in=cfg["emb_dim"],
            dim_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(p=cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


def transformer_block_impl():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)

    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print(f"Input shape: {x.shape}\nInput:\n{x}")
    print(f"Output shape: {output.shape}\nOutput:\n{output}")
