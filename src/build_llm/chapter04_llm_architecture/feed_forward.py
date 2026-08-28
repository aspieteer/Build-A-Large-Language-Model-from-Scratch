import torch
from torch import nn

from .gelu_activation import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        # Linear -> GELU -> Linear
        self.layers = nn.Sequential(
            nn.Linear(in_features=cfg["emb_dim"], out_features=4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(in_features=4 * cfg["emb_dim"], out_features=cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
