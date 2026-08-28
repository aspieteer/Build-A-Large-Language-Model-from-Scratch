import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


def norm_example():
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    batch_example = torch.randn(2, 5)
    # nn Layer stacked with a Linear layer, followed by a ReLU activation layer
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

    out = layer(batch_example)
    print("torch.size(2, 5) through Linear(5, 6) and ReLU:\n", out)

    mean = out.mean(
        dim=-1, keepdim=True
    )  # keepdim keeps result the same dimension as calculated before
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print(
        "=========================================================================\nNormalized layer outputs:\n",
        out_norm,
    )
    print("Mean:\n", mean)
    print("Variance:\n", var)

    emb_dim = 5
    ln = LayerNorm(emb_dim=emb_dim)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
    print(
        "=========================================================================\nLayerNorm outputs:\n",
        out_norm,
    )
    print("Mean:\n", mean)
    print("Variance:\n", var)
