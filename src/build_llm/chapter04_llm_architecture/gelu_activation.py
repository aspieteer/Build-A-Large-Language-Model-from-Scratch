import matplotlib.pyplot as plt
import torch
from torch import nn


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


def plot_activation_func():
    gelu, relu = GELU(), nn.ReLU()

    x = torch.linspace(start=-3, end=3, steps=100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))

    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(label=f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}{x}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
