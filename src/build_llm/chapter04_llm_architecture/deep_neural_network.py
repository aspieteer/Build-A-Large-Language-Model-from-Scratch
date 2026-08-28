import torch
from torch import nn


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: list, use_shortcut: bool) -> None:
        super().__init__()

        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.GELU())
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x


def deep_nn_gradients():
    sample_input = torch.ones(2, 6, 384)
    layer_sizes = [384, 768, 1536, 1536, 768, 384]

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes=layer_sizes, use_shortcut=False
    )
    print("--- Without Shortcut connection ---")
    print_gradients(model=model_without_shortcut, x=sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes=layer_sizes, use_shortcut=True
    )
    print("\n--- With Shortcut connection ---")
    print_gradients(model=model_with_shortcut, x=sample_input)


def print_gradients(model: nn.Module, x: torch.Tensor):
    output = model(x)
    target = torch.zeros(output.shape)

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
