import sys

import tiktoken
import torch

from build_llm.chapter04_llm_architecture.deep_neural_network import deep_nn_gradients
from build_llm.chapter04_llm_architecture.feed_forward import FeedForward
from build_llm.chapter04_llm_architecture.generate_text import generate_text_impl
from build_llm.chapter04_llm_architecture.gpt_model import gpt_model_impl
from build_llm.chapter04_llm_architecture.transformer_block import (
    transformer_block_impl,
)

from .config import GPT2_XL_CONFIG, GPT_CONFIG_124M
from .dummy_gpt_model import DummyGPTModel
from .layer_norm import norm_example


def dummy_model_impl(batch: torch.Tensor):
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print("Output:\n", logits)


def layer_norm_showcase():
    norm_example()


def feed_forward():
    ffn = FeedForward(GPT_CONFIG_124M)

    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(
        f"Torch Rand(2, 3, 768) Gone through FeedForward(Linear -> GELU -> Linear):\nOutput Shape: {out.shape}\nOutput:\n{out}"
    )


def deep_neural_network_showcase():
    deep_nn_gradients()


def transformer_block_showcase():
    transformer_block_impl()


def generate_text_showcase():
    generate_text_impl(cfg=GPT_CONFIG_124M)


def main():
    tiktok = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(data=tiktok.encode(txt1)))
    batch.append(torch.tensor(data=tiktok.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    print("Dummy GPT Model:\n--------------------------------------")
    dummy_model_impl(batch=batch)

    print(
        "\n--------------------------------------\nLayer Norm:\n--------------------------------------"
    )
    layer_norm_showcase()

    # plot_activation_func()

    print(
        "\n--------------------------------------\nFeedForward:\n--------------------------------------"
    )
    feed_forward()

    print(
        "\n--------------------------------------\nDeep Neural Network:\n--------------------------------------"
    )
    deep_nn_gradients()

    print(
        "\n--------------------------------------\nTransformer Block:\n--------------------------------------"
    )
    transformer_block_showcase()

    print(
        "\n--------------------------------------\nGPT Model:\n--------------------------------------"
    )
    gpt_model_impl(batch=batch, cfg=GPT_CONFIG_124M)
    # gpt_model_impl(batch=batch, cfg=GPT2_XL_CONFIG)

    print(
        "\n--------------------------------------\nGenerate Text Simple:\n--------------------------------------"
    )
    generate_text_showcase()


if __name__ == "__main__":
    sys.exit(main())
