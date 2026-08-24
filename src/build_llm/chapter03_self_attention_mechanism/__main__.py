import sys

import torch

from .naive_attention import (
    calc_context_vec_for_one_token,
    calc_whole_context_vecs,
)
from .self_attention import (
    CausalAttention,
    MultiHeadAttention,
    MultiHeadAttentionWrapper,
    SelfAttention_v1,
    SelfAttention_v2,
)


def main():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )

    print(
        "\n===============================\n=====   Naive Attention   =====\n==============================="
    )
    calc_context_vec_for_one_token(inputs, index=1)
    calc_whole_context_vecs(inputs)

    print(
        "\n===============================================\n===   Self Attention w/ trainable weights   ===\n==============================================="
    )
    print("[ manual seed: 123, dim_in = 3, dim_out = 2 ]\nSelf Attention v1")
    torch.manual_seed(123)
    dim_in = 3
    dim_out = 2

    sa_v1 = SelfAttention_v1(dim_in=dim_in, dim_out=dim_out)
    print(sa_v1(inputs))

    print("\n[ manual seed: 789, dim_in = 3, dim_out = 2 ]\nSelf Attention v2")
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(dim_in=dim_in, dim_out=dim_out)
    print(sa_v2(inputs))

    # Stack two inputs as batch size 2
    batch = torch.stack((inputs, inputs), dim=0)

    print(
        "\n[ manual seed: 123, dim_in = 3, dim_out = 2, dropout = 0.0 ]\nCausal Attention"
    )
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(
        dim_in=dim_in, dim_out=dim_out, context_length=context_length, dropout=0.0
    )
    context_vec = ca(batch)
    print(f"Context Vector shape: {context_vec.shape}\nContext Vector:\n{context_vec}")
    # delimiter
    print(
        "\n[ manual seed: 123, dim_in = 3, dim_out = 2, dropout = 0.0, num_heads = 2 ]\nMultiHead Attention Wrapper(heads concat in serial)"
    )
    torch.manual_seed(123)
    mhaw = MultiHeadAttentionWrapper(
        dim_in=dim_in,
        dim_out=dim_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=2,
    )
    context_vecs = mhaw(batch)
    print(
        f"Context Vector shape: {context_vecs.shape}\nContext Vector:\n{context_vecs}"
    )
    #
    print(
        "\n[ manual seed: 123, dim_in = 3, dim_out = 16, dropout = 0.0, num_heads = 4 ]\nMultiHead Attention"
    )
    batch = torch.stack((inputs, inputs, inputs, inputs), dim=0)
    batch_size, context_length, dim_in = batch.shape
    dim_out = 16
    torch.manual_seed(123)
    mha = MultiHeadAttention(
        dim_in=dim_in,
        dim_out=dim_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=4,
    )
    context_vecs = mha(batch)
    print(
        f"Context Vector shape: {context_vecs.shape}\nContext Vector:\n{context_vecs}"
    )


if __name__ == "__main__":
    sys.exit(main())
