import math

import torch
from torch import double, nn


class SelfAttention_v1(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_key = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_value = nn.Parameter(torch.rand(dim_in, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        # softmax(attn_scores / sqrt(dim_keys))
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(
            in_features=dim_in, out_features=dim_out, bias=qkv_bias
        )
        self.W_key = nn.Linear(in_features=dim_in, out_features=dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(
            in_features=dim_in, out_features=dim_out, bias=qkv_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


class CausalAttention(nn.Module):
    # Type declaration only: the actual attribute is set by register_buffer()
    # below (stored in self._buffers and resolved via Module.__getattr__).
    # Without it, pyright infers `self.mask` as `Tensor | Module` and rejects
    # slicing because Module has no __getitem__.
    mask: torch.Tensor

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.context_length = context_length
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, dim_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    context_length=context_length,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.context_length = context_length
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, dim_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # split dim_out into two dimensions: [num_heads * head_dim]
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        # From [b, num_tokens, num_heads, head_dim] to [b, num_heads, num_tokens, head_dim]
        queries = queries.transpose(-3, -2)
        keys = keys.transpose(-3, -2)
        values = values.transpose(-3, -2)
        # queries @(matmul) keys.transpose
        attn_scores = queries @ keys.transpose(-2, -1)
        # mask subsequent tokens, implementing causal attention
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # calculate context vector and transform back its dimension order
        context_vec = (attn_weights @ values).transpose(-3, -2)

        context_vec = context_vec.contiguous().view(batch, num_tokens, self.dim_out)
        # Adds an optional linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec


class ManualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.weight.shape[1]

        bound = 1 / math.sqrt(fan_in)

        # In-place init on an nn.Parameter (a grad-requiring leaf) must run
        # under no_grad, otherwise autograd raises:
        # "a leaf Variable that requires grad is being used in an in-place operation."
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input @ weights.T + bias
        out = input @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
