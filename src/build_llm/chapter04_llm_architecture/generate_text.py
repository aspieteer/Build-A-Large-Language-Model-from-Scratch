import tiktoken
import torch
from torch import nn

from build_llm.chapter04_llm_architecture.gpt_model import GPTModel


def generate_text_simple(
    model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int
):
    assert len(idx.shape) == 2, (
        "idx is a (batch, n_tokens) array of indices in the current context"
    )
    for _ in range(max_new_tokens):
        # idx is a (batch, n_tokens) array of indices in the current context
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Convert logits to 2d: (batch, context_length, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx


def generate_text_impl(cfg: dict):
    tiktok = tiktoken.get_encoding("gpt2")

    start_context = "Hello, I am"
    encoded = tiktok.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model = GPTModel(cfg=cfg)
    model.eval()  # Disable dropout whilst not training
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=cfg["context_length"],
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tiktok.decode(out.squeeze(0).tolist())
    print("Decoded text:", decoded_text)
