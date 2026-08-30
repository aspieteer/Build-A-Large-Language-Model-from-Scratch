import tiktoken
from torch import nn

from build_llm.chapter04_llm_architecture.generate_text import generate_text_simple

from .utils import text_to_token_ids, token_ids_to_text


def gen_text(model: nn.Module, tokenizer: tiktoken.Encoding, cfg: dict):
    model.eval()

    start_context = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text=start_context, tokenizer=tokenizer),
        max_new_tokens=10,
        context_size=cfg["context_length"],
    )

    print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))
