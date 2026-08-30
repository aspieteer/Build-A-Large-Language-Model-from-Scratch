import sys

import tiktoken
from torch import nn, torch

from build_llm.chapter04_llm_architecture.config import GPT_CONFIG_124M_SHORTEN
from build_llm.chapter04_llm_architecture.gpt_model import GPTModel
from build_llm.chapter05_pretraining.computing_loss import (
    calc_loss_loader,
    preparing_data,
)
from build_llm.chapter05_pretraining.text_eval import text_eval_impl


def calc_loss_impl(
    text_data: str,
    tokenizer: tiktoken.Encoding,
    model: nn.Module,
    device,
):
    train_loader, val_loader = preparing_data(text_data=text_data, tokenizer=tokenizer)

    model.to(device=device)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            data_loader=train_loader, model=model, device=device
        )
        val_loss = calc_loss_loader(data_loader=val_loader, model=model, device=device)

    print("\nTraining loss:", train_loss)
    print("\nValidation loss:", val_loss)


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(123)
    model = GPTModel(cfg=GPT_CONFIG_124M_SHORTEN)

    # gen_text(model=model, tokenizer=tokenizer, cfg=GPT_CONFIG_124M_SHORTEN)

    # text_eval_impl(model=model, tokenizer=tokenizer)

    file_path = "the-verdict.txt"
    with open(file_path, mode="r", encoding="utf-8") as f:
        text_data = f.read()

    calc_loss_impl(text_data=text_data, tokenizer=tokenizer, model=model, device=device)


if __name__ == "__main__":
    sys.exit(main())
