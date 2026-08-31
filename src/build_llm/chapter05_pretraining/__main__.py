import sys

import tiktoken
from torch import nn, torch
from torch.utils.data import DataLoader

from build_llm.chapter04_llm_architecture.config import GPT_CONFIG_124M_SHORTEN
from build_llm.chapter04_llm_architecture.gpt_model import GPTModel
from build_llm.chapter05_pretraining.computing_loss import (
    calc_loss_loader,
    preparing_data,
)
from build_llm.chapter05_pretraining.pretraining import plot_losses, train_model_simple
from build_llm.chapter05_pretraining.text_eval import text_eval_impl


def calc_loss_impl(
    text_data: str,
    tokenizer: tiktoken.Encoding,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device,
):
    model.to(device=device)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            data_loader=train_loader, model=model, device=device
        )
        val_loss = calc_loss_loader(data_loader=val_loader, model=model, device=device)

    print("\nTraining loss:", train_loss)
    print("\nValidation loss:", val_loss)


def train_model_impl(
    model: nn.Module,
    device,
    tokenizer: tiktoken.Encoding,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    # plot_losses(
    #     epochs_seen=epochs_tensor,
    #     tokens_seen=tokens_seen,
    #     train_losses=train_losses,
    #     val_losses=val_losses,
    # )


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")

    # Let the model has seeded weights for kind of reproducible implementation curves.
    # torch.manual_seed(123)
    model = GPTModel(cfg=GPT_CONFIG_124M_SHORTEN)

    # gen_text(model=model, tokenizer=tokenizer, cfg=GPT_CONFIG_124M_SHORTEN)

    # text_eval_impl(model=model, tokenizer=tokenizer)

    file_path = "the-verdict1.txt"
    with open(file_path, mode="r", encoding="utf-8") as f:
        text_data = f.read()

    train_loader, val_loader = preparing_data(text_data=text_data, tokenizer=tokenizer)

    # calc_loss_impl(
    #     text_data=text_data,
    #     tokenizer=tokenizer,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     model=model,
    #     device=device,
    # )

    train_model_impl(
        model=model,
        device=device,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    sys.exit(main())
