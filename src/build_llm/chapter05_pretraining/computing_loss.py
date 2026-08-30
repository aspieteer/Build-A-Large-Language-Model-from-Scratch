import tiktoken
from torch import nn, torch
from torch.utils.data import DataLoader

from build_llm.chapter02_data_sampling.dataloader import create_dataloader_v1
from build_llm.chapter04_llm_architecture.config import GPT_CONFIG_124M_SHORTEN
from build_llm.chapter05_pretraining.utils import calc_loss_batch


def preparing_data(
    text_data: str, tokenizer: tiktoken.Encoding
) -> tuple[DataLoader, DataLoader]:
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text=text_data))
    print("\nCharacters:", total_characters)
    print("Tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(total_characters * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        tokenizer=tokenizer,
        text=train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M_SHORTEN["context_length"],
        stride=GPT_CONFIG_124M_SHORTEN["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        tokenizer=tokenizer,
        text=val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M_SHORTEN["context_length"],
        stride=GPT_CONFIG_124M_SHORTEN["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("\nTrain loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    return train_loader, val_loader


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    device,
    num_batches: int | None = None,
):
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
