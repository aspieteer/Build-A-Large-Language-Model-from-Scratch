import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from tiktoken import Encoding
from torch import nn
from torch.utils.data import DataLoader

from build_llm.chapter04_llm_architecture.generate_text import generate_text_simple
from build_llm.chapter05_pretraining.computing_loss import calc_loss_loader
from build_llm.chapter05_pretraining.utils import (
    calc_loss_batch,
    text_to_token_ids,
    token_ids_to_text,
)


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: Encoding,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients

            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, Tokens seen {tokens_seen}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(
            model=model, tokenizer=tokenizer, device=device, start_context=start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    eval_iter: int,
):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(
            data_loader=train_loader, model=model, device=device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            data_loader=val_loader, model=model, device=device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: nn.Module, tokenizer: Encoding, device, start_context: str
):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    assert isinstance(context_size, int)

    encoded = text_to_token_ids(text=start_context, tokenizer=tokenizer).to(
        device=device
    )

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def plot_losses(
    epochs_seen: torch.Tensor, tokens_seen: list, train_losses: list, val_losses: list
):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
