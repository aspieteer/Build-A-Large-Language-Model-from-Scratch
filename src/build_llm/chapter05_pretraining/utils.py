from tiktoken import Encoding
from torch import nn, torch


def text_to_token_ids(text: str, tokenizer: Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add the batch dimension

    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Encoding) -> str:
    flattened = token_ids.squeeze(0).tolist()  # to list

    return tokenizer.decode(tokens=flattened)


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        input=logits.flatten(0, 1), target=target_batch.flatten()
    )

    return loss
