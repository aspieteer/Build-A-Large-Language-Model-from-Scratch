import tiktoken
import torch

from .dataloader import create_dataloader_v1


def token_embedding(
    tokenizer: tiktoken.Encoding,
    vocab_size: int,
    output_dim: int,
    text: str,
    batch_size: int = 8,
    max_length: int = 4,
    stride: int = 4,
    shuffle: bool = False,
):
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    dataloader = create_dataloader_v1(
        tokenizer=tokenizer,
        text=text,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
    )
    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)

    return token_embeddings


def position_embedding(context_length: int, output_dim: int):
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    return pos_embeddings
