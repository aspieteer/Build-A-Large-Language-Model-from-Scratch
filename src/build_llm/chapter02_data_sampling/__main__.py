import sys

import tiktoken

from .embedding import (
    position_embedding,
    token_embedding,
)


def main():
    tiktok = tiktoken.get_encoding("gpt2")
    vocab_size = tiktok.max_token_value + 1

    output_dim = 256
    batch_size = 8
    max_length = 4

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    token_embeddings = token_embedding(
        tokenizer=tiktok,
        vocab_size=vocab_size,
        output_dim=output_dim,
        text=raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    pos_embeddings = position_embedding(
        context_length=max_length, output_dim=output_dim
    )

    input_embeddings = token_embeddings + pos_embeddings

    print(
        f"\n====================================================\nToken embedding shape (given output_dim = {output_dim}):\n{token_embeddings.shape}"
    )
    print(
        f"\nPosition embedding shape (given output_dim = {output_dim}):\n{pos_embeddings.shape}"
    )

    print(
        "\nTotal Input embeddings are the sum of position_embeddings and token_embeddings with its each batch.\nFinal embeddings shape: ",
        input_embeddings.shape,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
