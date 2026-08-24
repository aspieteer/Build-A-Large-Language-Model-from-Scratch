import sys
from importlib.metadata import version

import tiktoken

from .naive_tokenizer import naive_tokenizer
from .simple_tokenizer import SimpleTokenizerV1, SimpleTokenizerV2


def main():
    print(f"tiktoken version: {version('tiktoken')}")

    vocab = naive_tokenizer()

    tokenizer1 = SimpleTokenizerV1(vocab)

    text = """
    "It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride.
    """

    ids = tokenizer1.encode(text)

    print(ids)
    print(tokenizer1.decode(ids))

    # v2
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = f"{text1} <|endoftext|> {text2}"
    print(text)

    tokenizer2 = SimpleTokenizerV2(vocab)

    ids = tokenizer2.encode(text)
    print(ids)
    print(tokenizer2.decode(ids))

    # tiktoken
    print("\n===== Using tiktoken =====")
    tiktok = tiktoken.get_encoding("gpt2")

    text = """Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPalace."""

    integers = tiktok.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    strings = tiktok.decode(integers)
    print(strings)

    print("\n===== Encoding the whole text =====")
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tiktok.encode(raw_text)
    print("Encoding length: ", len(enc_text))
    print(enc_text[:50])

    return 0


if __name__ == "__main__":
    sys.exit(main())
