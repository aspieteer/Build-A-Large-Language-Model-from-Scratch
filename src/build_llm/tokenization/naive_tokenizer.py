import re


def naive_tokenizer() -> dict[str, int]:
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of characters: ", len(raw_text))
    print(raw_text[:99])
    print()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    print(f"Before strip: {preprocessed[:50]}\n")

    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(f"Aftre strip: {preprocessed[:50]}")
    print(f"Total length: {len(preprocessed)}\n")

    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_tokens)
    print(f"Vocabulary size: {vocab_size}\n")

    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i >= 50:
    #         break

    return vocab
