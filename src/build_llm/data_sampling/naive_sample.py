import tiktoken


def context_sampling(
    tokenizer: tiktoken.Encoding, enc_sample: list[int], context_size: int = 4
):
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
