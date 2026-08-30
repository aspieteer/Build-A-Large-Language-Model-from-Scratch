import tiktoken
from torch import nn, torch

from .utils import token_ids_to_text


def text_eval_impl(model: nn.Module, tokenizer: tiktoken.Encoding):
    inputs = []
    txt1 = "every effort moves"
    txt2 = "I really like"
    inputs.append(torch.tensor(data=tokenizer.encode(txt1)))
    inputs.append(torch.tensor(data=tokenizer.encode(txt2)))
    inputs = torch.stack(inputs, dim=0)

    print("Inputs token ids:\n", inputs)

    targets = []
    txt3 = " effort moves you"
    txt4 = " really like chocolate"
    targets.append(torch.tensor(data=tokenizer.encode(txt3)))
    targets.append(torch.tensor(data=tokenizer.encode(txt4)))
    targets = torch.stack(targets, dim=0)

    print("Targets token ids:\n", targets)

    with torch.no_grad():
        logits = model(inputs)

    prbs = torch.softmax(logits, dim=-1)
    print("\nProbabilities shape:", prbs.shape)
    print("Probabilities:\n", prbs)

    token_ids_max = torch.argmax(input=prbs, dim=-1, keepdim=True)
    print("\nToken ids with max Probabilities:\n", token_ids_max)

    print(
        f"Targets batch 1: {token_ids_to_text(token_ids=targets[0], tokenizer=tokenizer)}"
    )
    print(
        f"Output batch 1: {token_ids_to_text(token_ids=token_ids_max[0].flatten(), tokenizer=tokenizer)}"
    )

    # choose the first batch
    text_idx = 0
    # select the same ids as the target ids we want from the Probabilities tensor
    target_prbs_1 = prbs[text_idx, [0, 1, 2], targets[text_idx]]
    print("\nText 1 Target token ids:", target_prbs_1)

    # choose the second batch
    text_idx = 1
    # select the same ids as the target ids we want from the Probabilities tensor
    target_prbs_2 = prbs[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2 Target token ids:", target_prbs_2)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    input_prbs = torch.cat((target_prbs_1, target_prbs_2))

    manual_x_entropy(input_prbs=input_prbs)
    torch_x_entropy(logits=logits_flat, targets=targets_flat)


def manual_x_entropy(input_prbs: torch.Tensor):
    log_prbs = torch.log(input=input_prbs)
    print("\nlog(p)s:\n", log_prbs)

    avg_log_prbs = torch.mean(log_prbs)
    print("mean of the log(p)s:\n", avg_log_prbs)

    neg_avg_log_prbs = avg_log_prbs * -1
    print("negative of the mean (the exact cross-entropy value):\n", neg_avg_log_prbs)


def torch_x_entropy(logits: torch.Tensor, targets: torch.Tensor):
    loss = torch.nn.functional.cross_entropy(input=logits, target=targets)
    print("\n Torch built-in cross entropy:\n", loss)
