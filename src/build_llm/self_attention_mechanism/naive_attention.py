import torch


def softmax_naive(x: torch.Tensor):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def calc_context_vec_for_one_token(input_embeddings: torch.Tensor, index: int):
    # Assert that 0 <= index < length of the list
    assert 0 <= index < len(input_embeddings), f"Index {index} out of bounds"

    query = input_embeddings[index]

    # For each input embedding in position, calculate the dot product with query token embedding.
    # Result: torch.Size([1, Length])
    attn_scores_i = torch.empty(input_embeddings.shape[0])
    for i, x_i in enumerate(input_embeddings):
        attn_scores_i[i] = torch.dot(x_i, query)

    print("Attention score:\n", attn_scores_i)

    # Get the output from softmax function for normalization
    attn_weights_i = torch.softmax(attn_scores_i, dim=0)
    print("Attention weights:\n", attn_weights_i)
    print("Sum:", attn_weights_i.sum())

    context_vec_i = torch.zeros(query.shape)
    for i, x_i in enumerate(input_embeddings):
        context_vec_i += attn_weights_i[i] * x_i

    print("Context Vector:\n", context_vec_i)


def calc_whole_context_vecs(input_embeddings: torch.Tensor):
    attn_scores = input_embeddings @ input_embeddings.T
    print("Attention score:\n", attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("Attention weights:\n", attn_weights)
    print("Sum:", attn_weights.sum(dim=-1))

    whole_context_vecs = attn_weights @ input_embeddings
    print("Context Vector:\n", whole_context_vecs)
