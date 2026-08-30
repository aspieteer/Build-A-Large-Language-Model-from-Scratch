import torch
from torch import nn

from build_llm.chapter04_llm_architecture.layer_norm import LayerNorm
from build_llm.chapter04_llm_architecture.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=cfg["context_length"], embedding_dim=cfg["emb_dim"]
        )
        self.drop_emb = nn.Dropout(p=cfg["drop_rate"])
        # Transformer Block layer
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # The last Normalization layer for the final output
        self.final_norm = LayerNorm(emb_dim=cfg["emb_dim"])
        # Convert the embedding space back to vocabulary space
        self.out_head = nn.Linear(
            in_features=cfg["emb_dim"], out_features=cfg["vocab_size"], bias=False
        )

    def forward(self, in_token_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_token_idx.shape

        tok_embeds = self.tok_emb(in_token_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_token_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


def gpt_model_impl(batch: torch.Tensor, cfg: dict):
    torch.manual_seed(123)
    model = GPTModel(cfg)

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print("Output:\n", out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")

    print("\nToken embedding layer shape:", model.tok_emb.weight.shape)
    print("Position embedding layer shape:", model.pos_emb.weight.shape)
    print(
        "Number of parameters in Transformer Blocks:",
        sum(p.numel() for p in model.trf_blocks.parameters()),
    )
    print("Inside one of a transformer block:")
    trf_block0 = model.trf_blocks[0]
    print(trf_block0.attn)
    print(
        "Multi-Head-Attention module's number of params:",
        sum(p.numel() for p in trf_block0.attn.parameters()),
    )
    print(trf_block0.ff)
    print(
        "Feed-Forward module's number of params:",
        sum(p.numel() for p in trf_block0.ff.parameters()),
    )
    print(
        "Two Layer Normalization modules:",
        sum(p.numel() for p in trf_block0.norm1.parameters()) * 2,
    )
    print("\nOutput layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(
        f"Number of trainable parameters (considering weight tying): {total_params_gpt2:,}"
    )

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\nTotal size of the model: {total_size_mb:.2f} MB")
