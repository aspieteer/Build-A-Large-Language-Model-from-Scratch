import torch
from torch import nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=cfg["context_length"], embedding_dim=cfg["emb_dim"]
        )
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # a placeholder for Transformer Blocks
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # a placeholder for LayerNorm at the final stage
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # Turn tokens from embedding space to vocab space
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


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: torch.Size, eps=1e-5) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
