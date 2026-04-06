from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    sequence: [B, L, D]
    mask: [B, L], True means a valid position.
    """
    mask = mask.float()
    masked_sequence = sequence * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked_sequence.sum(dim=1) / denom


def build_position_ids(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: [B, L], True means a valid position.
    returns:
        position_ids: [B, L], padding positions are 0, valid positions are 1..N
    """
    return torch.cumsum(mask.long(), dim=1) * mask.long()


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        sequence: [B, L, D]
        mask: [B, L], True means a valid position.
        """
        batch_size, seq_len, hidden_dim = sequence.shape
        pooled = sequence.new_zeros((batch_size, hidden_dim))
        weights = sequence.new_zeros((batch_size, seq_len))

        valid_rows = mask.any(dim=1)
        if valid_rows.any():
            valid_sequence = sequence[valid_rows]
            valid_mask = mask[valid_rows]

            scores = self.score(valid_sequence).squeeze(-1)
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
            valid_weights = torch.softmax(scores, dim=1)
            valid_weights = valid_weights.masked_fill(~valid_mask, 0.0)
            valid_weights = valid_weights / valid_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

            pooled[valid_rows] = (valid_sequence * valid_weights.unsqueeze(-1)).sum(dim=1)
            weights[valid_rows] = valid_weights

        if return_weights:
            return pooled, weights

        return pooled
