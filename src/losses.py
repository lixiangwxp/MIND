from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = candidate_mask.bool()

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask].float()

        if valid_logits.numel() == 0:
            raise ValueError("No valid candidates found for BCE loss.")

        return F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            reduction=self.reduction,
        )
