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


class ImpressionPairwiseLoss(nn.Module):
    """
    Pairwise ranking loss inside each impression.

    For every impression, compare each clicked candidate against each non-clicked
    candidate and encourage:
        score(clicked) > score(non_clicked)
    """

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
        pos_mask = (labels == 1) & valid_mask
        neg_mask = (labels == 0) & valid_mask

        # [B, K, K], only keep "clicked vs non-clicked" pairs inside each impression.
        pair_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        pair_mask_float = pair_mask.float()

        # diff[b, i, j] = score(clicked_i) - score(non_clicked_j)
        diff = logits.unsqueeze(2) - logits.unsqueeze(1)
        pairwise_loss = F.softplus(-diff)

        valid_pair_count = pair_mask_float.sum()
        if valid_pair_count <= 0:
            return logits.sum() * 0.0

        masked_loss = pairwise_loss * pair_mask_float

        if self.reduction == "sum":
            return masked_loss.sum()

        if self.reduction == "mean":
            return masked_loss.sum() / valid_pair_count

        raise ValueError(f"Unsupported reduction: {self.reduction}")


class ListNetTop(nn.Module):
    """
    ListNet Top-1 loss.

    For each impression, convert candidate logits into a Top-1 probability
    distribution with softmax, convert labels into a target Top-1 distribution,
    and minimize the cross-entropy between the two distributions.
    """

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
        valid_impressions = valid_mask.any(dim=1)

        if not valid_impressions.any():
            raise ValueError("No valid impressions found for ListNetTop loss.")

        mask_fill_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~valid_mask, mask_fill_value)
        masked_labels = labels.float().masked_fill(~valid_mask, mask_fill_value)

        pred_log_probs = F.log_softmax(masked_logits, dim=1).masked_fill(~valid_mask, 0.0)
        target_probs = F.softmax(masked_labels, dim=1).masked_fill(~valid_mask, 0.0)

        per_impression_loss = -(target_probs * pred_log_probs).sum(dim=1)
        valid_losses = per_impression_loss[valid_impressions]

        if self.reduction == "sum":
            return valid_losses.sum()

        if self.reduction == "mean":
            return valid_losses.mean()

        raise ValueError(f"Unsupported reduction: {self.reduction}")
