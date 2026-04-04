from __future__ import annotations

import math
from typing import Optional

import torch


def _get_valid_scores_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[list[float], list[int]]:
    valid_logits = logits[candidate_mask].detach().cpu().tolist()
    valid_labels = labels[candidate_mask].detach().cpu().tolist()
    valid_labels = [int(x) for x in valid_labels]
    return valid_logits, valid_labels


def auc_score(scores: list[float], labels: list[int]) -> Optional[float]:
    pos_scores = [score for score, label in zip(scores, labels) if label == 1]
    neg_scores = [score for score, label in zip(scores, labels) if label == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    correct = 0.0
    total = 0

    for pos_score in pos_scores:
        for neg_score in neg_scores:
            if pos_score > neg_score:
                correct += 1.0
            elif pos_score == neg_score:
                correct += 0.5
            total += 1

    return correct / total


def mrr_score(scores: list[float], labels: list[int]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)

    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            return 1.0 / rank

    return 0.0


def dcg_at_k(labels: list[int], k: int) -> float:
    dcg = 0.0
    for i, label in enumerate(labels[:k]):
        dcg += (2**label - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(scores: list[float], labels: list[int], k: int) -> float:
    ranked_labels = [
        label for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    ]
    ideal_labels = sorted(labels, reverse=True)

    dcg = dcg_at_k(ranked_labels, k)
    idcg = dcg_at_k(ideal_labels, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_batch_ranking_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, float]:
    auc_values = []
    mrr_values = []
    ndcg5_values = []
    ndcg10_values = []

    batch_size = logits.size(0)

    for i in range(batch_size):
        scores_i, labels_i = _get_valid_scores_and_labels(
            logits[i],
            labels[i],
            candidate_mask[i],
        )

        auc_i = auc_score(scores_i, labels_i)
        if auc_i is not None:
            auc_values.append(auc_i)

        mrr_values.append(mrr_score(scores_i, labels_i))
        ndcg5_values.append(ndcg_at_k(scores_i, labels_i, k=5))
        ndcg10_values.append(ndcg_at_k(scores_i, labels_i, k=10))

    return {
        "AUC": sum(auc_values) / len(auc_values) if auc_values else 0.0,
        "MRR": sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        "nDCG@5": sum(ndcg5_values) / len(ndcg5_values) if ndcg5_values else 0.0,
        "nDCG@10": sum(ndcg10_values) / len(ndcg10_values) if ndcg10_values else 0.0,
    }
