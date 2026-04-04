from __future__ import annotations

from typing import Optional

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


class NewsEncoder(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_subcategories: int,
        vocab_size: int,
        embedding_dim: int = 128,
        category_dim: int = 32,
        subcategory_dim: int = 32,
        token_dim: int = 128,
        num_entities: int = 0,
        entity_dim: int = 64,
        use_entities: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_entities = use_entities

        self.category_embedding = nn.Embedding(num_categories, category_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories, subcategory_dim, padding_idx=0)
        self.token_embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)

        input_dim = category_dim + subcategory_dim + token_dim

        if self.use_entities:
            self.entity_embedding = nn.Embedding(num_entities, entity_dim, padding_idx=0)
            input_dim += entity_dim
        else:
            self.entity_embedding = None

        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
        title_token_ids: torch.Tensor,
        title_token_mask: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        category_ids: [B, L] or [B, K]
        title_token_ids: [B, L, T] or [B, K, T]
        returns:
            news_vecs: [B, L, D] or [B, K, D]
        """
        original_shape = category_ids.shape
        flat_size = category_ids.numel()

        flat_category_ids = category_ids.reshape(-1)
        flat_subcategory_ids = subcategory_ids.reshape(-1)
        flat_title_token_ids = title_token_ids.reshape(flat_size, title_token_ids.size(-1))
        flat_title_token_mask = title_token_mask.reshape(flat_size, title_token_mask.size(-1))

        category_vec = self.category_embedding(flat_category_ids)
        subcategory_vec = self.subcategory_embedding(flat_subcategory_ids)

        title_token_vecs = self.token_embedding(flat_title_token_ids)
        title_vec = masked_mean_pool(title_token_vecs, flat_title_token_mask)

        features = [category_vec, subcategory_vec, title_vec]

        if self.use_entities:
            if entity_ids is None or entity_mask is None:
                raise ValueError("entity_ids and entity_mask are required when use_entities=True")

            flat_entity_ids = entity_ids.reshape(flat_size, entity_ids.size(-1))
            flat_entity_mask = entity_mask.reshape(flat_size, entity_mask.size(-1))
            entity_vecs = self.entity_embedding(flat_entity_ids)
            entity_vec = masked_mean_pool(entity_vecs, flat_entity_mask)
            features.append(entity_vec)

        news_input = torch.cat(features, dim=-1)
        news_vec = self.proj(news_input)
        return news_vec.view(*original_shape, -1)


class UserEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, history_news_vecs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """
        history_news_vecs: [B, H, D]
        history_mask: [B, H]
        returns:
            user_vec: [B, D]
        """
        return masked_mean_pool(history_news_vecs, history_mask)


class ClickScorer(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = embedding_dim * 4 + 1

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_vec: torch.Tensor, candidate_news_vecs: torch.Tensor) -> torch.Tensor:
        """
        user_vec: [B, D]
        candidate_news_vecs: [B, K, D]
        returns:
            logits: [B, K]
        """
        user_vec = user_vec.unsqueeze(1).expand_as(candidate_news_vecs)

        mul_feat = user_vec * candidate_news_vecs
        abs_diff_feat = torch.abs(user_vec - candidate_news_vecs)
        dot_feat = mul_feat.sum(dim=-1, keepdim=True)

        cross_features = torch.cat(
            [user_vec, candidate_news_vecs, mul_feat, abs_diff_feat, dot_feat],
            dim=-1,
        )
        logits = self.mlp(cross_features).squeeze(-1)
        return logits


class BaselineNewsRecModel(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_subcategories: int,
        vocab_size: int,
        news_category_ids: torch.Tensor,
        news_subcategory_ids: torch.Tensor,
        news_title_token_ids: torch.Tensor,
        news_title_mask: torch.Tensor,
        num_entities: int = 0,
        news_entity_ids: Optional[torch.Tensor] = None,
        news_entity_mask: Optional[torch.Tensor] = None,
        embedding_dim: int = 128,
        category_dim: int = 32,
        subcategory_dim: int = 32,
        token_dim: int = 128,
        entity_dim: int = 64,
        use_entities: bool = False,
        scorer_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_entities = use_entities

        self.news_encoder = NewsEncoder(
            num_categories=num_categories,
            num_subcategories=num_subcategories,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            category_dim=category_dim,
            subcategory_dim=subcategory_dim,
            token_dim=token_dim,
            num_entities=num_entities,
            entity_dim=entity_dim,
            use_entities=use_entities,
            dropout=dropout,
        )
        self.user_encoder = UserEncoder()
        self.scorer = ClickScorer(
            embedding_dim=embedding_dim,
            hidden_dim=scorer_hidden_dim,
            dropout=dropout,
        )

        self.register_buffer("news_category_ids", news_category_ids.long())
        self.register_buffer("news_subcategory_ids", news_subcategory_ids.long())
        self.register_buffer("news_title_token_ids", news_title_token_ids.long())
        self.register_buffer("news_title_mask", news_title_mask.bool())

        if self.use_entities:
            if news_entity_ids is None or news_entity_mask is None:
                raise ValueError("news entity tensors are required when use_entities=True")
            self.register_buffer("news_entity_ids", news_entity_ids.long())
            self.register_buffer("news_entity_mask", news_entity_mask.bool())
        else:
            self.news_entity_ids = None
            self.news_entity_mask = None

    def lookup_news_features(self, news_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {
            "category_ids": self.news_category_ids[news_ids],
            "subcategory_ids": self.news_subcategory_ids[news_ids],
            "title_token_ids": self.news_title_token_ids[news_ids],
            "title_token_mask": self.news_title_mask[news_ids],
        }

        if self.use_entities:#如果模型启用了 entity 特征，就把“这些新闻对应的实体特征”也一起查出来。
            features["entity_ids"] = self.news_entity_ids[news_ids]
            features["entity_mask"] = self.news_entity_mask[news_ids]

        return features

    def encode_news_batch(self, news_ids: torch.Tensor) -> torch.Tensor:
        features = self.lookup_news_features(news_ids)

        return self.news_encoder(
            category_ids=features["category_ids"],
            subcategory_ids=features["subcategory_ids"],
            title_token_ids=features["title_token_ids"],
            title_token_mask=features["title_token_mask"],
            entity_ids=features.get("entity_ids"),
            entity_mask=features.get("entity_mask"),
        )

    def forward(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        history_ids: [B, H]
        history_mask: [B, H]
        candidate_ids: [B, K]
        candidate_mask: [B, K]
        """
        history_news_vecs = self.encode_news_batch(history_ids)
        candidate_news_vecs = self.encode_news_batch(candidate_ids)

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        logits = self.scorer(user_vec, candidate_news_vecs)
        logits = logits.masked_fill(~candidate_mask, -1e9)

        return {
            "logits": logits,
            "user_vec": user_vec,
            "history_news_vecs": history_news_vecs,
            "candidate_news_vecs": candidate_news_vecs,
        }
