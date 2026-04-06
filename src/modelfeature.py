from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model import FeedForwardBlock, MaskedAttentionPooling, build_position_ids, masked_mean_pool


def build_transformer_encoder(
    hidden_dim: int,
    num_layers: int,
    num_attention_heads: int,
    ffn_dim: int,
    dropout: float,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_attention_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class FeatureNewsEncoder(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_subcategories: int,
        vocab_size: int,
        max_title_len: int,
        max_abstract_len: int,
        embedding_dim: int = 128,
        category_dim: int = 32,
        subcategory_dim: int = 32,
        token_dim: int = 128,
        num_entities: int = 0,
        entity_dim: int = 64,
        use_entities: bool = False,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_entities = use_entities

        self.category_embedding = nn.Embedding(num_categories, category_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories, subcategory_dim, padding_idx=0)
        self.token_embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.title_position_embedding = nn.Embedding(max_title_len + 1, token_dim, padding_idx=0)
        self.abstract_position_embedding = nn.Embedding(max_abstract_len + 1, token_dim, padding_idx=0)
        self.title_transformer = build_transformer_encoder(
            hidden_dim=token_dim,
            num_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=dropout,
        )
        self.abstract_transformer = build_transformer_encoder(
            hidden_dim=token_dim,
            num_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=dropout,
        )
        self.title_pool = MaskedAttentionPooling(token_dim)
        self.abstract_pool = MaskedAttentionPooling(token_dim)
        self.text_dropout = nn.Dropout(dropout)

        input_dim = category_dim + subcategory_dim + token_dim + token_dim
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

    def encode_text_sequence(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        position_embedding: nn.Embedding,
        transformer: nn.TransformerEncoder,
        pooler: MaskedAttentionPooling,
    ) -> torch.Tensor:
        embedded = self.token_embedding(token_ids)
        position_ids = build_position_ids(token_mask)
        embedded = self.text_dropout(embedded + position_embedding(position_ids))

        pooled = embedded.new_zeros((embedded.size(0), embedded.size(-1)))
        valid_rows = token_mask.any(dim=1)
        if valid_rows.any():
            valid_embedded = embedded[valid_rows]
            valid_mask = token_mask[valid_rows]
            contextualized = transformer(
                valid_embedded,
                src_key_padding_mask=~valid_mask,
            )
            pooled[valid_rows] = pooler(contextualized, valid_mask)

        return pooled

    def forward(
        self,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
        title_token_ids: torch.Tensor,
        title_token_mask: torch.Tensor,
        abstract_token_ids: torch.Tensor,
        abstract_token_mask: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = category_ids.shape
        flat_size = category_ids.numel()

        flat_category_ids = category_ids.reshape(-1)
        flat_subcategory_ids = subcategory_ids.reshape(-1)
        flat_title_token_ids = title_token_ids.reshape(flat_size, title_token_ids.size(-1))
        flat_title_token_mask = title_token_mask.reshape(flat_size, title_token_mask.size(-1))
        flat_abstract_token_ids = abstract_token_ids.reshape(flat_size, abstract_token_ids.size(-1))
        flat_abstract_token_mask = abstract_token_mask.reshape(flat_size, abstract_token_mask.size(-1))

        category_vec = self.category_embedding(flat_category_ids)
        subcategory_vec = self.subcategory_embedding(flat_subcategory_ids)
        title_vec = self.encode_text_sequence(
            flat_title_token_ids,
            flat_title_token_mask,
            self.title_position_embedding,
            self.title_transformer,
            self.title_pool,
        )
        abstract_vec = self.encode_text_sequence(
            flat_abstract_token_ids,
            flat_abstract_token_mask,
            self.abstract_position_embedding,
            self.abstract_transformer,
            self.abstract_pool,
        )

        features = [category_vec, subcategory_vec, title_vec, abstract_vec]

        if self.use_entities:
            if entity_ids is None or entity_mask is None:
                raise ValueError("entity_ids and entity_mask are required when use_entities=True")

            flat_entity_ids = entity_ids.reshape(flat_size, entity_ids.size(-1))
            flat_entity_mask = entity_mask.reshape(flat_size, entity_mask.size(-1))
            entity_vecs = self.entity_embedding(flat_entity_ids)
            entity_vec = masked_mean_pool(entity_vecs, flat_entity_mask)
            features.append(entity_vec)

        news_input = torch.cat(features, dim=-1)
        news_vec = self.proj(news_input)#########
        return news_vec.view(*original_shape, -1)


class MultiHeadTargetAttentionUserEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardBlock(embedding_dim, embedding_dim * 4, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.cold_start_user = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.cold_start_user, mean=0.0, std=0.02)

    def forward(
        self,
        history_news_vecs: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_news_vecs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_len, embedding_dim = candidate_news_vecs.shape
        history_len = history_news_vecs.size(1)

        user_vecs = self.cold_start_user.view(1, 1, embedding_dim).expand(
            batch_size,
            candidate_len,
            embedding_dim,
        ).clone()
        attention_weights = candidate_news_vecs.new_zeros((batch_size, candidate_len, history_len))

        valid_rows = history_mask.any(dim=1)
        if valid_rows.any():
            valid_candidates = candidate_news_vecs[valid_rows]
            valid_history = history_news_vecs[valid_rows]
            valid_history_mask = history_mask[valid_rows]

            attention_output, valid_attention_weights = self.attention(
                query=valid_candidates,
                key=valid_history,
                value=valid_history,
                key_padding_mask=~valid_history_mask,
                need_weights=True,
            )
            user_states = self.norm1(valid_candidates + self.dropout(attention_output))
            user_states = self.norm2(user_states + self.ffn(user_states))

            user_vecs[valid_rows] = user_states
            attention_weights[valid_rows] = valid_attention_weights

        return user_vecs, attention_weights


class FeatureClickScorer(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = embedding_dim * 4 + 1

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_vecs: torch.Tensor, candidate_news_vecs: torch.Tensor) -> torch.Tensor:
        mul_feat = user_vecs * candidate_news_vecs
        abs_diff_feat = torch.abs(user_vecs - candidate_news_vecs)
        dot_feat = mul_feat.sum(dim=-1, keepdim=True)

        cross_features = torch.cat(
            [user_vecs, candidate_news_vecs, mul_feat, abs_diff_feat, dot_feat],
            dim=-1,
        )
        logits = self.mlp(cross_features).squeeze(-1)
        return logits


class FeatureNewsRecModel(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_subcategories: int,
        vocab_size: int,
        news_category_ids: torch.Tensor,
        news_subcategory_ids: torch.Tensor,
        news_title_token_ids: torch.Tensor,
        news_title_mask: torch.Tensor,
        news_abstract_token_ids: torch.Tensor,
        news_abstract_mask: torch.Tensor,
        max_title_len: int,
        max_abstract_len: int,
        num_entities: int = 0,
        news_entity_ids: Optional[torch.Tensor] = None,
        news_entity_mask: Optional[torch.Tensor] = None,
        embedding_dim: int = 128,
        category_dim: int = 32,
        subcategory_dim: int = 32,
        token_dim: int = 128,
        entity_dim: int = 64,
        use_entities: bool = True,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_ffn_dim: int = 512,
        scorer_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_entities = use_entities

        self.news_encoder = FeatureNewsEncoder(
            num_categories=num_categories,
            num_subcategories=num_subcategories,
            vocab_size=vocab_size,
            max_title_len=max_title_len,
            max_abstract_len=max_abstract_len,
            embedding_dim=embedding_dim,
            category_dim=category_dim,
            subcategory_dim=subcategory_dim,
            token_dim=token_dim,
            num_entities=num_entities,
            entity_dim=entity_dim,
            use_entities=use_entities,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            transformer_ffn_dim=transformer_ffn_dim,
            dropout=dropout,
        )
        self.user_encoder = MultiHeadTargetAttentionUserEncoder(
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        self.scorer = FeatureClickScorer(
            embedding_dim=embedding_dim,
            hidden_dim=scorer_hidden_dim,
            dropout=dropout,
        )

        self.register_buffer("news_category_ids", news_category_ids.long())
        self.register_buffer("news_subcategory_ids", news_subcategory_ids.long())
        self.register_buffer("news_title_token_ids", news_title_token_ids.long())
        self.register_buffer("news_title_mask", news_title_mask.bool())
        self.register_buffer("news_abstract_token_ids", news_abstract_token_ids.long())
        self.register_buffer("news_abstract_mask", news_abstract_mask.bool())

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
            "abstract_token_ids": self.news_abstract_token_ids[news_ids],
            "abstract_token_mask": self.news_abstract_mask[news_ids],
        }

        if self.use_entities:
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
            abstract_token_ids=features["abstract_token_ids"],
            abstract_token_mask=features["abstract_token_mask"],
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
        history_news_vecs = self.encode_news_batch(history_ids)
        candidate_news_vecs = self.encode_news_batch(candidate_ids)
        user_vecs, history_attention_weights = self.user_encoder(
            history_news_vecs=history_news_vecs,
            history_mask=history_mask,
            candidate_news_vecs=candidate_news_vecs,
        )####target attention 真正发生的地方

        logits = self.scorer(user_vecs, candidate_news_vecs)
        logits = logits.masked_fill(~candidate_mask, -1e9)

        return {
            "logits": logits,
            "user_vecs": user_vecs,
            "history_news_vecs": history_news_vecs,
            "candidate_news_vecs": candidate_news_vecs,
            "history_attention_weights": history_attention_weights,
        }
