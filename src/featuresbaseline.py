from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
FEATURE_SCHEMA_VERSION = 2
REQUIRED_FEATURE_KEYS = {
    "feature_schema_version",
    "news_category_ids",
    "news_subcategory_ids",
    "news_title_token_ids",
    "news_title_mask",
    "news_abstract_token_ids",
    "news_abstract_mask",
    "news_entity_ids",
    "news_entity_mask",
    "category_to_index",
    "subcategory_to_index",
    "token_to_index",
    "entity_to_index",
}


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_list(text: str) -> list[str]:
    if isinstance(text, list):
        return [str(item) for item in text]
    if not isinstance(text, str) or not text.strip():
        return []
    return json.loads(text)


def build_index_mapping(values: list[str], add_unk: bool = False) -> dict[str, int]:
    mapping = {PAD_TOKEN: 0}
    if add_unk:
        mapping[UNK_TOKEN] = 1

    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)

    return mapping


def build_news_feature_tensors(
    news_parquet_path: str | Path,
    news_id_to_index_path: str | Path,
    max_title_len: int = 24,
    max_abstract_len: int = 24,
    max_entity_len: int = 5,
) -> dict[str, Any]:
    news_df = pd.read_parquet(news_parquet_path)
    news_id_to_index = load_json(news_id_to_index_path)

    if "abstract_tokens" not in news_df.columns:
        news_df["abstract_tokens"] = "[]"
    if "entities" not in news_df.columns:
        news_df["entities"] = "[]"

    categories = sorted(news_df["category"].dropna().unique().tolist())
    subcategories = sorted(news_df["subcategory"].dropna().unique().tolist())

    all_tokens: list[str] = []
    for text in news_df["title_tokens"]:
        all_tokens.extend(parse_json_list(text))
    for text in news_df["abstract_tokens"]:
        all_tokens.extend(parse_json_list(text))

    all_entities: list[str] = []
    for text in news_df["entities"]:
        all_entities.extend(parse_json_list(text))

    category_to_index = build_index_mapping(categories, add_unk=False)
    subcategory_to_index = build_index_mapping(subcategories, add_unk=False)
    token_to_index = build_index_mapping(all_tokens, add_unk=True)
    entity_to_index = build_index_mapping(all_entities, add_unk=True)

    num_news = len(news_id_to_index)

    news_category_ids = torch.zeros(num_news, dtype=torch.long)
    news_subcategory_ids = torch.zeros(num_news, dtype=torch.long)
    news_title_token_ids = torch.zeros((num_news, max_title_len), dtype=torch.long)
    news_title_mask = torch.zeros((num_news, max_title_len), dtype=torch.bool)
    news_abstract_token_ids = torch.zeros((num_news, max_abstract_len), dtype=torch.long)
    news_abstract_mask = torch.zeros((num_news, max_abstract_len), dtype=torch.bool)
    news_entity_ids = torch.zeros((num_news, max_entity_len), dtype=torch.long)
    news_entity_mask = torch.zeros((num_news, max_entity_len), dtype=torch.bool)

    unk_index = token_to_index[UNK_TOKEN]
    entity_unk_index = entity_to_index[UNK_TOKEN]

    for row in news_df.itertuples(index=False):
        news_id = row.news_id
        if news_id not in news_id_to_index:
            continue

        news_idx = news_id_to_index[news_id]

        news_category_ids[news_idx] = category_to_index.get(row.category, 0)
        news_subcategory_ids[news_idx] = subcategory_to_index.get(row.subcategory, 0)

        title_tokens = parse_json_list(row.title_tokens)[:max_title_len]
        token_ids = [token_to_index.get(token, unk_index) for token in title_tokens]

        if token_ids:
            news_title_token_ids[news_idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            news_title_mask[news_idx, : len(token_ids)] = True

        abstract_tokens = parse_json_list(row.abstract_tokens)[:max_abstract_len]
        abstract_token_ids = [token_to_index.get(token, unk_index) for token in abstract_tokens]
        if abstract_token_ids:
            news_abstract_token_ids[news_idx, : len(abstract_token_ids)] = torch.tensor(
                abstract_token_ids,
                dtype=torch.long,
            )
            news_abstract_mask[news_idx, : len(abstract_token_ids)] = True

        entity_ids = parse_json_list(row.entities)[:max_entity_len]
        encoded_entity_ids = [entity_to_index.get(entity_id, entity_unk_index) for entity_id in entity_ids]
        if encoded_entity_ids:
            news_entity_ids[news_idx, : len(encoded_entity_ids)] = torch.tensor(
                encoded_entity_ids,
                dtype=torch.long,
            )
            news_entity_mask[news_idx, : len(encoded_entity_ids)] = True

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "max_title_len": max_title_len,
        "max_abstract_len": max_abstract_len,
        "max_entity_len": max_entity_len,
        "news_category_ids": news_category_ids,
        "news_subcategory_ids": news_subcategory_ids,
        "news_title_token_ids": news_title_token_ids,
        "news_title_mask": news_title_mask,
        "news_abstract_token_ids": news_abstract_token_ids,
        "news_abstract_mask": news_abstract_mask,
        "news_entity_ids": news_entity_ids,
        "news_entity_mask": news_entity_mask,
        "category_to_index": category_to_index,
        "subcategory_to_index": subcategory_to_index,
        "token_to_index": token_to_index,
        "entity_to_index": entity_to_index,
    }


def load_baseline_news_features(
    processed_dir: str | Path = "data/processed",
    max_title_len: int = 24,
    max_abstract_len: int = 24,
    max_entity_len: int = 5,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    return build_news_feature_tensors(
        news_parquet_path=processed_dir / "news_dict.parquet",
        news_id_to_index_path=processed_dir / "news_id_to_index.json",
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
        max_entity_len=max_entity_len,
    )


def save_baseline_news_features(
    output_path: str | Path,
    processed_dir: str | Path = "data/processed",
    max_title_len: int = 24,
    max_abstract_len: int = 24,
    max_entity_len: int = 5,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = load_baseline_news_features(
        processed_dir=processed_dir,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
        max_entity_len=max_entity_len,
    )
    torch.save(features, output_path)
    return features


def load_saved_baseline_news_features(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return torch.load(path, map_location="cpu")


def is_feature_cache_compatible(
    features: dict[str, Any],
    max_title_len: int,
    max_abstract_len: int,
    max_entity_len: int,
) -> bool:
    if not REQUIRED_FEATURE_KEYS.issubset(features):
        return False

    if features.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        return False

    if features["news_title_token_ids"].size(1) != max_title_len:
        return False
    if features["news_abstract_token_ids"].size(1) != max_abstract_len:
        return False
    if features["news_entity_ids"].size(1) != max_entity_len:
        return False

    return True


def load_or_build_baseline_news_features(
    cache_path: str | Path,
    processed_dir: str | Path = "data/processed",
    max_title_len: int = 24,
    max_abstract_len: int = 24,
    max_entity_len: int = 5,
) -> dict[str, Any]:
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            features = load_saved_baseline_news_features(cache_path)
        except Exception:
            features = None
        else:
            if is_feature_cache_compatible(
                features,
                max_title_len=max_title_len,
                max_abstract_len=max_abstract_len,
                max_entity_len=max_entity_len,
            ):
                return features

    return save_baseline_news_features(
        output_path=cache_path,
        processed_dir=processed_dir,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
        max_entity_len=max_entity_len,
    )
