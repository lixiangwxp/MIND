from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
PAD_NEWS_ID = "<PAD>"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def parse_impressions(text: str) -> tuple[list[str], list[int]]:
    if not isinstance(text, str) or not text.strip():
        return [], []

    news_ids: list[str] = []
    labels: list[int] = []

    for token in text.split():
        nid, y = token.rsplit("-", 1)
        news_ids.append(nid)
        labels.append(int(y))

    return news_ids, labels


def parse_history(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def _parse_entity_blob(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip() or text.strip() == "[]":
        return []

    try:
        raw_entities = json.loads(text)
    except json.JSONDecodeError:
        return []

    entity_ids: list[str] = []
    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue

        wikidata_id = entity.get("WikidataId")
        if wikidata_id:
            entity_ids.append(wikidata_id)

    return entity_ids


def parse_entities(title_entities: str, abstract_entities: str) -> list[str]:
    entity_ids = _parse_entity_blob(title_entities) + _parse_entity_blob(abstract_entities)
    return list(dict.fromkeys(entity_ids))


def load_news_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS)


def load_behaviors_frame(path: str | Path) -> pd.DataFrame:
    behaviors = pd.read_csv(path, sep="\t", header=None, names=BEH_COLS)
    behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
    behaviors["history"] = behaviors["history"].apply(parse_history)

    parsed_impressions = behaviors["impressions"].apply(parse_impressions)
    behaviors["candidate_news_ids"] = parsed_impressions.map(lambda item: item[0])
    behaviors["labels"] = parsed_impressions.map(lambda item: item[1])
    return behaviors


def _truncate_prefix(items: list[str], max_len: int | None) -> list[str]:
    if max_len is None or max_len <= 0:
        return items
    return items[:max_len]


def _truncate_history(history_ids: list[str], max_history_len: int | None) -> list[str]:
    if max_history_len is None or max_history_len <= 0:
        return history_ids
    return history_ids[-max_history_len:]


def build_news_dict(
    news_df: pd.DataFrame,
    max_title_len: int | None = 24,
    max_abstract_len: int | None = 48,
    max_entity_len: int | None = 5,
) -> dict[str, dict[str, Any]]:
    news_dict: dict[str, dict[str, Any]] = {}

    for row in news_df.itertuples(index=False):
        title_tokens = _truncate_prefix(tokenize(row.title), max_title_len)
        abstract_tokens = _truncate_prefix(tokenize(row.abstract), max_abstract_len)
        entity_ids = _truncate_prefix(
            parse_entities(row.title_entities, row.abstract_entities),
            max_entity_len,
        )

        news_dict[row.news_id] = {
            "category": row.category,
            "subcategory": row.subcategory,
            "title_tokens": title_tokens,
            "abstract_tokens": abstract_tokens,
            "entities": entity_ids,
        }

    return news_dict


def _sample_candidates(
    candidate_news_ids: list[str],
    candidate_labels: list[int],
    negative_sample_size: int | None,
    negative_sample_ratio: int | None,
    negative_sample_max_size: int | None,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    pos_indices = [idx for idx, label in enumerate(candidate_labels) if label == 1]
    neg_indices = [idx for idx, label in enumerate(candidate_labels) if label == 0]

    if negative_sample_ratio is not None and negative_sample_ratio > 0:
        target_negative_count = len(pos_indices) * negative_sample_ratio
    elif negative_sample_size is not None and negative_sample_size >= 0:
        target_negative_count = negative_sample_size
    else:
        return candidate_news_ids, candidate_labels

    if negative_sample_max_size is not None and negative_sample_max_size > 0:
        target_negative_count = min(target_negative_count, negative_sample_max_size)

    if target_negative_count <= 0:
        selected_indices = pos_indices if pos_indices else list(range(len(candidate_news_ids)))
    elif len(neg_indices) <= target_negative_count:
        selected_indices = list(range(len(candidate_news_ids)))
    else:
        sampled_neg_indices = rng.sample(neg_indices, target_negative_count)
        selected_indices = sorted(pos_indices + sampled_neg_indices)

    sampled_news_ids = [candidate_news_ids[idx] for idx in selected_indices]
    sampled_labels = [candidate_labels[idx] for idx in selected_indices]
    return sampled_news_ids, sampled_labels


def build_impression_samples(
    behaviors_df: pd.DataFrame,
    max_history_len: int | None = 50,
    negative_sample_size: int | None = None,
    negative_sample_ratio: int | None = None,
    negative_sample_max_size: int | None = None,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    samples: list[dict[str, Any]] = []

    for row in behaviors_df.itertuples(index=False):
        history_ids = _truncate_history(row.history, max_history_len)
        candidate_news_ids, candidate_labels = _sample_candidates(
            row.candidate_news_ids,
            row.labels,
            negative_sample_size,
            negative_sample_ratio,
            negative_sample_max_size,
            rng,
        )

        samples.append(
            {
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "timestamp": row.time,
                "history": history_ids,
                "candidates": candidate_news_ids,
                "labels": candidate_labels,
            }
        )

    return samples


def merge_news_frames(*news_frames: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat(news_frames, ignore_index=True)
    return merged.drop_duplicates(subset="news_id", keep="first")


def build_news_id_mapping(news_dict: dict[str, dict[str, Any]]) -> dict[str, int]:
    news_id_to_index = {PAD_NEWS_ID: 0}

    for idx, news_id in enumerate(sorted(news_dict), start=1):
        news_id_to_index[news_id] = idx

    return news_id_to_index


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def build_news_feature_frame(news_dict: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for news_id, features in sorted(news_dict.items()):
        rows.append(
            {
                "news_id": news_id,
                "category": features["category"],
                "subcategory": features["subcategory"],
                "title_tokens": _json_dumps(features["title_tokens"]),
                "abstract_tokens": _json_dumps(features["abstract_tokens"]),
                "entities": _json_dumps(features["entities"]),
            }
        )

    return pd.DataFrame(rows)


def build_impression_frame(impression_samples: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sample in impression_samples:
        rows.append(
            {
                "impression_id": sample["impression_id"],
                "user_id": sample["user_id"],
                "timestamp": sample["timestamp"].isoformat(),
                "history": _json_dumps(sample["history"]),
                "candidates": _json_dumps(sample["candidates"]),
                "labels": _json_dumps(sample["labels"]),
                "history_len": len(sample["history"]),
                "candidate_len": len(sample["candidates"]),
                "positive_count": int(sum(sample["labels"])),
            }
        )

    return pd.DataFrame(rows)


def save_processed_artifacts(
    train_dir: str | Path,
    dev_dir: str | Path,
    output_dir: str | Path,
    max_history_len: int = 50,
    max_title_len: int = 24,
    max_abstract_len: int = 48,
    max_entity_len: int = 5,
    train_negative_sample_size: int | None = None,
    train_negative_sample_ratio: int | None = 8,
    train_negative_sample_max_size: int | None = 24,
    dev_negative_sample_size: int | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    train_dir = Path(train_dir)
    dev_dir = Path(dev_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_news_df = load_news_frame(train_dir / "news.tsv")
    dev_news_df = load_news_frame(dev_dir / "news.tsv")
    merged_news_df = merge_news_frames(train_news_df, dev_news_df)
    news_dict = build_news_dict(
        merged_news_df,
        max_title_len=max_title_len,
        max_abstract_len=max_abstract_len,
        max_entity_len=max_entity_len,
    )

    train_behaviors_df = load_behaviors_frame(train_dir / "behaviors.tsv")
    dev_behaviors_df = load_behaviors_frame(dev_dir / "behaviors.tsv")

    train_samples = build_impression_samples(
        train_behaviors_df,
        max_history_len=max_history_len,
        negative_sample_size=train_negative_sample_size,
        negative_sample_ratio=train_negative_sample_ratio,
        negative_sample_max_size=train_negative_sample_max_size,
        random_seed=random_seed,
    )
    dev_samples = build_impression_samples(
        dev_behaviors_df,
        max_history_len=max_history_len,
        negative_sample_size=dev_negative_sample_size,
        random_seed=random_seed,
    )

    news_id_to_index = build_news_id_mapping(news_dict)

    news_path = output_dir / "news_dict.parquet"
    train_path = output_dir / "train_impressions.parquet"
    dev_path = output_dir / "dev_impressions.parquet"
    mapping_path = output_dir / "news_id_to_index.json"
    meta_path = output_dir / "preprocess_meta.json"

    build_news_feature_frame(news_dict).to_parquet(news_path, index=False)
    build_impression_frame(train_samples).to_parquet(train_path, index=False)
    build_impression_frame(dev_samples).to_parquet(dev_path, index=False)

    mapping_path.write_text(
        json.dumps(news_id_to_index, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "train_dir": str(train_dir),
        "dev_dir": str(dev_dir),
        "output_dir": str(output_dir),
        "max_history_len": max_history_len,
        "max_title_len": max_title_len,
        "max_abstract_len": max_abstract_len,
        "max_entity_len": max_entity_len,
        "train_negative_sample_size": train_negative_sample_size,
        "train_negative_sample_ratio": train_negative_sample_ratio,
        "train_negative_sample_max_size": train_negative_sample_max_size,
        "dev_negative_sample_size": dev_negative_sample_size,
        "random_seed": random_seed,
        "news_count": len(news_dict),
        "train_impression_count": len(train_samples),
        "dev_impression_count": len(dev_samples),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "news_path": news_path,
        "train_path": train_path,
        "dev_path": dev_path,
        "mapping_path": mapping_path,
        "meta_path": meta_path,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate processed MIND-small training data.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/raw/MINDsmall_train"),
        help="Directory containing the MIND-small train split.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=Path("data/raw/MINDsmall_dev"),
        help="Directory containing the MIND-small dev split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed parquet/json files will be written.",
    )
    parser.add_argument(
        "--max-history-len",
        type=int,
        default=50,
        help="Keep only the most recent N clicked news per impression.",
    )
    parser.add_argument(
        "--max-title-len",
        type=int,
        default=24,
        help="Keep at most N title tokens per news article.",
    )
    parser.add_argument(
        "--max-abstract-len",
        type=int,
        default=48,
        help="Keep at most N abstract tokens per news article.",
    )
    parser.add_argument(
        "--max-entity-len",
        type=int,
        default=5,
        help="Keep at most N entity ids per news article.",
    )
    parser.add_argument(
        "--train-negative-sample-size",
        type=int,
        default=None,
        help="Fallback fixed-count negative sampling for training. Ignored when ratio > 0.",
    )
    parser.add_argument(
        "--train-negative-sample-ratio",
        type=int,
        default=8,
        help="Target negative-to-positive sampling ratio for training. Set to 0 to disable.",
    )
    parser.add_argument(
        "--train-negative-sample-max-size",
        type=int,
        default=24,
        help="Maximum number of negative candidates kept per training impression.",
    )
    parser.add_argument(
        "--dev-negative-sample-size",
        type=int,
        default=None,
        help="Optionally sample negatives for dev. Leave unset to keep the full candidate list.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for negative sampling.",
    )
    args = parser.parse_args()

    artifacts = save_processed_artifacts(
        train_dir=args.train_dir,
        dev_dir=args.dev_dir,
        output_dir=args.output_dir,
        max_history_len=args.max_history_len,
        max_title_len=args.max_title_len,
        max_abstract_len=args.max_abstract_len,
        max_entity_len=args.max_entity_len,
        train_negative_sample_size=args.train_negative_sample_size,
        train_negative_sample_ratio=args.train_negative_sample_ratio,
        train_negative_sample_max_size=args.train_negative_sample_max_size,
        dev_negative_sample_size=args.dev_negative_sample_size,
        random_seed=args.random_seed,
    )

    metadata = artifacts["metadata"]
    print(f"Saved news features to {artifacts['news_path']}")
    print(f"Saved train impressions to {artifacts['train_path']}")
    print(f"Saved dev impressions to {artifacts['dev_path']}")
    print(f"Saved news id mapping to {artifacts['mapping_path']}")
    print(f"Saved preprocess metadata to {artifacts['meta_path']}")
    print(f"News count: {metadata['news_count']}")
    print(f"Train impressions: {metadata['train_impression_count']}")
    print(f"Dev impressions: {metadata['dev_impression_count']}")


if __name__ == "__main__":
    main()
