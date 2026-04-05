from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
except ImportError:  # pragma: no cover - used in data-only environments
    torch = None
    DataLoader = Any  # type: ignore[misc,assignment]

    class TorchDataset:  # type: ignore[no-redef]
        pass

PAD_NEWS_ID = "<PAD>"
BUCKET_ORDER = ("short", "medium", "long")


def get_candidate_bucket(candidate_len: int) -> str:
    if candidate_len <= 10:
        return "short"
    if candidate_len <= 24:
        return "medium"
    return "long"


def build_news_id_mapping(
    news_dict: Mapping[str, Any],
    pad_token: str = PAD_NEWS_ID,
) -> dict[str, int]:
    news_id_to_index = {pad_token: 0}

    for idx, news_id in enumerate(sorted(news_dict), start=1):
        news_id_to_index[news_id] = idx

    return news_id_to_index


class ImpressionDataset(TorchDataset):
    def __init__(
        self,
        impression_samples: Sequence[Mapping[str, Any]],
        news_id_to_index: Mapping[str, int],
        unknown_index: int = 0,
    ) -> None:
        self.impression_samples = list(impression_samples)
        self.news_id_to_index = dict(news_id_to_index)
        self.unknown_index = unknown_index

    def __len__(self) -> int:
        return len(self.impression_samples)

    def _encode_news_ids(self, news_ids: Sequence[str]) -> list[int]:
        return [self.news_id_to_index.get(news_id, self.unknown_index) for news_id in news_ids]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.impression_samples[index])
        history = list(sample.get("history", []))
        candidates = list(sample.get("candidates", []))
        labels = list(sample.get("labels", []))

        sample["history"] = history
        sample["candidates"] = candidates
        sample["labels"] = labels
        sample["history_ids"] = self._encode_news_ids(history)
        sample["candidate_ids"] = self._encode_news_ids(candidates)
        return sample


@dataclass
class RequestCollator:
    pad_news_index: int = 0
    label_pad_value: float = -100.0

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if torch is None:
            raise ImportError("torch is required to collate batches. Please install torch first.")

        batch_size = len(batch)
        max_history_len = max((len(item["history_ids"]) for item in batch), default=0)
        max_candidate_len = max((len(item["candidate_ids"]) for item in batch), default=0)

        history_ids = torch.full(
            (batch_size, max_history_len),
            fill_value=self.pad_news_index,
            dtype=torch.long,
        )
        candidate_ids = torch.full(
            (batch_size, max_candidate_len),
            fill_value=self.pad_news_index,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_candidate_len),
            fill_value=self.label_pad_value,
            dtype=torch.float32,
        )
        history_mask = torch.zeros((batch_size, max_history_len), dtype=torch.bool)
        candidate_mask = torch.zeros((batch_size, max_candidate_len), dtype=torch.bool)

        for row_idx, item in enumerate(batch):
            history = item["history_ids"]
            candidates = item["candidate_ids"]
            target = item["labels"]

            if history:
                history_ids[row_idx, : len(history)] = torch.tensor(history, dtype=torch.long)
                history_mask[row_idx, : len(history)] = True

            if candidates:
                candidate_ids[row_idx, : len(candidates)] = torch.tensor(
                    candidates,
                    dtype=torch.long,
                )
                labels[row_idx, : len(target)] = torch.tensor(target, dtype=torch.float32)
                candidate_mask[row_idx, : len(candidates)] = True

        return {
            "impression_ids": [item["impression_id"] for item in batch],
            "user_ids": [item["user_id"] for item in batch],
            "timestamps": [item["timestamp"] for item in batch],
            "history_ids": history_ids,
            "history_mask": history_mask,
            "candidate_ids": candidate_ids,
            "candidate_mask": candidate_mask,
            "labels": labels,
        }


def build_dataloader(
    impression_samples: Sequence[Mapping[str, Any]],
    news_id_to_index: Mapping[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pad_news_index: int = 0,
    label_pad_value: float = -100.0,
    drop_last: bool = False,
) -> DataLoader:
    if torch is None:
        raise ImportError("torch is required to build a DataLoader. Please install torch first.")

    dataset = ImpressionDataset(
        impression_samples=impression_samples,
        news_id_to_index=news_id_to_index,
        unknown_index=pad_news_index,
    )
    collator = RequestCollator(
        pad_news_index=pad_news_index,
        label_pad_value=label_pad_value,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=drop_last,
    )


def build_bucketed_dataloaders(
    impression_samples: Sequence[Mapping[str, Any]],
    news_id_to_index: Mapping[str, int],
    batch_sizes: Mapping[str, int],
    shuffle: bool,
    num_workers: int = 0,
    pad_news_index: int = 0,
    label_pad_value: float = -100.0,
    drop_last: bool = False,
) -> dict[str, DataLoader]:
    buckets: dict[str, list[Mapping[str, Any]]] = {name: [] for name in BUCKET_ORDER}

    for sample in impression_samples:
        bucket_name = get_candidate_bucket(len(sample.get("candidates", [])))
        buckets[bucket_name].append(sample)

    dataloaders: dict[str, DataLoader] = {}
    for bucket_name in BUCKET_ORDER:
        samples = buckets[bucket_name]
        if not samples:
            continue

        dataloaders[bucket_name] = build_dataloader(
            impression_samples=samples,
            news_id_to_index=news_id_to_index,
            batch_size=batch_sizes[bucket_name],
            shuffle=shuffle,
            num_workers=num_workers,
            pad_news_index=pad_news_index,
            label_pad_value=label_pad_value,
            drop_last=drop_last,
        )

    return dataloaders
