from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW

try:
    import wandb
except ImportError:
    wandb = None

from dataset import build_dataloader
from eval import auc_score, mrr_score, ndcg_at_k
from featuresbaseline import load_or_build_baseline_news_features
from losses import MaskedBCEWithLogitsLoss
from modelbaseline import BaselineNewsRecModel


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@dataclass
class TrainConfig:
    processed_dir: Path = Path("data/processed")
    train_path: Path = Path("data/processed/train_impressions.parquet")
    dev_path: Path = Path("data/processed/dev_impressions.parquet")
    news_id_to_index_path: Path = Path("data/processed/news_id_to_index.json")
    feature_cache_path: Path = Path("data/processed/baseline_news_features.pt")
    checkpoint_path: Path = Path("outputs/baseline_best.pt")

    max_title_len: int = 24
    embedding_dim: int = 128
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 3
    num_workers: int = 0
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    log_interval: int = 200

    use_wandb: bool = True
    wandb_project: str = "mind"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_dir: Path = Path("outputs/wandb")

    device: str = detect_device()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MIND baseline recommender.")
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity or team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional display name for the Weights & Biases run.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Steps between console and W&B training logs.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return []


def load_impression_samples(path: str | Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    samples: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        samples.append(
            {
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "timestamp": row.timestamp,
                "history": ensure_list(row.history),
                "candidates": ensure_list(row.candidates),
                "labels": ensure_list(row.labels),
            }
        )

    return samples


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tensor_keys = ["history_ids", "history_mask", "candidate_ids", "candidate_mask", "labels"]

    moved = dict(batch)
    for key in tensor_keys:
        moved[key] = moved[key].to(device)

    return moved


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def serialize_config(config: TrainConfig) -> dict[str, Any]:
    serialized: dict[str, Any] = {}

    for key, value in config.__dict__.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value

    return serialized


def init_wandb_run(
    config: TrainConfig,
    train_sample_count: int,
    dev_sample_count: int,
    trainable_params: int,
) -> Any | None:
    if not config.use_wandb:
        print("wandb logging disabled for this run.")
        return None

    if wandb is None:
        print("wandb is not installed in the current environment, skipping experiment logging.")
        return None

    config.wandb_dir.mkdir(parents=True, exist_ok=True)

    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            dir=str(config.wandb_dir),
            config={
                **serialize_config(config),
                "train_samples": train_sample_count,
                "dev_samples": dev_sample_count,
                "trainable_params": trainable_params,
            },
        )
    except Exception as exc:
        print(f"wandb init failed: {exc}. continuing without wandb logging.")
        return None

    wandb.define_metric("train_step/global_step")
    wandb.define_metric("train_step/*", step_metric="train_step/global_step")
    wandb.define_metric("epoch/index")
    wandb.define_metric("epoch/*", step_metric="epoch/index")

    run.summary["checkpoint_path"] = str(config.checkpoint_path)

    print(f"wandb run name = {run.name}")
    if getattr(run, "url", None):
        print(f"wandb url = {run.url}")

    return run


@torch.no_grad()
def evaluate(
    model: BaselineNewsRecModel,
    data_loader,
    criterion: MaskedBCEWithLogitsLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_valid_candidates = 0

    auc_sum = 0.0
    auc_count = 0
    mrr_sum = 0.0
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    request_count = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(
            history_ids=batch["history_ids"],
            history_mask=batch["history_mask"],
            candidate_ids=batch["candidate_ids"],
            candidate_mask=batch["candidate_mask"],
        )

        logits = outputs["logits"]
        loss = criterion(logits, batch["labels"], batch["candidate_mask"])

        valid_count = int(batch["candidate_mask"].sum().item())
        total_loss += loss.item() * valid_count
        total_valid_candidates += valid_count

        batch_size = logits.size(0)
        for i in range(batch_size):
            mask_i = batch["candidate_mask"][i]
            scores_i = logits[i][mask_i].detach().cpu().tolist()
            labels_i = batch["labels"][i][mask_i].detach().cpu().tolist()
            labels_i = [int(x) for x in labels_i]

            request_count += 1

            auc_i = auc_score(scores_i, labels_i)
            if auc_i is not None:
                auc_sum += auc_i
                auc_count += 1

            mrr_sum += mrr_score(scores_i, labels_i)
            ndcg5_sum += ndcg_at_k(scores_i, labels_i, k=5)
            ndcg10_sum += ndcg_at_k(scores_i, labels_i, k=10)

    return {
        "loss": total_loss / max(total_valid_candidates, 1),
        "AUC": auc_sum / max(auc_count, 1),
        "MRR": mrr_sum / max(request_count, 1),
        "nDCG@5": ndcg5_sum / max(request_count, 1),
        "nDCG@10": ndcg10_sum / max(request_count, 1),
    }


def main() -> None:
    args = parse_args()
    config = TrainConfig()

    if args.disable_wandb:
        config.use_wandb = False
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if config.log_interval < 1:
        raise ValueError("log_interval must be at least 1")

    device = torch.device(config.device)

    features = load_or_build_baseline_news_features(
        cache_path=config.feature_cache_path,
        processed_dir=config.processed_dir,
        max_title_len=config.max_title_len,
    )
    news_id_to_index = load_json(config.news_id_to_index_path)

    train_samples = load_impression_samples(config.train_path)
    dev_samples = load_impression_samples(config.dev_path)

    train_loader = build_dataloader(
        impression_samples=train_samples,
        news_id_to_index=news_id_to_index,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dev_loader = build_dataloader(
        impression_samples=dev_samples,
        news_id_to_index=news_id_to_index,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = BaselineNewsRecModel(
        num_categories=len(features["category_to_index"]),
        num_subcategories=len(features["subcategory_to_index"]),
        vocab_size=len(features["token_to_index"]),
        news_category_ids=features["news_category_ids"],
        news_subcategory_ids=features["news_subcategory_ids"],
        news_title_token_ids=features["news_title_token_ids"],
        news_title_mask=features["news_title_mask"],
        embedding_dim=config.embedding_dim,
        use_entities=False,
        dropout=config.dropout,
    ).to(device)

    criterion = MaskedBCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    trainable_params = count_parameters(model)

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device = {device}")
    print(f"train samples = {len(train_samples)}")
    print(f"dev samples = {len(dev_samples)}")
    print(f"trainable params = {trainable_params:,}")

    wandb_run = init_wandb_run(
        config=config,
        train_sample_count=len(train_samples),
        dev_sample_count=len(dev_samples),
        trainable_params=trainable_params,
    )

    best_dev_mrr = -1.0
    global_step = 0

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()

            running_loss = 0.0
            running_valid_candidates = 0
            num_train_steps = len(train_loader)

            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)

                optimizer.zero_grad()

                outputs = model(
                    history_ids=batch["history_ids"],
                    history_mask=batch["history_mask"],
                    candidate_ids=batch["candidate_ids"],
                    candidate_mask=batch["candidate_mask"],
                )

                loss = criterion(
                    outputs["logits"],
                    batch["labels"],
                    batch["candidate_mask"],
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                global_step += 1

                valid_count = int(batch["candidate_mask"].sum().item())
                running_loss += loss.item() * valid_count
                running_valid_candidates += valid_count

                should_log_step = step % config.log_interval == 0 or step == num_train_steps
                if should_log_step:
                    avg_train_loss = running_loss / max(running_valid_candidates, 1)
                    print(f"[epoch {epoch}] step {step} train_loss={avg_train_loss:.4f}")

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train_step/global_step": global_step,
                                "train_step/epoch": epoch,
                                "train_step/step_in_epoch": step,
                                "train_step/loss": avg_train_loss,
                                "train_step/lr": optimizer.param_groups[0]["lr"],
                            }
                        )

            train_loss = running_loss / max(running_valid_candidates, 1)
            dev_metrics = evaluate(model, dev_loader, criterion, device)
            is_best_checkpoint = dev_metrics["MRR"] > best_dev_mrr

            print(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:.4f} "
                f"dev_loss={dev_metrics['loss']:.4f} "
                f"AUC={dev_metrics['AUC']:.4f} "
                f"MRR={dev_metrics['MRR']:.4f} "
                f"nDCG@5={dev_metrics['nDCG@5']:.4f} "
                f"nDCG@10={dev_metrics['nDCG@10']:.4f}"
            )

            if is_best_checkpoint:
                best_dev_mrr = dev_metrics["MRR"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "dev_metrics": dev_metrics,
                    },
                    config.checkpoint_path,
                )
                print(f"saved best checkpoint to {config.checkpoint_path}")

                if wandb_run is not None:
                    wandb_run.summary["best/dev_MRR"] = best_dev_mrr
                    wandb_run.summary["best/epoch"] = epoch
                    wandb_run.summary["best/checkpoint_path"] = str(config.checkpoint_path)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch/index": epoch,
                        "epoch/train_loss": train_loss,
                        "epoch/dev_loss": dev_metrics["loss"],
                        "epoch/dev_AUC": dev_metrics["AUC"],
                        "epoch/dev_MRR": dev_metrics["MRR"],
                        "epoch/dev_nDCG@5": dev_metrics["nDCG@5"],
                        "epoch/dev_nDCG@10": dev_metrics["nDCG@10"],
                        "epoch/best_dev_MRR": best_dev_mrr,
                        "epoch/is_best_checkpoint": int(is_best_checkpoint),
                    }
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
