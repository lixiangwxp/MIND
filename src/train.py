from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    import wandb
except ImportError:
    wandb = None

from dataset import BUCKET_ORDER, build_bucketed_dataloaders, build_dataloader
from eval import auc_score, brier_score, mrr_score, ndcg_at_k, recall_at_k
from featuresbaseline import load_or_build_baseline_news_features
from losses import ImpressionPairwiseLoss, ListNetTop, MaskedBCEWithLogitsLoss
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
    max_abstract_len: int = 24
    max_entity_len: int = 5
    embedding_dim: int = 128
    batch_size: int = 8
    bce_short_batch_size: int = 8
    bce_medium_batch_size: int = 4
    bce_long_batch_size: int = 2
    pairwise_short_batch_size: int = 4
    pairwise_medium_batch_size: int = 2
    pairwise_long_batch_size: int = 1
    lr: float = 5e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 1
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    epochs: int = 20
    num_workers: int = 0
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    log_interval: int = 200
    loss_type: str = "bce"
    early_stopping_min_epochs: int = 3
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3
    use_entities: bool = True

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
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["bce", "pairwise", "listnet_top"],
        default=None,
        help="Training loss to use. Default keeps the original BCE baseline.",
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


def build_criterion(loss_type: str) -> torch.nn.Module:
    if loss_type == "bce":
        return MaskedBCEWithLogitsLoss()

    if loss_type == "pairwise":
        return ImpressionPairwiseLoss()

    if loss_type == "listnet_top":
        return ListNetTop()

    raise ValueError(f"Unsupported loss type: {loss_type}")


def get_loss_weight(loss_type: str, candidate_mask: torch.Tensor) -> int:
    if loss_type == "listnet_top":
        # Listwise loss is averaged per impression, not per candidate.
        return int(candidate_mask.bool().any(dim=1).sum().item())

    return int(candidate_mask.sum().item())


def get_train_bucket_batch_sizes(config: TrainConfig, loss_type: str) -> dict[str, int]:
    if loss_type == "pairwise":
        return {
            "short": config.pairwise_short_batch_size,
            "medium": config.pairwise_medium_batch_size,
            "long": config.pairwise_long_batch_size,
        }

    return {
        "short": config.bce_short_batch_size,
        "medium": config.bce_medium_batch_size,
        "long": config.bce_long_batch_size,
    }


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
    wandb.define_metric("epoch/dev_loss", step_metric="epoch/index")
    wandb.define_metric("epoch/dev_MRR", step_metric="epoch/index")
    wandb.define_metric("epoch/dev_nDCG@10", step_metric="epoch/index")
    wandb.define_metric("epoch/dev_BrierScore", step_metric="epoch/index")
    wandb.define_metric("epoch/dev_Recall@10", step_metric="epoch/index")
    wandb.define_metric("epoch/dev_GlobalCTR", step_metric="epoch/index")
    wandb.define_metric("epoch/dev_BrierRef", step_metric="epoch/index")
    wandb.define_metric("epoch/ranking_score", step_metric="epoch/index")
    wandb.define_metric("epoch/calibration_score", step_metric="epoch/index")
    wandb.define_metric("epoch/selection_score", step_metric="epoch/index")
    wandb.define_metric("epoch/lr", step_metric="epoch/index")

    run.summary["checkpoint_path"] = str(config.checkpoint_path)

    print(f"wandb run name = {run.name}")
    if getattr(run, "url", None):
        print(f"wandb url = {run.url}")

    return run


@torch.no_grad()
def evaluate(
    model: BaselineNewsRecModel,
    data_loader,
    criterion: torch.nn.Module,
    loss_type: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_loss_weight = 0
    total_valid_candidates = 0

    auc_sum = 0.0
    auc_count = 0
    mrr_sum = 0.0
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    brier_error_sum = 0.0
    brier_item_count = 0
    recall10_sum = 0.0
    request_count = 0
    total_positive_candidates = 0

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
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss encountered.")

        loss_weight = get_loss_weight(loss_type, batch["candidate_mask"])
        valid_count = int(batch["candidate_mask"].sum().item())
        total_loss += loss.item() * loss_weight
        total_loss_weight += loss_weight
        total_valid_candidates += valid_count

        batch_size = logits.size(0)
        for i in range(batch_size):
            mask_i = batch["candidate_mask"][i]
            scores_i = logits[i][mask_i].detach().cpu().tolist()
            labels_i = batch["labels"][i][mask_i].detach().cpu().tolist()
            labels_i = [int(x) for x in labels_i]
            total_positive_candidates += sum(labels_i)

            request_count += 1

            auc_i = auc_score(scores_i, labels_i)
            if auc_i is not None:
                auc_sum += auc_i
                auc_count += 1

            mrr_sum += mrr_score(scores_i, labels_i)
            ndcg5_sum += ndcg_at_k(scores_i, labels_i, k=5)
            ndcg10_sum += ndcg_at_k(scores_i, labels_i, k=10)
            brier_error_sum += brier_score(scores_i, labels_i) * len(labels_i)
            brier_item_count += len(labels_i)
            recall10_sum += recall_at_k(scores_i, labels_i, k=10)

    global_ctr = total_positive_candidates / max(total_valid_candidates, 1)
    brier_score_value = brier_error_sum / max(brier_item_count, 1)
    brier_ref = global_ctr * (1.0 - global_ctr)
    calibration_score = max(0.0, 1.0 - (brier_score_value / brier_ref)) if brier_ref > 0 else 0.0
    ranking_score = ndcg10_sum / max(request_count, 1)
    selection_score = 0.5 * ranking_score + 0.5 * calibration_score

    return {
        "loss": total_loss / max(total_loss_weight, 1),
        "AUC": auc_sum / max(auc_count, 1),
        "MRR": mrr_sum / max(request_count, 1),
        "nDCG@5": ndcg5_sum / max(request_count, 1),
        "nDCG@10": ranking_score,
        "BrierScore": brier_score_value,
        "Recall@10": recall10_sum / max(request_count, 1),
        "GlobalCTR": global_ctr,
        "BrierRef": brier_ref,
        "CalibrationScore": calibration_score,
        "RankingScore": ranking_score,
        "SelectionScore": selection_score,
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
    if args.loss_type:
        config.loss_type = args.loss_type
    if config.log_interval < 1:
        raise ValueError("log_interval must be at least 1")

    device = torch.device(config.device)

    features = load_or_build_baseline_news_features(
        cache_path=config.feature_cache_path,
        processed_dir=config.processed_dir,
        max_title_len=config.max_title_len,
        max_abstract_len=config.max_abstract_len,
        max_entity_len=config.max_entity_len,
    )
    news_id_to_index = load_json(config.news_id_to_index_path)

    train_samples = load_impression_samples(config.train_path)
    dev_samples = load_impression_samples(config.dev_path)

    train_bucket_batch_sizes = get_train_bucket_batch_sizes(config, config.loss_type)
    train_loaders = build_bucketed_dataloaders(
        impression_samples=train_samples,
        news_id_to_index=news_id_to_index,
        batch_sizes=train_bucket_batch_sizes,
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
        news_abstract_token_ids=features["news_abstract_token_ids"],
        news_abstract_mask=features["news_abstract_mask"],
        num_entities=len(features["entity_to_index"]),
        news_entity_ids=features["news_entity_ids"],
        news_entity_mask=features["news_entity_mask"],
        embedding_dim=config.embedding_dim,
        use_entities=config.use_entities,
        dropout=config.dropout,
    ).to(device)

    criterion = build_criterion(config.loss_type)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_lr,
    )
    trainable_params = count_parameters(model)

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device = {device}")
    print(f"train samples = {len(train_samples)}")
    print(f"dev samples = {len(dev_samples)}")
    print(f"trainable params = {trainable_params:,}")
    print(f"loss type = {config.loss_type}")
    print(f"use entities = {config.use_entities}")
    print(f"dev batch size = {config.batch_size}")
    for bucket_name in BUCKET_ORDER:
        loader = train_loaders.get(bucket_name)
        if loader is None:
            print(f"train bucket[{bucket_name}] samples=0 batch_size={train_bucket_batch_sizes[bucket_name]}")
            continue

        print(
            f"train bucket[{bucket_name}] "
            f"samples={len(loader.dataset)} batch_size={train_bucket_batch_sizes[bucket_name]} "
            f"steps={len(loader)}"
        )
    print(
        "lr scheduler = ReduceLROnPlateau("
        f"mode=min, factor={config.lr_scheduler_factor}, "
        f"patience={config.lr_scheduler_patience}, min_lr={config.min_lr}"
        ")"
    )

    wandb_run = init_wandb_run(
        config=config,
        train_sample_count=len(train_samples),
        dev_sample_count=len(dev_samples),
        trainable_params=trainable_params,
    )

    best_selection_score = float("-inf")
    global_step = 0
    epochs_without_improvement = 0

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()

            running_loss = 0.0
            running_loss_weight = 0
            num_train_steps = sum(len(loader) for loader in train_loaders.values())
            step = 0
            epoch_bucket_order = list(BUCKET_ORDER)
            random.Random(epoch).shuffle(epoch_bucket_order)

            for bucket_name in epoch_bucket_order:
                train_loader = train_loaders.get(bucket_name)
                if train_loader is None:
                    continue

                for batch in train_loader:
                    step += 1
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
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss encountered at epoch={epoch}, "
                            f"step={step}, bucket={bucket_name}."
                        )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    global_step += 1

                    loss_weight = get_loss_weight(config.loss_type, batch["candidate_mask"])
                    running_loss += loss.item() * loss_weight
                    running_loss_weight += loss_weight

                    should_log_step = step % config.log_interval == 0 or step == num_train_steps
                    if should_log_step:
                        avg_train_loss = running_loss / max(running_loss_weight, 1)
                        print(
                            f"[epoch {epoch}] step {step} "
                            f"bucket={bucket_name} train_loss={avg_train_loss:.4f}"
                        )

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

            train_loss = running_loss / max(running_loss_weight, 1)
            dev_metrics = evaluate(model, dev_loader, criterion, config.loss_type, device)
            current_selection_score = dev_metrics["SelectionScore"]
            is_best_checkpoint = current_selection_score > best_selection_score
            has_meaningful_improvement = current_selection_score > (
                best_selection_score + config.early_stopping_min_delta
            )

            print(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:.4f} "
                f"dev_loss={dev_metrics['loss']:.4f} "
                f"AUC={dev_metrics['AUC']:.4f} "
                f"MRR={dev_metrics['MRR']:.4f} "
                f"nDCG@5={dev_metrics['nDCG@5']:.4f} "
                f"nDCG@10={dev_metrics['nDCG@10']:.4f} "
                f"BrierScore={dev_metrics['BrierScore']:.4f} "
                f"Recall@10={dev_metrics['Recall@10']:.4f} "
                f"selection_score={dev_metrics['SelectionScore']:.4f}"
            )

            scheduler.step(dev_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[epoch {epoch}] lr={current_lr:.6g}")

            if is_best_checkpoint:
                best_selection_score = current_selection_score
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
                    wandb_run.summary["best/selection_score"] = best_selection_score
                    wandb_run.summary["best/epoch"] = epoch
                    wandb_run.summary["best/checkpoint_path"] = str(config.checkpoint_path)
                    wandb_run.summary["best/dev_nDCG@10"] = dev_metrics["nDCG@10"]
                    wandb_run.summary["best/dev_BrierScore"] = dev_metrics["BrierScore"]
                    wandb_run.summary["best/dev_Recall@10"] = dev_metrics["Recall@10"]

            if epoch >= config.early_stopping_min_epochs:
                if has_meaningful_improvement:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(
                    "early stopping monitor: "
                    f"{epochs_without_improvement}/{config.early_stopping_patience} "
                    f"epochs without selection_score improvement > {config.early_stopping_min_delta:.4f}"
                )

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
                        "epoch/dev_BrierScore": dev_metrics["BrierScore"],
                        "epoch/dev_Recall@10": dev_metrics["Recall@10"],
                        "epoch/dev_GlobalCTR": dev_metrics["GlobalCTR"],
                        "epoch/dev_BrierRef": dev_metrics["BrierRef"],
                        "epoch/ranking_score": dev_metrics["RankingScore"],
                        "epoch/calibration_score": dev_metrics["CalibrationScore"],
                        "epoch/selection_score": dev_metrics["SelectionScore"],
                        "epoch/lr": current_lr,
                        "epoch/best_selection_score": best_selection_score,
                        "epoch/is_best_checkpoint": int(is_best_checkpoint),
                        "epoch/has_meaningful_improvement": int(has_meaningful_improvement),
                        "epoch/epochs_without_improvement": epochs_without_improvement,
                    }
                )
                wandb_run.summary["latest/epoch"] = epoch
                wandb_run.summary["latest/dev_loss"] = dev_metrics["loss"]
                wandb_run.summary["latest/dev_MRR"] = dev_metrics["MRR"]
                wandb_run.summary["latest/dev_nDCG@10"] = dev_metrics["nDCG@10"]
                wandb_run.summary["latest/dev_BrierScore"] = dev_metrics["BrierScore"]
                wandb_run.summary["latest/dev_Recall@10"] = dev_metrics["Recall@10"]
                wandb_run.summary["latest/dev_GlobalCTR"] = dev_metrics["GlobalCTR"]
                wandb_run.summary["latest/dev_BrierRef"] = dev_metrics["BrierRef"]
                wandb_run.summary["latest/ranking_score"] = dev_metrics["RankingScore"]
                wandb_run.summary["latest/calibration_score"] = dev_metrics["CalibrationScore"]
                wandb_run.summary["latest/selection_score"] = dev_metrics["SelectionScore"]
                wandb_run.summary["latest/lr"] = current_lr

            should_early_stop = (
                epoch >= config.early_stopping_min_epochs
                and epochs_without_improvement >= config.early_stopping_patience
            )
            if should_early_stop:
                print(
                    "early stopping triggered: "
                    f"selection_score did not improve by more than {config.early_stopping_min_delta:.4f} "
                    f"for {config.early_stopping_patience} consecutive epochs."
                )
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
