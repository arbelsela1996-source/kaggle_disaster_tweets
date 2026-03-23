from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import config
from src.data import load_train_dataframe
from src.datasets import TweetDataset
from src.modeling import create_model_and_tokenizer
from src.plots import plot_confusion_matrix, plot_loss_curves, plot_metric_curves
from src.train_utils import (
    compute_classification_metrics,
    format_classification_report,
    set_seed,
    train_one_epoch,
    evaluate,
)


@dataclass
class ExperimentSpec:
    name: str
    description: str
    model_name: str
    learning_rate: float
    num_epochs: int
    max_length: int
    seeds: List[int]
    threshold_tuning: bool = False
    use_kfold: bool = False
    n_splits: int = 3


def ensure_output_dirs() -> None:
    os.makedirs(config.outputs_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)


def collect_val_probabilities(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_classification_metrics(y_true.tolist(), (y_prob >= 0.5).astype(int).tolist())
    best_score = best_metrics["f1_macro"]

    for thr in np.arange(0.30, 0.71, 0.02):
        preds = (y_prob >= thr).astype(int)
        metrics = compute_classification_metrics(y_true.tolist(), preds.tolist())
        score = metrics["f1_macro"]
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def train_single_split(
    train_df,
    val_df,
    spec: ExperimentSpec,
    seed: int,
    device: torch.device,
    output_dir: str,
    save_artifacts: bool = False,
) -> Dict[str, float]:
    set_seed(seed)
    model, tokenizer = create_model_and_tokenizer(spec.model_name)
    model.to(device)

    train_dataset = TweetDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=spec.max_length,
    )
    val_dataset = TweetDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["target"].tolist(),
        tokenizer=tokenizer,
        max_length=spec.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * spec.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_score = -1.0
    best_model_state = None
    best_epoch_metrics: Dict[str, float] = {}
    best_epoch_cm = None
    best_epoch_labels = None
    best_epoch_preds = None

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_f1_macros: List[float] = []

    for epoch in range(1, spec.num_epochs + 1):
        print(f"\n[{spec.name}] seed={seed} epoch {epoch}/{spec.num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_loss, metrics, y_true, y_pred, cm = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_macros.append(metrics["f1_macro"])

        print(
            f"Train loss={train_loss:.4f}, "
            f"Val loss={val_loss:.4f}, "
            f"Val acc={metrics['accuracy']:.4f}, "
            f"Val F1={metrics['f1']:.4f}, "
            f"Val macro-F1={metrics['f1_macro']:.4f}"
        )

        # Select best checkpoint by macro-F1 to reduce class bias.
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_metrics = metrics
            best_epoch_cm = cm
            best_epoch_labels = y_true
            best_epoch_preds = y_pred

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_metrics = dict(best_epoch_metrics)
    final_threshold = 0.5

    if spec.threshold_tuning:
        y_true, y_prob = collect_val_probabilities(model, val_loader, device)
        tuned_threshold, tuned_metrics = tune_threshold(y_true, y_prob)
        tuned_preds = (y_prob >= tuned_threshold).astype(int)
        final_threshold = tuned_threshold
        final_metrics = tuned_metrics
        best_epoch_cm = confusion_matrix(y_true, tuned_preds)
        best_epoch_labels = y_true
        best_epoch_preds = tuned_preds
        print(
            f"[{spec.name}] seed={seed} threshold tuning: "
            f"best_threshold={tuned_threshold:.2f}, macro-F1={tuned_metrics['f1_macro']:.4f}"
        )

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "best_model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        plot_loss_curves(
            train_losses,
            val_losses,
            save_path=os.path.join(output_dir, "loss_curves.png"),
        )
        plot_metric_curves(
            val_f1_macros,
            val_f1_macros,
            metric_name="Macro-F1",
            save_path=os.path.join(output_dir, "macro_f1_curve.png"),
        )
        plot_confusion_matrix(
            best_epoch_cm,
            {0: "Not disaster", 1: "Disaster"},
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

        report = format_classification_report(best_epoch_labels.tolist(), best_epoch_preds.tolist())
        with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    final_metrics["threshold"] = final_threshold
    return final_metrics


def run_experiment(
    df,
    spec: ExperimentSpec,
    device: torch.device,
    output_root: str,
) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(f"Running {spec.name}: {spec.description}")
    print("=" * 80)

    split_results: List[Dict[str, float]] = []
    result_dir = os.path.join(output_root, spec.name)
    os.makedirs(result_dir, exist_ok=True)

    if spec.use_kfold:
        skf = StratifiedKFold(n_splits=spec.n_splits, shuffle=True, random_state=42)
        texts = df["text"].values
        labels = df["target"].values
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
            fold_train = df.iloc[train_idx].reset_index(drop=True)
            fold_val = df.iloc[val_idx].reset_index(drop=True)
            fold_result = train_single_split(
                fold_train,
                fold_val,
                spec,
                seed=42,
                device=device,
                output_dir=os.path.join(result_dir, f"fold_{fold_idx}"),
                save_artifacts=False,
            )
            split_results.append(fold_result)
            print(
                f"[{spec.name}] fold={fold_idx}/{spec.n_splits}, "
                f"macro-F1={fold_result['f1_macro']:.4f}"
            )
    else:
        for seed in spec.seeds:
            train_df, val_df = train_test_split(
                df,
                test_size=config.val_size,
                random_state=seed,
                stratify=df["target"],
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            run_result = train_single_split(
                train_df,
                val_df,
                spec,
                seed=seed,
                device=device,
                output_dir=os.path.join(result_dir, f"seed_{seed}"),
                save_artifacts=False,
            )
            split_results.append(run_result)
            print(
                f"[{spec.name}] seed={seed}, "
                f"acc={run_result['accuracy']:.4f}, "
                f"macro-F1={run_result['f1_macro']:.4f}, "
                f"threshold={run_result['threshold']:.2f}"
            )

    macro_f1_values = np.array([r["f1_macro"] for r in split_results], dtype=float)
    acc_values = np.array([r["accuracy"] for r in split_results], dtype=float)
    threshold_values = np.array([r.get("threshold", 0.5) for r in split_results], dtype=float)

    summary = {
        "name": spec.name,
        "description": spec.description,
        "mean_accuracy": float(acc_values.mean()),
        "std_accuracy": float(acc_values.std()),
        "mean_macro_f1": float(macro_f1_values.mean()),
        "std_macro_f1": float(macro_f1_values.std()),
        "mean_threshold": float(threshold_values.mean()),
    }

    # Stability-aware score: reward high macro-F1 while penalizing volatility.
    summary["selection_score"] = summary["mean_macro_f1"] - 0.25 * summary["std_macro_f1"]

    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(
        f"[{spec.name}] mean acc={summary['mean_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f}), "
        f"mean macro-F1={summary['mean_macro_f1']:.4f} (+/- {summary['std_macro_f1']:.4f})"
    )
    return summary


def build_experiments() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="E1_epochs_5",
            description="Phase 1: baseline with longer training",
            model_name="bert-base-uncased",
            learning_rate=2e-5,
            num_epochs=5,
            max_length=128,
            seeds=[42, 123, 777],
        ),
        ExperimentSpec(
            name="E2_lr_1e5",
            description="Phase 1: longer training with lower learning rate",
            model_name="bert-base-uncased",
            learning_rate=1e-5,
            num_epochs=5,
            max_length=128,
            seeds=[42, 123, 777],
        ),
        ExperimentSpec(
            name="E3_longer_seq",
            description="Phase 1: lower LR plus longer sequence length",
            model_name="bert-base-uncased",
            learning_rate=1e-5,
            num_epochs=5,
            max_length=160,
            seeds=[42, 123, 777],
        ),
        ExperimentSpec(
            name="E4_bertweet",
            description="Phase 2: tweet-specific encoder",
            model_name="vinai/bertweet-base",
            learning_rate=1e-5,
            num_epochs=5,
            max_length=160,
            seeds=[42, 123, 777],
        ),
        ExperimentSpec(
            name="E5_bertweet_threshold",
            description="Phase 4: tweet-specific encoder plus threshold tuning",
            model_name="vinai/bertweet-base",
            learning_rate=1e-5,
            num_epochs=5,
            max_length=160,
            seeds=[42, 123, 777],
            threshold_tuning=True,
        ),
        ExperimentSpec(
            name="E6_kfold_reliability",
            description="Phase 3: reliability check with stratified k-fold",
            model_name="vinai/bertweet-base",
            learning_rate=1e-5,
            num_epochs=3,
            max_length=160,
            seeds=[42],
            threshold_tuning=True,
            use_kfold=True,
            n_splits=3,
        ),
    ]


def main() -> None:
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading full dataset...")
    df = load_train_dataframe(config.train_file)
    print(f"Loaded {len(df)} rows")

    experiments = build_experiments()
    all_results: List[Dict[str, float]] = []
    benchmark_root = os.path.join(config.outputs_dir, "benchmark_runs")
    os.makedirs(benchmark_root, exist_ok=True)

    for spec in experiments:
        summary = run_experiment(df, spec, device, benchmark_root)
        all_results.append(summary)

    all_results_sorted = sorted(all_results, key=lambda x: x["selection_score"], reverse=True)
    best = all_results_sorted[0]

    print("\n" + "#" * 80)
    print("EXPERIMENT SUMMARY (ranked by stability-aware score)")
    print("#" * 80)
    for rank, res in enumerate(all_results_sorted, start=1):
        print(
            f"{rank}. {res['name']}: "
            f"mean_macro_f1={res['mean_macro_f1']:.4f}, "
            f"mean_acc={res['mean_accuracy']:.4f}, "
            f"std_macro_f1={res['std_macro_f1']:.4f}, "
            f"selection_score={res['selection_score']:.4f}"
        )

    print("\nBEST APPROACH")
    print(
        f"- Experiment: {best['name']}\n"
        f"- Why: highest stability-aware score using macro-F1 and variance penalty.\n"
        f"- Expected macro-F1: {best['mean_macro_f1']:.4f} (+/- {best['std_macro_f1']:.4f})\n"
        f"- Expected accuracy: {best['mean_accuracy']:.4f} (+/- {best['std_accuracy']:.4f})\n"
        f"- Suggested threshold: {best['mean_threshold']:.2f}"
    )


if __name__ == "__main__":
    main()

