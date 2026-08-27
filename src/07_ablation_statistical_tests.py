"""
Ablation and statistical tests for the Auto-MCDM framework on the held-out test
split.

The script evaluates whether the automated components add defensible value:
- manual fixed MCDM baselines,
- exhaustive Auto-MCDM baseline,
- single-optimizer archive solutions,
- merged five-optimizer Pareto archive,
- selected knee point Pareto Auto-MCDM solution.

Paired BCa bootstrap confidence intervals are reported for average precision and
F1 at the deployment budget. Holm-Bonferroni correction is applied across the
strategy comparisons.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


TEST_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_test_predictions.parquet")
BASELINE_RANKING_DEFAULT = Path("outputs/auto_mcdm/rankings/selected_auto_mcdm_ranking.parquet")
PARETO_RANKING_DEFAULT = Path("outputs/pareto_ensemble/rankings/pareto_selected_auto_mcdm_ranking.parquet")
PARETO_ARCHIVE_DEFAULT = Path("outputs/pareto_ensemble/tables/table_03_unique_archive.csv")
PARETO_SELECTED_DEFAULT = Path("outputs/pareto_ensemble/tables/table_05_selected_pareto_solutions.csv")
OUTPUT_DEFAULT = Path("outputs/ablation")
AUTO_MCDM_PATH = Path(__file__).with_name("04_auto_mcdm.py")

FIXED_THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20]


def load_auto_mcdm_module() -> Any:
    spec = importlib.util.spec_from_file_location("auto_mcdm_lib", AUTO_MCDM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Auto-MCDM module at {AUTO_MCDM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MCDM = load_auto_mcdm_module()


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def composite_label(frame: pd.DataFrame) -> pd.Series:
    labels = frame[MCDM.LABEL_COLUMNS].copy()
    available = labels.notna().any(axis=1)
    composite = labels.fillna(0).max(axis=1)
    return composite.where(available, np.nan)


def score_manual_mcdm(criteria: pd.DataFrame, method: str, normalization: str, weights: np.ndarray | None = None) -> np.ndarray:
    raw_matrix = criteria.to_numpy(dtype=np.float64)
    matrix = MCDM.normalize_matrix(raw_matrix, normalization, reference=raw_matrix)
    if weights is None:
        weights = np.ones(matrix.shape[1], dtype=float) / matrix.shape[1]
    scores = MCDM.MCDM_FUNCTIONS[method](matrix, weights)
    return MCDM.safe_minmax(scores.reshape(-1, 1)).ravel()


def decode_archive_row(row: pd.Series, criteria_names: list[str]) -> dict[str, Any]:
    mask = np.array(
        [str(row[f"criterion_selected_{criterion}"]).lower() == "true" for criterion in criteria_names],
        dtype=bool,
    )
    weights = np.array([float(row[f"weight_{criterion}"]) for criterion in criteria_names], dtype=float)
    selected_weights = weights[mask]
    selected_weights = selected_weights / max(selected_weights.sum(), 1e-12)
    return {
        "mask": mask,
        "weights": selected_weights,
        "method": str(row["method"]),
        "normalization": str(row["normalization"]),
        "top_k_rate": float(row["top_k_rate"]),
    }


def score_archive_config(criteria: pd.DataFrame, row: pd.Series) -> np.ndarray:
    criteria_names = criteria.columns.tolist()
    decoded = decode_archive_row(row, criteria_names)
    raw_matrix = criteria.to_numpy(dtype=np.float64)
    matrix = MCDM.normalize_matrix(raw_matrix, decoded["normalization"], reference=raw_matrix)
    matrix = matrix[:, decoded["mask"]]
    scores = MCDM.MCDM_FUNCTIONS[decoded["method"]](matrix, decoded["weights"])
    return MCDM.safe_minmax(scores.reshape(-1, 1)).ravel()


def metric_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    k = max(int(math.ceil(len(y_true) * threshold)), 1)
    order = np.argsort(-scores)[:k]
    hits = int(y_true[order].sum())
    positives = max(int(y_true.sum()), 1)
    precision = hits / k
    recall = hits / positives
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    baseline = positives / len(y_true)
    return {
        "threshold_rate": threshold,
        "top_k": k,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift": precision / baseline if baseline > 0 else math.nan,
        "ndcg": MCDM.ndcg_at_k(y_true, scores, k),
    }


def evaluate_scores(name: str, scores: np.ndarray, labels: pd.Series, thresholds: list[float]) -> tuple[dict[str, Any], pd.DataFrame]:
    mask = labels.notna().to_numpy()
    y_true = labels.loc[mask].astype(int).to_numpy()
    y_scores = scores[mask]
    threshold_rows = []
    for threshold in thresholds:
        row = metric_at_threshold(y_true, y_scores, threshold)
        row["strategy"] = name
        threshold_rows.append(row)
    threshold_frame = pd.DataFrame(threshold_rows)
    best = threshold_frame.sort_values(["f1", "ndcg", "precision"], ascending=False).iloc[0]
    summary = {
        "strategy": name,
        "average_precision": float(average_precision_score(y_true, y_scores)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) == 2 else math.nan,
        "best_threshold_rate": float(best["threshold_rate"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
        "best_ndcg": float(best["ndcg"]),
        "best_lift": float(best["lift"]),
    }
    for threshold in [0.05, 0.10, 0.20]:
        row = threshold_frame[np.isclose(threshold_frame["threshold_rate"], threshold)]
        if not row.empty:
            selected = row.iloc[0]
            suffix = f"{int(threshold * 100)}pct"
            summary[f"precision_at_{suffix}"] = float(selected["precision"])
            summary[f"recall_at_{suffix}"] = float(selected["recall"])
            summary[f"f1_at_{suffix}"] = float(selected["f1"])
            summary[f"ndcg_at_{suffix}"] = float(selected["ndcg"])
    return summary, threshold_frame


def evaluate_archive_full(
    archive: pd.DataFrame,
    criteria: pd.DataFrame,
    labels: pd.Series,
    max_configs: int = 0,
) -> pd.DataFrame:
    if max_configs > 0 and len(archive) > max_configs:
        archive = archive.sort_values("primary_average_precision", ascending=False).head(max_configs).copy()

    mask = labels.notna().to_numpy()
    y_true = labels.loc[mask].astype(int).to_numpy()
    rows = []
    for index, row in archive.iterrows():
        scores = score_archive_config(criteria, row)[mask]
        selected_threshold = float(row["top_k_rate"])
        selected_metrics = metric_at_threshold(y_true, scores, selected_threshold)
        fixed_metrics = metric_at_threshold(y_true, scores, 0.20)
        rows.append(
            {
                "archive_row": int(index),
                "algorithm": row["algorithm"],
                "method": row["method"],
                "normalization": row["normalization"],
                "selected_criteria_count": int(float(row["selected_criteria_count"])),
                "top_k_rate": selected_threshold,
                "test_average_precision": float(average_precision_score(y_true, scores)),
                "test_roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else math.nan,
                "test_selected_f1": selected_metrics["f1"],
                "test_selected_precision": selected_metrics["precision"],
                "test_selected_recall": selected_metrics["recall"],
                "test_selected_ndcg": selected_metrics["ndcg"],
                "test_f1_at_20pct": fixed_metrics["f1"],
                "test_precision_at_20pct": fixed_metrics["precision"],
                "test_recall_at_20pct": fixed_metrics["recall"],
                "test_ndcg_at_20pct": fixed_metrics["ndcg"],
            }
        )
    return pd.DataFrame(rows)


def bca_interval(samples: np.ndarray, point_estimate: float, alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap confidence interval."""
    from scipy.stats import norm

    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return math.nan, math.nan

    proportion_less = float(np.mean(samples < point_estimate))
    proportion_less = min(max(proportion_less, 1e-6), 1 - 1e-6)
    z0 = float(norm.ppf(proportion_less))

    jackknife = []
    n = len(samples)
    if n > 1:
        sum_samples = samples.sum()
        for index in range(n):
            jackknife.append((sum_samples - samples[index]) / (n - 1))
        jackknife = np.asarray(jackknife)
        jackknife_mean = jackknife.mean()
        numerator = np.sum((jackknife_mean - jackknife) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife) ** 2)) ** 1.5
        acceleration = float(numerator / denominator) if denominator > 0 else 0.0
    else:
        acceleration = 0.0

    z_alpha_lower = float(norm.ppf(alpha / 2))
    z_alpha_upper = float(norm.ppf(1 - alpha / 2))
    lower_quantile = float(norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower))))
    upper_quantile = float(norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper))))
    lower_quantile = min(max(lower_quantile, 0.0), 1.0)
    upper_quantile = min(max(upper_quantile, 0.0), 1.0)
    return float(np.quantile(samples, lower_quantile)), float(np.quantile(samples, upper_quantile))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1] if not math.isnan(item[1]) else 1.0)
    m = len(p_values)
    adjusted = [math.nan] * m
    running_max = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        if math.isnan(p):
            adjusted[original_index] = math.nan
            continue
        adjusted_value = min(max((m - rank) * p, running_max), 1.0)
        adjusted[original_index] = adjusted_value
        running_max = adjusted_value
    return adjusted


def paired_bootstrap(
    labels: pd.Series,
    scores_by_strategy: dict[str, np.ndarray],
    reference_strategy: str,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = labels.notna().to_numpy()
    y_true = labels.loc[mask].astype(int).to_numpy()
    scores = {name: values[mask] for name, values in scores_by_strategy.items()}
    rng = np.random.default_rng(seed)
    n = len(y_true)

    metric_rows = []
    iteration_count = 0
    while iteration_count < iterations:
        sample = rng.integers(0, n, size=n)
        y_sample = y_true[sample]
        if len(np.unique(y_sample)) < 2:
            continue
        for strategy, strategy_scores in scores.items():
            sampled_scores = strategy_scores[sample]
            ap = average_precision_score(y_sample, sampled_scores)
            f1_20 = metric_at_threshold(y_sample, sampled_scores, 0.20)["f1"]
            metric_rows.append(
                {
                    "iteration": iteration_count,
                    "strategy": strategy,
                    "average_precision": ap,
                    "f1_at_20pct": f1_20,
                }
            )
        iteration_count += 1

    metrics = pd.DataFrame(metric_rows)
    comparison_rows = []
    reference = metrics[metrics["strategy"].eq(reference_strategy)].sort_values("iteration")
    ap_observed = {
        strategy: float(average_precision_score(y_true, scores[strategy]))
        for strategy in scores
    }
    f1_observed = {
        strategy: float(metric_at_threshold(y_true, scores[strategy], 0.20)["f1"])
        for strategy in scores
    }

    raw_p_values: dict[tuple[str, str], float] = {}
    for strategy in scores:
        if strategy == reference_strategy:
            continue
        current = metrics[metrics["strategy"].eq(strategy)].sort_values("iteration")
        merged = reference.merge(current, on="iteration", suffixes=("_reference", "_comparison"))
        for metric in ["average_precision", "f1_at_20pct"]:
            differences = (merged[f"{metric}_comparison"] - merged[f"{metric}_reference"]).to_numpy()
            differences = differences[np.isfinite(differences)]
            mean_diff = float(np.mean(differences)) if len(differences) > 0 else math.nan
            observed_diff = (
                ap_observed[strategy] - ap_observed[reference_strategy]
                if metric == "average_precision"
                else f1_observed[strategy] - f1_observed[reference_strategy]
            )
            lower, upper = bca_interval(differences, observed_diff)
            prob_better = float((differences > 0).mean()) if len(differences) > 0 else math.nan
            p_value = float(2 * min(prob_better, 1 - prob_better)) if not math.isnan(prob_better) else math.nan
            raw_p_values[(strategy, metric)] = p_value
            comparison_rows.append(
                {
                    "reference_strategy": reference_strategy,
                    "comparison_strategy": strategy,
                    "metric": metric,
                    "difference_definition": "comparison minus reference on the test split",
                    "observed_difference": float(observed_diff),
                    "mean_bootstrap_difference": mean_diff,
                    "bca_ci_lower": lower,
                    "bca_ci_upper": upper,
                    "probability_comparison_better": prob_better,
                    "favoured_strategy": strategy if (not math.isnan(mean_diff)) and mean_diff > 0 else reference_strategy,
                    "raw_p_value": p_value,
                }
            )

    if comparison_rows:
        keys = list(raw_p_values.keys())
        p_values = [raw_p_values[key] for key in keys]
        adjusted = holm_bonferroni(p_values)
        adjusted_lookup = dict(zip(keys, adjusted))
        for row in comparison_rows:
            row["holm_bonferroni_p_value"] = adjusted_lookup[(row["comparison_strategy"], row["metric"])]

    return metrics, pd.DataFrame(comparison_rows)


def save_ablation_bar(summary: pd.DataFrame, output_path: Path) -> None:
    plot = summary.sort_values("average_precision", ascending=False)
    labels = [clean_caption(value) for value in plot["strategy"]]
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10.8, 5.7), dpi=180)
    width = 0.36
    ax.bar(x - width / 2, plot["average_precision"], width=width, label="Average precision", color="#5b7f95", edgecolor="#25323a", linewidth=0.6)
    ax.bar(x + width / 2, plot["best_f1"], width=width, label="Best F1", color="#d1a15f", edgecolor="#5f4a2b", linewidth=0.6)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Metric value on test split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.16, 1, 1))
    fig.text(0.5, 0.025, "Figure 1. Ablation performance comparison on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_algorithm_box(archive_eval: pd.DataFrame, output_path: Path) -> None:
    algorithms = sorted(archive_eval["algorithm"].unique())
    values = [archive_eval.loc[archive_eval["algorithm"].eq(algorithm), "test_f1_at_20pct"].to_numpy() for algorithm in algorithms]
    fig, ax = plt.subplots(figsize=(9.0, 5.3), dpi=180)
    ax.boxplot(values, tick_labels=[clean_caption(value) for value in algorithms], patch_artist=True)
    for patch in ax.artists:
        patch.set_facecolor("#6f8795")
    ax.set_ylabel("Test F1 at 20 percent budget")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 2. Single optimizer archive distribution on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_bootstrap_diff(comparisons: pd.DataFrame, output_path: Path) -> None:
    plot = comparisons[comparisons["metric"].eq("average_precision")].copy()
    plot = plot.sort_values("mean_bootstrap_difference")
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=180)
    y = np.arange(len(plot))
    ax.errorbar(
        plot["mean_bootstrap_difference"],
        y,
        xerr=[
            plot["mean_bootstrap_difference"] - plot["bca_ci_lower"],
            plot["bca_ci_upper"] - plot["mean_bootstrap_difference"],
        ],
        fmt="o",
        color="#315f72",
        ecolor="#7c8f63",
        capsize=3,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, [clean_caption(value) for value in plot["comparison_strategy"]])
    ax.set_xlabel("Average precision difference versus Pareto Auto MCDM on test split")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 3. Bootstrap average precision differences with BCa intervals", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_threshold_lines(thresholds: pd.DataFrame, output_path: Path) -> None:
    subset = thresholds[thresholds["strategy"].isin(["manual topsis equal", "baseline auto mcdm", "pareto auto mcdm"])].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    for strategy, group in subset.groupby("strategy"):
        ax.plot(group["threshold_rate"] * 100, group["f1"], marker="o", label=clean_caption(strategy))
    ax.set_xlabel("Review threshold percent")
    ax.set_ylabel("F1 on test split")
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 4. Threshold sensitivity of ablation strategies on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_ablation(
    test_predictions_path: Path,
    baseline_ranking_path: Path,
    pareto_ranking_path: Path,
    pareto_archive_path: Path,
    pareto_selected_path: Path,
    output_dir: Path,
    bootstrap_iterations: int,
    seed: int,
) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(test_predictions_path)
    predictions["y_any_risk"] = composite_label(predictions)

    baseline_ranking = pd.read_parquet(baseline_ranking_path)
    pareto_ranking = pd.read_parquet(pareto_ranking_path)
    baseline_test = baseline_ranking[baseline_ranking["split"].eq("test")][["sample_row_id", "auto_mcdm_score"]]
    pareto_test = pareto_ranking[pareto_ranking["split"].eq("test")][["sample_row_id", "pareto_auto_mcdm_score"]]

    frame = predictions.merge(baseline_test, on="sample_row_id", how="inner").merge(pareto_test, on="sample_row_id", how="inner")
    target = frame["y_cri_high"]
    criteria, _ = MCDM.make_criteria(frame)

    archive = pd.read_csv(pareto_archive_path)
    selected_solutions = pd.read_csv(pareto_selected_path)
    selected_knee = selected_solutions[selected_solutions["selector"].eq("knee")].iloc[0]

    manual_topsis = score_manual_mcdm(criteria, method="topsis", normalization="minmax")
    manual_vikor = score_manual_mcdm(criteria, method="vikor", normalization="minmax")
    manual_waspas = score_manual_mcdm(criteria, method="waspas", normalization="minmax")
    pareto_selected_from_archive = score_archive_config(criteria, selected_knee)

    scores_by_strategy = {
        "raw cri probability": frame["p_cri_high"].to_numpy(dtype=float),
        "manual topsis equal": manual_topsis,
        "manual vikor equal": manual_vikor,
        "manual waspas equal": manual_waspas,
        "baseline auto mcdm": frame["auto_mcdm_score"].to_numpy(dtype=float),
        "pareto auto mcdm": frame["pareto_auto_mcdm_score"].to_numpy(dtype=float),
    }

    reproduction_difference = float(np.max(np.abs(pareto_selected_from_archive - scores_by_strategy["pareto auto mcdm"])))

    summary_rows = []
    threshold_frames = []
    for strategy, scores in scores_by_strategy.items():
        summary, threshold_metrics = evaluate_scores(strategy, scores, target, FIXED_THRESHOLDS)
        summary_rows.append(summary)
        threshold_frames.append(threshold_metrics)
    strategy_summary = pd.DataFrame(summary_rows)
    threshold_summary = pd.concat(threshold_frames, ignore_index=True)

    archive_full = evaluate_archive_full(archive, criteria, target)
    best_by_algorithm_ap = archive_full.sort_values("test_average_precision", ascending=False).groupby("algorithm").head(1)
    best_by_algorithm_f1 = archive_full.sort_values("test_f1_at_20pct", ascending=False).groupby("algorithm").head(1)

    bootstrap_metrics, bootstrap_comparisons = paired_bootstrap(
        labels=target,
        scores_by_strategy=scores_by_strategy,
        reference_strategy="pareto auto mcdm",
        iterations=bootstrap_iterations,
        seed=seed,
    )

    reproduction = pd.DataFrame(
        [
            {
                "check": "pareto selected score reproduction",
                "status": "pass" if reproduction_difference < 1e-5 else "review",
                "max_absolute_difference": reproduction_difference,
            }
        ]
    )

    strategy_summary.to_csv(output_dir / "tables" / "table_01_ablation_strategy_summary.csv", index=False)
    threshold_summary.to_csv(output_dir / "tables" / "table_02_threshold_ablation.csv", index=False)
    archive_full.to_csv(output_dir / "tables" / "table_03_archive_full_test_validation.csv", index=False)
    best_by_algorithm_ap.to_csv(output_dir / "tables" / "table_04_best_single_optimizer_by_ap.csv", index=False)
    best_by_algorithm_f1.to_csv(output_dir / "tables" / "table_05_best_single_optimizer_by_f1.csv", index=False)
    bootstrap_metrics.to_csv(output_dir / "tables" / "table_06_bootstrap_metrics.csv", index=False)
    bootstrap_comparisons.to_csv(output_dir / "tables" / "table_07_bootstrap_comparisons.csv", index=False)
    reproduction.to_csv(output_dir / "tables" / "table_08_reproduction_check.csv", index=False)

    write_markdown_table(strategy_summary, output_dir / "tables" / "table_01_ablation_strategy_summary.md")
    write_markdown_table(best_by_algorithm_ap, output_dir / "tables" / "table_04_best_single_optimizer_by_ap.md")
    write_markdown_table(bootstrap_comparisons, output_dir / "tables" / "table_07_bootstrap_comparisons.md")
    write_markdown_table(reproduction, output_dir / "tables" / "table_08_reproduction_check.md")

    save_ablation_bar(strategy_summary, output_dir / "figures" / "figure_01_ablation_performance.png")
    save_algorithm_box(archive_full, output_dir / "figures" / "figure_02_single_optimizer_archive_distribution.png")
    save_bootstrap_diff(bootstrap_comparisons, output_dir / "figures" / "figure_03_bootstrap_differences.png")
    save_threshold_lines(threshold_summary, output_dir / "figures" / "figure_04_threshold_sensitivity.png")

    best_strategy_ap = strategy_summary.sort_values("average_precision", ascending=False).iloc[0]
    best_strategy_f1 = strategy_summary.sort_values("best_f1", ascending=False).iloc[0]
    pareto_row = strategy_summary[strategy_summary["strategy"].eq("pareto auto mcdm")].iloc[0]
    baseline_row = strategy_summary[strategy_summary["strategy"].eq("baseline auto mcdm")].iloc[0]
    manual_row = strategy_summary[strategy_summary["strategy"].eq("manual topsis equal")].iloc[0]

    summary = {
        "test_rows": int(len(frame)),
        "bootstrap_iterations": bootstrap_iterations,
        "reproduction_difference": reproduction_difference,
        "best_strategy_by_average_precision": str(best_strategy_ap["strategy"]),
        "best_strategy_by_best_f1": str(best_strategy_f1["strategy"]),
        "pareto_auto_mcdm": {
            "average_precision": float(pareto_row["average_precision"]),
            "best_f1": float(pareto_row["best_f1"]),
            "f1_at_20pct": float(pareto_row["f1_at_20pct"]),
        },
        "baseline_auto_mcdm": {
            "average_precision": float(baseline_row["average_precision"]),
            "best_f1": float(baseline_row["best_f1"]),
            "f1_at_20pct": float(baseline_row["f1_at_20pct"]),
        },
        "manual_topsis_equal": {
            "average_precision": float(manual_row["average_precision"]),
            "best_f1": float(manual_row["best_f1"]),
            "f1_at_20pct": float(manual_row["f1_at_20pct"]),
        },
        "archive_configurations_evaluated": int(len(archive_full)),
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Ablation Summary",
        "",
        f"- Test rows evaluated: {len(frame):,}",
        f"- Bootstrap iterations: {bootstrap_iterations:,}",
        f"- Best strategy by AP on test: {best_strategy_ap['strategy']}",
        f"- Best strategy by F1 on test: {best_strategy_f1['strategy']}",
        f"- Pareto score reproduction difference: {reproduction_difference:.3g}",
        "",
        "Generated tables are in `outputs/ablation/tables`.",
        "Generated figures are in `outputs/ablation/figures`.",
    ]
    (output_dir / "ablation_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation and statistical tests on the held-out test split.")
    parser.add_argument("--test-predictions-path", type=Path, default=TEST_PREDICTIONS_DEFAULT)
    parser.add_argument("--baseline-ranking-path", type=Path, default=BASELINE_RANKING_DEFAULT)
    parser.add_argument("--pareto-ranking-path", type=Path, default=PARETO_RANKING_DEFAULT)
    parser.add_argument("--pareto-archive-path", type=Path, default=PARETO_ARCHIVE_DEFAULT)
    parser.add_argument("--pareto-selected-path", type=Path, default=PARETO_SELECTED_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ablation(
        test_predictions_path=args.test_predictions_path,
        baseline_ranking_path=args.baseline_ranking_path,
        pareto_ranking_path=args.pareto_ranking_path,
        pareto_archive_path=args.pareto_archive_path,
        pareto_selected_path=args.pareto_selected_path,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
