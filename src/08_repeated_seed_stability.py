"""
Repeated-seed stability evaluation for the five-algorithm Pareto Auto-MCDM.

The script reruns the Pareto archive search for multiple random seeds and
evaluates the knee point selection on the disjoint test split. It quantifies
whether the optimizer-level conclusions are stable rather than seed specific.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination


VAL_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_val_predictions.parquet")
TEST_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_test_predictions.parquet")
OUTPUT_DEFAULT = Path("outputs/repeated_seed_stability")
PARETO_MODULE_PATH = Path(__file__).with_name("05_pymoo_ensemble.py")


def load_pareto_module() -> Any:
    spec = importlib.util.spec_from_file_location("pareto_ensemble_lib", PARETO_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Pareto module at {PARETO_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARETO = load_pareto_module()


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def topk_set(scores: np.ndarray, rate: float) -> set[int]:
    k = max(int(math.ceil(len(scores) * rate)), 1)
    return set(np.argsort(-scores)[:k].tolist())


def mean_pairwise_overlap(ranking_scores: dict[int, np.ndarray], rate: float) -> float:
    seeds = sorted(ranking_scores)
    if len(seeds) < 2:
        return float("nan")
    sets = {seed: topk_set(ranking_scores[seed], rate) for seed in seeds}
    overlaps = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            left = sets[left_seed]
            right = sets[right_seed]
            overlaps.append(len(left.intersection(right)) / max(len(left), 1))
    return float(np.mean(overlaps))


def run_single_seed(
    seed: int,
    val_path: Path,
    test_path: Path,
    optimization_rows: int,
    pop_size: int,
    generations: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    data = PARETO.load_split_data(val_path, test_path, max_rows=optimization_rows, seed=seed)
    problem = PARETO.AutoMCDMProblem(data)
    termination = get_termination("n_gen", generations)

    archive_parts: list[pd.DataFrame] = []
    algorithm_rows = []
    seed_start = time.perf_counter()

    for offset, name in enumerate(PARETO.ALGORITHMS):
        algorithm = PARETO.make_algorithm(name, pop_size=pop_size, n_obj=problem.n_obj, seed=seed + offset)
        start = time.perf_counter()
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed + offset,
            verbose=False,
            save_history=False,
        )
        runtime = time.perf_counter() - start
        x_values, f_values = PARETO.collect_result_population(result)
        archive_part = pd.DataFrame(PARETO.archive_records(problem, name, x_values, f_values))
        archive_parts.append(archive_part)
        try:
            hypervolume = float(HV(ref_point=np.ones(problem.n_obj) * 1.1)(f_values))
        except Exception:
            hypervolume = math.nan
        algorithm_rows.append(
            {
                "seed": seed,
                "algorithm": name,
                "solutions_collected": len(archive_part),
                "runtime_seconds": runtime,
                "hypervolume": hypervolume,
                "best_primary_average_precision": float(archive_part["primary_average_precision"].max()),
                "best_primary_f1": float(archive_part["primary_f1"].max()),
                "best_any_f1": float(archive_part["any_f1"].max()),
            }
        )

    raw_archive = pd.concat(archive_parts, ignore_index=True)
    unique_archive, _ = PARETO.remove_duplicate_solutions(raw_archive)
    non_dominated = PARETO.extract_non_dominated(unique_archive)
    knee_index, _ = PARETO.knee_point_selection(non_dominated)
    selected = non_dominated.iloc[knee_index]

    decoded = PARETO.decoded_from_row(problem, selected)
    test_metrics = problem.evaluate_test(decoded)
    test_scores = problem.score_test(decoded)
    selected_top_k = max(int(math.ceil(len(test_scores) * float(selected["top_k_rate"]))), 1)

    summary = {
        "seed": seed,
        "runtime_seconds": time.perf_counter() - seed_start,
        "optimization_rows": len(data.val_frame),
        "test_rows": len(data.test_frame),
        "raw_archive_solutions": len(raw_archive),
        "unique_archive_solutions": len(unique_archive),
        "non_dominated_solutions": len(non_dominated),
        "selected_algorithm": str(selected["algorithm"]),
        "selected_method": str(selected["method"]),
        "selected_normalization": str(selected["normalization"]),
        "selected_top_k_rate": float(selected["top_k_rate"]),
        "selected_top_k_count_test": selected_top_k,
        "selected_criteria_count": int(selected["selected_criteria_count"]),
        "val_primary_average_precision": float(selected["primary_average_precision"]),
        "val_primary_f1": float(selected["primary_f1"]),
        "val_any_f1": float(selected["any_f1"]),
        "test_primary_average_precision": float(test_metrics["test_primary_average_precision"]),
        "test_primary_precision": float(test_metrics["test_primary_precision"]),
        "test_primary_recall": float(test_metrics["test_primary_recall"]),
        "test_primary_f1": float(test_metrics["test_primary_f1"]),
        "test_primary_ndcg": float(test_metrics["test_primary_ndcg"]),
        "test_any_precision": float(test_metrics["test_any_precision"]),
        "test_any_recall": float(test_metrics["test_any_recall"]),
        "test_any_f1": float(test_metrics["test_any_f1"]),
        "test_any_ndcg": float(test_metrics["test_any_ndcg"]),
    }

    algorithm_summary = pd.DataFrame(algorithm_rows)
    algorithm_summary["selected_seed_algorithm"] = algorithm_summary["algorithm"].eq(summary["selected_algorithm"])
    return summary, algorithm_summary, test_scores


def save_metric_stability_figure(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["test_primary_average_precision", "test_primary_f1", "test_any_f1"]
    labels = ["Test Primary AP", "Test Primary F1", "Test Any risk F1"]
    means = summary[metrics].mean()
    stds = summary[metrics].std(ddof=1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=stds, capsize=4, color="#5b7f95", edgecolor="#25323a", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Metric value on test split")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 1. Repeated seed metric stability on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_selection_frequency_figure(summary: pd.DataFrame, output_path: Path) -> None:
    counts = summary["selected_method"].value_counts()
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    bars = ax.bar(
        [clean_caption(label) for label in counts.index],
        counts.values,
        color="#d1a15f",
        edgecolor="#5f4a2b",
        linewidth=0.6,
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Selected runs")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 2. Selected MCDM method frequency", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_overlap_figure(overlap: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    ax.plot(overlap["threshold_rate"] * 100, overlap["mean_pairwise_overlap"], marker="o", color="#7c8f63")
    ax.set_xlabel("Review threshold percent")
    ax.set_ylabel("Mean pairwise top k overlap on test split")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 3. Top k overlap across repeated seeds on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_runtime_figure(algorithm_summary: pd.DataFrame, output_path: Path) -> None:
    grouped = algorithm_summary.groupby("algorithm")["runtime_seconds"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=180)
    ax.bar(
        [clean_caption(label) for label in grouped.index],
        grouped.values,
        color="#6f8795",
        edgecolor="#25323a",
        linewidth=0.6,
    )
    ax.set_ylabel("Mean runtime seconds")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 4. Optimizer runtime across repeated seeds", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_repeated_seeds(
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    seeds: list[int],
    optimization_rows: int,
    pop_size: int,
    generations: int,
) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    algorithm_frames = []
    ranking_scores = {}

    for seed in seeds:
        summary, algorithm_summary, scores = run_single_seed(
            seed=seed,
            val_path=val_path,
            test_path=test_path,
            optimization_rows=optimization_rows,
            pop_size=pop_size,
            generations=generations,
        )
        summary_rows.append(summary)
        algorithm_frames.append(algorithm_summary)
        ranking_scores[seed] = scores

    summary_df = pd.DataFrame(summary_rows)
    algorithm_df = pd.concat(algorithm_frames, ignore_index=True)
    overlap_df = pd.DataFrame(
        [
            {
                "threshold_rate": rate,
                "mean_pairwise_overlap": mean_pairwise_overlap(ranking_scores, rate),
            }
            for rate in [0.01, 0.03, 0.05, 0.10, 0.20]
        ]
    )

    metric_summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": float(summary_df[metric].mean()),
                "std": float(summary_df[metric].std(ddof=1)),
                "min": float(summary_df[metric].min()),
                "max": float(summary_df[metric].max()),
            }
            for metric in [
                "test_primary_average_precision",
                "test_primary_f1",
                "test_primary_ndcg",
                "test_any_f1",
                "runtime_seconds",
            ]
        ]
    )

    selection_frequency = summary_df.groupby(
        ["selected_algorithm", "selected_method", "selected_normalization"],
        dropna=False,
    ).size().reset_index(name="runs")

    summary_df.to_csv(output_dir / "tables" / "table_01_repeated_seed_selected_solutions.csv", index=False)
    metric_summary.to_csv(output_dir / "tables" / "table_02_repeated_seed_metric_summary.csv", index=False)
    selection_frequency.to_csv(output_dir / "tables" / "table_03_selection_frequency.csv", index=False)
    algorithm_df.to_csv(output_dir / "tables" / "table_04_algorithm_runtime_and_performance.csv", index=False)
    overlap_df.to_csv(output_dir / "tables" / "table_05_top_k_overlap.csv", index=False)

    write_markdown_table(summary_df, output_dir / "tables" / "table_01_repeated_seed_selected_solutions.md")
    write_markdown_table(metric_summary, output_dir / "tables" / "table_02_repeated_seed_metric_summary.md")
    write_markdown_table(selection_frequency, output_dir / "tables" / "table_03_selection_frequency.md")
    write_markdown_table(overlap_df, output_dir / "tables" / "table_05_top_k_overlap.md")

    save_metric_stability_figure(summary_df, output_dir / "figures" / "figure_01_repeated_seed_metric_stability.png")
    save_selection_frequency_figure(summary_df, output_dir / "figures" / "figure_02_method_selection_frequency.png")
    save_overlap_figure(overlap_df, output_dir / "figures" / "figure_03_top_k_overlap.png")
    save_runtime_figure(algorithm_df, output_dir / "figures" / "figure_04_optimizer_runtime.png")

    report = {
        "seeds": seeds,
        "optimization_rows": optimization_rows,
        "population_size": pop_size,
        "generations": generations,
        "runs": len(seeds),
        "mean_test_primary_average_precision": float(summary_df["test_primary_average_precision"].mean()),
        "std_test_primary_average_precision": float(summary_df["test_primary_average_precision"].std(ddof=1)),
        "mean_test_primary_f1": float(summary_df["test_primary_f1"].mean()),
        "std_test_primary_f1": float(summary_df["test_primary_f1"].std(ddof=1)),
        "mean_test_any_f1": float(summary_df["test_any_f1"].mean()),
        "std_test_any_f1": float(summary_df["test_any_f1"].std(ddof=1)),
        "mean_top_20_overlap": float(overlap_df.loc[np.isclose(overlap_df["threshold_rate"], 0.20), "mean_pairwise_overlap"].iloc[0]),
    }
    (output_dir / "repeated_seed_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Repeated Seed Stability Summary",
        "",
        f"- Runs: {len(seeds):,}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Optimization rows per run: {optimization_rows:,}",
        f"- Population size: {pop_size:,}",
        f"- Generations: {generations:,}",
        f"- Mean test primary AP: {report['mean_test_primary_average_precision']:.4f}",
        f"- Mean test primary F1: {report['mean_test_primary_f1']:.4f}",
        f"- Mean test any-risk F1: {report['mean_test_any_f1']:.4f}",
        "",
        "Generated tables are in `outputs/repeated_seed_stability/tables`.",
        "Generated figures are in `outputs/repeated_seed_stability/figures`.",
    ]
    (output_dir / "repeated_seed_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed Pareto Auto-MCDM stability evaluation on test split.")
    parser.add_argument("--val-path", type=Path, default=VAL_PREDICTIONS_DEFAULT)
    parser.add_argument("--test-path", type=Path, default=TEST_PREDICTIONS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--optimization-rows", type=int, default=12_000)
    parser.add_argument("--pop-size", type=int, default=60)
    parser.add_argument("--generations", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    run_repeated_seeds(
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        seeds=seeds,
        optimization_rows=args.optimization_rows,
        pop_size=args.pop_size,
        generations=args.generations,
    )


if __name__ == "__main__":
    main()
