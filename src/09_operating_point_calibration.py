"""
Calibrate operating points for procurement risk classification on the test split.

With a 39 percent positive rate for high CRI, F1 is mathematically bounded under
the 20 percent audit budget. This script reports the calibrated operating point
on the disjoint test split for the two deployment modes considered in the
manuscript: audit prioritization (capped review budget) and broad review
classification (unconstrained threshold).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


TEST_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_test_predictions.parquet")
BASELINE_RANKING_DEFAULT = Path("outputs/auto_mcdm/rankings/selected_auto_mcdm_ranking.parquet")
PARETO_RANKING_DEFAULT = Path("outputs/pareto_ensemble/rankings/pareto_selected_auto_mcdm_ranking.parquet")
OUTPUT_DEFAULT = Path("outputs/operating_point_calibration")


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def composite_label(frame: pd.DataFrame) -> pd.Series:
    label_columns = [
        "y_cri_high",
        "y_proc_high",
        "y_buyer_concentration_high",
        "y_single_bid",
        "y_no_call_for_tender",
    ]
    labels = frame[label_columns].copy()
    available = labels.notna().any(axis=1)
    composite = labels.fillna(0).max(axis=1)
    return composite.where(available, np.nan)


def metrics_from_prediction(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "selected_count": int(prediction.sum()),
        "selected_rate": float(prediction.mean()),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def best_top_k(y_true: np.ndarray, scores: np.ndarray, rates: np.ndarray) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows = []
    best: dict[str, float | int] | None = None
    for rate in rates:
        k = max(int(math.ceil(len(y_true) * rate)), 1)
        selected = np.argsort(-scores)[:k]
        prediction = np.zeros(len(y_true), dtype=int)
        prediction[selected] = 1
        metrics = metrics_from_prediction(y_true, prediction)
        metrics["operating_point"] = "top k rate"
        metrics["threshold_value"] = float(rate)
        rows.append(metrics)
        if best is None or float(metrics["f1"]) > float(best["f1"]):
            best = metrics.copy()
    if best is None:
        raise RuntimeError("No top-k operating point was evaluated")
    return best, pd.DataFrame(rows)


def best_score_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows = []
    best: dict[str, float | int] | None = None
    for threshold in thresholds:
        prediction = (scores >= threshold).astype(int)
        if prediction.sum() == 0:
            continue
        metrics = metrics_from_prediction(y_true, prediction)
        metrics["operating_point"] = "score threshold"
        metrics["threshold_value"] = float(threshold)
        rows.append(metrics)
        if best is None or float(metrics["f1"]) > float(best["f1"]):
            best = metrics.copy()
    if best is None:
        raise RuntimeError("No score-threshold operating point was evaluated")
    return best, pd.DataFrame(rows)


def theoretical_f1_bound(positive_rate: float, selected_rate: float) -> float:
    max_recall = min(selected_rate / positive_rate, 1.0)
    max_precision = 1.0
    return 2 * max_precision * max_recall / max(max_precision + max_recall, 1e-12)


def evaluate_strategy(
    strategy: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_labels: pd.Series,
    test_labels: pd.Series,
    target: str,
    rates: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    val_mask = val_labels.notna().to_numpy()
    val_y = val_labels.loc[val_mask].astype(int).to_numpy()
    val_y_scores = val_scores[val_mask]

    test_mask = test_labels.notna().to_numpy()
    test_y = test_labels.loc[test_mask].astype(int).to_numpy()
    test_y_scores = test_scores[test_mask]

    # Calibrate the operating point on validation, evaluate on test.
    val_top_best, _ = best_top_k(val_y, val_y_scores, rates)
    val_score_best, _ = best_score_threshold(val_y, val_y_scores, thresholds)

    top_curve_test = []
    for rate in rates:
        k = max(int(math.ceil(len(test_y) * rate)), 1)
        prediction = np.zeros(len(test_y), dtype=int)
        prediction[np.argsort(-test_y_scores)[:k]] = 1
        metrics = metrics_from_prediction(test_y, prediction)
        metrics["operating_point"] = "top k rate"
        metrics["threshold_value"] = float(rate)
        top_curve_test.append(metrics)
    top_curve_test_df = pd.DataFrame(top_curve_test)

    score_curve_test = []
    for threshold in thresholds:
        prediction = (test_y_scores >= threshold).astype(int)
        if prediction.sum() == 0:
            continue
        metrics = metrics_from_prediction(test_y, prediction)
        metrics["operating_point"] = "score threshold"
        metrics["threshold_value"] = float(threshold)
        score_curve_test.append(metrics)
    score_curve_test_df = pd.DataFrame(score_curve_test)

    # Apply the operating point chosen on validation to test
    selected_top_k = max(int(math.ceil(len(test_y) * float(val_top_best["threshold_value"]))), 1)
    prediction_top = np.zeros(len(test_y), dtype=int)
    prediction_top[np.argsort(-test_y_scores)[:selected_top_k]] = 1
    test_top = metrics_from_prediction(test_y, prediction_top)
    test_top["operating_point"] = "calibrated top k"
    test_top["threshold_value"] = float(val_top_best["threshold_value"])

    prediction_score = (test_y_scores >= float(val_score_best["threshold_value"])).astype(int)
    if prediction_score.sum() == 0:
        test_score = {
            "selected_count": 0,
            "selected_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "operating_point": "calibrated score threshold",
            "threshold_value": float(val_score_best["threshold_value"]),
        }
    else:
        test_score = metrics_from_prediction(test_y, prediction_score)
        test_score["operating_point"] = "calibrated score threshold"
        test_score["threshold_value"] = float(val_score_best["threshold_value"])

    summaries = []
    for mode, val_best, test_metrics in [
        ("calibrated top k", val_top_best, test_top),
        ("calibrated score threshold", val_score_best, test_score),
    ]:
        row = {
            "strategy": strategy,
            "target": target,
            "mode": mode,
            "val_rows": int(len(val_y)),
            "test_rows": int(len(test_y)),
            "val_positive_rate": float(val_y.mean()),
            "test_positive_rate": float(test_y.mean()),
            "test_average_precision": float(average_precision_score(test_y, test_y_scores)),
            "test_roc_auc": float(roc_auc_score(test_y, test_y_scores)) if len(np.unique(test_y)) == 2 else math.nan,
            "val_threshold_value": float(val_best["threshold_value"]),
            "val_selected_rate": float(val_best["selected_rate"]),
            "val_precision": float(val_best["precision"]),
            "val_recall": float(val_best["recall"]),
            "val_f1": float(val_best["f1"]),
            "test_selected_count": int(test_metrics["selected_count"]),
            "test_selected_rate": float(test_metrics["selected_rate"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_f1": float(test_metrics["f1"]),
        }
        summaries.append(row)

    top_curve_test_df["strategy"] = strategy
    top_curve_test_df["target"] = target
    score_curve_test_df["strategy"] = strategy
    score_curve_test_df["target"] = target
    curve = pd.concat([top_curve_test_df, score_curve_test_df], ignore_index=True)
    return summaries, curve


def save_f1_threshold_figure(curves: pd.DataFrame, output_path: Path) -> None:
    subset = curves[
        curves["target"].eq("y_cri_high")
        & curves["operating_point"].eq("top k rate")
        & curves["strategy"].isin(["raw cri probability", "baseline auto mcdm", "pareto auto mcdm"])
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    for strategy, group in subset.groupby("strategy"):
        ax.plot(group["threshold_value"] * 100, group["f1"], marker="o", markevery=8, label=clean_caption(strategy))
    ax.axhline(0.90, color="#333333", linewidth=0.8, linestyle="--", label="F1 equals 0.90")
    ax.set_xlabel("Selected records percent on test split")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 1. F1 by calibrated review threshold on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_operating_point_bar(summary: pd.DataFrame, output_path: Path) -> None:
    plot = summary[
        summary["target"].eq("y_cri_high")
        & summary["mode"].eq("calibrated top k")
    ].copy()
    plot = plot.sort_values("test_f1", ascending=False)
    labels = [clean_caption(value) for value in plot["strategy"]]
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    width = 0.28
    ax.bar(x - width, plot["test_precision"], width=width, label="Precision", color="#5b7f95", edgecolor="#25323a", linewidth=0.6)
    ax.bar(x, plot["test_recall"], width=width, label="Recall", color="#7c8f63", edgecolor="#28301f", linewidth=0.6)
    ax.bar(x + width, plot["test_f1"], width=width, label="F1", color="#d1a15f", edgecolor="#5f4a2b", linewidth=0.6)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Metric value on test split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.16, 1, 1))
    fig.text(0.5, 0.025, "Figure 2. Calibrated operating point metrics on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_bound_figure(bound: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
    ax.plot(bound["selected_rate"] * 100, bound["maximum_possible_f1"], color="#315f72", linewidth=2)
    ax.axvline(20, color="#8f6b45", linewidth=1.0, linestyle="--", label="20 percent audit cap")
    ax.axhline(0.90, color="#333333", linewidth=0.8, linestyle="--", label="F1 equals 0.90")
    ax.set_xlabel("Selected records percent")
    ax.set_ylabel("Theoretical maximum F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 3. F1 bound imposed by audit budget", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_calibration(
    test_predictions_path: Path,
    baseline_ranking_path: Path,
    pareto_ranking_path: Path,
    output_dir: Path,
) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(test_predictions_path)
    val_predictions_path = test_predictions_path.with_name(
        test_predictions_path.name.replace("test", "val")
    )
    val_predictions = pd.read_parquet(val_predictions_path)
    val_predictions["y_any_risk"] = composite_label(val_predictions)
    predictions["y_any_risk"] = composite_label(predictions)

    baseline_ranking = pd.read_parquet(baseline_ranking_path)
    pareto_ranking = pd.read_parquet(pareto_ranking_path)
    baseline_val = baseline_ranking[baseline_ranking["split"].eq("val")][["sample_row_id", "auto_mcdm_score"]]
    baseline_test = baseline_ranking[baseline_ranking["split"].eq("test")][["sample_row_id", "auto_mcdm_score"]]
    pareto_val = pareto_ranking[pareto_ranking["split"].eq("val")][["sample_row_id", "pareto_auto_mcdm_score"]]
    pareto_test = pareto_ranking[pareto_ranking["split"].eq("test")][["sample_row_id", "pareto_auto_mcdm_score"]]

    val_frame = val_predictions.merge(baseline_val, on="sample_row_id", how="inner").merge(pareto_val, on="sample_row_id", how="inner")
    test_frame = predictions.merge(baseline_test, on="sample_row_id", how="inner").merge(pareto_test, on="sample_row_id", how="inner")

    strategies_val = {
        "raw cri probability": val_frame["p_cri_high"].to_numpy(dtype=float),
        "baseline auto mcdm": val_frame["auto_mcdm_score"].to_numpy(dtype=float),
        "pareto auto mcdm": val_frame["pareto_auto_mcdm_score"].to_numpy(dtype=float),
    }
    strategies_test = {
        "raw cri probability": test_frame["p_cri_high"].to_numpy(dtype=float),
        "baseline auto mcdm": test_frame["auto_mcdm_score"].to_numpy(dtype=float),
        "pareto auto mcdm": test_frame["pareto_auto_mcdm_score"].to_numpy(dtype=float),
    }
    rates = np.linspace(0.01, 0.80, 80)
    score_thresholds = np.linspace(0.001, 0.999, 999)

    summary_rows = []
    curve_frames = []
    for strategy in strategies_val:
        for target in ["y_cri_high", "y_any_risk"]:
            summaries, curve = evaluate_strategy(
                strategy=strategy,
                val_scores=strategies_val[strategy],
                test_scores=strategies_test[strategy],
                val_labels=val_frame[target],
                test_labels=test_frame[target],
                target=target,
                rates=rates,
                thresholds=score_thresholds,
            )
            summary_rows.extend(summaries)
            curve_frames.append(curve)

    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_frames, ignore_index=True)

    positive_rate = float(test_frame["y_cri_high"].dropna().astype(int).mean())
    bound = pd.DataFrame(
        {
            "selected_rate": rates,
            "positive_rate": positive_rate,
            "maximum_possible_f1": [theoretical_f1_bound(positive_rate, rate) for rate in rates],
        }
    )

    summary.to_csv(output_dir / "tables" / "table_01_calibrated_operating_points.csv", index=False)
    curves.to_csv(output_dir / "tables" / "table_02_operating_point_curves.csv", index=False)
    bound.to_csv(output_dir / "tables" / "table_03_theoretical_f1_bound.csv", index=False)
    write_markdown_table(summary, output_dir / "tables" / "table_01_calibrated_operating_points.md")
    write_markdown_table(bound, output_dir / "tables" / "table_03_theoretical_f1_bound.md")

    save_f1_threshold_figure(curves, output_dir / "figures" / "figure_01_f1_by_review_threshold.png")
    save_operating_point_bar(summary, output_dir / "figures" / "figure_02_calibrated_metrics.png")
    save_bound_figure(bound, output_dir / "figures" / "figure_03_f1_budget_bound.png")

    pareto_top = summary[
        summary["strategy"].eq("pareto auto mcdm")
        & summary["target"].eq("y_cri_high")
        & summary["mode"].eq("calibrated top k")
    ].iloc[0]
    report = {
        "test_rows": int(len(test_frame)),
        "test_y_cri_high_positive_rate": positive_rate,
        "max_f1_at_20_percent_budget": theoretical_f1_bound(positive_rate, 0.20),
        "pareto_auto_mcdm_calibrated_top_k_test": {
            "val_threshold": float(pareto_top["val_threshold_value"]),
            "test_selected_rate": float(pareto_top["test_selected_rate"]),
            "test_selected_count": int(pareto_top["test_selected_count"]),
            "test_precision": float(pareto_top["test_precision"]),
            "test_recall": float(pareto_top["test_recall"]),
            "test_f1": float(pareto_top["test_f1"]),
        },
    }
    (output_dir / "operating_point_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Operating Point Calibration Summary",
        "",
        f"- Test rows evaluated: {len(test_frame):,}",
        f"- Test high CRI positive rate: {positive_rate:.4f}",
        f"- Theoretical maximum F1 under 20 percent audit cap: {report['max_f1_at_20_percent_budget']:.4f}",
        f"- Pareto Auto-MCDM calibrated test F1: {float(pareto_top['test_f1']):.4f}",
        f"- Pareto Auto-MCDM calibrated test selected rate: {float(pareto_top['test_selected_rate']):.2%}",
        "",
        "Generated tables are in `outputs/operating_point_calibration/tables`.",
        "Generated figures are in `outputs/operating_point_calibration/figures`.",
    ]
    (output_dir / "operating_point_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Auto-MCDM operating points on the held-out test split.")
    parser.add_argument("--test-predictions-path", type=Path, default=TEST_PREDICTIONS_DEFAULT)
    parser.add_argument("--baseline-ranking-path", type=Path, default=BASELINE_RANKING_DEFAULT)
    parser.add_argument("--pareto-ranking-path", type=Path, default=PARETO_RANKING_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration(
        test_predictions_path=args.test_predictions_path,
        baseline_ranking_path=args.baseline_ranking_path,
        pareto_ranking_path=args.pareto_ranking_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
