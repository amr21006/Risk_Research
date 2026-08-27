"""
Validate and verify the automated MCDM outputs on the held-out test split.

Three ranking strategies are compared on the disjoint test split:
1. raw predicted CRI probability from the strict_ex_ante risk model;
2. baseline Auto-MCDM from exhaustive configuration search;
3. Pareto Auto-MCDM from the five-algorithm archive selected with the knee point
   rule.

The MCDM stack is framed here as a multi-criterion decision support layer; the
raw probability remains the strongest single signal for the primary outcome but
does not natively support multi-objective audit prioritization. The numbers
below quantify the marginal cost of that additional capability.
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
PREPROCESS_FORBIDDEN_DEFAULT = Path("outputs/preprocess/tables/table_05_forbidden_feature_verification.csv")
OUTPUT_DEFAULT = Path("outputs/validation")
AUTO_MCDM_PATH = Path(__file__).with_name("04_auto_mcdm.py")

THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20]


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


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    positives = max(int(y_true.sum()), 1)
    baseline = positives / len(y_true)
    order = np.argsort(-scores)
    for threshold in thresholds:
        k = max(int(math.ceil(len(y_true) * threshold)), 1)
        selected = order[:k]
        hits = int(y_true[selected].sum())
        precision = hits / k
        recall = hits / positives
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold_rate": threshold,
                "top_k": k,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "lift": precision / baseline if baseline else math.nan,
                "ndcg": MCDM.ndcg_at_k(y_true, scores, k),
            }
        )
    return pd.DataFrame(rows)


def evaluate_strategy(
    name: str,
    scores: np.ndarray,
    labels: pd.Series,
    thresholds: list[float],
    target_name: str,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    mask = labels.notna().to_numpy()
    y_true = labels.loc[mask].astype(int).to_numpy()
    y_scores = scores[mask]
    topk = ranking_metrics(y_true, y_scores, thresholds)
    best = topk.sort_values(["f1", "ndcg", "precision"], ascending=False).iloc[0]
    summary = {
        "strategy": name,
        "target": target_name,
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, y_scores)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) == 2 else math.nan,
        "best_threshold_rate": float(best["threshold_rate"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
        "best_ndcg": float(best["ndcg"]),
        "best_lift": float(best["lift"]),
    }
    topk.insert(0, "target", target_name)
    topk.insert(0, "strategy", name)
    return summary, topk


def spearman_pair(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    ranks_a = pd.Series(scores_a).rank().to_numpy()
    ranks_b = pd.Series(scores_b).rank().to_numpy()
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def topk_jaccard(scores_a: np.ndarray, scores_b: np.ndarray, rate: float) -> float:
    k = max(int(math.ceil(len(scores_a) * rate)), 1)
    set_a = set(np.argsort(-scores_a)[:k].tolist())
    set_b = set(np.argsort(-scores_b)[:k].tolist())
    if not set_a and not set_b:
        return float("nan")
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def feature_drop_robustness(
    criteria: pd.DataFrame,
    selected_config: dict[str, Any],
    labels: pd.Series,
    drop_rates: list[float],
    seed: int,
    iterations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mask_avail = labels.notna().to_numpy()
    y_true = labels.loc[mask_avail].astype(int).to_numpy()
    matrix_full = MCDM.normalize_matrix(
        criteria.to_numpy(dtype=np.float64),
        selected_config["normalization"],
        reference=criteria.to_numpy(dtype=np.float64),
    )
    base_scores = MCDM.MCDM_FUNCTIONS[selected_config["method"]](matrix_full, selected_config["weights_full"])
    base_scores = MCDM.safe_minmax(base_scores.reshape(-1, 1)).ravel()[mask_avail]
    base_ap = float(average_precision_score(y_true, base_scores))

    rows = []
    n_criteria = matrix_full.shape[1]
    for rate in drop_rates:
        ap_deltas = []
        jaccards = []
        spearmans = []
        for _ in range(iterations):
            drop_count = max(int(round(n_criteria * rate)), 1)
            drop_indices = rng.choice(n_criteria, size=drop_count, replace=False)
            keep_mask = np.ones(n_criteria, dtype=bool)
            keep_mask[drop_indices] = False
            if keep_mask.sum() == 0:
                continue
            sub_matrix = matrix_full[:, keep_mask]
            sub_weights = selected_config["weights_full"][keep_mask]
            if sub_weights.sum() <= 0:
                continue
            sub_weights = sub_weights / sub_weights.sum()
            sub_scores = MCDM.MCDM_FUNCTIONS[selected_config["method"]](sub_matrix, sub_weights)
            sub_scores = MCDM.safe_minmax(sub_scores.reshape(-1, 1)).ravel()[mask_avail]
            try:
                ap_deltas.append(float(average_precision_score(y_true, sub_scores)) - base_ap)
            except Exception:
                continue
            jaccards.append(topk_jaccard(base_scores, sub_scores, 0.20))
            spearmans.append(spearman_pair(base_scores, sub_scores))
        rows.append(
            {
                "drop_rate": rate,
                "iterations": iterations,
                "ap_delta_mean": float(np.mean(ap_deltas)) if ap_deltas else math.nan,
                "ap_delta_std": float(np.std(ap_deltas, ddof=1)) if len(ap_deltas) > 1 else math.nan,
                "ap_delta_min": float(np.min(ap_deltas)) if ap_deltas else math.nan,
                "ap_delta_max": float(np.max(ap_deltas)) if ap_deltas else math.nan,
                "top20pct_jaccard_mean": float(np.mean(jaccards)) if jaccards else math.nan,
                "spearman_rank_mean": float(np.mean(spearmans)) if spearmans else math.nan,
            }
        )
    return pd.DataFrame(rows)


def decision_support_metrics(
    raw_scores: np.ndarray,
    baseline_scores: np.ndarray,
    pareto_scores: np.ndarray,
    test_frame: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    targets = ["y_cri_high", "y_buyer_concentration_high", "y_no_call_for_tender", "y_proc_high", "y_single_bid"]
    for target in targets:
        if target not in test_frame.columns:
            continue
        mask = test_frame[target].notna().to_numpy()
        y_true = test_frame.loc[mask, target].astype(int).to_numpy()
        for strategy_name, scores in {
            "raw cri probability": raw_scores,
            "baseline auto mcdm": baseline_scores,
            "pareto auto mcdm": pareto_scores,
        }.items():
            y_scores = scores[mask]
            try:
                ap = float(average_precision_score(y_true, y_scores))
            except Exception:
                ap = math.nan
            row = {
                "strategy": strategy_name,
                "target": target,
                "average_precision": ap,
                "positive_rate": float(y_true.mean()),
            }
            for rate in thresholds:
                k = max(int(math.ceil(len(y_true) * rate)), 1)
                selected = np.argsort(-y_scores)[:k]
                hits = int(y_true[selected].sum())
                row[f"precision_at_{int(rate * 100)}pct"] = hits / k
            rows.append(row)
    return pd.DataFrame(rows)


def save_validation_figure(summary: pd.DataFrame, output_path: Path) -> None:
    plot = summary[summary["target"].eq("y_cri_high")].copy()
    labels = [clean_caption(value) for value in plot["strategy"]]
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    width = 0.36
    ax.bar(x - width / 2, plot["average_precision"], width=width, label="Average precision", color="#5b7f95", edgecolor="#25323a", linewidth=0.6)
    ax.bar(x + width / 2, plot["best_f1"], width=width, label="Best F1", color="#d1a15f", edgecolor="#5f4a2b", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Metric value on held out test split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 1. Test ranking comparison for the primary target", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_topk_figure(topk: pd.DataFrame, output_path: Path) -> None:
    subset = topk[topk["target"].eq("y_cri_high")].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    for strategy, group in subset.groupby("strategy"):
        ax.plot(group["threshold_rate"] * 100, group["precision"], marker="o", label=clean_caption(strategy))
    ax.set_xlabel("Review threshold percent")
    ax.set_ylabel("Precision on test split")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 2. Top k precision comparison on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_robustness_figure(robustness: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.0), dpi=180)
    ax.plot(robustness["drop_rate"] * 100, robustness["top20pct_jaccard_mean"], marker="o", color="#7c8f63", label="Top 20% Jaccard")
    ax.plot(robustness["drop_rate"] * 100, robustness["spearman_rank_mean"], marker="s", color="#315f72", label="Spearman rank correlation")
    ax.set_xlabel("Criteria dropped percent")
    ax.set_ylabel("Robustness score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 3. Ranking robustness under criterion ablation", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_score_overlay_figure(frame: pd.DataFrame, output_path: Path) -> None:
    plot = frame[frame["y_cri_high"].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), dpi=180, sharey=True)
    score_columns = [
        ("p_cri_high", "Raw CRI probability"),
        ("auto_mcdm_score", "Baseline Auto MCDM"),
        ("pareto_auto_mcdm_score", "Pareto Auto MCDM"),
    ]
    for ax, (column, title) in zip(axes, score_columns):
        ax.hist(plot.loc[plot["y_cri_high"].eq(0), column], bins=30, alpha=0.65, color="#6f8795", label="Lower risk")
        ax.hist(plot.loc[plot["y_cri_high"].eq(1), column], bins=30, alpha=0.60, color="#d1a15f", label="High risk")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Records")
    axes[1].set_xlabel("Ranking score on test split")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.03, 0.13, 1, 1))
    fig.text(0.5, 0.025, "Figure 4. Score separation by primary label on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_decision_support_figure(decision_support: pd.DataFrame, output_path: Path) -> None:
    targets = decision_support["target"].unique()
    strategies = decision_support["strategy"].unique()
    width = 0.27
    fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=180)
    x = np.arange(len(targets))
    for offset, strategy in enumerate(strategies):
        subset = decision_support[decision_support["strategy"].eq(strategy)].set_index("target").reindex(targets)
        ax.bar(
            x + (offset - 1) * width,
            subset["average_precision"],
            width=width,
            label=clean_caption(strategy),
            edgecolor="#1f2d33",
            linewidth=0.5,
        )
    ax.set_xticks(x, [clean_caption(value) for value in targets], rotation=15, ha="right")
    ax.set_ylabel("Average precision on test split")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.13, 1, 1))
    fig.text(0.5, 0.025, "Figure 5. Decision support coverage across risk dimensions", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_selected_pareto_config(pareto_ranking_path: Path) -> dict[str, Any] | None:
    selected_path = pareto_ranking_path.parent.parent / "tables" / "table_05_selected_pareto_solutions.csv"
    if not selected_path.exists():
        return None
    selections = pd.read_csv(selected_path)
    if selections.empty:
        return None
    selected_row = selections[selections["selector"].eq("knee")].iloc[0]
    mask = []
    weights = []
    for column in selected_row.index:
        if column.startswith("criterion_selected_"):
            criterion = column.removeprefix("criterion_selected_")
            mask.append(bool(selected_row[column]))
            weights.append(float(selected_row[f"weight_{criterion}"]))
    mask_arr = np.array(mask, dtype=bool)
    weights_arr = np.array(weights, dtype=float)
    weights_full = np.where(mask_arr, weights_arr, 0.0)
    if weights_full.sum() > 0:
        weights_full = weights_full / weights_full.sum()
    return {
        "method": str(selected_row["method"]),
        "normalization": str(selected_row["normalization"]),
        "weights_full": weights_full,
        "top_k_rate": float(selected_row["top_k_rate"]),
    }


def run_validation(
    test_predictions_path: Path,
    baseline_ranking_path: Path,
    pareto_ranking_path: Path,
    forbidden_path: Path,
    output_dir: Path,
    seed: int,
    robustness_iterations: int,
) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(test_predictions_path)
    baseline_ranking = pd.read_parquet(baseline_ranking_path)
    pareto_ranking = pd.read_parquet(pareto_ranking_path)
    baseline_test = baseline_ranking[baseline_ranking["split"].eq("test")][["sample_row_id", "auto_mcdm_score", "selected_for_review"]]
    pareto_test = pareto_ranking[pareto_ranking["split"].eq("test")][["sample_row_id", "pareto_auto_mcdm_score", "selected_for_review"]]
    pareto_test = pareto_test.rename(columns={"selected_for_review": "pareto_selected_for_review"})
    baseline_test = baseline_test.rename(columns={"selected_for_review": "baseline_selected_for_review"})

    frame = predictions.merge(baseline_test, on="sample_row_id", how="inner").merge(pareto_test, on="sample_row_id", how="inner")
    frame["y_any_risk"] = composite_label(frame)
    selected_rates = [
        float(frame["baseline_selected_for_review"].sum() / len(frame)),
        float(frame["pareto_selected_for_review"].sum() / len(frame)),
    ]
    thresholds = sorted({round(rate, 8) for rate in THRESHOLDS + selected_rates if rate > 0})

    strategies = {
        "raw cri probability": frame["p_cri_high"].to_numpy(dtype=float),
        "baseline auto mcdm": frame["auto_mcdm_score"].to_numpy(dtype=float),
        "pareto auto mcdm": frame["pareto_auto_mcdm_score"].to_numpy(dtype=float),
    }

    summary_rows = []
    topk_frames = []
    for strategy, scores in strategies.items():
        for target in ["y_cri_high", "y_any_risk"]:
            summary, topk = evaluate_strategy(strategy, scores, frame[target], thresholds, target)
            summary_rows.append(summary)
            topk_frames.append(topk)

    summary_df = pd.DataFrame(summary_rows)
    topk_df = pd.concat(topk_frames, ignore_index=True)

    decision_support_df = decision_support_metrics(
        strategies["raw cri probability"],
        strategies["baseline auto mcdm"],
        strategies["pareto auto mcdm"],
        frame,
        thresholds=[0.05, 0.10, 0.20],
    )

    pareto_config = load_selected_pareto_config(pareto_ranking_path)
    if pareto_config is not None:
        criteria, _ = MCDM.make_criteria(frame)
        robustness_df = feature_drop_robustness(
            criteria=criteria,
            selected_config=pareto_config,
            labels=frame["y_cri_high"],
            drop_rates=[0.05, 0.10, 0.20, 0.30],
            seed=seed,
            iterations=robustness_iterations,
        )
    else:
        robustness_df = pd.DataFrame(
            [{"drop_rate": rate, "iterations": 0, "ap_delta_mean": math.nan} for rate in [0.05, 0.10, 0.20, 0.30]]
        )

    forbidden_features = pd.read_csv(forbidden_path)
    forbidden_count = int(forbidden_features["forbidden_detected"].astype(str).str.lower().eq("true").sum())
    verification = pd.DataFrame(
        [
            {
                "check": "forbidden temporal geographic identifier and raw label source features",
                "status": "pass" if forbidden_count == 0 else "fail",
                "evidence": f"{forbidden_count} forbidden features detected in final feature sets",
            },
            {
                "check": "test split rows aligned across ranking outputs",
                "status": "pass" if len(frame) == len(predictions) else "fail",
                "evidence": f"{len(frame):,} merged test rows from {len(predictions):,} prediction rows",
            },
            {
                "check": "pareto output includes review selection",
                "status": "pass" if frame["pareto_selected_for_review"].any() else "fail",
                "evidence": f"{int(frame['pareto_selected_for_review'].sum()):,} selected records",
            },
            {
                "check": "baseline output includes review selection",
                "status": "pass" if frame["baseline_selected_for_review"].any() else "fail",
                "evidence": f"{int(frame['baseline_selected_for_review'].sum()):,} selected records",
            },
        ]
    )

    summary_df.to_csv(output_dir / "tables" / "table_01_ranking_validation_summary.csv", index=False)
    topk_df.to_csv(output_dir / "tables" / "table_02_top_k_validation.csv", index=False)
    robustness_df.to_csv(output_dir / "tables" / "table_03_ranking_robustness.csv", index=False)
    verification.to_csv(output_dir / "tables" / "table_04_verification_checklist.csv", index=False)
    decision_support_df.to_csv(output_dir / "tables" / "table_05_decision_support_coverage.csv", index=False)

    write_markdown_table(summary_df, output_dir / "tables" / "table_01_ranking_validation_summary.md")
    write_markdown_table(robustness_df, output_dir / "tables" / "table_03_ranking_robustness.md")
    write_markdown_table(verification, output_dir / "tables" / "table_04_verification_checklist.md")
    write_markdown_table(decision_support_df, output_dir / "tables" / "table_05_decision_support_coverage.md")

    save_validation_figure(summary_df, output_dir / "figures" / "figure_01_ranking_validation_comparison.png")
    save_topk_figure(topk_df, output_dir / "figures" / "figure_02_top_k_precision_comparison.png")
    save_robustness_figure(robustness_df, output_dir / "figures" / "figure_03_ranking_robustness.png")
    save_score_overlay_figure(frame, output_dir / "figures" / "figure_04_score_separation.png")
    save_decision_support_figure(decision_support_df, output_dir / "figures" / "figure_05_decision_support_coverage.png")

    summary = {
        "test_rows": int(len(frame)),
        "strategies": list(strategies.keys()),
        "targets": ["y_cri_high", "y_any_risk"],
        "verification_passed": bool(verification["status"].eq("pass").all()),
        "best_y_cri_strategy_by_average_precision": summary_df.loc[
            summary_df[summary_df["target"].eq("y_cri_high")]["average_precision"].idxmax(),
            "strategy",
        ],
        "best_y_cri_strategy_by_best_f1": summary_df.loc[
            summary_df[summary_df["target"].eq("y_cri_high")]["best_f1"].idxmax(),
            "strategy",
        ],
        "outputs": {
            "tables": str(output_dir / "tables"),
            "figures": str(output_dir / "figures"),
        },
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Validation Summary",
        "",
        f"- Test rows validated: {len(frame):,}",
        f"- Strategies compared: {', '.join(strategies.keys())}",
        f"- Verification passed: {verification['status'].eq('pass').all()}",
        "",
        "Generated tables are in `outputs/validation/tables`.",
        "Generated figures are in `outputs/validation/figures`.",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and verify ranking outputs on held-out test split.")
    parser.add_argument("--test-predictions-path", type=Path, default=TEST_PREDICTIONS_DEFAULT)
    parser.add_argument("--baseline-ranking-path", type=Path, default=BASELINE_RANKING_DEFAULT)
    parser.add_argument("--pareto-ranking-path", type=Path, default=PARETO_RANKING_DEFAULT)
    parser.add_argument("--forbidden-path", type=Path, default=PREPROCESS_FORBIDDEN_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--robustness-iterations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(
        test_predictions_path=args.test_predictions_path,
        baseline_ranking_path=args.baseline_ranking_path,
        pareto_ranking_path=args.pareto_ranking_path,
        forbidden_path=args.forbidden_path,
        output_dir=args.output_dir,
        seed=args.seed,
        robustness_iterations=args.robustness_iterations,
    )


if __name__ == "__main__":
    main()
