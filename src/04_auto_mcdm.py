"""
Automated MCDM baseline for construction procurement risk prioritization.

Criterion weights and configuration are fitted on the validation split, then
applied to the disjoint test split. The strict_ex_ante feature set provides the
headline probabilities; an audit_priority view is produced for the ex-post audit
comparison reported in the discussion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


VAL_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_val_predictions.parquet")
TEST_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_test_predictions.parquet")
OUTPUT_DEFAULT = Path("outputs/auto_mcdm")

PROBABILITY_COLUMNS = [
    "p_cri_high",
    "p_proc_high",
    "p_buyer_concentration_high",
    "p_single_bid",
    "p_no_call_for_tender",
]

LABEL_COLUMNS = [
    "y_cri_high",
    "y_proc_high",
    "y_buyer_concentration_high",
    "y_single_bid",
    "y_no_call_for_tender",
]

NORMALIZATION_METHODS = ["minmax", "vector", "sum", "robust"]
WEIGHT_METHODS = ["equal", "entropy", "critic", "validation_ap"]
MCDM_METHODS = ["topsis", "vikor", "edas", "cocoso", "waspas"]
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def safe_minmax(values: np.ndarray) -> np.ndarray:
    minimum = np.nanmin(values, axis=0)
    maximum = np.nanmax(values, axis=0)
    span = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
    return (values - minimum) / span


def normalize_matrix(values: np.ndarray, method: str, reference: np.ndarray | None = None) -> np.ndarray:
    """Normalize values using statistics estimated from reference (defaults to values).

    Separating reference from values is required to fit normalization on the
    validation split and apply identical scaling to the test split.
    """
    values = np.asarray(values, dtype=np.float64)
    if reference is None:
        reference = values
    reference = np.asarray(reference, dtype=np.float64)
    fill = np.nanmedian(reference, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    values = np.where(np.isnan(values), fill, values)
    reference_clean = np.where(np.isnan(reference), fill, reference)

    if method == "minmax":
        minimum = np.nanmin(reference_clean, axis=0)
        maximum = np.nanmax(reference_clean, axis=0)
        span = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
        normalized = (values - minimum) / span
    elif method == "vector":
        denominator = np.sqrt(np.sum(reference_clean ** 2, axis=0))
        denominator = np.where(denominator == 0, 1.0, denominator)
        normalized = values / denominator
        ref_normalized = reference_clean / denominator
        minimum = np.nanmin(ref_normalized, axis=0)
        maximum = np.nanmax(ref_normalized, axis=0)
        span = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
        normalized = (normalized - minimum) / span
    elif method == "sum":
        denominator = np.sum(reference_clean, axis=0)
        denominator = np.where(denominator == 0, 1.0, denominator)
        normalized = values / denominator
        ref_normalized = reference_clean / denominator
        minimum = np.nanmin(ref_normalized, axis=0)
        maximum = np.nanmax(ref_normalized, axis=0)
        span = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
        normalized = (normalized - minimum) / span
    elif method == "robust":
        lower = np.nanpercentile(reference_clean, 1, axis=0)
        upper = np.nanpercentile(reference_clean, 99, axis=0)
        clipped = np.clip(values, lower, upper)
        span = np.where(upper - lower == 0, 1.0, upper - lower)
        normalized = (clipped - lower) / span
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return np.clip(np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def equal_weights(matrix: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    return np.ones(matrix.shape[1], dtype=np.float64) / matrix.shape[1]


def entropy_weights(matrix: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    x = np.clip(matrix, 1e-12, None)
    p = x / np.clip(x.sum(axis=0), 1e-12, None)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(matrix.shape[0])
    diversity = 1 - entropy
    if np.allclose(diversity.sum(), 0):
        return equal_weights(matrix)
    return diversity / diversity.sum()


def critic_weights(matrix: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    std = matrix.std(axis=0)
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    conflict = np.sum(1 - corr, axis=0)
    information = std * conflict
    if np.allclose(information.sum(), 0):
        return equal_weights(matrix)
    return information / information.sum()


def validation_ap_weights(matrix: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    if labels is None:
        return equal_weights(matrix)
    weights = []
    for column_index in range(matrix.shape[1]):
        try:
            weights.append(average_precision_score(labels, matrix[:, column_index]))
        except Exception:
            weights.append(0.0)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0)
    if np.allclose(weights.sum(), 0):
        return equal_weights(matrix)
    return weights / weights.sum()


WEIGHT_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray | None], np.ndarray]] = {
    "equal": equal_weights,
    "entropy": entropy_weights,
    "critic": critic_weights,
    "validation_ap": validation_ap_weights,
}


def topsis_score(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = matrix * weights
    ideal = weighted.max(axis=0)
    anti_ideal = weighted.min(axis=0)
    d_pos = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))
    return d_neg / np.clip(d_pos + d_neg, 1e-12, None)


def vikor_score(matrix: np.ndarray, weights: np.ndarray, v: float = 0.5) -> np.ndarray:
    best = matrix.max(axis=0)
    worst = matrix.min(axis=0)
    span = np.where(best - worst == 0, 1.0, best - worst)
    regret_matrix = weights * (best - matrix) / span
    s = regret_matrix.sum(axis=1)
    r = regret_matrix.max(axis=1)
    s_best, s_worst = s.min(), s.max()
    r_best, r_worst = r.min(), r.max()
    q = v * (s - s_best) / max(s_worst - s_best, 1e-12)
    q += (1 - v) * (r - r_best) / max(r_worst - r_best, 1e-12)
    return 1 - q


def edas_score(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    average = np.clip(matrix.mean(axis=0), 1e-12, None)
    positive = np.maximum(0, matrix - average) / average
    negative = np.maximum(0, average - matrix) / average
    sp = positive @ weights
    sn = negative @ weights
    nsp = sp / max(sp.max(), 1e-12)
    nsn = 1 - sn / max(sn.max(), 1e-12)
    return 0.5 * (nsp + nsn)


def cocoso_score(matrix: np.ndarray, weights: np.ndarray, lambd: float = 0.5) -> np.ndarray:
    x = np.clip(matrix, 1e-9, 1.0)
    s = x @ weights
    p = np.sum(np.power(x, weights), axis=1)
    k_a = (s + p) / max(np.sum(s + p), 1e-12)
    k_b = s / max(np.min(s), 1e-12) + p / max(np.min(p), 1e-12)
    k_c = (lambd * s + (1 - lambd) * p) / max(lambd * np.max(s) + (1 - lambd) * np.max(p), 1e-12)
    score = np.cbrt(np.clip(k_a * k_b * k_c, 0, None)) + (k_a + k_b + k_c) / 3
    return safe_minmax(score.reshape(-1, 1)).ravel()


def waspas_score(matrix: np.ndarray, weights: np.ndarray, lambd: float = 0.5) -> np.ndarray:
    x = np.clip(matrix, 1e-9, 1.0)
    weighted_sum = x @ weights
    weighted_product = np.prod(np.power(x, weights), axis=1)
    return lambd * weighted_sum + (1 - lambd) * weighted_product


MCDM_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "topsis": topsis_score,
    "vikor": vikor_score,
    "edas": edas_score,
    "cocoso": cocoso_score,
    "waspas": waspas_score,
}


def make_criteria(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    criteria = frame[PROBABILITY_COLUMNS].copy()
    criteria["risk_consensus_mean"] = criteria[PROBABILITY_COLUMNS].mean(axis=1)
    criteria["risk_consensus_max"] = criteria[PROBABILITY_COLUMNS].max(axis=1)
    uncertainty = 1 - 2 * (criteria[PROBABILITY_COLUMNS] - 0.5).abs()
    criteria["uncertainty_mean"] = uncertainty.mean(axis=1)
    criteria["uncertainty_max"] = uncertainty.max(axis=1)

    manifest = pd.DataFrame(
        [
            {
                "criterion": column,
                "orientation": "benefit",
                "construction_interpretation": interpretation,
            }
            for column, interpretation in {
                "p_cri_high": "predicted composite integrity risk",
                "p_proc_high": "predicted procedural risk",
                "p_buyer_concentration_high": "predicted buyer concentration risk",
                "p_single_bid": "predicted single bid vulnerability",
                "p_no_call_for_tender": "predicted no call for tender risk",
                "risk_consensus_mean": "mean predicted risk across outcomes",
                "risk_consensus_max": "maximum predicted risk across outcomes",
                "uncertainty_mean": "mean model uncertainty across outcomes",
                "uncertainty_max": "maximum model uncertainty across outcomes",
            }.items()
        ]
    )
    return criteria, manifest


def composite_label(frame: pd.DataFrame) -> pd.Series:
    labels = frame[LABEL_COLUMNS].copy()
    available = labels.notna().any(axis=1)
    composite = labels.fillna(0).max(axis=1)
    return composite.where(available, np.nan)


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def top_k_metrics(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
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
                "lift": precision / baseline if baseline > 0 else math.nan,
                "ndcg": ndcg_at_k(y_true, scores, k),
            }
        )
    return pd.DataFrame(rows)


def evaluate_score(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    threshold_metrics = top_k_metrics(y_true, scores, thresholds)
    best = threshold_metrics.sort_values(["f1", "ndcg", "precision"], ascending=False).iloc[0]
    result = {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else math.nan,
        "selected_threshold_rate": float(best["threshold_rate"]),
        "selected_top_k": int(best["top_k"]),
        "selected_precision": float(best["precision"]),
        "selected_recall": float(best["recall"]),
        "selected_f1": float(best["f1"]),
        "selected_lift": float(best["lift"]),
        "selected_ndcg": float(best["ndcg"]),
    }
    for rate in [0.01, 0.05, 0.10]:
        row = threshold_metrics.loc[np.isclose(threshold_metrics["threshold_rate"], rate)]
        if not row.empty:
            selected = row.iloc[0]
            suffix = f"{int(rate * 100)}pct"
            result[f"precision_at_{suffix}"] = float(selected["precision"])
            result[f"recall_at_{suffix}"] = float(selected["recall"])
            result[f"ndcg_at_{suffix}"] = float(selected["ndcg"])
    return result


def run_auto_mcdm(val_path: Path, test_path: Path, output_dir: Path) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "rankings").mkdir(parents=True, exist_ok=True)

    val_frame = pd.read_parquet(val_path)
    test_frame = pd.read_parquet(test_path)

    val_criteria, criterion_manifest = make_criteria(val_frame)
    test_criteria, _ = make_criteria(test_frame)
    val_frame["y_any_risk"] = composite_label(val_frame)
    test_frame["y_any_risk"] = composite_label(test_frame)

    primary_target = "y_cri_high"
    objective_targets = ["y_cri_high", "y_any_risk"]
    val_masks = {target: val_frame[target].notna().to_numpy() for target in objective_targets}
    test_masks = {target: test_frame[target].notna().to_numpy() for target in objective_targets}

    val_matrix_raw = val_criteria.to_numpy(dtype=np.float64)
    test_matrix_raw = test_criteria.to_numpy(dtype=np.float64)

    val_rows = []
    test_rows = []
    threshold_rows = []
    weight_rows = []
    val_score_cache: dict[tuple[str, str, str], np.ndarray] = {}
    test_score_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for normalization in NORMALIZATION_METHODS:
        val_normalized = normalize_matrix(val_matrix_raw, normalization, reference=val_matrix_raw)
        test_normalized = normalize_matrix(test_matrix_raw, normalization, reference=val_matrix_raw)
        for weight_method in WEIGHT_METHODS:
            primary_mask = val_masks[primary_target]
            weights = WEIGHT_FUNCTIONS[weight_method](
                val_normalized[primary_mask],
                val_frame.loc[primary_mask, primary_target].astype(int).to_numpy()
                if weight_method == "validation_ap"
                else None,
            )
            for criterion, weight in zip(val_criteria.columns, weights):
                weight_rows.append(
                    {
                        "normalization": normalization,
                        "weight_method": weight_method,
                        "criterion": criterion,
                        "weight": weight,
                    }
                )

            for method in MCDM_METHODS:
                val_scores = MCDM_FUNCTIONS[method](val_normalized, weights)
                val_scores = safe_minmax(val_scores.reshape(-1, 1)).ravel()
                test_scores = MCDM_FUNCTIONS[method](test_normalized, weights)
                test_scores = safe_minmax(test_scores.reshape(-1, 1)).ravel()
                val_score_cache[(method, normalization, weight_method)] = val_scores
                test_score_cache[(method, normalization, weight_method)] = test_scores

                val_record = {
                    "mcdm_method": method,
                    "normalization": normalization,
                    "weight_method": weight_method,
                }
                test_record = dict(val_record)
                for target in objective_targets:
                    val_mask = val_masks[target]
                    val_y = val_frame.loc[val_mask, target].astype(int).to_numpy()
                    val_target_scores = val_scores[val_mask]
                    val_metrics = evaluate_score(val_y, val_target_scores, THRESHOLDS)
                    val_record.update({f"{target}_{key}": value for key, value in val_metrics.items()})

                    test_mask = test_masks[target]
                    test_y = test_frame.loc[test_mask, target].astype(int).to_numpy()
                    test_target_scores = test_scores[test_mask]
                    test_metrics = evaluate_score(test_y, test_target_scores, THRESHOLDS)
                    test_record.update({f"{target}_{key}": value for key, value in test_metrics.items()})

                    val_threshold_metric = top_k_metrics(val_y, val_target_scores, THRESHOLDS)
                    val_threshold_metric.insert(0, "target", target)
                    val_threshold_metric.insert(0, "weight_method", weight_method)
                    val_threshold_metric.insert(0, "normalization", normalization)
                    val_threshold_metric.insert(0, "mcdm_method", method)
                    val_threshold_metric.insert(0, "split", "val")
                    threshold_rows.append(val_threshold_metric)

                    test_threshold_metric = top_k_metrics(test_y, test_target_scores, THRESHOLDS)
                    test_threshold_metric.insert(0, "target", target)
                    test_threshold_metric.insert(0, "weight_method", weight_method)
                    test_threshold_metric.insert(0, "normalization", normalization)
                    test_threshold_metric.insert(0, "mcdm_method", method)
                    test_threshold_metric.insert(0, "split", "test")
                    threshold_rows.append(test_threshold_metric)

                val_rows.append(val_record)
                test_rows.append(test_record)

    val_performance = pd.DataFrame(val_rows)
    test_performance = pd.DataFrame(test_rows)
    thresholds_frame = pd.concat(threshold_rows, ignore_index=True)
    weights_frame = pd.DataFrame(weight_rows)

    selection_columns = [
        "y_cri_high_selected_f1",
        "y_cri_high_selected_ndcg",
        "y_cri_high_average_precision",
        "y_any_risk_selected_f1",
        "y_any_risk_selected_ndcg",
    ]
    val_performance["auto_selection_score"] = (
        0.35 * val_performance["y_cri_high_selected_f1"]
        + 0.25 * val_performance["y_cri_high_selected_ndcg"]
        + 0.20 * val_performance["y_cri_high_average_precision"]
        + 0.10 * val_performance["y_any_risk_selected_f1"]
        + 0.10 * val_performance["y_any_risk_selected_ndcg"]
    )
    val_selected = val_performance.sort_values(["auto_selection_score"] + selection_columns, ascending=False).iloc[0].copy()
    selected_config = (
        str(val_selected["mcdm_method"]),
        str(val_selected["normalization"]),
        str(val_selected["weight_method"]),
    )
    val_selected_scores = val_score_cache[selected_config]
    test_selected_scores = test_score_cache[selected_config]
    selected_threshold = float(val_selected["y_cri_high_selected_threshold_rate"])
    selected_top_k_test = max(int(math.ceil(len(test_selected_scores) * selected_threshold)), 1)

    test_match = test_performance[
        test_performance["mcdm_method"].eq(selected_config[0])
        & test_performance["normalization"].eq(selected_config[1])
        & test_performance["weight_method"].eq(selected_config[2])
    ].iloc[0].to_dict()

    val_ranking = val_frame[["sample_row_id"] + LABEL_COLUMNS + PROBABILITY_COLUMNS + ["y_any_risk"]].copy()
    val_ranking["auto_mcdm_score"] = val_selected_scores.astype("float32")
    val_ranking["auto_mcdm_rank"] = val_ranking["auto_mcdm_score"].rank(ascending=False, method="first").astype(int)
    val_ranking["split"] = "val"

    test_ranking = test_frame[["sample_row_id"] + LABEL_COLUMNS + PROBABILITY_COLUMNS + ["y_any_risk"]].copy()
    test_ranking["auto_mcdm_score"] = test_selected_scores.astype("float32")
    test_ranking["auto_mcdm_rank"] = test_ranking["auto_mcdm_score"].rank(ascending=False, method="first").astype(int)
    test_ranking["selected_for_review"] = test_ranking["auto_mcdm_rank"] <= selected_top_k_test
    test_ranking["split"] = "test"

    selected_table = val_selected.to_frame().T
    selected_table["selected_top_k_rate"] = selected_threshold
    selected_table["selected_top_k_count_test"] = selected_top_k_test
    for key, value in test_match.items():
        if key in {"mcdm_method", "normalization", "weight_method"}:
            continue
        selected_table[f"test_{key}"] = value

    criterion_manifest.to_csv(output_dir / "tables" / "table_01_criterion_manifest.csv", index=False)
    weights_frame.to_csv(output_dir / "tables" / "table_02_weight_schemes.csv", index=False)
    val_performance.to_csv(output_dir / "tables" / "table_03_val_mcdm_configuration_performance.csv", index=False)
    test_performance.to_csv(output_dir / "tables" / "table_03b_test_mcdm_configuration_performance.csv", index=False)
    selected_table.to_csv(output_dir / "tables" / "table_04_selected_auto_mcdm_configuration.csv", index=False)
    thresholds_frame.to_csv(output_dir / "tables" / "table_05_threshold_search.csv", index=False)
    test_ranking.head(100).to_csv(output_dir / "tables" / "table_06_top_ranked_tenders_test.csv", index=False)

    write_markdown_table(criterion_manifest, output_dir / "tables" / "table_01_criterion_manifest.md")
    write_markdown_table(selected_table, output_dir / "tables" / "table_04_selected_auto_mcdm_configuration.md")
    write_markdown_table(test_ranking.head(30), output_dir / "tables" / "table_06_top_ranked_tenders_test.md")

    pd.concat([val_ranking, test_ranking], ignore_index=True).to_parquet(
        output_dir / "rankings" / "selected_auto_mcdm_ranking.parquet", index=False
    )

    save_performance_figure(test_performance, output_dir / "figures" / "figure_01_mcdm_method_performance.png")
    save_threshold_figure(
        thresholds_frame[thresholds_frame["split"].eq("test")],
        selected_config,
        output_dir / "figures" / "figure_02_threshold_search.png",
    )
    save_weight_figure(weights_frame, selected_config, output_dir / "figures" / "figure_03_selected_criterion_weights.png")
    save_score_distribution_figure(test_ranking, output_dir / "figures" / "figure_04_selected_score_distribution.png")

    summary = {
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "criteria": val_criteria.columns.tolist(),
        "evaluated_configurations": int(len(val_performance)),
        "selected_configuration": {
            "mcdm_method": selected_config[0],
            "normalization": selected_config[1],
            "weight_method": selected_config[2],
            "top_k_rate": selected_threshold,
            "val_selection_score": float(val_selected["auto_selection_score"]),
            "test_y_cri_high_average_precision": float(test_match["y_cri_high_average_precision"]),
            "test_y_cri_high_selected_f1": float(test_match["y_cri_high_selected_f1"]),
            "test_y_cri_high_selected_precision": float(test_match["y_cri_high_selected_precision"]),
            "test_y_cri_high_selected_recall": float(test_match["y_cri_high_selected_recall"]),
            "test_y_cri_high_selected_ndcg": float(test_match["y_cri_high_selected_ndcg"]),
            "test_y_any_risk_average_precision": float(test_match["y_any_risk_average_precision"]),
            "test_y_any_risk_selected_f1": float(test_match["y_any_risk_selected_f1"]),
        },
        "ranking_path": str(output_dir / "rankings" / "selected_auto_mcdm_ranking.parquet"),
    }
    (output_dir / "auto_mcdm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Auto-MCDM Summary",
        "",
        f"- Validation rows used for weight learning and configuration selection: {len(val_frame):,}",
        f"- Test rows used for reporting: {len(test_frame):,}",
        f"- Criteria evaluated: {len(val_criteria.columns):,}",
        f"- MCDM configurations evaluated: {len(val_performance):,}",
        f"- Selected method: {selected_config[0]}",
        f"- Selected normalization: {selected_config[1]}",
        f"- Selected weighting: {selected_config[2]}",
        f"- Selected top-k rate: {selected_threshold:.0%}",
        f"- Test PR-AUC for y_cri_high: {float(test_match['y_cri_high_average_precision']):.4f}",
        f"- Test selected F1 for y_cri_high: {float(test_match['y_cri_high_selected_f1']):.4f}",
        "",
        "Generated tables are in `outputs/auto_mcdm/tables`.",
        "Generated figures are in `outputs/auto_mcdm/figures`.",
    ]
    (output_dir / "auto_mcdm_report.md").write_text("\n".join(report), encoding="utf-8")


def save_performance_figure(test_performance: pd.DataFrame, output_path: Path) -> None:
    grouped = test_performance.groupby("mcdm_method")[
        ["y_cri_high_average_precision", "y_cri_high_selected_f1"]
    ].agg(["mean", "max"]).reset_index()
    grouped.columns = ["mcdm_method", "ap_mean", "ap_max", "f1_mean", "f1_max"]
    grouped = grouped.sort_values("ap_max", ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=180)
    x = np.arange(len(grouped))
    width = 0.36
    ax.bar(x - width / 2, grouped["ap_max"], width=width, label="Best test AP", color="#5b7f95", edgecolor="#25323a", linewidth=0.6)
    ax.bar(x + width / 2, grouped["f1_max"], width=width, label="Best test F1", color="#d1a15f", edgecolor="#5f4a2b", linewidth=0.6)
    ax.set_xticks(x, [clean_caption(value) for value in grouped["mcdm_method"]])
    ax.set_ylabel("Metric value on held out test split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 1. MCDM method performance on held out test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_threshold_figure(thresholds: pd.DataFrame, selected_config: tuple[str, str, str], output_path: Path) -> None:
    method, normalization, weight_method = selected_config
    subset = thresholds[
        thresholds["mcdm_method"].eq(method)
        & thresholds["normalization"].eq(normalization)
        & thresholds["weight_method"].eq(weight_method)
        & thresholds["target"].eq("y_cri_high")
    ].copy()

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.plot(subset["threshold_rate"] * 100, subset["precision"], marker="o", label="Precision", color="#315f72")
    ax.plot(subset["threshold_rate"] * 100, subset["recall"], marker="s", label="Recall", color="#8f6b45")
    ax.plot(subset["threshold_rate"] * 100, subset["f1"], marker="^", label="F1", color="#7c8f63")
    ax.set_xlabel("Review threshold percent")
    ax.set_ylabel("Metric value on held out test split")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 2. Selected configuration threshold curve on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_weight_figure(weights: pd.DataFrame, selected_config: tuple[str, str, str], output_path: Path) -> None:
    _, normalization, weight_method = selected_config
    subset = weights[
        weights["normalization"].eq(normalization)
        & weights["weight_method"].eq(weight_method)
    ].copy()
    subset = subset.sort_values("weight", ascending=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=180)
    ax.barh(
        [clean_caption(value) for value in subset["criterion"]],
        subset["weight"],
        color="#5b7f95",
        edgecolor="#25323a",
        linewidth=0.6,
    )
    ax.set_xlabel("Criterion weight learned on validation split")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.08, 1, 1))
    fig.text(0.5, 0.02, "Figure 3. Selected criterion weights", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution_figure(ranking: pd.DataFrame, output_path: Path) -> None:
    plot = ranking[ranking["y_cri_high"].notna()].copy()
    negative = plot.loc[plot["y_cri_high"].eq(0), "auto_mcdm_score"]
    positive = plot.loc[plot["y_cri_high"].eq(1), "auto_mcdm_score"]

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.hist(negative, bins=35, alpha=0.70, label="Lower risk label", color="#6f8795")
    ax.hist(positive, bins=35, alpha=0.65, label="High risk label", color="#d1a15f")
    ax.set_xlabel("Auto MCDM score on test split")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 4. Selected score distribution on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated MCDM baseline with held-out test split.")
    parser.add_argument("--val-path", type=Path, default=VAL_PREDICTIONS_DEFAULT)
    parser.add_argument("--test-path", type=Path, default=TEST_PREDICTIONS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_auto_mcdm(val_path=args.val_path, test_path=args.test_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
