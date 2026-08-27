"""
Train risk-prediction models with a held-out test split.

The script reads the pre-assigned split (train, val, test) produced by
02_preprocess_aggregate.py. For each target it trains candidate classifiers on
the training split, selects the best model by validation PR-AUC, and exports
calibrated probabilities for both the validation and test splits.

The strict_ex_ante feature set is the headline configuration. The audit_priority
feature set is retained as a supplementary ex-post audit view.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_DIR_DEFAULT = Path("data/modeling")
PREPROCESS_TABLE_DEFAULT = Path("outputs/preprocess/tables/table_02_clean_feature_manifest.csv")
OUTPUT_DEFAULT = Path("outputs/risk_prediction")

HEADLINE_FEATURE_SET = "strict_ex_ante_enhanced"
SUPPLEMENTARY_FEATURE_SET = "audit_priority"
BASELINE_EX_ANTE_FEATURE_SET = "strict_ex_ante"

BINARY_TARGETS = [
    "y_cri_high",
    "y_proc_high",
    "y_buyer_concentration_high",
    "y_single_bid",
    "y_no_call_for_tender",
]

RANKING_K = (0.01, 0.05, 0.10)
SPLIT_LABELS = ("train", "val", "test")


@dataclass
class PreparedData:
    frame: pd.DataFrame
    features: list[str]
    targets: list[str]


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def feature_columns(manifest_path: Path, feature_set: str) -> list[str]:
    manifest = pd.read_csv(manifest_path)
    flag = f"{feature_set}_feature"
    if flag not in manifest.columns:
        raise ValueError(f"Feature-set flag not found in manifest: {flag}")
    return manifest.loc[manifest[flag].astype(str).str.lower().eq("true"), "column"].tolist()


def stratified_sample_per_split(
    path: Path,
    columns: list[str],
    split_caps: dict[str, int],
    seed: int,
    batch_size: int,
    primary_target: str,
) -> pd.DataFrame:
    """Sample up to split_caps[label] rows per split label preserving stratification."""
    parquet = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    needed_columns = list(dict.fromkeys(["row_id", "split", primary_target] + columns))

    sampled_parts: list[pd.DataFrame] = []
    seen_per_split: dict[str, int] = {label: 0 for label in split_caps}

    for batch in parquet.iter_batches(columns=needed_columns, batch_size=batch_size):
        frame = batch.to_pandas()
        for label, cap in split_caps.items():
            mask = frame["split"].eq(label)
            if not mask.any():
                continue
            available = frame.loc[mask].copy()
            seen_per_split[label] += len(available)
            sampled_parts.append(available)

    if not sampled_parts:
        raise RuntimeError(f"No rows sampled from {path}")

    pooled = pd.concat(sampled_parts, ignore_index=True)
    cuts: list[pd.DataFrame] = []
    for label, cap in split_caps.items():
        subset = pooled.loc[pooled["split"].eq(label)]
        if cap <= 0 or len(subset) <= cap:
            cuts.append(subset)
            continue
        target_values = pd.to_numeric(subset[primary_target], errors="coerce")
        stratum = target_values.where(target_values.isin([0, 1]), other=-1).astype(int)
        keep_indices: list[int] = []
        for value in (-1, 0, 1):
            stratum_mask = stratum.eq(value)
            stratum_rows = subset.loc[stratum_mask]
            if stratum_rows.empty:
                continue
            stratum_cap = max(int(round(cap * (len(stratum_rows) / len(subset)))), 1)
            stratum_cap = min(stratum_cap, len(stratum_rows))
            indices = rng.choice(stratum_rows.index.to_numpy(), size=stratum_cap, replace=False)
            keep_indices.extend(indices.tolist())
        cuts.append(subset.loc[sorted(keep_indices)])

    sampled = pd.concat(cuts, ignore_index=True)
    sampled = sampled.sort_values(["split", "row_id"]).reset_index(drop=True)
    sampled.insert(0, "sample_row_id", np.arange(len(sampled), dtype=np.int64))
    return sampled


def load_prepared_data(
    data_dir: Path,
    manifest_path: Path,
    feature_set: str,
    split_caps: dict[str, int],
    seed: int,
    batch_size: int,
) -> PreparedData:
    if feature_set == HEADLINE_FEATURE_SET:
        path = data_dir / "tender_lot_strict_ex_ante_enhanced.parquet"
    elif feature_set == BASELINE_EX_ANTE_FEATURE_SET:
        path = data_dir / "tender_lot_strict_ex_ante.parquet"
    elif feature_set == SUPPLEMENTARY_FEATURE_SET:
        path = data_dir / "tender_lot_audit_priority.parquet"
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    features = feature_columns(manifest_path, feature_set)
    target_columns = [target for target in BINARY_TARGETS]
    columns = list(dict.fromkeys(features + target_columns))
    frame = stratified_sample_per_split(
        path=path,
        columns=columns,
        split_caps=split_caps,
        seed=seed,
        batch_size=batch_size,
        primary_target="y_cri_high",
    )

    for target in BINARY_TARGETS:
        if target in frame:
            frame[target] = pd.to_numeric(frame[target], errors="coerce")

    return PreparedData(frame=frame, features=features, targets=BINARY_TARGETS)


def build_preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [
        column
        for column in features
        if str(frame[column].dtype) in {"object", "string", "category"} or frame[column].dtype.name == "string"
    ]
    numeric = [column for column in features if column not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def sanitize_for_sklearn(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in features:
        if str(sanitized[column].dtype) in {"object", "string", "category"} or sanitized[column].dtype.name == "string":
            sanitized[column] = sanitized[column].astype(object).where(sanitized[column].notna(), np.nan)
    return sanitized


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not mask.any():
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray, rates: Iterable[float]) -> dict[str, float]:
    order = np.argsort(-y_prob)
    positives = max(int(y_true.sum()), 1)
    metrics: dict[str, float] = {}
    for rate in rates:
        k = max(int(math.ceil(len(y_true) * rate)), 1)
        selected = order[:k]
        hits = int(y_true[selected].sum())
        metrics[f"precision_at_{int(rate * 100)}pct"] = hits / k
        metrics[f"recall_at_{int(rate * 100)}pct"] = hits / positives
        gain = hits / k
        baseline = positives / len(y_true) if len(y_true) else math.nan
        metrics[f"lift_at_{int(rate * 100)}pct"] = gain / baseline if baseline else math.nan
    return metrics


def model_candidates(seed: int, pos_weight: float, use_cuda: bool) -> dict[str, object]:
    return {
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=700,
            tol=1e-3,
            random_state=seed,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=240,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        ),
        "xgboost": XGBClassifier(
            n_estimators=240,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda" if use_cuda else "cpu",
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=pos_weight,
        ),
    }


def predict_probability(model: object, x_values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_values)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_values)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model does not expose probability estimates: {type(model)}")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else math.nan,
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1_at_0_5": f1_score(y_true, y_pred, zero_division=0),
        "precision_at_0_5": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_0_5": recall_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    metrics.update(ranking_metrics(y_true, y_prob, RANKING_K))
    return metrics


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"feature_{index}" for index in range(len(preprocessor.get_feature_names_out()))]


def feature_importance(model: object, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    underlying = model
    if hasattr(model, "calibrated_classifiers_"):
        try:
            estimators = [item.estimator for item in model.calibrated_classifiers_ if hasattr(item, "estimator")]
        except Exception:
            estimators = []
        if estimators and hasattr(estimators[0], "feature_importances_"):
            values = np.mean(
                np.stack([np.asarray(estimator.feature_importances_, dtype=float) for estimator in estimators]),
                axis=0,
            )
            order = np.argsort(-values)[:top_n]
            total = values.sum()
            if total > 0:
                values = values / total
            return pd.DataFrame(
                {
                    "feature": [feature_names[index] for index in order],
                    "importance": values[order],
                }
            )

    if hasattr(underlying, "feature_importances_"):
        values = np.asarray(underlying.feature_importances_, dtype=float)
    elif hasattr(underlying, "coef_"):
        values = np.abs(np.asarray(underlying.coef_).ravel())
    else:
        values = np.zeros(len(feature_names), dtype=float)

    if len(values) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    total = values.sum()
    if total > 0:
        values = values / total
    order = np.argsort(-values)[:top_n]
    return pd.DataFrame(
        {
            "feature": [feature_names[index] for index in order],
            "importance": values[order],
        }
    )


def train_for_target(
    prepared: PreparedData,
    feature_set: str,
    target: str,
    seed: int,
    use_cuda: bool,
) -> tuple[list[dict[str, float | str | int]], pd.DataFrame, np.ndarray, np.ndarray, object]:
    frame = prepared.frame
    target_values = pd.to_numeric(frame[target], errors="coerce")
    available = target_values.notna()

    splits = frame["split"]
    train_mask = splits.eq("train") & available
    val_mask = splits.eq("val") & available

    train_frame = frame.loc[train_mask, prepared.features + [target]].copy()
    val_frame = frame.loc[val_mask, prepared.features + [target]].copy()
    train_frame = sanitize_for_sklearn(train_frame, prepared.features)
    val_frame = sanitize_for_sklearn(val_frame, prepared.features)

    if train_frame[target].nunique(dropna=True) < 2 or val_frame[target].nunique(dropna=True) < 2:
        raise ValueError(f"Target lacks both classes after split: {feature_set} {target}")

    y_train = train_frame[target].astype(int).to_numpy()
    y_val = val_frame[target].astype(int).to_numpy()
    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    pos_weight = neg / pos

    preprocessor = build_preprocessor(train_frame, prepared.features)
    x_train = preprocessor.fit_transform(train_frame[prepared.features]).astype("float32")
    x_val = preprocessor.transform(val_frame[prepared.features]).astype("float32")
    x_all = preprocessor.transform(
        sanitize_for_sklearn(frame[prepared.features], prepared.features)
    ).astype("float32")

    results: list[dict[str, float | str | int]] = []
    fitted_models: dict[str, object] = {}
    warnings.filterwarnings("ignore", category=UserWarning)

    for model_name, base_model in model_candidates(seed, pos_weight=pos_weight, use_cuda=use_cuda).items():
        try:
            base_model.fit(x_train, y_train)
            y_prob_uncal = predict_probability(base_model, x_val)
            uncalibrated_metrics = evaluate_predictions(y_val, y_prob_uncal)

            # Calibrate the trained estimator on the validation split by freezing
            # the base estimator so that the post-calibration probabilities are
            # produced without re-training the underlying model.
            calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="isotonic")
            calibrator.fit(x_val, y_val)
            y_prob_cal = predict_probability(calibrator, x_val)
            calibrated_metrics = evaluate_predictions(y_val, y_prob_cal)

            record = {
                "feature_set": feature_set,
                "target": target,
                "model": model_name,
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_val": float(y_val.mean()),
                **{f"val_uncalibrated_{key}": value for key, value in uncalibrated_metrics.items()},
                **{f"val_calibrated_{key}": value for key, value in calibrated_metrics.items()},
            }
            record["pr_auc"] = float(calibrated_metrics["pr_auc"])
            record["roc_auc"] = float(calibrated_metrics["roc_auc"])
            record["brier_score"] = float(calibrated_metrics["brier_score"])
            record["ece"] = float(calibrated_metrics["ece"])
            results.append(record)
            fitted_models[model_name] = calibrator
        except Exception as exc:
            results.append(
                {
                    "feature_set": feature_set,
                    "target": target,
                    "model": model_name,
                    "train_rows": len(y_train),
                    "val_rows": len(y_val),
                    "positive_rate_train": float(y_train.mean()),
                    "positive_rate_val": float(y_val.mean()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result_frame = pd.DataFrame(results)
    if "pr_auc" not in result_frame.columns:
        raise RuntimeError(f"No valid model completed for {feature_set} {target}")
    valid_results = result_frame.dropna(subset=["pr_auc"])
    if valid_results.empty:
        raise RuntimeError(f"No valid model completed for {feature_set} {target}")

    selected_name = valid_results.sort_values(
        ["pr_auc", "roc_auc", "brier_score"],
        ascending=[False, False, True],
    ).iloc[0]["model"]
    selected_model = fitted_models[str(selected_name)]
    selected_probabilities_all = predict_probability(selected_model, x_all)

    importances = feature_importance(selected_model, transformed_feature_names(preprocessor), top_n=20)
    importances.insert(0, "target", target)
    importances.insert(0, "feature_set", feature_set)
    importances.insert(0, "model", selected_name)

    return results, importances, selected_probabilities_all, frame["sample_row_id"].to_numpy(), selected_model


def save_metric_figure(performance: pd.DataFrame, output_path: Path) -> None:
    selected = performance.dropna(subset=["pr_auc"]).copy()
    selected["label"] = selected["feature_set"] + " | " + selected["target"] + " | " + selected["model"]
    selected = selected.sort_values("pr_auc", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10.8, 6.4), dpi=180)
    ax.barh(
        [clean_caption(label) for label in selected["label"]],
        selected["pr_auc"],
        color="#5b7f95",
        edgecolor="#25323a",
        linewidth=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Calibrated validation PR AUC")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.08, 1, 1))
    fig.text(0.5, 0.02, "Figure 1. Candidate model comparison by calibrated validation PR AUC", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_selected_precision_figure(selected: pd.DataFrame, output_path: Path) -> None:
    plot = selected.copy()
    plot["label"] = plot["feature_set"] + " | " + plot["target"]
    plot = plot.sort_values("val_calibrated_precision_at_5pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10.4, 6.0), dpi=180)
    ax.barh(
        [clean_caption(label) for label in plot["label"]],
        plot["val_calibrated_precision_at_5pct"],
        color="#7c8f63",
        edgecolor="#28301f",
        linewidth=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Calibrated validation precision at 5 percent")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.08, 1, 1))
    fig.text(0.5, 0.02, "Figure 2. Selected model top five percent precision on validation split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_importance_figure(importances: pd.DataFrame, feature_set: str, output_path: Path) -> None:
    subset = importances[(importances["target"].eq("y_cri_high")) & (importances["feature_set"].eq(feature_set))].copy()
    if subset.empty:
        subset = importances.copy()
    subset = subset.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=180)
    ax.barh(
        [clean_caption(label) for label in subset["feature"]],
        subset["importance"],
        color="#d1a15f",
        edgecolor="#5f4a2b",
        linewidth=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Normalized importance")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.08, 1, 1))
    fig.text(0.5, 0.02, f"Figure 3. Selected model feature importance for {clean_caption(feature_set)}", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_calibration_figure(
    val_probabilities: dict[tuple[str, str], np.ndarray],
    val_labels: dict[tuple[str, str], np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=180)
    for (feature_set, target), probs in val_probabilities.items():
        labels = val_labels[(feature_set, target)]
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        mean_pred = []
        mean_true = []
        for bin_index in range(len(bins) - 1):
            mask = bin_indices == bin_index
            if not mask.any():
                continue
            mean_pred.append(float(probs[mask].mean()))
            mean_true.append(float(labels[mask].mean()))
        if mean_pred:
            ax.plot(mean_pred, mean_true, marker="o", label=clean_caption(f"{feature_set} | {target}"), alpha=0.85)
    ax.plot([0, 1], [0, 1], color="#333333", linewidth=0.8, linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.08, 1, 1))
    fig.text(0.5, 0.02, "Figure 4. Reliability diagrams for calibrated probabilities on validation split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    data_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    max_rows_train: int,
    max_rows_val: int,
    max_rows_test: int,
    seed: int,
    batch_size: int,
) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    use_cuda = cuda_available()
    feature_set_targets = {
        HEADLINE_FEATURE_SET: BINARY_TARGETS,
        BASELINE_EX_ANTE_FEATURE_SET: BINARY_TARGETS,
        SUPPLEMENTARY_FEATURE_SET: BINARY_TARGETS,
    }

    performance_records: list[dict[str, float | str | int]] = []
    importance_frames: list[pd.DataFrame] = []
    selected_records: list[dict[str, float | str | int]] = []
    val_predictions_by_feature_set: dict[str, pd.DataFrame] = {}
    test_predictions_by_feature_set: dict[str, pd.DataFrame] = {}
    calibration_probabilities: dict[tuple[str, str], np.ndarray] = {}
    calibration_labels: dict[tuple[str, str], np.ndarray] = {}

    split_caps = {"train": max_rows_train, "val": max_rows_val, "test": max_rows_test}

    for feature_set, targets in feature_set_targets.items():
        prepared = load_prepared_data(
            data_dir=data_dir,
            manifest_path=manifest_path,
            feature_set=feature_set,
            split_caps=split_caps,
            seed=seed,
            batch_size=batch_size,
        )

        val_mask = prepared.frame["split"].eq("val")
        test_mask = prepared.frame["split"].eq("test")
        val_predictions = prepared.frame.loc[val_mask, ["sample_row_id"] + BINARY_TARGETS].copy()
        test_predictions = prepared.frame.loc[test_mask, ["sample_row_id"] + BINARY_TARGETS].copy()
        val_predictions["split"] = "val"
        test_predictions["split"] = "test"

        for target in targets:
            results, importances, probabilities_all, sample_ids, model = train_for_target(
                prepared=prepared,
                feature_set=feature_set,
                target=target,
                seed=seed,
                use_cuda=use_cuda,
            )
            performance_records.extend(results)
            importance_frames.append(importances)

            result_frame = pd.DataFrame(results)
            valid = result_frame.dropna(subset=["pr_auc"])
            selected_row = valid.sort_values(
                ["pr_auc", "roc_auc", "brier_score"],
                ascending=[False, False, True],
            ).iloc[0].to_dict()
            selected_row["selection_metric"] = "highest calibrated validation PR AUC"
            selected_records.append(selected_row)

            sample_id_to_prob = pd.Series(probabilities_all.astype("float32"), index=sample_ids)
            short = target.removeprefix("y_")
            val_predictions[f"p_{short}"] = (
                val_predictions["sample_row_id"].map(sample_id_to_prob).astype("float32")
            )
            test_predictions[f"p_{short}"] = (
                test_predictions["sample_row_id"].map(sample_id_to_prob).astype("float32")
            )

            val_target_values = pd.to_numeric(val_predictions[target], errors="coerce")
            val_target_available = val_target_values.notna()
            calibration_probabilities[(feature_set, target)] = (
                val_predictions.loc[val_target_available, f"p_{short}"].to_numpy(dtype=float)
            )
            calibration_labels[(feature_set, target)] = (
                val_target_values.loc[val_target_available].astype(int).to_numpy()
            )

        val_predictions_by_feature_set[feature_set] = val_predictions.reset_index(drop=True)
        test_predictions_by_feature_set[feature_set] = test_predictions.reset_index(drop=True)

    performance = pd.DataFrame(performance_records)
    selected = pd.DataFrame(selected_records)
    importances = pd.concat(importance_frames, ignore_index=True)

    performance.to_csv(output_dir / "tables" / "table_01_candidate_model_performance.csv", index=False)
    selected.to_csv(output_dir / "tables" / "table_02_selected_models.csv", index=False)
    importances.to_csv(output_dir / "tables" / "table_03_selected_model_feature_importance.csv", index=False)
    write_markdown_table(performance, output_dir / "tables" / "table_01_candidate_model_performance.md")
    write_markdown_table(selected, output_dir / "tables" / "table_02_selected_models.md")
    write_markdown_table(importances, output_dir / "tables" / "table_03_selected_model_feature_importance.md")

    headline_val_path = output_dir / "predictions" / "strict_ex_ante_val_predictions.parquet"
    headline_test_path = output_dir / "predictions" / "strict_ex_ante_test_predictions.parquet"
    baseline_val_path = output_dir / "predictions" / "strict_ex_ante_basic_val_predictions.parquet"
    baseline_test_path = output_dir / "predictions" / "strict_ex_ante_basic_test_predictions.parquet"
    supp_val_path = output_dir / "predictions" / "audit_priority_val_predictions.parquet"
    supp_test_path = output_dir / "predictions" / "audit_priority_test_predictions.parquet"

    val_predictions_by_feature_set[HEADLINE_FEATURE_SET].to_parquet(headline_val_path, index=False)
    test_predictions_by_feature_set[HEADLINE_FEATURE_SET].to_parquet(headline_test_path, index=False)
    val_predictions_by_feature_set[BASELINE_EX_ANTE_FEATURE_SET].to_parquet(baseline_val_path, index=False)
    test_predictions_by_feature_set[BASELINE_EX_ANTE_FEATURE_SET].to_parquet(baseline_test_path, index=False)
    val_predictions_by_feature_set[SUPPLEMENTARY_FEATURE_SET].to_parquet(supp_val_path, index=False)
    test_predictions_by_feature_set[SUPPLEMENTARY_FEATURE_SET].to_parquet(supp_test_path, index=False)

    save_metric_figure(performance, output_dir / "figures" / "figure_01_candidate_model_comparison.png")
    save_selected_precision_figure(selected, output_dir / "figures" / "figure_02_selected_model_precision.png")
    save_importance_figure(importances, HEADLINE_FEATURE_SET, output_dir / "figures" / "figure_03_feature_importance.png")
    save_calibration_figure(
        calibration_probabilities,
        calibration_labels,
        output_dir / "figures" / "figure_04_calibration_diagrams.png",
    )

    summary = {
        "cuda_available": use_cuda,
        "split_caps": split_caps,
        "candidate_models": ["sgd_logistic", "lightgbm", "xgboost"],
        "calibration_method": "isotonic regression on validation split with cv='prefit'",
        "headline_feature_set": HEADLINE_FEATURE_SET,
        "selected_models": selected[
            [
                "feature_set",
                "target",
                "model",
                "pr_auc",
                "roc_auc",
                "brier_score",
                "ece",
                "val_calibrated_precision_at_5pct",
                "val_calibrated_recall_at_5pct",
            ]
        ].to_dict("records"),
        "prediction_paths": {
            "strict_ex_ante_enhanced_val": str(headline_val_path),
            "strict_ex_ante_enhanced_test": str(headline_test_path),
            "strict_ex_ante_basic_val": str(baseline_val_path),
            "strict_ex_ante_basic_test": str(baseline_test_path),
            "audit_priority_val": str(supp_val_path),
            "audit_priority_test": str(supp_test_path),
        },
    }
    (output_dir / "risk_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Risk Prediction Summary",
        "",
        f"- CUDA available: {use_cuda}",
        f"- Headline feature set: {HEADLINE_FEATURE_SET}",
        f"- Split caps train/val/test: {max_rows_train:,}/{max_rows_val:,}/{max_rows_test:,}",
        f"- Calibration method: isotonic regression on validation split with cv='prefit'",
        f"- Prediction outputs: see `outputs/risk_prediction/predictions/`",
        "",
        "Generated tables are in `outputs/risk_prediction/tables`.",
        "Generated figures are in `outputs/risk_prediction/figures`.",
    ]
    (output_dir / "risk_prediction_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train risk prediction models with held-out test split.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--manifest-path", type=Path, default=PREPROCESS_TABLE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--max-rows-train", type=int, default=120_000)
    parser.add_argument("--max-rows-val", type=int, default=40_000)
    parser.add_argument("--max-rows-test", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=150_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        max_rows_train=args.max_rows_train,
        max_rows_val=args.max_rows_val,
        max_rows_test=args.max_rows_test,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
