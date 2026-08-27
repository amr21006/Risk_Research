"""Shared loaders for the Revision 2 supporting analyses.

Every analysis in this folder reuses the exact criterion construction,
normalization, and aggregation code of the submitted pipeline
(``src/04_auto_mcdm.py`` and ``src/05_pymoo_ensemble.py``) so that the new
results are computed on the same artifacts as the manuscript tables.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OPTI_ROOT = REPO_ROOT  # kept for readability in the analysis scripts
SRC = OPTI_ROOT / "src"
OUT = REPO_ROOT / "outputs" / "revision2"
OUT.mkdir(parents=True, exist_ok=True)

VAL_PREDICTIONS = OPTI_ROOT / "outputs/risk_prediction/predictions/strict_ex_ante_val_predictions.parquet"
TEST_PREDICTIONS = OPTI_ROOT / "outputs/risk_prediction/predictions/strict_ex_ante_test_predictions.parquet"
ARCHIVE = OPTI_ROOT / "outputs/pareto_ensemble/tables/table_04_global_non_dominated_archive.csv"
PARETO_RANKING = OPTI_ROOT / "outputs/pareto_ensemble/rankings/pareto_selected_auto_mcdm_ranking.parquet"
AUTO_RANKING = OPTI_ROOT / "outputs/auto_mcdm/rankings/selected_auto_mcdm_ranking.parquet"

OPTIMIZATION_ROWS = 15_000
SEED = 20260525

OBJECTIVE_COLUMNS = [
    "objective_ap_loss",
    "objective_ndcg_loss",
    "objective_f1_loss",
    "objective_any_f1_loss",
    "objective_review_burden",
    "objective_complexity",
]

TARGETS = [
    ("Composite integrity risk", "y_cri_high", "p_cri_high"),
    ("Market concentration risk", "y_buyer_concentration_high", "p_buyer_concentration_high"),
    ("No-call-for-tender risk", "y_no_call_for_tender", "p_no_call_for_tender"),
    ("Procedural risk", "y_proc_high", "p_proc_high"),
    ("Single-bid risk", "y_single_bid", "p_single_bid"),
]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MCDM = _load_module(SRC / "04_auto_mcdm.py", "auto_mcdm_module")


class Bench:
    """Validation/test criterion matrices, normalized exactly as in the pipeline."""

    def __init__(self, optimization_rows: int = OPTIMIZATION_ROWS, seed: int = SEED) -> None:
        val = pd.read_parquet(VAL_PREDICTIONS)
        test = pd.read_parquet(TEST_PREDICTIONS)
        val["y_any_risk"] = MCDM.composite_label(val)
        test["y_any_risk"] = MCDM.composite_label(test)

        if optimization_rows > 0 and len(val) > optimization_rows:
            val = val.sample(n=optimization_rows, random_state=seed).sort_values("sample_row_id").reset_index(drop=True)

        self.val_frame = val
        self.test_frame = test
        self.criteria_names = list(MCDM.make_criteria(val)[0].columns)

        val_matrix = MCDM.make_criteria(val)[0].to_numpy(dtype=np.float64)
        test_matrix = MCDM.make_criteria(test)[0].to_numpy(dtype=np.float64)
        self.norm_val = {
            name: MCDM.normalize_matrix(val_matrix, name, reference=val_matrix)
            for name in MCDM.NORMALIZATION_METHODS
        }
        self.norm_test = {
            name: MCDM.normalize_matrix(test_matrix, name, reference=val_matrix)
            for name in MCDM.NORMALIZATION_METHODS
        }

    def decode_row(self, row: pd.Series) -> dict[str, Any]:
        mask = np.array([bool(row[f"criterion_selected_{c}"]) for c in self.criteria_names], dtype=bool)
        weights = np.array([float(row[f"weight_{c}"]) for c in self.criteria_names], dtype=float)[mask]
        weights = weights / max(weights.sum(), 1e-12)
        return {
            "selected_mask": mask,
            "weights": weights,
            "method": str(row["method"]),
            "normalization": str(row["normalization"]),
            "top_k_rate": float(row["top_k_rate"]),
        }

    def score_test(self, decoded: dict[str, Any]) -> np.ndarray:
        matrix = self.norm_test[decoded["normalization"]][:, decoded["selected_mask"]]
        scores = MCDM.MCDM_FUNCTIONS[decoded["method"]](matrix, decoded["weights"])
        return MCDM.safe_minmax(scores.reshape(-1, 1)).ravel()

    def score_val(self, decoded: dict[str, Any]) -> np.ndarray:
        matrix = self.norm_val[decoded["normalization"]][:, decoded["selected_mask"]]
        scores = MCDM.MCDM_FUNCTIONS[decoded["method"]](matrix, decoded["weights"])
        return MCDM.safe_minmax(scores.reshape(-1, 1)).ravel()


def load_archive() -> pd.DataFrame:
    return pd.read_csv(ARCHIVE)


def scaled_objective_matrix(front: pd.DataFrame) -> np.ndarray:
    matrix = front[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    span = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
    return (matrix - minimum) / span


def knee_point_index(front: pd.DataFrame) -> int:
    return int(np.argmin(np.linalg.norm(scaled_objective_matrix(front), axis=1)))


def pseudo_weight_index(front: pd.DataFrame, target_weights: np.ndarray) -> int:
    """Deb and Sundar (2006) pseudo-weight selector, identical to src/05_pymoo_ensemble.py."""
    scaled = scaled_objective_matrix(front)
    inverse = 1 - scaled
    denominators = inverse.sum(axis=1, keepdims=True)
    denominators = np.where(denominators == 0, 1.0, denominators)
    pseudo_weights = inverse / denominators
    target = np.asarray(target_weights, dtype=float)
    target = target / target.sum()
    return int(np.argmin(np.linalg.norm(pseudo_weights - target[None, :], axis=1)))


def precision_at_k(y: np.ndarray, score: np.ndarray, k_ratio: float) -> float:
    k = max(int(np.ceil(len(score) * k_ratio)), 1)
    order = np.argsort(-score)[:k]
    return float(np.asarray(y)[order].sum() / k)


def top_k_set(score: np.ndarray, k_ratio: float) -> np.ndarray:
    k = max(int(np.ceil(len(score) * k_ratio)), 1)
    return np.argsort(-score)[:k]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


def f1_at_rate(y: np.ndarray, score: np.ndarray, rate: float) -> float:
    y = np.asarray(y)
    k = max(int(np.ceil(len(score) * rate)), 1)
    order = np.argsort(-score)[:k]
    hits = y[order].sum()
    precision = hits / k
    recall = hits / max(y.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def write(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"[written] {stem}.csv / .md  ({len(df)} rows)")
