"""
Five-algorithm Pareto optimization for automated MCDM configuration.

The script optimizes criterion masks, criterion weights, normalization choices,
MCDM method choices, and top-k review thresholds. It runs five pymoo algorithms
(NSGA-II, NSGA-III, R-NSGA-II, SMS-EMOA, AGE-MOEA), merges their candidate
archives, removes duplicates, extracts the global non-dominated front, and
selects one compromise solution using two principled multi-criterion selectors:
a knee-point selector (minimum distance to the ideal in scaled objective space)
and a pseudo-weight selector following Deb and Sundar (2006).

Optimization objectives are evaluated on the validation split. Final reporting
uses the disjoint test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.metrics import average_precision_score


VAL_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_val_predictions.parquet")
TEST_PREDICTIONS_DEFAULT = Path("outputs/risk_prediction/predictions/strict_ex_ante_test_predictions.parquet")
OUTPUT_DEFAULT = Path("outputs/pareto_ensemble")
AUTO_MCDM_PATH = Path(__file__).with_name("04_auto_mcdm.py")

ALGORITHMS = ["NSGA-II", "NSGA-III", "R-NSGA-II", "SMS-EMOA", "AGE-MOEA"]


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


@dataclass
class SplitData:
    frame: pd.DataFrame
    criteria: pd.DataFrame
    normalized_matrices_val: dict[str, np.ndarray]
    normalized_matrices_test: dict[str, np.ndarray]
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    val_y_primary: np.ndarray
    val_y_any: np.ndarray
    val_primary_mask: np.ndarray
    val_any_mask: np.ndarray
    test_y_primary: np.ndarray
    test_y_any: np.ndarray
    test_primary_mask: np.ndarray
    test_any_mask: np.ndarray


def load_split_data(val_path: Path, test_path: Path, max_rows: int, seed: int) -> SplitData:
    val_frame = pd.read_parquet(val_path)
    test_frame = pd.read_parquet(test_path)
    val_frame["y_any_risk"] = MCDM.composite_label(val_frame)
    test_frame["y_any_risk"] = MCDM.composite_label(test_frame)

    if max_rows > 0 and len(val_frame) > max_rows:
        sampled = val_frame.sample(n=max_rows, random_state=seed).sort_values("sample_row_id")
        val_frame_opt = sampled.reset_index(drop=True)
    else:
        val_frame_opt = val_frame

    val_criteria, _ = MCDM.make_criteria(val_frame_opt)
    test_criteria, _ = MCDM.make_criteria(test_frame)

    val_primary_mask = val_frame_opt["y_cri_high"].notna().to_numpy()
    val_any_mask = val_frame_opt["y_any_risk"].notna().to_numpy()
    test_primary_mask = test_frame["y_cri_high"].notna().to_numpy()
    test_any_mask = test_frame["y_any_risk"].notna().to_numpy()

    val_y_primary = val_frame_opt.loc[val_primary_mask, "y_cri_high"].astype(int).to_numpy()
    val_y_any = val_frame_opt.loc[val_any_mask, "y_any_risk"].astype(int).to_numpy()
    test_y_primary = test_frame.loc[test_primary_mask, "y_cri_high"].astype(int).to_numpy()
    test_y_any = test_frame.loc[test_any_mask, "y_any_risk"].astype(int).to_numpy()

    val_matrix = val_criteria.to_numpy(dtype=np.float64)
    test_matrix = test_criteria.to_numpy(dtype=np.float64)
    normalized_val = {
        name: MCDM.normalize_matrix(val_matrix, name, reference=val_matrix)
        for name in MCDM.NORMALIZATION_METHODS
    }
    normalized_test = {
        name: MCDM.normalize_matrix(test_matrix, name, reference=val_matrix)
        for name in MCDM.NORMALIZATION_METHODS
    }

    return SplitData(
        frame=val_frame_opt,
        criteria=val_criteria,
        normalized_matrices_val=normalized_val,
        normalized_matrices_test=normalized_test,
        val_frame=val_frame_opt,
        test_frame=test_frame,
        val_y_primary=val_y_primary,
        val_y_any=val_y_any,
        val_primary_mask=val_primary_mask,
        val_any_mask=val_any_mask,
        test_y_primary=test_y_primary,
        test_y_any=test_y_any,
        test_primary_mask=test_primary_mask,
        test_any_mask=test_any_mask,
    )


def f1_at_k(y_true: np.ndarray, scores: np.ndarray, rate: float) -> tuple[float, float, float, float]:
    k = max(int(math.ceil(len(y_true) * rate)), 1)
    order = np.argsort(-scores)[:k]
    hits = int(y_true[order].sum())
    positives = max(int(y_true.sum()), 1)
    precision = hits / k
    recall = hits / positives
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ndcg = MCDM.ndcg_at_k(y_true, scores, k)
    return precision, recall, f1, ndcg


class AutoMCDMProblem(ElementwiseProblem):
    def __init__(self, data: SplitData, min_criteria: int = 3):
        self.opt_data = data
        self.criteria_names = data.criteria.columns.tolist()
        self.n_criteria = len(self.criteria_names)
        self.min_criteria = min_criteria
        self.n_methods = len(MCDM.MCDM_METHODS)
        self.n_normalizations = len(MCDM.NORMALIZATION_METHODS)
        n_var = self.n_criteria * 2 + 3
        super().__init__(
            n_var=n_var,
            n_obj=6,
            n_ieq_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        mask_values = x[: self.n_criteria]
        weight_values = x[self.n_criteria : 2 * self.n_criteria]
        selected_mask = mask_values >= 0.35
        if int(selected_mask.sum()) < self.min_criteria:
            top_indices = np.argsort(-mask_values)[: self.min_criteria]
            selected_mask = np.zeros(self.n_criteria, dtype=bool)
            selected_mask[top_indices] = True

        raw_weights = np.clip(weight_values[selected_mask], 1e-9, None)
        weights = raw_weights / raw_weights.sum()

        method_index = int(np.clip(np.floor(x[-3] * self.n_methods), 0, self.n_methods - 1))
        normalization_index = int(np.clip(np.floor(x[-2] * self.n_normalizations), 0, self.n_normalizations - 1))
        top_k_rate = 0.01 + float(x[-1]) * 0.19

        return {
            "selected_mask": selected_mask,
            "weights": weights,
            "method": MCDM.MCDM_METHODS[method_index],
            "normalization": MCDM.NORMALIZATION_METHODS[normalization_index],
            "top_k_rate": top_k_rate,
        }

    def score_val(self, decoded: dict[str, Any]) -> np.ndarray:
        matrix = self.opt_data.normalized_matrices_val[decoded["normalization"]][:, decoded["selected_mask"]]
        scores = MCDM.MCDM_FUNCTIONS[decoded["method"]](matrix, decoded["weights"])
        return MCDM.safe_minmax(scores.reshape(-1, 1)).ravel()

    def score_test(self, decoded: dict[str, Any]) -> np.ndarray:
        matrix = self.opt_data.normalized_matrices_test[decoded["normalization"]][:, decoded["selected_mask"]]
        scores = MCDM.MCDM_FUNCTIONS[decoded["method"]](matrix, decoded["weights"])
        return MCDM.safe_minmax(scores.reshape(-1, 1)).ravel()

    def evaluate_val(self, decoded: dict[str, Any]) -> dict[str, float]:
        scores = self.score_val(decoded)
        primary_scores = scores[self.opt_data.val_primary_mask]
        any_scores = scores[self.opt_data.val_any_mask]

        primary_precision, primary_recall, primary_f1, primary_ndcg = f1_at_k(
            self.opt_data.val_y_primary, primary_scores, decoded["top_k_rate"]
        )
        any_precision, any_recall, any_f1, any_ndcg = f1_at_k(
            self.opt_data.val_y_any, any_scores, decoded["top_k_rate"]
        )
        try:
            average_precision = average_precision_score(self.opt_data.val_y_primary, primary_scores)
        except Exception:
            average_precision = 0.0

        complexity = float(decoded["selected_mask"].sum() / self.n_criteria)
        review_burden = float(decoded["top_k_rate"])

        return {
            "primary_average_precision": float(average_precision),
            "primary_precision": primary_precision,
            "primary_recall": primary_recall,
            "primary_f1": primary_f1,
            "primary_ndcg": primary_ndcg,
            "any_precision": any_precision,
            "any_recall": any_recall,
            "any_f1": any_f1,
            "any_ndcg": any_ndcg,
            "review_burden": review_burden,
            "complexity": complexity,
        }

    def evaluate_test(self, decoded: dict[str, Any]) -> dict[str, float]:
        scores = self.score_test(decoded)
        primary_scores = scores[self.opt_data.test_primary_mask]
        any_scores = scores[self.opt_data.test_any_mask]

        primary_precision, primary_recall, primary_f1, primary_ndcg = f1_at_k(
            self.opt_data.test_y_primary, primary_scores, decoded["top_k_rate"]
        )
        any_precision, any_recall, any_f1, any_ndcg = f1_at_k(
            self.opt_data.test_y_any, any_scores, decoded["top_k_rate"]
        )
        try:
            average_precision = average_precision_score(self.opt_data.test_y_primary, primary_scores)
        except Exception:
            average_precision = 0.0

        complexity = float(decoded["selected_mask"].sum() / self.n_criteria)
        review_burden = float(decoded["top_k_rate"])

        return {
            "test_primary_average_precision": float(average_precision),
            "test_primary_precision": primary_precision,
            "test_primary_recall": primary_recall,
            "test_primary_f1": primary_f1,
            "test_primary_ndcg": primary_ndcg,
            "test_any_precision": any_precision,
            "test_any_recall": any_recall,
            "test_any_f1": any_f1,
            "test_any_ndcg": any_ndcg,
            "test_review_burden": review_burden,
            "test_complexity": complexity,
        }

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        decoded = self.decode(x)
        metrics = self.evaluate_val(decoded)
        out["F"] = np.array(
            [
                1 - metrics["primary_average_precision"],
                1 - metrics["primary_ndcg"],
                1 - metrics["primary_f1"],
                1 - metrics["any_f1"],
                metrics["review_burden"],
                metrics["complexity"],
            ],
            dtype=np.float64,
        )


class HypervolumeCallback(Callback):
    def __init__(self, reference_point: np.ndarray):
        super().__init__()
        self.reference_point = reference_point
        self.history: list[dict[str, float]] = []
        self.indicator = HV(ref_point=self.reference_point)

    def notify(self, algorithm: Any) -> None:
        try:
            f_values = algorithm.pop.get("F")
            hv = float(self.indicator(f_values)) if f_values is not None and len(f_values) > 0 else math.nan
        except Exception:
            hv = math.nan
        self.history.append({"n_gen": int(algorithm.n_gen), "n_eval": int(algorithm.evaluator.n_eval), "hypervolume": hv})


def make_algorithm(name: str, pop_size: int, n_obj: int, seed: int) -> Any:
    if name == "NSGA-II":
        return NSGA2(pop_size=pop_size, eliminate_duplicates=True)
    if name == "NSGA-III":
        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=seed)
        return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
    if name == "R-NSGA-II":
        reference = np.zeros((1, n_obj))
        return RNSGA2(ref_points=reference, pop_size=pop_size, epsilon=0.01, eliminate_duplicates=True)
    if name == "SMS-EMOA":
        return SMSEMOA(pop_size=pop_size, eliminate_duplicates=True)
    if name == "AGE-MOEA":
        return AGEMOEA(pop_size=pop_size, eliminate_duplicates=True)
    raise ValueError(f"Unsupported algorithm: {name}")


def collect_result_population(result: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(result, "pop", None) is not None:
        x = result.pop.get("X")
        f = result.pop.get("F")
        if x is not None and f is not None:
            return np.asarray(x), np.asarray(f)
    return np.asarray(result.X), np.asarray(result.F)


def decision_signature(problem: AutoMCDMProblem, x: np.ndarray) -> str:
    decoded = problem.decode(x)
    selected_indices = np.where(decoded["selected_mask"])[0].tolist()
    weights = np.round(decoded["weights"], 4).tolist()
    return json.dumps(
        {
            "method": decoded["method"],
            "normalization": decoded["normalization"],
            "top_k_rate": round(decoded["top_k_rate"], 4),
            "criteria": selected_indices,
            "weights": weights,
        },
        sort_keys=True,
    )


def archive_records(problem: AutoMCDMProblem, algorithm_name: str, x_values: np.ndarray, f_values: np.ndarray) -> list[dict[str, Any]]:
    records = []
    for index, (x, f) in enumerate(zip(x_values, f_values)):
        decoded = problem.decode(x)
        val_metrics = problem.evaluate_val(decoded)
        weights_full = np.zeros(problem.n_criteria, dtype=float)
        weights_full[decoded["selected_mask"]] = decoded["weights"]
        record = {
            "algorithm": algorithm_name,
            "solution_index": index,
            "method": decoded["method"],
            "normalization": decoded["normalization"],
            "top_k_rate": decoded["top_k_rate"],
            "selected_criteria_count": int(decoded["selected_mask"].sum()),
            "decision_signature": decision_signature(problem, x),
            "objective_ap_loss": f[0],
            "objective_ndcg_loss": f[1],
            "objective_f1_loss": f[2],
            "objective_any_f1_loss": f[3],
            "objective_review_burden": f[4],
            "objective_complexity": f[5],
            **val_metrics,
        }
        for criterion, selected, weight in zip(problem.criteria_names, decoded["selected_mask"], weights_full):
            record[f"criterion_selected_{criterion}"] = bool(selected)
            record[f"weight_{criterion}"] = weight
        records.append(record)
    return records


def remove_duplicate_solutions(archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = len(archive)
    decision_unique = archive.drop_duplicates(subset=["decision_signature"]).copy()
    after_decision = len(decision_unique)

    objective_columns = [column for column in archive.columns if column.startswith("objective_")]
    rounded = decision_unique[objective_columns].round(5).astype(str).agg("|".join, axis=1)
    objective_unique = decision_unique.loc[~rounded.duplicated()].copy()
    after_objective = len(objective_unique)

    filtering = pd.DataFrame(
        [
            {"stage": "raw merged archive", "solutions": start},
            {"stage": "decision space unique", "solutions": after_decision},
            {"stage": "objective space unique", "solutions": after_objective},
        ]
    )
    return objective_unique.reset_index(drop=True), filtering


def extract_non_dominated(archive: pd.DataFrame) -> pd.DataFrame:
    objective_columns = [column for column in archive.columns if column.startswith("objective_")]
    values = archive[objective_columns].to_numpy(dtype=float)
    fronts = NonDominatedSorting().do(values, only_non_dominated_front=False)
    archive = archive.copy()
    archive["pareto_front"] = np.nan
    for front_index, front in enumerate(fronts):
        archive.loc[front, "pareto_front"] = front_index
    return archive.loc[archive["pareto_front"].eq(0)].reset_index(drop=True)


def scaled_objective_matrix(front: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    objective_columns = [column for column in front.columns if column.startswith("objective_")]
    matrix = front[objective_columns].to_numpy(dtype=float)
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    span = np.where(maximum - minimum == 0, 1.0, maximum - minimum)
    scaled = (matrix - minimum) / span
    return scaled, objective_columns


def knee_point_selection(front: pd.DataFrame) -> tuple[int, dict[str, Any]]:
    scaled, objective_columns = scaled_objective_matrix(front)
    distances = np.linalg.norm(scaled, axis=1)
    knee_index = int(np.argmin(distances))
    return knee_index, {
        "selector": "knee point minimum distance to ideal",
        "objective_columns": objective_columns,
        "distance_to_ideal": float(distances[knee_index]),
    }


def pseudo_weight_selection(front: pd.DataFrame, target_weights: np.ndarray) -> tuple[int, dict[str, Any]]:
    """Deb and Sundar (2006) pseudo weights selector.

    For each solution, the pseudo weight on objective i is
    w_i = (max_i - F_i) / (max_i - min_i) / sum_j ((max_j - F_j) / (max_j - min_j)).
    The selected solution is the one whose pseudo-weight vector is closest to
    the supplied target weights under L_2 distance.
    """
    scaled, objective_columns = scaled_objective_matrix(front)
    inverse = 1 - scaled
    denominators = inverse.sum(axis=1, keepdims=True)
    denominators = np.where(denominators == 0, 1.0, denominators)
    pseudo_weights = inverse / denominators
    target = target_weights / target_weights.sum()
    distances = np.linalg.norm(pseudo_weights - target[None, :], axis=1)
    selected_index = int(np.argmin(distances))
    return selected_index, {
        "selector": "pseudo weights L2 to target",
        "objective_columns": objective_columns,
        "target_weights": target.tolist(),
        "distance_to_target": float(distances[selected_index]),
    }


def decoded_from_row(problem: AutoMCDMProblem, selected: pd.Series) -> dict[str, Any]:
    mask = np.array(
        [bool(selected[f"criterion_selected_{criterion}"]) for criterion in problem.criteria_names],
        dtype=bool,
    )
    weights = np.array([float(selected[f"weight_{criterion}"]) for criterion in problem.criteria_names], dtype=float)
    selected_weights = weights[mask]
    selected_weights = selected_weights / max(selected_weights.sum(), 1e-12)
    return {
        "selected_mask": mask,
        "weights": selected_weights,
        "method": str(selected["method"]),
        "normalization": str(selected["normalization"]),
        "top_k_rate": float(selected["top_k_rate"]),
    }


def save_pareto_figure(non_dominated: pd.DataFrame, selected_knee: pd.Series, selected_pw: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=180)
    for algorithm, subset in non_dominated.groupby("algorithm"):
        ax.scatter(subset["primary_average_precision"], subset["primary_f1"], s=22, alpha=0.72, label=algorithm)
    ax.scatter(
        [selected_knee["primary_average_precision"]],
        [selected_knee["primary_f1"]],
        s=100,
        marker="*",
        color="#1f2d33",
        label="Knee point",
        zorder=5,
    )
    ax.scatter(
        [selected_pw["primary_average_precision"]],
        [selected_pw["primary_f1"]],
        s=120,
        marker="D",
        edgecolor="#8f1f2f",
        facecolor="none",
        linewidth=1.6,
        label="Pseudo weights",
        zorder=5,
    )
    ax.set_xlabel("Primary average precision on validation split")
    ax.set_ylabel("Primary F1 at selected threshold on validation split")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.10, 1, 1))
    fig.text(0.5, 0.025, "Figure 1. Global Pareto archive with principled compromise selectors", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_contribution_figure(non_dominated: pd.DataFrame, output_path: Path) -> None:
    counts = non_dominated["algorithm"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=180)
    bars = ax.bar(
        [clean_caption(label) for label in counts.index],
        counts.values,
        color="#5b7f95",
        edgecolor="#25323a",
        linewidth=0.6,
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Non dominated solutions")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 2. Algorithm contribution to final archive", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_selected_weight_figure(problem: AutoMCDMProblem, selected: pd.Series, output_path: Path) -> None:
    weights = pd.DataFrame(
        {
            "criterion": problem.criteria_names,
            "weight": [float(selected[f"weight_{criterion}"]) for criterion in problem.criteria_names],
        }
    )
    weights = weights.loc[weights["weight"] > 0].sort_values("weight", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=180)
    ax.barh(
        [clean_caption(value) for value in weights["criterion"]],
        weights["weight"],
        color="#d1a15f",
        edgecolor="#5f4a2b",
        linewidth=0.6,
    )
    ax.set_xlabel("Knee point criterion weight")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.08, 1, 1))
    fig.text(0.5, 0.02, "Figure 3. Knee point selected criterion weights", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution_figure(ranking: pd.DataFrame, output_path: Path) -> None:
    plot = ranking[ranking["y_cri_high"].notna()].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.hist(plot.loc[plot["y_cri_high"].eq(0), "pareto_auto_mcdm_score"], bins=35, alpha=0.70, label="Lower risk label", color="#6f8795")
    ax.hist(plot.loc[plot["y_cri_high"].eq(1), "pareto_auto_mcdm_score"], bins=35, alpha=0.65, label="High risk label", color="#d1a15f")
    ax.set_xlabel("Pareto Auto MCDM score on test split")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 4. Pareto Auto MCDM score distribution on test split", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_convergence_figure(convergence: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.1), dpi=180)
    for algorithm, subset in convergence.groupby("algorithm"):
        ax.plot(subset["n_gen"], subset["hypervolume"], label=algorithm, linewidth=1.4, alpha=0.9)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 5. Hypervolume convergence by algorithm", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_ensemble(
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    optimization_rows: int,
    pop_size: int,
    generations: int,
    seed: int,
) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "rankings").mkdir(parents=True, exist_ok=True)

    data = load_split_data(val_path, test_path, max_rows=optimization_rows, seed=seed)
    problem = AutoMCDMProblem(data)
    termination = get_termination("n_gen", generations)
    reference_point = np.ones(problem.n_obj) * 1.1

    archive_parts: list[pd.DataFrame] = []
    algorithm_rows = []
    convergence_rows = []

    for offset, name in enumerate(ALGORITHMS):
        algorithm = make_algorithm(name, pop_size=pop_size, n_obj=problem.n_obj, seed=seed + offset)
        callback = HypervolumeCallback(reference_point)
        start = time.perf_counter()
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed + offset,
            verbose=False,
            save_history=False,
            callback=callback,
        )
        runtime = time.perf_counter() - start
        x_values, f_values = collect_result_population(result)
        records = archive_records(problem, name, x_values, f_values)
        archive_part = pd.DataFrame(records)
        archive_parts.append(archive_part)

        try:
            hv = float(HV(ref_point=reference_point)(f_values))
        except Exception:
            hv = math.nan
        algorithm_rows.append(
            {
                "algorithm": name,
                "solutions_collected": len(archive_part),
                "runtime_seconds": runtime,
                "final_hypervolume": hv,
                "best_primary_average_precision": float(archive_part["primary_average_precision"].max()),
                "best_primary_f1": float(archive_part["primary_f1"].max()),
                "best_any_f1": float(archive_part["any_f1"].max()),
            }
        )
        for row in callback.history:
            convergence_rows.append({"algorithm": name, **row})

    raw_archive = pd.concat(archive_parts, ignore_index=True)
    unique_archive, filtering = remove_duplicate_solutions(raw_archive)
    non_dominated = extract_non_dominated(unique_archive)
    convergence = pd.DataFrame(convergence_rows)

    knee_index, knee_meta = knee_point_selection(non_dominated)
    selected_knee = non_dominated.iloc[knee_index].copy()
    target_weights = np.array([0.30, 0.20, 0.20, 0.15, 0.10, 0.05])
    pw_index, pw_meta = pseudo_weight_selection(non_dominated, target_weights)
    selected_pw = non_dominated.iloc[pw_index].copy()

    selected_knee_decoded = decoded_from_row(problem, selected_knee)
    selected_pw_decoded = decoded_from_row(problem, selected_pw)
    test_metrics_knee = problem.evaluate_test(selected_knee_decoded)
    test_metrics_pw = problem.evaluate_test(selected_pw_decoded)

    test_scores_knee = problem.score_test(selected_knee_decoded)
    test_top_k_knee = max(int(math.ceil(len(test_scores_knee) * float(selected_knee["top_k_rate"]))), 1)
    test_scores_pw = problem.score_test(selected_pw_decoded)
    test_top_k_pw = max(int(math.ceil(len(test_scores_pw) * float(selected_pw["top_k_rate"]))), 1)

    val_scores_knee = problem.score_val(selected_knee_decoded)
    val_top_k_knee = max(int(math.ceil(len(val_scores_knee) * float(selected_knee["top_k_rate"]))), 1)

    val_ranking = data.val_frame[["sample_row_id"] + MCDM.LABEL_COLUMNS + MCDM.PROBABILITY_COLUMNS + ["y_any_risk"]].copy()
    val_ranking["pareto_auto_mcdm_score"] = val_scores_knee.astype("float32")
    val_ranking["pareto_auto_mcdm_rank"] = val_ranking["pareto_auto_mcdm_score"].rank(ascending=False, method="first").astype(int)
    val_ranking["selected_for_review"] = val_ranking["pareto_auto_mcdm_rank"] <= val_top_k_knee
    val_ranking["split"] = "val"

    test_ranking = data.test_frame[["sample_row_id"] + MCDM.LABEL_COLUMNS + MCDM.PROBABILITY_COLUMNS + ["y_any_risk"]].copy()
    test_ranking["pareto_auto_mcdm_score"] = test_scores_knee.astype("float32")
    test_ranking["pareto_auto_mcdm_rank"] = test_ranking["pareto_auto_mcdm_score"].rank(ascending=False, method="first").astype(int)
    test_ranking["selected_for_review"] = test_ranking["pareto_auto_mcdm_rank"] <= test_top_k_knee
    test_ranking["split"] = "test"

    pd.concat([val_ranking, test_ranking], ignore_index=True).to_parquet(
        output_dir / "rankings" / "pareto_selected_auto_mcdm_ranking.parquet", index=False
    )

    algorithm_summary = pd.DataFrame(algorithm_rows)
    selected_records = []
    for label, selected_row, decoded, test_metrics, val_top_k, test_top_k, meta in (
        ("knee", selected_knee, selected_knee_decoded, test_metrics_knee, val_top_k_knee, test_top_k_knee, knee_meta),
        ("pseudo_weights", selected_pw, selected_pw_decoded, test_metrics_pw, None, test_top_k_pw, pw_meta),
    ):
        row = selected_row.to_dict()
        row["selector"] = label
        row["selector_metadata"] = json.dumps(meta)
        for key, value in test_metrics.items():
            row[key] = value
        row["test_selected_top_k_count"] = test_top_k
        if val_top_k is not None:
            row["val_selected_top_k_count"] = val_top_k
        selected_records.append(row)
    selected_solutions = pd.DataFrame(selected_records)

    algorithm_summary.to_csv(output_dir / "tables" / "table_01_algorithm_summary.csv", index=False)
    filtering.to_csv(output_dir / "tables" / "table_02_duplicate_filtering.csv", index=False)
    unique_archive.to_csv(output_dir / "tables" / "table_03_unique_archive.csv", index=False)
    non_dominated.to_csv(output_dir / "tables" / "table_04_global_non_dominated_archive.csv", index=False)
    selected_solutions.to_csv(output_dir / "tables" / "table_05_selected_pareto_solutions.csv", index=False)
    convergence.to_csv(output_dir / "tables" / "table_06_hypervolume_convergence.csv", index=False)
    test_ranking.head(100).to_csv(output_dir / "tables" / "table_07_top_ranked_tenders_test.csv", index=False)

    write_markdown_table(algorithm_summary, output_dir / "tables" / "table_01_algorithm_summary.md")
    write_markdown_table(filtering, output_dir / "tables" / "table_02_duplicate_filtering.md")
    write_markdown_table(selected_solutions, output_dir / "tables" / "table_05_selected_pareto_solutions.md")
    write_markdown_table(test_ranking.head(30), output_dir / "tables" / "table_07_top_ranked_tenders_test.md")

    save_pareto_figure(non_dominated, selected_knee, selected_pw, output_dir / "figures" / "figure_01_global_pareto_archive.png")
    save_contribution_figure(non_dominated, output_dir / "figures" / "figure_02_algorithm_contribution.png")
    save_selected_weight_figure(problem, selected_knee, output_dir / "figures" / "figure_03_pareto_selected_weights.png")
    save_score_distribution_figure(test_ranking, output_dir / "figures" / "figure_04_pareto_score_distribution.png")
    save_convergence_figure(convergence, output_dir / "figures" / "figure_05_hypervolume_convergence.png")

    summary = {
        "optimization_rows": int(len(data.val_frame)),
        "test_rows": int(len(data.test_frame)),
        "algorithms": ALGORITHMS,
        "population_size": pop_size,
        "generations": generations,
        "raw_archive_solutions": int(len(raw_archive)),
        "unique_archive_solutions": int(len(unique_archive)),
        "non_dominated_solutions": int(len(non_dominated)),
        "selected_solution_knee": {
            "algorithm": str(selected_knee["algorithm"]),
            "method": str(selected_knee["method"]),
            "normalization": str(selected_knee["normalization"]),
            "top_k_rate": float(selected_knee["top_k_rate"]),
            "val_top_k_count": int(val_top_k_knee),
            "test_top_k_count": int(test_top_k_knee),
            "val_primary_average_precision": float(selected_knee["primary_average_precision"]),
            "val_primary_f1": float(selected_knee["primary_f1"]),
            "test_primary_average_precision": float(test_metrics_knee["test_primary_average_precision"]),
            "test_primary_f1": float(test_metrics_knee["test_primary_f1"]),
            "test_any_f1": float(test_metrics_knee["test_any_f1"]),
            "selected_criteria_count": int(selected_knee["selected_criteria_count"]),
            "selector_metadata": knee_meta,
        },
        "selected_solution_pseudo_weights": {
            "algorithm": str(selected_pw["algorithm"]),
            "method": str(selected_pw["method"]),
            "normalization": str(selected_pw["normalization"]),
            "top_k_rate": float(selected_pw["top_k_rate"]),
            "test_top_k_count": int(test_top_k_pw),
            "test_primary_average_precision": float(test_metrics_pw["test_primary_average_precision"]),
            "test_primary_f1": float(test_metrics_pw["test_primary_f1"]),
            "test_any_f1": float(test_metrics_pw["test_any_f1"]),
            "selected_criteria_count": int(selected_pw["selected_criteria_count"]),
            "selector_metadata": pw_meta,
        },
        "ranking_path": str(output_dir / "rankings" / "pareto_selected_auto_mcdm_ranking.parquet"),
    }
    (output_dir / "pareto_ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Pareto Ensemble Summary",
        "",
        f"- Validation rows used for optimization: {len(data.val_frame):,}",
        f"- Test rows used for reporting: {len(data.test_frame):,}",
        f"- Algorithms: {', '.join(ALGORITHMS)}",
        f"- Population size: {pop_size}, generations: {generations}",
        f"- Raw archive solutions: {len(raw_archive):,}",
        f"- Unique archive solutions: {len(unique_archive):,}",
        f"- Global non-dominated solutions: {len(non_dominated):,}",
        f"- Knee selected method: {selected_knee['method']}",
        f"- Knee selected normalization: {selected_knee['normalization']}",
        f"- Knee selected top-k rate: {float(selected_knee['top_k_rate']):.2%}",
        f"- Test PR-AUC at knee point: {float(test_metrics_knee['test_primary_average_precision']):.4f}",
        f"- Test F1 at knee point: {float(test_metrics_knee['test_primary_f1']):.4f}",
        "",
        "Generated tables are in `outputs/pareto_ensemble/tables`.",
        "Generated figures are in `outputs/pareto_ensemble/figures`.",
    ]
    (output_dir / "pareto_ensemble_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-algorithm pymoo Auto-MCDM ensemble with principled compromise selectors.")
    parser.add_argument("--val-path", type=Path, default=VAL_PREDICTIONS_DEFAULT)
    parser.add_argument("--test-path", type=Path, default=TEST_PREDICTIONS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--optimization-rows", type=int, default=15_000)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ensemble(
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        optimization_rows=args.optimization_rows,
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
