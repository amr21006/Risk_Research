"""A4 - Dependence structure and incremental ranking value of the five predicted
probabilities, and a direct comparison against re-estimating the composite.

Addresses Reviewer #2, Comment 7 (follow-up), parts (a) and (b).

(a) Rank and linear dependence among the five calibrated probabilities on the
    held-out test sample, the principal-component structure, and the incremental
    test average precision each probability adds beyond the primary probability
    alone.  Increments are estimated by fitting the combination rule on the
    validation split and applying it once to the test split, with a paired
    bootstrap interval on the difference.

(b) A direct re-estimation comparator: a logistic re-estimation of the composite
    outcome on the five validation-split probabilities, evaluated on the test
    split alongside the raw primary probability, the Auto-MCDM baseline and the
    Pareto-optimized configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import r2_common as C

PROBS = [
    ("p_cri_high", "Composite integrity risk"),
    ("p_proc_high", "Procedural risk"),
    ("p_buyer_concentration_high", "Market concentration risk"),
    ("p_single_bid", "Single-bid risk"),
    ("p_no_call_for_tender", "No-call-for-tender risk"),
]
LABELS = [name for _, name in PROBS]
COLUMNS = [col for col, _ in PROBS]
BOOTSTRAP = 1000
SEED = 20260820


def paired_bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, iterations: int = BOOTSTRAP) -> tuple[float, float]:
    """Percentile interval on AP(a) - AP(b) under paired resampling of test rows."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    deltas = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            deltas[i] = np.nan
            continue
        deltas[i] = average_precision_score(yb, a[idx]) - average_precision_score(yb, b[idx])
    deltas = deltas[~np.isnan(deltas)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    val = pd.read_parquet(C.VAL_PREDICTIONS)
    test = pd.read_parquet(C.TEST_PREDICTIONS)

    # dependence structure on the held-out test sample
    matrix = test[COLUMNS]
    spearman = matrix.corr(method="spearman")
    pearson = matrix.corr(method="pearson")
    for name, corr in (("spearman", spearman), ("pearson", pearson)):
        table = corr.copy()
        table.index = LABELS
        table.columns = LABELS
        table = table.round(3).reset_index().rename(columns={"index": "predicted probability"})
        C.write(table, f"t_probability_correlation_{name}")
        print(f"\n{name} correlation")
        print(table.to_string(index=False))

    scaled = StandardScaler().fit_transform(matrix.to_numpy(dtype=float))
    pca = PCA().fit(scaled)
    pca_table = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "variance explained (%)": np.round(100 * pca.explained_variance_ratio_, 1),
            "cumulative variance explained (%)": np.round(100 * np.cumsum(pca.explained_variance_ratio_), 1),
        }
    )
    C.write(pca_table, "t_probability_pca")
    print("\nprincipal components")
    print(pca_table.to_string(index=False))

    # incremental ranking value beyond the primary probability
    val_ok = val["y_cri_high"].notna()
    test_ok = test["y_cri_high"].notna()
    y_val = val.loc[val_ok, "y_cri_high"].astype(int).to_numpy()
    y_test = test.loc[test_ok, "y_cri_high"].astype(int).to_numpy()

    primary_test = test.loc[test_ok, "p_cri_high"].to_numpy()
    base_ap = average_precision_score(y_test, primary_test)

    rows = []
    for col, name in PROBS:
        if col == "p_cri_high":
            rows.append(
                {
                    "criterion added to primary probability": "none (primary probability alone)",
                    "standalone test AP (%)": round(100 * base_ap, 2),
                    "combined test AP (%)": round(100 * base_ap, 2),
                    "incremental AP (pp)": "0.00",
                    "95% paired bootstrap interval (pp)": "reference",
                }
            )
            continue
        features = ["p_cri_high", col]
        model = LogisticRegression(max_iter=1000)
        model.fit(val.loc[val_ok, features].to_numpy(dtype=float), y_val)
        combined = model.predict_proba(test.loc[test_ok, features].to_numpy(dtype=float))[:, 1]
        standalone = test.loc[test_ok, col].to_numpy()
        lo, hi = paired_bootstrap_delta(y_test, combined, primary_test)
        rows.append(
            {
                "criterion added to primary probability": name,
                "standalone test AP (%)": round(100 * average_precision_score(y_test, standalone), 2),
                "combined test AP (%)": round(100 * average_precision_score(y_test, combined), 2),
                "incremental AP (pp)": f"{100 * (average_precision_score(y_test, combined) - base_ap):+.2f}",
                "95% paired bootstrap interval (pp)": f"[{100*lo:+.2f}, {100*hi:+.2f}]",
            }
        )

    model_all = LogisticRegression(max_iter=1000)
    model_all.fit(val.loc[val_ok, COLUMNS].to_numpy(dtype=float), y_val)
    all_test = model_all.predict_proba(test.loc[test_ok, COLUMNS].to_numpy(dtype=float))[:, 1]
    lo, hi = paired_bootstrap_delta(y_test, all_test, primary_test)
    rows.append(
        {
            "criterion added to primary probability": "all four secondary probabilities",
            "standalone test AP (%)": "n/a",
            "combined test AP (%)": round(100 * average_precision_score(y_test, all_test), 2),
            "incremental AP (pp)": f"{100 * (average_precision_score(y_test, all_test) - base_ap):+.2f}",
            "95% paired bootstrap interval (pp)": f"[{100*lo:+.2f}, {100*hi:+.2f}]",
        }
    )
    increment = pd.DataFrame(rows)
    C.write(increment, "t_incremental_ranking_value")
    print("\nincremental ranking value beyond the primary probability")
    print(increment.to_string(index=False))

    # direct re-estimation comparator against the deployed strategies
    pareto = pd.read_parquet(C.PARETO_RANKING)
    auto = pd.read_parquet(C.AUTO_RANKING)
    pareto = pareto[pareto["split"] == "test"]
    auto = auto[auto["split"] == "test"]
    merged = pareto.merge(auto[["sample_row_id", "auto_mcdm_score"]], on="sample_row_id", how="inner")
    merged = merged[merged["y_cri_high"].notna()]
    order = merged["sample_row_id"].to_numpy()
    lookup = pd.Series(all_test, index=test.loc[test_ok, "sample_row_id"].to_numpy())
    reest = lookup.reindex(order).to_numpy()
    y_merged = merged["y_cri_high"].astype(int).to_numpy()

    strategies = {
        "Raw primary probability": merged["p_cri_high"].to_numpy(),
        "Logistic re-estimation of the composite on five probabilities": reest,
        "Auto-MCDM baseline": merged["auto_mcdm_score"].to_numpy(),
        "Pareto-optimized configuration": merged["pareto_auto_mcdm_score"].to_numpy(),
    }
    comparison = []
    for name, score in strategies.items():
        comparison.append(
            {
                "strategy": name,
                "test AP (%)": round(100 * average_precision_score(y_merged, score), 2),
                "test F1 at 20% review (%)": round(100 * C.f1_at_rate(y_merged, score, 0.20), 2),
                "precision at 5% review (%)": round(100 * C.precision_at_k(y_merged, score, 0.05), 1),
                "criteria an inspector must read": {
                    "Raw primary probability": 1,
                    "Logistic re-estimation of the composite on five probabilities": 5,
                    "Auto-MCDM baseline": 9,
                    "Pareto-optimized configuration": 3,
                }[name],
            }
        )
    comparison_df = pd.DataFrame(comparison)
    C.write(comparison_df, "t_reestimation_comparison")
    print("\ndirect re-estimation compared with the deployed strategies")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
