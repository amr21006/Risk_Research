"""A1 - Regenerate the multi-dimensional coverage and bootstrap tables.

Addresses Reviewer #1 (Table 5 / narrative reconciliation and the definition of
"raw probability") and the Additional Comment (consistent regeneration from one
pipeline run).

Two distinct comparators are produced and reported separately:

  single-queue  one ranking is produced by a strategy and scored against each of
                the five risk dimensions.  "raw probability" here is the raw
                primary-outcome (CRI-high) calibrated probability, which is the
                comparator defined in the Methods and used in every other table.

  per-target    each dimension is ranked by its own calibrated probability.
                This requires five separate queues and is therefore a
                per-dimension reference ceiling, not a deployable single queue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import r2_common as C

STRATEGY_LABEL = {
    "raw cri probability": "Raw primary probability",
    "baseline auto mcdm": "Auto-MCDM baseline",
    "pareto auto mcdm": "Pareto-optimized",
}


def merged_test_frame() -> pd.DataFrame:
    pareto = pd.read_parquet(C.PARETO_RANKING)
    auto = pd.read_parquet(C.AUTO_RANKING)
    pareto = pareto[pareto["split"] == "test"]
    auto = auto[auto["split"] == "test"]
    return pareto.merge(auto[["sample_row_id", "auto_mcdm_score"]], on="sample_row_id", how="inner")


def single_queue_coverage(df: pd.DataFrame) -> pd.DataFrame:
    queues = {
        "raw cri probability": df["p_cri_high"].to_numpy(),
        "baseline auto mcdm": df["auto_mcdm_score"].to_numpy(),
        "pareto auto mcdm": df["pareto_auto_mcdm_score"].to_numpy(),
    }
    rows = []
    for label, y_col, _ in C.TARGETS:
        valid = df[y_col].notna().to_numpy()
        y = df.loc[valid, y_col].astype(int).to_numpy()
        for strategy, score in queues.items():
            s = score[valid]
            rows.append(
                {
                    "strategy": STRATEGY_LABEL[strategy],
                    "risk dimension": label,
                    "evaluated rows": int(valid.sum()),
                    "positive rate (%)": round(100 * y.mean(), 1),
                    "test AP (%)": round(100 * average_precision_score(y, s), 1),
                    "precision at 5% (%)": round(100 * C.precision_at_k(y, s, 0.05), 1),
                    "precision at 10% (%)": round(100 * C.precision_at_k(y, s, 0.10), 1),
                    "precision at 20% (%)": round(100 * C.precision_at_k(y, s, 0.20), 1),
                }
            )
    order = {label: i for i, (label, _, _) in enumerate(C.TARGETS)}
    out = pd.DataFrame(rows)
    out["_o"] = out["risk dimension"].map(order)
    return out.sort_values(["_o", "strategy"]).drop(columns="_o").reset_index(drop=True)


def per_target_reference(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, y_col, p_col in C.TARGETS:
        valid = df[y_col].notna().to_numpy()
        y = df.loc[valid, y_col].astype(int).to_numpy()
        own = df.loc[valid, p_col].to_numpy()
        primary = df.loc[valid, "p_cri_high"].to_numpy()
        pareto = df.loc[valid, "pareto_auto_mcdm_score"].to_numpy()
        rows.append(
            {
                "risk dimension": label,
                "evaluated rows": int(valid.sum()),
                "positive rate (%)": round(100 * y.mean(), 1),
                "own-target probability AP (%)": round(100 * average_precision_score(y, own), 1),
                "raw primary probability AP (%)": round(100 * average_precision_score(y, primary), 1),
                "Pareto queue AP (%)": round(100 * average_precision_score(y, pareto), 1),
                "reference gap to Pareto queue (pp)": round(
                    100 * (average_precision_score(y, own) - average_precision_score(y, pareto)), 1
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_table() -> pd.DataFrame:
    src = pd.read_csv(C.OPTI_ROOT / "outputs/ablation/tables/table_07_bootstrap_comparisons.csv")
    label = {
        "raw cri probability": "Raw primary probability",
        "manual topsis equal": "Manual TOPSIS (equal weights)",
        "manual vikor equal": "Manual VIKOR (equal weights)",
        "manual waspas equal": "Manual WASPAS (equal weights)",
        "baseline auto mcdm": "Auto-MCDM baseline",
    }
    metric = {"average_precision": "Average precision", "f1_at_20pct": "F1 at 20% review"}
    rows = []
    for _, r in src.iterrows():
        # the artifact stores "comparison minus reference"; the Pareto advantage is its negative
        rows.append(
            {
                "comparison strategy": label[r["comparison_strategy"]],
                "metric": metric[r["metric"]],
                "observed Pareto advantage (pp)": f"{-100 * r['observed_difference']:+.2f}",
                "bootstrap mean Pareto advantage (pp)": f"{-100 * r['mean_bootstrap_difference']:+.2f}",
                "BCa CI lower (pp)": f"{-100 * r['bca_ci_upper']:+.2f}",
                "BCa CI upper (pp)": f"{-100 * r['bca_ci_lower']:+.2f}",
                "Holm p value": "<0.001" if r["holm_bonferroni_p_value"] < 0.001 else f"{r['holm_bonferroni_p_value']:.3f}",
                "favored strategy": (
                    "Pareto-optimized"
                    if r["favoured_strategy"] == "pareto auto mcdm"
                    else f"{label[r['comparison_strategy']]} (n.s.)"
                ),
            }
        )
    return pd.DataFrame(rows)


def reproduction_check(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm the stored Pareto ranking is reproduced by re-decoding the knee solution."""
    bench = C.Bench()
    archive = C.load_archive()
    knee = archive.iloc[C.knee_point_index(archive)]
    decoded = bench.decode_row(knee)
    scores = bench.score_test(decoded)
    stored = pd.read_parquet(C.PARETO_RANKING)
    stored = stored[stored["split"] == "test"].reset_index(drop=True)
    recomputed = pd.DataFrame({"sample_row_id": bench.test_frame["sample_row_id"].to_numpy(), "recomputed": scores})
    joined = stored[["sample_row_id", "pareto_auto_mcdm_score"]].merge(recomputed, on="sample_row_id")
    spearman = joined["pareto_auto_mcdm_score"].corr(joined["recomputed"], method="spearman")
    max_abs = float(np.abs(joined["pareto_auto_mcdm_score"] - joined["recomputed"]).max())
    return pd.DataFrame(
        [
            {"check": "knee configuration re-decoded from archive", "value": f"{knee['algorithm']} / {knee['method']} / {knee['normalization']} / {knee['selected_criteria_count']} criteria / {100*knee['top_k_rate']:.1f}% review"},
            {"check": "rows matched against stored ranking", "value": f"{len(joined):,}"},
            {"check": "Spearman correlation, recomputed vs stored score", "value": f"{spearman:.6f}"},
            {"check": "maximum absolute score difference", "value": f"{max_abs:.2e}"},
        ]
    )


def main() -> None:
    df = merged_test_frame()
    print(f"merged test rows: {len(df):,}")

    single = single_queue_coverage(df)
    C.write(single, "t_coverage_single_queue")
    print(single.to_string(index=False))

    reference = per_target_reference(df)
    C.write(reference, "t_coverage_per_target_reference")
    print(reference.to_string(index=False))

    boot = bootstrap_table()
    C.write(boot, "t_bootstrap_comparisons")
    print(boot.to_string(index=False))

    check = reproduction_check(df)
    C.write(check, "t_reproduction_check")
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
