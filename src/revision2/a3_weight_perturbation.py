"""A3 - Sensitivity of the selected configuration to the pseudo-weight vector.

Addresses Reviewer #2, Comment 6 (follow-up).  The pre-specified target vector
(0.30, 0.20, 0.20, 0.15, 0.10, 0.05) is perturbed and the pseudo-weight selector
is re-run over the 428-solution global non-dominated archive.  Each selected
configuration is decoded and re-scored on the held-out test sample, and the
resulting inspection queue is compared with the knee-point queue.

Two perturbation schemes are reported:
  one-at-a-time  each component scaled by 0.5 and by 1.5 in turn, renormalized
                 (12 variants)
  neighbourhood  500 Dirichlet draws centred on the base vector with a
                 concentration that yields roughly +/-50% component variation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import r2_common as C

BASE = np.array([0.30, 0.20, 0.20, 0.15, 0.10, 0.05])
COMPONENT = [
    "primary AP",
    "primary nDCG",
    "primary F1",
    "secondary F1",
    "review burden",
    "configuration complexity",
]
DIRICHLET_DRAWS = 500
DIRICHLET_CONCENTRATION = 12.0
SEED = 20260820


def describe(row: pd.Series) -> str:
    return (
        f"{row['algorithm']} / {row['method']} / {row['normalization']} / "
        f"{int(row['selected_criteria_count'])} criteria / {100 * row['top_k_rate']:.1f}%"
    )


def main() -> None:
    bench = C.Bench()
    archive = C.load_archive()
    y = bench.test_frame["y_cri_high"]
    valid = y.notna().to_numpy()
    y_valid = y[valid].astype(int).to_numpy()

    knee_idx = C.knee_point_index(archive)
    knee_row = archive.iloc[knee_idx]
    knee_scores = bench.score_test(bench.decode_row(knee_row))
    knee_queue = C.top_k_set(knee_scores, float(knee_row["top_k_rate"]))

    base_idx = C.pseudo_weight_index(archive, BASE)
    print(f"knee point   : #{knee_idx} {describe(knee_row)}")
    print(f"base pseudo-w: #{base_idx} {describe(archive.iloc[base_idx])}")

    def evaluate(name: str, weights: np.ndarray) -> dict:
        idx = C.pseudo_weight_index(archive, weights)
        row = archive.iloc[idx]
        scores = bench.score_test(bench.decode_row(row))
        queue = C.top_k_set(scores, float(row["top_k_rate"]))
        rho = spearmanr(scores, knee_scores).statistic
        return {
            "perturbation": name,
            "target vector": " / ".join(f"{w:.3f}" for w in weights / weights.sum()),
            "archive solution": int(idx),
            "selected configuration": describe(row),
            "review rate (%)": round(100 * float(row["top_k_rate"]), 1),
            "test primary AP (%)": round(100 * average_precision_score(y_valid, scores[valid]), 2),
            "queue Jaccard vs knee point": round(C.jaccard(queue, knee_queue), 3),
            "rank Spearman vs knee point": round(float(rho), 4),
        }

    rows = [evaluate("Base vector (pre-specified)", BASE)]
    for i, name in enumerate(COMPONENT):
        for factor, tag in ((0.5, "-50%"), (1.5, "+50%")):
            weights = BASE.copy()
            weights[i] = weights[i] * factor
            rows.append(evaluate(f"{name} {tag}", weights))

    oat = pd.DataFrame(rows)
    C.write(oat, "t_weight_perturbation_one_at_a_time")
    print()
    print(oat.to_string(index=False))

    # Dirichlet neighbourhood
    rng = np.random.default_rng(SEED)
    draws = rng.dirichlet(BASE * DIRICHLET_CONCENTRATION / BASE.sum() * len(BASE), size=DIRICHLET_DRAWS)
    cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    picks, jaccards, rhos, aps, rates = [], [], [], [], []
    for w in draws:
        idx = C.pseudo_weight_index(archive, w)
        if idx not in cache:
            row = archive.iloc[idx]
            scores = bench.score_test(bench.decode_row(row))
            cache[idx] = (
                C.top_k_set(scores, float(row["top_k_rate"])),
                scores,
                100 * average_precision_score(y_valid, scores[valid]),
            )
        queue, scores, ap = cache[idx]
        picks.append(idx)
        jaccards.append(C.jaccard(queue, knee_queue))
        rhos.append(float(spearmanr(scores, knee_scores).statistic))
        aps.append(ap)
        rates.append(100 * float(archive.iloc[idx]["top_k_rate"]))

    counts = pd.Series(picks).value_counts()
    modal_idx = int(counts.index[0])
    neighbourhood = pd.DataFrame(
        [
            {"statistic": "Dirichlet draws", "value": f"{DIRICHLET_DRAWS}"},
            {"statistic": "distinct archive solutions selected", "value": f"{len(counts)}"},
            {"statistic": "modal solution share (%)", "value": f"{100 * counts.iloc[0] / DIRICHLET_DRAWS:.1f}"},
            {"statistic": "modal solution", "value": describe(archive.iloc[modal_idx])},
            {"statistic": "test primary AP, min to max (%)", "value": f"{min(aps):.2f} to {max(aps):.2f}"},
            {"statistic": "review rate, min to max (%)", "value": f"{min(rates):.1f} to {max(rates):.1f}"},
            {
                "statistic": "queue Jaccard vs knee point, 5th to 95th percentile",
                "value": f"{np.percentile(jaccards, 5):.3f} to {np.percentile(jaccards, 95):.3f}",
            },
            {"statistic": "queue Jaccard vs knee point, median", "value": f"{np.median(jaccards):.3f}"},
            {"statistic": "rank Spearman vs knee point, median", "value": f"{np.median(rhos):.4f}"},
            {"statistic": "rank Spearman vs knee point, minimum", "value": f"{min(rhos):.4f}"},
        ]
    )
    C.write(neighbourhood, "t_weight_perturbation_neighbourhood")
    print()
    print(neighbourhood.to_string(index=False))

    detail = pd.DataFrame(
        [
            {
                "archive solution": int(i),
                "selected configuration": describe(archive.iloc[int(i)]),
                "draws selecting it": int(n),
                "share of draws (%)": round(100 * n / DIRICHLET_DRAWS, 1),
                "test primary AP (%)": round(cache[int(i)][2], 2),
                "queue Jaccard vs knee point": round(C.jaccard(cache[int(i)][0], knee_queue), 3),
            }
            for i, n in counts.items()
        ]
    )
    C.write(detail, "t_weight_perturbation_selected_solutions")
    print()
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
