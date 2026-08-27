"""A5 - A realistic capacity and preference setting in which the Pareto-selected
queue materially differs from the raw-probability queue.

Addresses Reviewer #2, Comment 7 (follow-up), part (c).

Scenario: an inspectorate with a multi-risk mandate can review 5% of the tender
lot portfolio in a cycle and is accountable for competition-related risk (single
bid, no call for tender) as well as composite integrity risk.

The configuration is chosen from the global non-dominated archive using
validation-split objective values only - the archive stores the validation
objectives, so no test information enters the choice.  Selection rule: among
archive solutions whose review burden lies within the agency's capacity band
(4% to 6%), take the solution with the lowest secondary-outcome F1 loss subject
to a primary ranking-quality floor.  The selected configuration is then scored
once on the held-out test sample and compared with the raw primary-probability
queue at the same capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import r2_common as C

CAPACITY = 0.05
BAND = (0.04, 0.06)
AP_FLOOR_MARGIN = 0.05  # allow up to 5 pp validation AP loss relative to the best in band


def main() -> None:
    bench = C.Bench()
    archive = C.load_archive()

    in_band = archive[
        (archive["review_burden"] >= BAND[0]) & (archive["review_burden"] <= BAND[1])
    ].copy()
    print(f"archive solutions in the {100*BAND[0]:.0f}-{100*BAND[1]:.0f}% capacity band: {len(in_band)}")

    floor = in_band["primary_average_precision"].max() - AP_FLOOR_MARGIN
    eligible = in_band[in_band["primary_average_precision"] >= floor].copy()
    print(f"eligible after the validation primary-AP floor ({100*floor:.1f}%): {len(eligible)}")

    chosen = eligible.sort_values("objective_any_f1_loss").iloc[0]
    print(
        "selected multi-risk configuration: "
        f"{chosen['algorithm']} / {chosen['method']} / {chosen['normalization']} / "
        f"{int(chosen['selected_criteria_count'])} criteria / {100*chosen['top_k_rate']:.1f}% review"
    )

    weights = {
        c.replace("weight_", ""): float(chosen[c])
        for c in archive.columns
        if c.startswith("weight_") and float(chosen[c]) > 0
    }
    weight_table = pd.DataFrame(
        [{"criterion": k.replace("_", " "), "weight (%)": round(100 * v, 1)} for k, v in sorted(weights.items(), key=lambda kv: -kv[1])]
    )
    C.write(weight_table, "t_multirisk_selected_weights")

    scenario_scores = bench.score_test(bench.decode_row(chosen))
    knee = archive.iloc[C.knee_point_index(archive)]
    knee_scores = bench.score_test(bench.decode_row(knee))
    raw_scores = bench.test_frame["p_cri_high"].to_numpy()

    queues = {
        "Raw primary probability": C.top_k_set(raw_scores, CAPACITY),
        "Knee-point configuration": C.top_k_set(knee_scores, CAPACITY),
        "Multi-risk mandate configuration": C.top_k_set(scenario_scores, CAPACITY),
    }
    scores = {
        "Raw primary probability": raw_scores,
        "Knee-point configuration": knee_scores,
        "Multi-risk mandate configuration": scenario_scores,
    }

    rows = []
    frame = bench.test_frame
    raw_queue = queues["Raw primary probability"]
    for name, queue in queues.items():
        record = {
            "queue": name,
            "lots reviewed": len(queue),
            "overlap with raw-probability queue (%)": round(100 * len(set(queue) & set(raw_queue)) / len(raw_queue), 1),
            "queue Jaccard vs raw probability": round(C.jaccard(queue, raw_queue), 3),
        }
        for label, y_col, _ in C.TARGETS:
            y = frame[y_col]
            captured = int(y.iloc[queue].fillna(0).sum())
            record[f"{label}: positives captured"] = captured
        distinct = frame.iloc[queue][[y for _, y, _ in C.TARGETS]].fillna(0).sum(axis=1)
        record["lots flagged on 2 or more dimensions"] = int((distinct >= 2).sum())
        record["total risk positives captured across dimensions"] = int(distinct.sum())
        valid = frame["y_cri_high"].notna().to_numpy()
        record["test primary AP (%)"] = round(
            100 * average_precision_score(frame.loc[valid, "y_cri_high"].astype(int), scores[name][valid]), 2
        )
        rows.append(record)

    scenario = pd.DataFrame(rows)
    C.write(scenario, "t_multirisk_scenario")
    print()
    print(scenario.to_string(index=False))

    # compact headline table for the manuscript
    headline = scenario[
        [
            "queue",
            "lots reviewed",
            "overlap with raw-probability queue (%)",
            "test primary AP (%)",
            "Single-bid risk: positives captured",
            "No-call-for-tender risk: positives captured",
            "Market concentration risk: positives captured",
            "total risk positives captured across dimensions",
            "lots flagged on 2 or more dimensions",
        ]
    ]
    C.write(headline, "t_multirisk_scenario_headline")
    print()
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
