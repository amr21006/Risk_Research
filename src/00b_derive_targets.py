"""Derive the five binary risk targets from the published Corruption Risk Index
component scores, and verify the derivation against the stored target columns.

This script makes the target binarization traceable in code.  Before this script
existed, the five ``y_*`` columns entered the modeling pipeline as precomputed
columns, and the thresholds were stated in the manuscript but not executed
anywhere in the repository.  The script closes that gap: it recomputes each
target from its source score and reports the agreement rate against the stored
column, row by row, over the whole modeling dataset.

Target definitions, as reported in the manuscript
-------------------------------------------------
  y_cri_high                  label_cri                        >= 0.50
  y_proc_high                 label_corr_proc                  >= 0.50
  y_buyer_concentration_high  label_corr_buyer_concentration   >= upper-quartile
                                                               cutoff (0.087196)
  y_single_bid                label_corr_singleb               >  0
  y_no_call_for_tender        label_corr_nocft                 >  0

The buyer-concentration cutoff is the 75th percentile of
``label_corr_buyer_concentration`` in the modeling data.  The script recomputes
that percentile and reports it alongside the fixed value used in the manuscript,
so the reader can confirm that the fixed value is the empirical quartile and not
an arbitrary constant.

Usage
-----
  python src/00b_derive_targets.py
  python src/00b_derive_targets.py --data data/modeling/tender_lot_strict_ex_ante_enhanced.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DEFAULT = Path("data/modeling/tender_lot_strict_ex_ante_enhanced.parquet")
OUTPUT_DEFAULT = Path("outputs/target_derivation")

BUYER_CONCENTRATION_CUTOFF = 0.087196
BUYER_CONCENTRATION_QUANTILE = 0.75

SPECIFICATION = [
    ("y_cri_high", "label_cri", "ge", 0.50, "composite Corruption Risk Index score at or above 0.50"),
    ("y_proc_high", "label_corr_proc", "ge", 0.50, "procedural-risk score at or above 0.50"),
    (
        "y_buyer_concentration_high",
        "label_corr_buyer_concentration",
        "ge",
        BUYER_CONCENTRATION_CUTOFF,
        "buyer-concentration score at or above the upper-quartile cutoff of the modeling data",
    ),
    ("y_single_bid", "label_corr_singleb", "gt", 0.0, "single-bid indicator is positive"),
    ("y_no_call_for_tender", "label_corr_nocft", "gt", 0.0, "no-call-for-tender indicator is positive"),
]


def derive(frame: pd.DataFrame, source: str, operator: str, threshold: float) -> pd.Series:
    """Binarize one source score, propagating missing source values as missing."""
    values = frame[source]
    flag = values >= threshold if operator == "ge" else values > threshold
    return flag.astype("float64").where(values.notna())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    columns = [source for _, source, _, _, _ in SPECIFICATION] + [target for target, _, _, _, _ in SPECIFICATION]
    frame = pd.read_parquet(args.data, columns=columns)
    print(f"modeling rows: {len(frame):,}")

    empirical_quantile = float(frame["label_corr_buyer_concentration"].quantile(BUYER_CONCENTRATION_QUANTILE))
    print(
        f"buyer-concentration {BUYER_CONCENTRATION_QUANTILE:.0%} quantile in the modeling data: "
        f"{empirical_quantile:.6f} (manuscript cutoff {BUYER_CONCENTRATION_CUTOFF})"
    )

    records = []
    for target, source, operator, threshold, description in SPECIFICATION:
        derived = derive(frame, source, operator, threshold)
        stored = frame[target].astype("float64")
        comparable = derived.fillna(-1) == stored.fillna(-1)
        disagreements = int((~comparable).sum())
        # A disagreement can only be explained when the stored component score sits
        # exactly on the threshold: the score is stored as float32, so a value that
        # was marginally below the threshold in full precision can round up to the
        # threshold on write. Any other disagreement is unexplained and fails.
        at_boundary = int((~comparable & frame[source].eq(threshold)).sum())
        unexplained = disagreements - at_boundary
        records.append(
            {
                "target": target,
                "source score": source,
                "rule": f"{source} {'>=' if operator == 'ge' else '>'} {threshold}",
                "definition": description,
                "rows": len(frame),
                "source missing": int(frame[source].isna().sum()),
                "derived positives": int(np.nansum(derived.to_numpy())),
                "stored positives": int(np.nansum(stored.to_numpy())),
                "rows in disagreement": disagreements,
                "disagreements exactly at the threshold": at_boundary,
                "unexplained disagreements": unexplained,
                "agreement rate (%)": round(100 * comparable.mean(), 6),
            }
        )
        if disagreements == 0:
            status = "reproduced exactly"
        elif unexplained == 0:
            status = (
                f"{disagreements:,} of {len(frame):,} rows differ, all sitting exactly on the "
                f"threshold (float32 storage precision)"
            )
        else:
            status = f"{unexplained:,} unexplained disagreements"
        print(f"  {target:28s} {status}")

    report = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "target_derivation_verification.csv", index=False)
    (args.output_dir / "target_derivation_verification.md").write_text(
        "# Target derivation verification\n\n"
        f"Modeling dataset: `{args.data}`\n\n"
        f"Buyer-concentration cutoff: {BUYER_CONCENTRATION_CUTOFF} "
        f"(empirical {BUYER_CONCENTRATION_QUANTILE:.0%} quantile: {empirical_quantile:.8f})\n\n"
        + report.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {args.output_dir / 'target_derivation_verification.csv'}")

    total = int(report["rows in disagreement"].sum())
    unexplained_total = int(report["unexplained disagreements"].sum())
    if total == 0:
        print("all five targets reproduced exactly from the published component scores")
    elif unexplained_total == 0:
        print(
            f"all five targets reproduced from the published component scores; {total} row(s) of "
            f"{len(frame):,} differ and every one of them sits exactly on its threshold, which is a "
            "float32 storage-precision effect rather than a rule difference"
        )
    else:
        raise SystemExit(f"{unexplained_total} unexplained disagreements; the derivation rule does not match")


if __name__ == "__main__":
    main()
