"""A16 - Provenance for the dataset-composition statements in the manuscript.

The Introduction reports the tender-lot count distribution and the Methodology
reports the country, award-year, CPV, and procedure composition of the source
records. Those numbers were previously stated in the text without a published
artifact behind them. This script recomputes each of them and writes one table
so that every printed value in the manuscript has a generating artifact.

Sources
  - tender-lot level (3,830,910 rows): data/modeling/tender_lot_audit_priority.parquet
  - source records  (4,337,081 rows): All_Construction_Data.csv
"""

from __future__ import annotations

import sys
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

import r2_common as C

sys.stdout.reconfigure(encoding="utf-8")

MODELING = C.OPTI_ROOT / "data" / "modeling" / "tender_lot_audit_priority.parquet"
RAW = C.OPTI_ROOT / "All_Construction_Data.csv"

rows: list[dict] = []


def add(statement: str, value: str, source: str) -> None:
    rows.append({"manuscript statement": statement, "recomputed value": value, "source": source})


# ---------------------------------------------------------------- tender-lot level
df = pq.read_table(
    MODELING, columns=["tender_lotscount", "cpv_group", "tender_proceduretype"]
).to_pandas()
n = len(df)
lots = df["tender_lotscount"]
src_m = "tender_lot_audit_priority.parquet"

add("modeling records", f"{n:,}", src_m)
add("median tender lot count", f"{lots.median():.0f}", src_m)
add("records in tenders with more than one lot", f"{(lots > 1).mean() * 100:.1f}%", src_m)
add("records in tenders with at least 10 lots", f"{(lots >= 10).mean() * 100:.1f}%", src_m)
add("99th percentile of tender lot count", f"{lots.quantile(0.99):.0f}", src_m)

for grp, share in (df["cpv_group"].value_counts(dropna=True) / n * 100).head(5).items():
    add(f"CPV group {grp} share", f"{share:.1f}%", src_m)

for proc in ("OPEN", "RESTRICTED", "NEGOTIATED_WITHOUT_PUBLICATION"):
    add(
        f"procedure type {proc} share of all records",
        f"{(df['tender_proceduretype'] == proc).sum() / n * 100:.1f}%",
        src_m,
    )

# ---------------------------------------------------------------- source records
countries: Counter = Counter()
years: Counter = Counter()
raw_n = 0
for chunk in pd.read_csv(
    RAW,
    usecols=["buyer_country", "tender_year"],
    dtype={"buyer_country": "string"},
    chunksize=500_000,
    low_memory=False,
):
    raw_n += len(chunk)
    countries.update(chunk["buyer_country"].dropna().tolist())
    years.update(pd.to_numeric(chunk["tender_year"], errors="coerce").dropna().astype(int).tolist())

src_r = "All_Construction_Data.csv"
total_c = sum(countries.values())
total_y = sum(years.values())

add("source records", f"{raw_n:,}", src_r)
add("distinct buyer countries", f"{len(countries)}", src_r)
for code, count in countries.most_common(6):
    add(f"buyer country {code} share", f"{count / total_c * 100:.1f}%", src_r)
add("award year range", f"{min(years)} to {max(years)}", src_r)
add(
    "share of award years in 2008 to 2021",
    f"{sum(v for k, v in years.items() if 2008 <= k <= 2021) / total_y * 100:.1f}%",
    src_r,
)

C.write(pd.DataFrame(rows), "t_dataset_composition")
print(f"\n{len(rows)} composition statistics recomputed")
for r in rows:
    print(f"  {r['manuscript statement']:52} {r['recomputed value']}")
