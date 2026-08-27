"""
Audit the construction procurement dataset before modeling.

The script is intentionally read-only for source data. It creates publication-ready
audit tables and figures under outputs/audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


RAW_DEFAULT = Path("All_Construction_Data.csv")
PROCESSED_DEFAULT = Path("data/processed/tender_lot_dataset.csv")
MANIFEST_DEFAULT = Path("data/processed/feature_manifest.csv")
OUTPUT_DEFAULT = Path("outputs/audit")

TEMPORAL_PATTERNS = (
    "year",
    "date",
    "deadline",
    "period",
    "duration",
)

GEO_PATTERNS = (
    "country",
    "city",
    "nuts",
    "postcode",
    "street",
    "address",
)

IDENTIFIER_PATTERNS = (
    "id",
    "name",
    "email",
    "phone",
    "url",
    "source",
    "contact",
)

TARGET_COLUMNS = {
    "corr_singleb",
    "corr_proc",
    "corr_nocft",
    "corr_buyer_concentration",
    "cri",
}

PROCESSED_TARGET_PREFIXES = (
    "label_",
    "y_",
)

SELECTED_AUDIT_COLUMNS = [
    "tender_proceduretype",
    "tender_nationalproceduretype",
    "tender_supplytype",
    "tender_isjointprocurement",
    "tender_isframeworkagreement",
    "tender_isdps",
    "tender_awardcriteria_count",
    "tender_corrections_count",
    "lot_bidscount",
    "lot_validbidscount",
    "lot_smebidscount",
    "bid_iswinning",
    "bid_isconsortium",
    "bid_issubcontracted",
    "bid_priceUsd",
    "tender_finalpriceUsd",
    "tender_estimatedpriceUsd",
    "lot_estimatedpriceUsd",
    "corr_singleb",
    "corr_proc",
    "corr_nocft",
    "corr_buyer_concentration",
    "cri",
    "buyer_country",
    "bidder_country",
    "tender_year",
]


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def classify_raw_column(column: str) -> tuple[str, str]:
    lower = column.lower()
    reasons: list[str] = []

    if column in TARGET_COLUMNS:
        reasons.append("target or validation label")
    if lower.startswith("filter_"):
        reasons.append("filter or derived screening field")
    if "currency" in lower:
        reasons.append("currency proxy")
    if any(pattern in lower for pattern in TEMPORAL_PATTERNS):
        reasons.append("temporal variable")
    if any(pattern in lower for pattern in GEO_PATTERNS):
        reasons.append("geographic variable")
    if any(pattern in lower for pattern in IDENTIFIER_PATTERNS):
        reasons.append("identifier or contact field")

    if reasons:
        return "exclude from model features", "; ".join(sorted(set(reasons)))
    return "candidate feature after preprocessing", "no automatic exclusion"


def classify_processed_column(column: str) -> tuple[str, str]:
    lower = column.lower()
    reasons: list[str] = []

    if column.startswith(PROCESSED_TARGET_PREFIXES):
        reasons.append("label or evaluation target")
    if "federated" in lower or "client" in lower:
        reasons.append("federated grouping field")
    if "trace" in lower or lower.endswith("_id") or lower == "id":
        reasons.append("trace or identifier field")
    if any(pattern in lower for pattern in TEMPORAL_PATTERNS):
        reasons.append("temporal variable")
    if any(pattern in lower for pattern in GEO_PATTERNS):
        reasons.append("geographic variable")

    if reasons:
        return "exclude from model features", "; ".join(sorted(set(reasons)))
    return "candidate model feature", "no automatic exclusion"


def truthy_count(series: pd.Series) -> int:
    values = series.dropna().astype(str).str.strip().str.lower()
    return int(values.isin({"1", "1.0", "true", "t", "yes", "y"}).sum())


def truthy_counter_count(counter: Counter) -> int:
    truthy = {"1", "1.0", "true", "t", "yes", "y"}
    return sum(count for value, count in counter.items() if str(value).strip().lower() in truthy)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "non_missing": 0,
            "mean": None,
            "p25": None,
            "median": None,
            "p75": None,
            "high_risk_count_at_p75": None,
        }

    series = pd.Series(values, dtype="float64")
    p75 = float(series.quantile(0.75))
    return {
        "non_missing": int(series.notna().sum()),
        "mean": float(series.mean()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "p75": p75,
        "high_risk_count_at_p75": int((series >= p75).sum()),
    }


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def save_bar_figure(
    values: dict[str, float],
    output_path: Path,
    caption: str,
    ylabel: str,
    color: str = "#315f72",
) -> None:
    labels = [clean_caption(k) for k in values.keys()]
    heights = list(values.values())
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    bars = ax.bar(labels, heights, color=color, edgecolor="#1f2d33", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout(rect=(0.03, 0.10, 1, 1))
    fig.text(0.5, 0.025, clean_caption(caption), ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_pie_figure(values: dict[str, int], output_path: Path, caption: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=180)
    labels = [clean_caption(k) for k in values.keys()]
    ax.pie(
        values.values(),
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#315f72", "#8fbc8f", "#d1a15f", "#a65d5d", "#6b6f8f", "#9a7b68"],
        textprops={"fontsize": 8},
    )
    ax.axis("equal")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.5, 0.025, clean_caption(caption), ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_selected_raw_columns(
    path: Path,
    columns: Iterable[str],
    chunksize: int,
) -> tuple[int, dict[str, int], dict[str, Counter], dict[str, dict[str, float | int | None]]]:
    header = set(read_header(path))
    usecols = [column for column in columns if column in header]
    missing_counts = defaultdict(int)
    category_counts: dict[str, Counter] = defaultdict(Counter)
    numeric_values: dict[str, list[float]] = defaultdict(list)
    row_count = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        row_count += len(chunk)
        for column in usecols:
            series = chunk[column]
            missing_counts[column] += int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())

            if column in TARGET_COLUMNS or column in {
                "tender_proceduretype",
                "tender_nationalproceduretype",
                "tender_supplytype",
                "buyer_country",
                "bidder_country",
                "bid_iswinning",
            }:
                values = series.dropna().astype(str).str.strip()
                values = values[values != ""]
                category_counts[column].update(values.tolist())

            if column in {
                "bid_priceUsd",
                "tender_finalpriceUsd",
                "tender_estimatedpriceUsd",
                "lot_estimatedpriceUsd",
                "corr_proc",
                "corr_buyer_concentration",
                "cri",
            }:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                numeric_values[column].extend(numeric.tolist())

    summaries = {column: numeric_summary(values) for column, values in numeric_values.items()}
    return row_count, dict(missing_counts), category_counts, summaries


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def build_audit(
    raw_path: Path,
    processed_path: Path,
    manifest_path: Path,
    output_dir: Path,
    chunksize: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    raw_header = read_header(raw_path)
    raw_rows, raw_missing, raw_counts, raw_numeric = summarize_selected_raw_columns(
        raw_path,
        SELECTED_AUDIT_COLUMNS,
        chunksize,
    )

    inventory_records = [
        {
            "dataset": "raw bid level data",
            "path": str(raw_path),
            "rows": raw_rows,
            "columns": len(raw_header),
            "size_mb": round(file_size_mb(raw_path), 2),
        }
    ]

    processed_header: list[str] = []
    processed_rows: int | None = None
    if processed_path.exists():
        processed_header = read_header(processed_path)
        processed_rows = max(count_lines(processed_path) - 1, 0)
        inventory_records.append(
            {
                "dataset": "processed tender lot data",
                "path": str(processed_path),
                "rows": processed_rows,
                "columns": len(processed_header),
                "size_mb": round(file_size_mb(processed_path), 2),
            }
        )

    inventory = pd.DataFrame(inventory_records)
    inventory.to_csv(output_dir / "tables" / "table_01_dataset_inventory.csv", index=False)
    write_markdown_table(inventory, output_dir / "tables" / "table_01_dataset_inventory.md")

    governance_rows = []
    for column in raw_header:
        decision, reason = classify_raw_column(column)
        governance_rows.append(
            {
                "dataset": "raw",
                "column": column,
                "decision": decision,
                "reason": reason,
            }
        )

    for column in processed_header:
        decision, reason = classify_processed_column(column)
        governance_rows.append(
            {
                "dataset": "processed",
                "column": column,
                "decision": decision,
                "reason": reason,
            }
        )

    governance = pd.DataFrame(governance_rows)
    governance.to_csv(output_dir / "tables" / "table_02_variable_governance.csv", index=False)
    write_markdown_table(governance, output_dir / "tables" / "table_02_variable_governance.md")

    label_rows = []
    for label in TARGET_COLUMNS:
        if label in {"corr_singleb", "corr_nocft"} and label in raw_counts:
            positive = truthy_counter_count(raw_counts[label])
            label_rows.append(
                {
                    "label": label,
                    "label_type": "binary indicator",
                    "non_missing": sum(raw_counts[label].values()),
                    "positive_or_high_risk_count": positive,
                    "threshold": "truthy values",
                }
            )
        elif label in raw_numeric:
            summary = raw_numeric[label]
            label_rows.append(
                {
                    "label": label,
                    "label_type": "numeric risk score",
                    "non_missing": summary["non_missing"],
                    "positive_or_high_risk_count": summary["high_risk_count_at_p75"],
                    "threshold": f"p75={summary['p75']:.6g}"
                    if summary["p75"] is not None
                    else "numeric label",
                }
            )

    labels = pd.DataFrame(label_rows).drop_duplicates(subset=["label"])
    labels.to_csv(output_dir / "tables" / "table_03_label_feasibility.csv", index=False)
    write_markdown_table(labels, output_dir / "tables" / "table_03_label_feasibility.md")

    missing_rows = []
    for column, missing in raw_missing.items():
        missing_rows.append(
            {
                "column": column,
                "missing_count": missing,
                "missing_rate": missing / raw_rows if raw_rows else math.nan,
            }
        )
    missingness = pd.DataFrame(missing_rows).sort_values("missing_rate", ascending=False)
    missingness.to_csv(output_dir / "tables" / "table_04_selected_column_missingness.csv", index=False)
    write_markdown_table(missingness, output_dir / "tables" / "table_04_selected_column_missingness.md")

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest.to_csv(output_dir / "tables" / "table_05_existing_feature_manifest.csv", index=False)
        write_markdown_table(manifest, output_dir / "tables" / "table_05_existing_feature_manifest.md")

    raw_decision_counts = governance[governance["dataset"] == "raw"]["decision"].value_counts().to_dict()
    save_pie_figure(
        raw_decision_counts,
        output_dir / "figures" / "figure_01_variable_governance.png",
        "Figure 1. Variable governance audit",
    )

    if not labels.empty:
        label_values = {
            row["label"]: row["non_missing"]
            for _, row in labels.drop_duplicates("label").iterrows()
            if pd.notna(row["non_missing"])
        }
        save_bar_figure(
            label_values,
            output_dir / "figures" / "figure_02_label_availability.png",
            "Figure 2. Label availability for procurement risk outcomes",
            "Non missing records",
            color="#5b7f95",
        )

    procedure_counts = raw_counts.get("tender_proceduretype", Counter()).most_common(8)
    if procedure_counts:
        save_bar_figure(
            dict(procedure_counts),
            output_dir / "figures" / "figure_03_procedure_type_distribution.png",
            "Figure 3. Tender procedure type distribution",
            "Records",
            color="#7c8f63",
        )

    report = {
        "raw_rows": raw_rows,
        "raw_columns": len(raw_header),
        "processed_rows": processed_rows,
        "processed_columns": len(processed_header) if processed_header else None,
        "raw_size_mb": round(file_size_mb(raw_path), 2),
        "processed_size_mb": round(file_size_mb(processed_path), 2) if processed_path.exists() else None,
        "outputs": {
            "tables": str(output_dir / "tables"),
            "figures": str(output_dir / "figures"),
        },
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_lines = [
        "# Data Audit Summary",
        "",
        f"- Raw rows: {raw_rows:,}",
        f"- Raw columns: {len(raw_header):,}",
        f"- Raw size: {file_size_mb(raw_path):,.2f} MB",
    ]
    if processed_rows is not None:
        report_lines.extend(
            [
                f"- Processed tender lot rows: {processed_rows:,}",
                f"- Processed columns: {len(processed_header):,}",
                f"- Processed size: {file_size_mb(processed_path):,.2f} MB",
            ]
        )
    report_lines.extend(
        [
            "",
            "Generated tables are in `outputs/audit/tables`.",
            "Generated figures are in `outputs/audit/figures`.",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit construction procurement data.")
    parser.add_argument("--raw-path", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_DEFAULT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_audit(
        raw_path=args.raw_path,
        processed_path=args.processed_path,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
