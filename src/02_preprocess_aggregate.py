"""
Build leakage-audited non-federated modeling datasets with a held-out test split.

The script starts from the existing tender-lot dataset produced from the raw
bid-level data and creates two feature sets:

1. audit_priority: post-award audit prioritization features.
2. strict_ex_ante: pre-outcome tender design features.

A reproducible stratified three-way split is assigned at this stage so that
downstream model fitting, criterion-weight learning, configuration selection,
and final reporting use disjoint subsets of rows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INPUT_DEFAULT = Path("data/processed/tender_lot_dataset.csv")
MANIFEST_DEFAULT = Path("data/processed/feature_manifest.csv")
DATA_OUTPUT_DEFAULT = Path("data/modeling")
REPORT_OUTPUT_DEFAULT = Path("outputs/preprocess")

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
    "trace",
    "federated",
    "client",
    "email",
    "phone",
    "url",
    "source",
    "contact",
)

TARGET_PREFIXES = ("label_", "y_")

EXPLICIT_LABEL_SOURCES = {
    "cri",
    "corr_singleb",
    "corr_proc",
    "corr_nocft",
    "corr_buyer_concentration",
}

STRICT_EX_ANTE_ALLOWED = {
    "tender_proceduretype",
    "tender_nationalproceduretype",
    "tender_supplytype",
    "tender_isjointprocurement",
    "tender_isframeworkagreement",
    "tender_isdps",
    "tender_maincpv",
    "tender_iseufunded",
    "tender_selectionmethod",
    "buyer_mainactivities",
    "buyer_buyertype",
    "buyer_type",
    "cpv_division",
    "cpv_group",
    "tender_lotscount",
    "tender_awardcriteria_count",
    "lot_row_nr",
    "tender_description_length",
    "lot_description_length",
    "tender_personalrequirements_length",
    "tender_technicalrequirements_length",
    "tender_economicrequirements_length",
}

BINARY_TARGETS = (
    "y_cri_high",
    "y_proc_high",
    "y_buyer_concentration_high",
    "y_single_bid",
    "y_no_call_for_tender",
)

SPLIT_LABELS = ("train", "val", "test")


def clean_caption(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def is_forbidden(column: str) -> tuple[bool, str]:
    lower = column.lower()
    reasons: list[str] = []
    if column.startswith(TARGET_PREFIXES):
        reasons.append("target")
    if column in EXPLICIT_LABEL_SOURCES:
        reasons.append("raw label source")
    if any(pattern in lower for pattern in TEMPORAL_PATTERNS):
        reasons.append("temporal")
    if any(pattern in lower for pattern in GEO_PATTERNS):
        reasons.append("geographic")
    if any(pattern in lower for pattern in IDENTIFIER_PATTERNS):
        reasons.append("identifier or grouping")
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest["used_as_feature"] = manifest["used_as_feature"].astype(str).str.lower().eq("true")
    return manifest


def build_feature_sets(manifest: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    records = []
    audit_features = []
    strict_features = []

    for row in manifest.to_dict("records"):
        column = row["column"]
        used = bool(row["used_as_feature"])
        forbidden, reason = is_forbidden(column)
        in_audit = used and not forbidden
        in_strict = in_audit and column in STRICT_EX_ANTE_ALLOWED

        if in_audit:
            audit_features.append(column)
        if in_strict:
            strict_features.append(column)

        records.append(
            {
                "column": column,
                "manifest_role": row.get("role", ""),
                "manifest_type": row.get("type", ""),
                "manifest_used_as_feature": used,
                "audit_priority_feature": in_audit,
                "strict_ex_ante_feature": in_strict,
                "exclusion_reason": reason if forbidden else "",
            }
        )

    return audit_features, strict_features, pd.DataFrame(records)


def infer_column_types(manifest: pd.DataFrame, columns: Iterable[str]) -> tuple[set[str], set[str]]:
    numeric: set[str] = set()
    categorical: set[str] = set()
    lookup = dict(zip(manifest["column"], manifest["type"]))
    for column in columns:
        column_type = str(lookup.get(column, "")).lower()
        if column.startswith(TARGET_PREFIXES):
            numeric.add(column)
        elif "numeric" in column_type or "risk outcome" in column_type:
            numeric.add(column)
        elif column.startswith("y_"):
            numeric.add(column)
        else:
            categorical.add(column)
    return numeric, categorical


def clean_chunk(chunk: pd.DataFrame, numeric_columns: set[str], categorical_columns: set[str]) -> pd.DataFrame:
    chunk = chunk.copy()
    for column in numeric_columns.intersection(chunk.columns):
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float32")
    for column in categorical_columns.intersection(chunk.columns):
        values = chunk[column].astype("string").str.strip()
        chunk[column] = values.mask(values == "", pd.NA)
    return chunk


def assign_split(
    chunk: pd.DataFrame,
    primary_target: str,
    train_rate: float,
    val_rate: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Assign train, val, or test labels by stratified random draw on the primary target.

    Rows with the primary target missing are stratified together as a separate
    pseudo-class to preserve the conditional distribution of available labels
    across splits.
    """
    target = pd.to_numeric(chunk[primary_target], errors="coerce")
    stratum = target.where(target.isin([0, 1]), other=-1).astype(int)
    splits = pd.Series(index=chunk.index, dtype="object")
    boundaries = np.cumsum([train_rate, val_rate])
    for value in (-1, 0, 1):
        mask = stratum.eq(value)
        n = int(mask.sum())
        if n == 0:
            continue
        draws = rng.random(n)
        assignment = np.where(
            draws < boundaries[0],
            SPLIT_LABELS[0],
            np.where(draws < boundaries[1], SPLIT_LABELS[1], SPLIT_LABELS[2]),
        )
        splits.loc[mask] = assignment
    return splits.astype("string")


def remove_existing(path: Path) -> None:
    if path.exists():
        path.unlink()


def write_parquet_chunk(writer: pq.ParquetWriter | None, path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def save_bar_figure(
    values: dict[str, float],
    output_path: Path,
    caption: str,
    ylabel: str,
    color: str = "#315f72",
) -> None:
    labels = [clean_caption(label) for label in values.keys()]
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    bars = ax.bar(labels, values.values(), color=color, edgecolor="#1f2d33", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
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
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, clean_caption(caption), ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_grouped_bar_figure(df: pd.DataFrame, output_path: Path, caption: str) -> None:
    labels = [clean_caption(label) for label in df["target"]]
    negatives = df["negative_count"].astype(float).to_numpy()
    positives = df["positive_count"].astype(float).to_numpy()
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 5.3), dpi=180)
    ax.bar(x, negatives, label="Negative", color="#6f8795", edgecolor="#25323a", linewidth=0.5)
    ax.bar(x, positives, bottom=negatives, label="Positive", color="#d1a15f", edgecolor="#5f4a2b", linewidth=0.5)
    ax.set_ylabel("Records")
    ax.set_xticks(list(x), labels, rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.13, 1, 1))
    fig.text(0.5, 0.025, clean_caption(caption), ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_feature_type_figure(feature_manifest: pd.DataFrame, output_path: Path) -> None:
    summary = []
    for feature_set in ["audit_priority_feature", "strict_ex_ante_feature"]:
        subset = feature_manifest[feature_manifest[feature_set]]
        counts = subset["manifest_type"].replace("", "unknown").value_counts()
        summary.append(
            {
                "feature_set": feature_set,
                "categorical": int(counts.get("categorical", 0)),
                "numeric": int(counts.get("numeric", 0)),
            }
        )
    df = pd.DataFrame(summary)
    labels = [clean_caption(label.replace("_feature", "")) for label in df["feature_set"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)
    ax.bar(labels, df["categorical"], label="Categorical", color="#7c8f63", edgecolor="#28301f", linewidth=0.5)
    ax.bar(labels, df["numeric"], bottom=df["categorical"], label="Numeric", color="#5b7f95", edgecolor="#25323a", linewidth=0.5)
    ax.set_ylabel("Feature count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 3. Feature type composition", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_split_figure(split_counts: dict[str, int], output_path: Path) -> None:
    labels = [clean_caption(value) for value in split_counts.keys()]
    values = list(split_counts.values())
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=180)
    bars = ax.bar(labels, values, color=["#5b7f95", "#7c8f63", "#d1a15f"], edgecolor="#1f2d33", linewidth=0.6)
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout(rect=(0.03, 0.11, 1, 1))
    fig.text(0.5, 0.025, "Figure 4. Three way split sizes", ha="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_dataset(
    input_path: Path,
    manifest_path: Path,
    data_output_dir: Path,
    report_output_dir: Path,
    chunksize: int,
    train_rate: float,
    val_rate: float,
    seed: int,
) -> None:
    if not (0 < train_rate < 1 and 0 < val_rate < 1 and train_rate + val_rate < 1):
        raise ValueError("train_rate and val_rate must define a valid three-way split")

    data_output_dir.mkdir(parents=True, exist_ok=True)
    (report_output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_output_dir / "figures").mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    audit_features, strict_features, feature_manifest = build_feature_sets(manifest)
    target_columns = [column for column in manifest["column"] if column.startswith(TARGET_PREFIXES)]

    all_columns = list(dict.fromkeys(audit_features + strict_features + target_columns))
    numeric_columns, categorical_columns = infer_column_types(manifest, all_columns)
    primary_target = "y_cri_high"
    if primary_target not in target_columns:
        raise ValueError(f"Primary target {primary_target} missing from manifest")

    audit_path = data_output_dir / "tender_lot_audit_priority.parquet"
    strict_path = data_output_dir / "tender_lot_strict_ex_ante.parquet"
    remove_existing(audit_path)
    remove_existing(strict_path)

    audit_writer: pq.ParquetWriter | None = None
    strict_writer: pq.ParquetWriter | None = None

    row_count = 0
    missing_counts = defaultdict(int)
    non_missing_counts = defaultdict(int)
    categorical_cardinality: dict[str, set[str]] = defaultdict(set)
    target_positive = Counter()
    target_non_missing = Counter()
    target_negative = Counter()
    split_counts = Counter()
    split_target_positive: dict[str, int] = {label: 0 for label in SPLIT_LABELS}
    split_target_non_missing: dict[str, int] = {label: 0 for label in SPLIT_LABELS}

    rng = np.random.default_rng(seed)
    next_row_id = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=all_columns,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        chunk = clean_chunk(chunk, numeric_columns, categorical_columns)
        row_count += len(chunk)

        for column in all_columns:
            missing = int(chunk[column].isna().sum())
            missing_counts[column] += missing
            non_missing_counts[column] += len(chunk) - missing

        for column in categorical_columns:
            if column in chunk:
                values = chunk[column].dropna().unique().tolist()
                categorical_cardinality[column].update(str(value) for value in values)

        for target in BINARY_TARGETS:
            if target in chunk:
                values = pd.to_numeric(chunk[target], errors="coerce")
                target_non_missing[target] += int(values.notna().sum())
                target_positive[target] += int((values == 1).sum())
                target_negative[target] += int((values == 0).sum())

        chunk = chunk.copy()
        chunk["row_id"] = np.arange(next_row_id, next_row_id + len(chunk), dtype=np.int64)
        next_row_id += len(chunk)
        chunk["split"] = assign_split(chunk, primary_target, train_rate, val_rate, rng)
        split_counts.update(chunk["split"].dropna().tolist())

        primary_values = pd.to_numeric(chunk[primary_target], errors="coerce")
        for label in SPLIT_LABELS:
            mask = chunk["split"].eq(label)
            split_target_non_missing[label] += int(primary_values[mask].notna().sum())
            split_target_positive[label] += int((primary_values[mask] == 1).sum())

        audit_writer = write_parquet_chunk(
            audit_writer,
            audit_path,
            chunk[["row_id", "split"] + audit_features + target_columns],
        )
        strict_writer = write_parquet_chunk(
            strict_writer,
            strict_path,
            chunk[["row_id", "split"] + strict_features + target_columns],
        )

    if audit_writer is not None:
        audit_writer.close()
    if strict_writer is not None:
        strict_writer.close()

    dataset_inventory = pd.DataFrame(
        [
            {
                "dataset": "audit priority",
                "path": str(audit_path),
                "rows": row_count,
                "features": len(audit_features),
                "targets": len(target_columns),
                "size_mb": round(audit_path.stat().st_size / (1024 * 1024), 2),
            },
            {
                "dataset": "strict ex ante",
                "path": str(strict_path),
                "rows": row_count,
                "features": len(strict_features),
                "targets": len(target_columns),
                "size_mb": round(strict_path.stat().st_size / (1024 * 1024), 2),
            },
        ]
    )
    dataset_inventory.to_csv(report_output_dir / "tables" / "table_01_modeling_dataset_inventory.csv", index=False)
    write_markdown_table(dataset_inventory, report_output_dir / "tables" / "table_01_modeling_dataset_inventory.md")

    feature_manifest.to_csv(report_output_dir / "tables" / "table_02_clean_feature_manifest.csv", index=False)
    write_markdown_table(feature_manifest, report_output_dir / "tables" / "table_02_clean_feature_manifest.md")

    missingness_records = []
    for column in all_columns:
        missingness_records.append(
            {
                "column": column,
                "non_missing_count": non_missing_counts[column],
                "missing_count": missing_counts[column],
                "missing_rate": missing_counts[column] / row_count if row_count else math.nan,
                "categorical_cardinality": len(categorical_cardinality[column]) if column in categorical_columns else "",
            }
        )
    missingness = pd.DataFrame(missingness_records).sort_values("missing_rate", ascending=False)
    missingness.to_csv(report_output_dir / "tables" / "table_03_modeling_column_missingness.csv", index=False)
    write_markdown_table(missingness, report_output_dir / "tables" / "table_03_modeling_column_missingness.md")

    target_summary = pd.DataFrame(
        [
            {
                "target": target,
                "non_missing_count": target_non_missing[target],
                "positive_count": target_positive[target],
                "negative_count": target_negative[target],
                "positive_rate": target_positive[target] / target_non_missing[target]
                if target_non_missing[target]
                else math.nan,
            }
            for target in BINARY_TARGETS
        ]
    )
    target_summary.to_csv(report_output_dir / "tables" / "table_04_binary_target_summary.csv", index=False)
    write_markdown_table(target_summary, report_output_dir / "tables" / "table_04_binary_target_summary.md")

    forbidden_records = []
    for feature_set_name, features in {
        "audit_priority": audit_features,
        "strict_ex_ante": strict_features,
    }.items():
        for column in features:
            forbidden, reason = is_forbidden(column)
            forbidden_records.append(
                {
                    "feature_set": feature_set_name,
                    "column": column,
                    "forbidden_detected": forbidden,
                    "reason": reason,
                }
            )
    forbidden = pd.DataFrame(forbidden_records)
    forbidden.to_csv(report_output_dir / "tables" / "table_05_forbidden_feature_verification.csv", index=False)
    write_markdown_table(forbidden, report_output_dir / "tables" / "table_05_forbidden_feature_verification.md")

    split_summary = pd.DataFrame(
        [
            {
                "split": label,
                "rows": int(split_counts[label]),
                "share": float(split_counts[label] / row_count) if row_count else math.nan,
                "primary_target_non_missing": int(split_target_non_missing[label]),
                "primary_target_positive": int(split_target_positive[label]),
                "primary_target_positive_rate": (
                    split_target_positive[label] / split_target_non_missing[label]
                    if split_target_non_missing[label]
                    else math.nan
                ),
            }
            for label in SPLIT_LABELS
        ]
    )
    split_summary.to_csv(report_output_dir / "tables" / "table_06_split_summary.csv", index=False)
    write_markdown_table(split_summary, report_output_dir / "tables" / "table_06_split_summary.md")

    save_bar_figure(
        {
            "audit priority": len(audit_features),
            "strict ex ante": len(strict_features),
        },
        report_output_dir / "figures" / "figure_01_feature_set_sizes.png",
        "Figure 1. Modeling feature set sizes",
        "Feature count",
    )
    save_grouped_bar_figure(
        target_summary,
        report_output_dir / "figures" / "figure_02_binary_target_balance.png",
        "Figure 2. Binary target balance",
    )
    save_feature_type_figure(
        feature_manifest,
        report_output_dir / "figures" / "figure_03_feature_type_composition.png",
    )
    save_split_figure(
        {label: int(split_counts[label]) for label in SPLIT_LABELS},
        report_output_dir / "figures" / "figure_04_split_sizes.png",
    )

    summary = {
        "rows": row_count,
        "audit_priority_features": len(audit_features),
        "strict_ex_ante_features": len(strict_features),
        "targets": len(target_columns),
        "audit_priority_path": str(audit_path),
        "strict_ex_ante_path": str(strict_path),
        "forbidden_features_detected": int(forbidden["forbidden_detected"].sum()),
        "split_rates": {"train": train_rate, "val": val_rate, "test": round(1 - train_rate - val_rate, 6)},
        "split_seed": seed,
        "split_counts": {label: int(split_counts[label]) for label in SPLIT_LABELS},
    }
    (report_output_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Preprocessing Summary",
        "",
        f"- Rows processed: {row_count:,}",
        f"- Audit priority features: {len(audit_features):,}",
        f"- Strict ex ante features: {len(strict_features):,}",
        f"- Targets retained: {len(target_columns):,}",
        f"- Forbidden features detected in final feature sets: {int(forbidden['forbidden_detected'].sum()):,}",
        f"- Split rates train/val/test: {train_rate:.2f}/{val_rate:.2f}/{round(1 - train_rate - val_rate, 4):.2f}",
        f"- Split counts train/val/test: {split_counts[SPLIT_LABELS[0]]:,}/{split_counts[SPLIT_LABELS[1]]:,}/{split_counts[SPLIT_LABELS[2]]:,}",
        "",
        "Generated tables are in `outputs/preprocess/tables`.",
        "Generated figures are in `outputs/preprocess/figures`.",
    ]
    (report_output_dir / "preprocess_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-federated modeling datasets with three-way split.")
    parser.add_argument("--input-path", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--data-output-dir", type=Path, default=DATA_OUTPUT_DEFAULT)
    parser.add_argument("--report-output-dir", type=Path, default=REPORT_OUTPUT_DEFAULT)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--train-rate", type=float, default=0.60)
    parser.add_argument("--val-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        input_path=args.input_path,
        manifest_path=args.manifest_path,
        data_output_dir=args.data_output_dir,
        report_output_dir=args.report_output_dir,
        chunksize=args.chunksize,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
