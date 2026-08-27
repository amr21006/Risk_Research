"""
Engineer additional ex-ante features for the strict_ex_ante feature set.

The script reads the strict_ex_ante parquet produced by 02_preprocess_aggregate
and adds three families of leakage-safe features:

1. Out-of-fold target-encoded versions of high-cardinality categoricals using
   sklearn TargetEncoder with cv=5. The encoder is fit on training rows only and
   applied to validation and test rows via the full-train fit.
2. Pairwise categorical interactions (string concatenation) that are themselves
   target encoded.
3. Numeric derived features from the existing length columns.

The output is tender_lot_strict_ex_ante_enhanced.parquet with the original split,
row_id, all original features, all targets, and the new engineered features.
The feature manifest is updated with a strict_ex_ante_enhanced flag so that the
downstream risk prediction stage can consume the enhanced view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder


INPUT_DEFAULT = Path("data/modeling/tender_lot_strict_ex_ante.parquet")
OUTPUT_DEFAULT = Path("data/modeling/tender_lot_strict_ex_ante_enhanced.parquet")
MANIFEST_DEFAULT = Path("outputs/preprocess/tables/table_02_clean_feature_manifest.csv")
REPORT_DEFAULT = Path("outputs/feature_engineering")

TE_TARGET = "y_cri_high"
TE_SUFFIX = "_te_cri"

TE_CATEGORICALS = [
    "tender_proceduretype",
    "tender_nationalproceduretype",
    "tender_maincpv",
    "tender_supplytype",
    "tender_selectionmethod",
    "buyer_mainactivities",
    "buyer_buyertype",
    "buyer_type",
    "cpv_division",
    "cpv_group",
]

INTERACTIONS = [
    ("tender_proceduretype", "tender_supplytype"),
    ("buyer_buyertype", "cpv_division"),
    ("tender_maincpv", "tender_proceduretype"),
]

NUMERIC_DERIVED = [
    "log_tender_lotscount",
    "requirements_total_length",
    "description_length_ratio",
    "criteria_per_lot",
]


def fit_target_encoder(
    values_train: pd.Series,
    y_train: pd.Series,
    seed: int,
) -> tuple[TargetEncoder, np.ndarray]:
    encoder = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed, smooth="auto")
    series = values_train.fillna("missing").astype(str).to_numpy().reshape(-1, 1)
    encoded_train = encoder.fit_transform(series, y_train.to_numpy()).ravel().astype("float32")
    return encoder, encoded_train


def transform_target_encoder(encoder: TargetEncoder, values: pd.Series) -> np.ndarray:
    series = values.fillna("missing").astype(str).to_numpy().reshape(-1, 1)
    return encoder.transform(series).ravel().astype("float32")


def build_enhanced_features(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    train_mask = frame["split"].eq("train") & frame[TE_TARGET].notna()
    val_mask = frame["split"].eq("val")
    test_mask = frame["split"].eq("test")
    if train_mask.sum() == 0:
        raise ValueError("Train split lacks labeled rows for target encoding")

    y_train = frame.loc[train_mask, TE_TARGET].astype(int)
    new_columns: dict[str, pd.Series] = {}
    new_descriptions: dict[str, str] = {}

    for column in TE_CATEGORICALS:
        if column not in frame.columns:
            continue
        new_name = f"{column}{TE_SUFFIX}"
        encoder, encoded_train = fit_target_encoder(frame.loc[train_mask, column], y_train, seed)
        encoded_full = np.full(len(frame), np.nan, dtype="float32")
        encoded_full[train_mask.to_numpy()] = encoded_train
        encoded_full[val_mask.to_numpy()] = transform_target_encoder(encoder, frame.loc[val_mask, column])
        encoded_full[test_mask.to_numpy()] = transform_target_encoder(encoder, frame.loc[test_mask, column])
        new_columns[new_name] = pd.Series(encoded_full, index=frame.index)
        new_descriptions[new_name] = f"out of fold target encoded {column} against {TE_TARGET}"

    interaction_keys: dict[str, pd.Series] = {}
    for left, right in INTERACTIONS:
        if left not in frame.columns or right not in frame.columns:
            continue
        interaction_name = f"{left}_x_{right}"
        joined = (
            frame[left].fillna("missing").astype(str)
            + "||"
            + frame[right].fillna("missing").astype(str)
        )
        interaction_keys[interaction_name] = joined
        encoded_name = f"{interaction_name}{TE_SUFFIX}"
        encoder, encoded_train = fit_target_encoder(joined.loc[train_mask], y_train, seed)
        encoded_full = np.full(len(frame), np.nan, dtype="float32")
        encoded_full[train_mask.to_numpy()] = encoded_train
        encoded_full[val_mask.to_numpy()] = transform_target_encoder(encoder, joined.loc[val_mask])
        encoded_full[test_mask.to_numpy()] = transform_target_encoder(encoder, joined.loc[test_mask])
        new_columns[encoded_name] = pd.Series(encoded_full, index=frame.index)
        new_descriptions[encoded_name] = f"out of fold target encoded interaction of {left} and {right}"

    if "tender_lotscount" in frame.columns:
        lots = pd.to_numeric(frame["tender_lotscount"], errors="coerce")
        new_columns["log_tender_lotscount"] = np.log1p(lots.clip(lower=0)).astype("float32")
        new_descriptions["log_tender_lotscount"] = "log of tender lots count"
    else:
        new_columns["log_tender_lotscount"] = pd.Series(np.nan, index=frame.index, dtype="float32")
        new_descriptions["log_tender_lotscount"] = "log of tender lots count"

    requirements_columns = [
        "tender_personalrequirements_length",
        "tender_technicalrequirements_length",
        "tender_economicrequirements_length",
    ]
    if all(column in frame.columns for column in requirements_columns):
        totals = sum(pd.to_numeric(frame[column], errors="coerce").fillna(0) for column in requirements_columns)
        new_columns["requirements_total_length"] = totals.astype("float32")
        new_descriptions["requirements_total_length"] = "sum of personal, technical, and economic requirement lengths"
    else:
        new_columns["requirements_total_length"] = pd.Series(np.nan, index=frame.index, dtype="float32")
        new_descriptions["requirements_total_length"] = "sum of personal, technical, and economic requirement lengths"

    if "tender_description_length" in frame.columns and "lot_description_length" in frame.columns:
        tender_len = pd.to_numeric(frame["tender_description_length"], errors="coerce").fillna(0)
        lot_len = pd.to_numeric(frame["lot_description_length"], errors="coerce").fillna(0)
        ratio = tender_len / (lot_len + 1.0)
        new_columns["description_length_ratio"] = ratio.astype("float32")
        new_descriptions["description_length_ratio"] = "ratio of tender description length to lot description length"
    else:
        new_columns["description_length_ratio"] = pd.Series(np.nan, index=frame.index, dtype="float32")
        new_descriptions["description_length_ratio"] = "ratio of tender description length to lot description length"

    if "tender_awardcriteria_count" in frame.columns and "tender_lotscount" in frame.columns:
        criteria = pd.to_numeric(frame["tender_awardcriteria_count"], errors="coerce").fillna(0)
        lots = pd.to_numeric(frame["tender_lotscount"], errors="coerce").fillna(1).clip(lower=1)
        new_columns["criteria_per_lot"] = (criteria / lots).astype("float32")
        new_descriptions["criteria_per_lot"] = "award criteria per lot"
    else:
        new_columns["criteria_per_lot"] = pd.Series(np.nan, index=frame.index, dtype="float32")
        new_descriptions["criteria_per_lot"] = "award criteria per lot"

    enhanced = frame.copy()
    for column, series in new_columns.items():
        enhanced[column] = series

    return enhanced, new_descriptions


def update_manifest(manifest_path: Path, new_columns: list[str], new_descriptions: dict[str, str]) -> None:
    manifest = pd.read_csv(manifest_path)
    manifest["audit_priority_feature"] = manifest["audit_priority_feature"].astype(str).str.lower().eq("true")
    manifest["strict_ex_ante_feature"] = manifest["strict_ex_ante_feature"].astype(str).str.lower().eq("true")
    if "strict_ex_ante_enhanced_feature" not in manifest.columns:
        manifest["strict_ex_ante_enhanced_feature"] = manifest["strict_ex_ante_feature"]
    else:
        manifest["strict_ex_ante_enhanced_feature"] = (
            manifest["strict_ex_ante_enhanced_feature"].astype(str).str.lower().eq("true")
            | manifest["strict_ex_ante_feature"]
        )

    existing = set(manifest["column"])
    additions = []
    for column in new_columns:
        if column in existing:
            manifest.loc[manifest["column"].eq(column), "strict_ex_ante_enhanced_feature"] = True
            continue
        additions.append(
            {
                "column": column,
                "manifest_role": "engineered feature",
                "manifest_type": "numeric",
                "manifest_used_as_feature": True,
                "audit_priority_feature": False,
                "strict_ex_ante_feature": False,
                "strict_ex_ante_enhanced_feature": True,
                "exclusion_reason": "",
            }
        )
    if additions:
        manifest = pd.concat([manifest, pd.DataFrame(additions)], ignore_index=True)

    manifest.to_csv(manifest_path, index=False)
    manifest.to_markdown(manifest_path.with_suffix(".md"), index=False)


def run_feature_engineering(input_path: Path, output_path: Path, manifest_path: Path, report_dir: Path, seed: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(input_path)
    if "split" not in frame.columns or "row_id" not in frame.columns:
        raise ValueError("Input parquet must contain split and row_id columns")

    enhanced, descriptions = build_enhanced_features(frame, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.to_parquet(output_path, index=False)

    update_manifest(manifest_path, list(descriptions.keys()), descriptions)

    manifest_rows = [
        {"column": column, "description": description}
        for column, description in descriptions.items()
    ]
    pd.DataFrame(manifest_rows).to_csv(report_dir / "engineered_feature_manifest.csv", index=False)
    pd.DataFrame(manifest_rows).to_markdown(report_dir / "engineered_feature_manifest.md", index=False)

    summary = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(enhanced)),
        "original_columns": int(len(frame.columns)),
        "enhanced_columns": int(len(enhanced.columns)),
        "added_features": list(descriptions.keys()),
        "added_feature_count": len(descriptions),
        "target_encoding_target": TE_TARGET,
        "target_encoder_cv": 5,
        "seed": seed,
        "output_path": str(output_path),
    }
    (report_dir / "feature_engineering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (report_dir / "feature_engineering_report.md").write_text(
        "\n".join(
            [
                "# Feature Engineering Summary",
                "",
                f"- Input parquet: `{input_path}`",
                f"- Output parquet: `{output_path}`",
                f"- Rows: {len(frame):,}",
                f"- Added features: {len(descriptions):,}",
                f"- Target encoding target: {TE_TARGET}",
                f"- Target encoding cross-validation folds: 5",
                "",
                "Generated artefacts are in `outputs/feature_engineering`.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer additional ex-ante features.")
    parser.add_argument("--input-path", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_feature_engineering(
        input_path=args.input_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        report_dir=args.report_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
