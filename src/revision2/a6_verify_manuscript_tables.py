"""A6 - Verify every numeric cell of the Revision 2 manuscript tables against the
pipeline artifact that produces it.

Addresses Reviewer #2, Additional Comment: "confirm that the repository outputs
correspond to the revised tables".  The output of this script is published in the
repository as the table-to-artifact verification record.
"""

from __future__ import annotations

import os
from pathlib import Path

import docx
import pandas as pd

import r2_common as C

DOC = Path(os.environ.get("MANUSCRIPT_DOCX", C.REPO_ROOT / "manuscript_JCEM_R2.docx"))
OPS = C.REPO_ROOT / "outputs/operating_point_calibration/tables/table_01_calibrated_operating_points.csv"
SPLITS = C.REPO_ROOT / "outputs/preprocess/tables"
RANKING = C.REPO_ROOT / "outputs/validation/tables/table_01_ranking_validation_summary.csv"
SELECTED = C.REPO_ROOT / "outputs/pareto_ensemble/tables/table_05_selected_pareto_solutions.csv"

checks: list[dict] = []


def check(table: str, cell: str, printed, artifact, source: str, tol: float = 0.051) -> None:
    try:
        ok = abs(float(printed) - float(artifact)) <= tol
    except (TypeError, ValueError):
        ok = str(printed).strip() == str(artifact).strip()
    checks.append(
        {
            "table": table,
            "cell": cell,
            "printed in manuscript": printed,
            "value in artifact": round(float(artifact), 3) if isinstance(artifact, float) else artifact,
            "artifact": source,
            "status": "match" if ok else "MISMATCH",
        }
    )


def main() -> None:
    if not DOC.exists():
        raise SystemExit(
            f"Manuscript file not found: {DOC}
"
            "This script verifies the submitted manuscript against the artifacts in this
"
            "repository. The manuscript is distributed by the journal, not here. Place it
"
            "at the repository root, or set MANUSCRIPT_DOCX to its path, then re-run."
        )
    doc = docx.Document(DOC)
    tables = doc.tables

    def grid(idx: int) -> list[list[str]]:
        return [[c.text.strip() for c in row.cells] for row in tables[idx].rows]

    # ---- Table 5, selected compromise solutions
    sel = pd.read_csv(SELECTED)
    t5 = grid(4)
    for row, (_, art) in zip(t5[1:], sel.iterrows()):
        tag = row[0]
        check("Table 5", f"{tag}: test primary AP", row[6], 100 * art["test_primary_average_precision"], "pareto_ensemble/table_05_selected_pareto_solutions.csv")
        check("Table 5", f"{tag}: test primary F1", row[7], 100 * art["test_primary_f1"], "pareto_ensemble/table_05_selected_pareto_solutions.csv")
        check("Table 5", f"{tag}: test any F1", row[8], 100 * art["test_any_f1"], "pareto_ensemble/table_05_selected_pareto_solutions.csv")
        check("Table 5", f"{tag}: top k rate", row[5], 100 * art["top_k_rate"], "pareto_ensemble/table_05_selected_pareto_solutions.csv")

    # ---- Table 7, ranking validation summary
    rank = pd.read_csv(RANKING)
    name = {"raw primary probability": "raw cri probability", "baseline auto MCDM": "baseline auto mcdm", "Pareto auto MCDM": "pareto auto mcdm"}
    target = {"Composite integrity risk": "y_cri_high", "Any modeled risk": "y_any_risk"}
    for row in grid(6)[1:]:
        art = rank[(rank.strategy == name[row[0]]) & (rank.target == target[row[1]])].iloc[0]
        tag = f"{row[0]} / {row[1]}"
        check("Table 7", f"{tag}: AP", row[4], 100 * art["average_precision"], "validation/table_01_ranking_validation_summary.csv")
        check("Table 7", f"{tag}: ROC AUC", row[5], 100 * art["roc_auc"], "validation/table_01_ranking_validation_summary.csv")
        check("Table 7", f"{tag}: best F1", row[6], 100 * art["best_f1"], "validation/table_01_ranking_validation_summary.csv")
        check("Table 7", f"{tag}: best precision", row[7], 100 * art["best_precision"], "validation/table_01_ranking_validation_summary.csv")
        check("Table 7", f"{tag}: best recall", row[8], 100 * art["best_recall"], "validation/table_01_ranking_validation_summary.csv")

    # ---- Table 8, coverage
    cov = pd.read_csv(C.OUT / "t_coverage_single_queue.csv")
    label = {"raw primary probability": "Raw primary probability", "baseline auto MCDM": "Auto-MCDM baseline", "Pareto auto MCDM": "Pareto-optimized"}
    for row in grid(7)[1:]:
        art = cov[(cov.strategy == label[row[0]]) & (cov["risk dimension"] == row[1])].iloc[0]
        tag = f"{row[0]} / {row[1]}"
        check("Table 8", f"{tag}: AP", row[4], art["test AP (%)"], "Revision 2/supporting_tables/t_coverage_single_queue.csv")
        check("Table 8", f"{tag}: P@5", row[5], art["precision at 5% (%)"], "Revision 2/supporting_tables/t_coverage_single_queue.csv")
        check("Table 8", f"{tag}: P@10", row[6], art["precision at 10% (%)"], "Revision 2/supporting_tables/t_coverage_single_queue.csv")
        check("Table 8", f"{tag}: P@20", row[7], art["precision at 20% (%)"], "Revision 2/supporting_tables/t_coverage_single_queue.csv")

    # ---- Table 12, bootstrap
    boot = pd.read_csv(C.OUT / "t_bootstrap_comparisons.csv")
    for row, (_, art) in zip(grid(11)[1:], boot.iterrows()):
        tag = f"{row[0]} / {row[1]}"
        for j, col in enumerate(
            ["observed Pareto advantage (pp)", "bootstrap mean Pareto advantage (pp)", "BCa CI lower (pp)", "BCa CI upper (pp)"], start=2
        ):
            check("Table 12", f"{tag}: {col}", row[j].replace("+", ""), str(art[col]).replace("+", ""), "Revision 2/supporting_tables/t_bootstrap_comparisons.csv")
        check("Table 12", f"{tag}: Holm p", row[6], art["Holm p value"], "Revision 2/supporting_tables/t_bootstrap_comparisons.csv")

    # ---- Table 13, operating points
    ops = pd.read_csv(OPS)
    ops = ops[ops.strategy == "pareto auto mcdm"]
    tmap = {"Composite integrity risk": "y_cri_high", "Any modeled risk": "y_any_risk"}
    mmap = {"calibrated top k": "calibrated top k", "calibrated score threshold": "calibrated score threshold"}
    for row in grid(12)[1:]:
        art = ops[(ops.target == tmap[row[0]]) & (ops["mode"] == mmap[row[1]])].iloc[0]
        tag = f"{row[0]} / {row[1]}"
        check("Table 13", f"{tag}: val threshold", row[2].replace("%", ""), 100 * art["val_threshold_value"], "operating_point_calibration/table_01_calibrated_operating_points.csv")
        check("Table 13", f"{tag}: val F1", row[3], 100 * art["val_f1"], "operating_point_calibration/table_01_calibrated_operating_points.csv")
        check("Table 13", f"{tag}: test selected rate", row[4], 100 * art["test_selected_rate"], "operating_point_calibration/table_01_calibrated_operating_points.csv")
        check("Table 13", f"{tag}: test selected count", row[5].replace(",", ""), art["test_selected_count"], "operating_point_calibration/table_01_calibrated_operating_points.csv")
        check("Table 13", f"{tag}: test precision", row[6], 100 * art["test_precision"], "operating_point_calibration/table_01_calibrated_operating_points.csv")
        check("Table 13", f"{tag}: test recall", row[7], 100 * art["test_recall"], "operating_point_calibration/table_01_calibrated_operating_points.csv")
        check("Table 13", f"{tag}: test F1", row[8], 100 * art["test_f1"], "operating_point_calibration/table_01_calibrated_operating_points.csv")

    result = pd.DataFrame(checks)
    C.write(result, "t_table_artifact_verification")
    bad = result[result.status == "MISMATCH"]
    print(f"\nchecked {len(result)} cells | mismatches: {len(bad)}")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
