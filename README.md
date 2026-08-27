# Pareto-Optimized Multi-Criterion Decision Support for Construction Procurement Inspection

Code and result artifacts for the study *Pareto-Optimized Multi-Criterion Decision Support for
Construction Procurement Inspection Under Review Capacity Constraints*.

The repository supports the manuscript's Data Availability Statement. It contains the full analysis
pipeline, the derivation of every modeling target, all result tables reported in the paper, and
scripts that verify the manuscript's numbers against the artifacts that produce them.

## What is here

```
src/                          analysis pipeline, run in numeric order
  01_data_audit.py            source data audit
  02_preprocess_aggregate.py  cleaning, feature manifest, stratified three-way split
  02b_feature_engineering.py  out-of-fold target encodings and engineered features
  00b_derive_targets.py       derives and verifies the five binary risk targets
  03_risk_prediction.py       candidate models, selection, isotonic calibration
  04_auto_mcdm.py             exhaustive multi-criterion grid search
  05_pymoo_ensemble.py        five-algorithm evolutionary Pareto archive
  06_validation_verification.py   held-out ranking validation and leakage checks
  07_ablation_statistical_tests.py  paired BCa bootstrap, Holm-Bonferroni
  08_repeated_seed_stability.py     seed stability
  09_operating_point_calibration.py review-budget operating points
  revision2/                  analyses added in the second revision (see below)

outputs/                      every table and figure behind the reported results
  preprocess/ risk_prediction/ auto_mcdm/ pareto_ensemble/ validation/
  ablation/ operating_point_calibration/ repeated_seed_stability/ audit/
  revision_full_test/         full held-out test partition check (765,877 records)
  target_derivation/          target derivation verification report
  revision1/                  outputs cited in the first response letter
  revision2/                  supporting tables for the second revision

RESULTS_MAP.md                maps every manuscript table to the artifact producing it
DATA_AVAILABILITY.md          how to obtain the source data
README_R2_UPDATE.md           what this second-revision update adds
docs/                         pointer to README_R2_UPDATE.md
```

## Reproducing the results

```bash
pip install -r requirements.txt

# obtain the source data first - see DATA_AVAILABILITY.md
python src/01_data_audit.py
python src/02_preprocess_aggregate.py
python src/02b_feature_engineering.py
python src/00b_derive_targets.py          # verifies the five targets against their source scores
python src/03_risk_prediction.py
python src/04_auto_mcdm.py
python src/05_pymoo_ensemble.py
python src/06_validation_verification.py
python src/07_ablation_statistical_tests.py
python src/08_repeated_seed_stability.py
python src/09_operating_point_calibration.py
```

Every stage writes to `outputs/`. Splits are assigned once at preprocessing with a fixed seed and
stored as a column on the modeling parquet, so all downstream stages operate on identical row-level
partitions.

### Second-revision analyses

These run against the stored artifacts and do not require refitting the pipeline, except for
`a2`, which refits the primary-target base learner across a hyperparameter grid.

```bash
python src/revision2/a1_coverage_and_bootstrap.py     # per-dimension coverage, bootstrap, reproduction check
python src/revision2/a2_hyperparameter_search.py      # 54-configuration validation search
python src/revision2/a3_weight_perturbation.py        # preference-vector sensitivity
python src/revision2/a4_dependence_and_increment.py   # dependence, incremental value, re-estimation comparator
python src/revision2/a5_multirisk_scenario.py         # multi-risk mandate at 5% review capacity
```

They import the pipeline's own criterion construction, normalization, and aggregation code from
`src/04_auto_mcdm.py`, so they run through the same code path as the reported results rather than a
re-implementation.

### Verification scripts

```bash
python src/revision2/a6_verify_manuscript_tables.py   # manuscript cells vs artifacts
python src/revision2/a16_dataset_composition.py       # dataset-composition statements
```

`a6_verify_manuscript_tables.py` reads the manuscript file, which is not distributed here.
Place `manuscript_JCEM_R2.docx` at the repository root, or point the script at it:

```bash
MANUSCRIPT_DOCX=/path/to/manuscript_JCEM_R2.docx python src/revision2/a6_verify_manuscript_tables.py
```

## Verification status

| Check | Result | Artifact |
|---|---|---|
| Target derivation reproduces the stored targets | 5 targets over 3,830,910 rows; one row differs, sitting exactly on its threshold (single-precision storage) | `outputs/target_derivation/` |
| Manuscript table cells match their artifacts | 176 cells checked, 0 mismatches | `outputs/revision2/t_table_artifact_verification.csv` |
| Stored Pareto ranking reproduces from the archive | Spearman 1.000000, maximum score difference 3.0e-08 | `outputs/revision2/t_reproduction_check.csv` |
| Dataset-composition statements in the text | 23 statements recomputed, all reproduce the printed value | `outputs/revision2/t_dataset_composition.csv` |

All reported results derive from a single execution of the pipeline. `RESULTS_MAP.md` links each
manuscript table to the artifact that produces it.

## Source dataset

The procurement records are not original to this study. They are the construction subset (Common
Procurement Vocabulary division 45) of the Global Contract-level Public Procurement Dataset:

> Fazekas, M., Tóth, B., Abdou, A., and Al-Shaibani, A. (2024). "Global contract-level public
> procurement dataset." *Data in Brief*, 54, 110412. https://doi.org/10.1016/j.dib.2024.110412

Please cite that dataset when using the underlying records. See `DATA_AVAILABILITY.md`.

## Scope of the results

The five modeled outcomes are Corruption Risk Index proxy indicators computed from procurement
records. They are rule-based risk flags, not confirmed instances of corruption, collusion, or
misconduct. The ranked output is intended for inspection triage ahead of professional document
review, not for enforcement, contractor disqualification, or any determination of wrongdoing.

## Status

This repository accompanies an unpublished manuscript and should not be cited as a publication.
For the underlying procurement records, cite the source dataset given above.
