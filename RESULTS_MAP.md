# Results map: manuscript table to generating artifact

Every numeric value printed in the manuscript is produced by one of the artifacts
listed below. All artifacts come from a single execution of the pipeline; no table
mixes values from different runs.

`src/revision2/a6_verify_manuscript_tables.py` re-reads the submitted DOCX and
compares its cells against these artifacts. Its output is
`outputs/revision2/t_table_artifact_verification.csv` — **176 cells checked, 0
mismatches**.

| Manuscript table | Content | Generating artifact |
|---|---|---|
| Table 1 | Positioning against prior empirical machine-learning studies | Narrative table; no computed values |
| Table 2 | Model input features, target definitions, leakage controls | `outputs/preprocess/tables/table_02_clean_feature_manifest.csv`; target rules executed by `src/00b_derive_targets.py` |
| Table 3 | Modeling dataset inventory and split sizes | `outputs/preprocess/tables/` split inventory |
| Table 4 | Validation hyperparameter search, primary target | `outputs/revision2/t_hyperparameter_search_full_grid.csv`, `t_hyperparameter_search_summary.csv`, `t_hyperparameter_search_spread.csv` |
| Table 5 | Selected compromise solutions | `outputs/pareto_ensemble/tables/table_05_selected_pareto_solutions.csv` |
| Table 6 | Preference-vector perturbation and queue stability | `outputs/revision2/t_weight_perturbation_one_at_a_time.csv`; neighbourhood figures in `t_weight_perturbation_neighbourhood.csv` and `t_weight_perturbation_selected_solutions.csv` |
| Table 7 | Test-split ranking validation summary | `outputs/validation/tables/table_01_ranking_validation_summary.csv` |
| Table 8 | Decision-support coverage, five risk dimensions, single queue | `outputs/revision2/t_coverage_single_queue.csv`, which reproduces `outputs/validation/tables/table_05_decision_support_coverage.csv` |
| Table 9 | Incremental ranking value beyond the primary probability | `outputs/revision2/t_incremental_ranking_value.csv` |
| Table 10 | Direct composite re-estimation vs the deployed strategies | `outputs/revision2/t_reestimation_comparison.csv` |
| Table 11 | Multi-risk mandate at 5% review capacity | `outputs/revision2/t_multirisk_scenario_headline.csv`; full detail in `t_multirisk_scenario.csv`, selected weights in `t_multirisk_selected_weights.csv` |
| Table 12 | Paired bootstrap Pareto advantages | `outputs/revision2/t_bootstrap_comparisons.csv`, derived from `outputs/ablation/tables/table_07_bootstrap_comparisons.csv` |
| Table 13 | Calibrated operating points | `outputs/operating_point_calibration/tables/table_01_calibrated_operating_points.csv` |

## Values reported in the text but not tabulated

| Manuscript statement | Artifact |
|---|---|
| Rank and linear dependence among the five predicted probabilities | `outputs/revision2/t_probability_correlation_spearman.csv`, `t_probability_correlation_pearson.csv` |
| Principal-component variance shares (51.6%, 73.4%) | `outputs/revision2/t_probability_pca.csv` |
| Per-dimension reference ceilings (48.0 / 93.2 / 99.9 / 71.9%) | `outputs/revision2/t_coverage_per_target_reference.csv` |
| Dirichlet neighbourhood summary for the preference vector | `outputs/revision2/t_weight_perturbation_neighbourhood.csv` |
| Full-test check on 765,877 labeled records (90.2% AP) | `outputs/revision1/full_test_knee_validation.csv` |
| Working-sample representativeness against the full test partition | `outputs/revision1/subsample_representativeness.csv` |
| Five binary targets reproduced from the published component scores | `outputs/target_derivation/target_derivation_verification.csv` |
| Tender-lot count distribution (median 1 lot, 27.9%, 11.2%, 99th percentile 52) | `outputs/revision2/t_dataset_composition.csv` |
| Buyer-country count and shares, award-year range, CPV group shares, procedure-type shares | `outputs/revision2/t_dataset_composition.csv` |

## Reproduction check

`outputs/revision2/t_reproduction_check.csv` re-decodes the knee-point solution
directly from `outputs/pareto_ensemble/tables/table_04_global_non_dominated_archive.csv`,
re-scores the 40,000-row test sample through the pipeline's own normalization and
aggregation code, and compares the result with the stored ranking:

| Check | Value |
|---|---|
| Knee configuration re-decoded from the archive | NSGA-III / CoCoSo / robust / 3 criteria / 11.7% review |
| Rows matched against the stored ranking | 40,000 |
| Spearman correlation, recomputed vs stored score | 1.000000 |
| Maximum absolute score difference | 2.97e-08 |

## How to regenerate everything

```bash
python src/00b_derive_targets.py                     # target derivation and verification
python src/revision2/a1_coverage_and_bootstrap.py    # Tables 8 and 12, reproduction check
python src/revision2/a2_hyperparameter_search.py     # Table 4
python src/revision2/a3_weight_perturbation.py       # Table 6
python src/revision2/a4_dependence_and_increment.py  # Tables 9 and 10, correlations, PCA
python src/revision2/a5_multirisk_scenario.py        # Table 11
python src/revision2/a6_verify_manuscript_tables.py  # cell-by-cell verification of the DOCX
python src/revision2/a16_dataset_composition.py      # dataset-composition statements
```
