# Repository update for the second revision

This update responds to the reviewers' request that the study's data availability
path actually carry what the manuscript cites. It adds three things the repository
did not previously contain.

## 1. The target derivation is now traceable in code

`src/00b_derive_targets.py` derives all five binary risk targets from the published
Corruption Risk Index component scores and verifies them against the stored target
columns over the whole modeling dataset. Previously the five `y_*` columns entered
the pipeline as precomputed columns and the thresholds appeared only in the
manuscript text.

Verification result (`outputs/target_derivation/target_derivation_verification.csv`),
over 3,830,910 rows:

| Target | Rule | Result |
|---|---|---|
| `y_cri_high` | `label_cri >= 0.50` | 1 row differs, sitting exactly on the threshold |
| `y_proc_high` | `label_corr_proc >= 0.50` | reproduced exactly |
| `y_buyer_concentration_high` | `label_corr_buyer_concentration >= 0.087196` | reproduced exactly |
| `y_single_bid` | `label_corr_singleb > 0` | reproduced exactly |
| `y_no_call_for_tender` | `label_corr_nocft > 0` | reproduced exactly |

The buyer-concentration cutoff of 0.087196 is recomputed by the script as the 75th
percentile of the component score in the modeling data, confirming that it is the
empirical upper quartile and not an arbitrary constant.

The single `y_cri_high` disagreement is one row whose stored component score is
exactly 0.5. Component scores are stored in single precision, so a value marginally
below the threshold in full precision can round up to the threshold on write. The
script reports such rows separately and fails only on disagreements that are not at
a threshold; there are none.

## 2. The outputs cited in the response letters are present

- `outputs/revision1/full_test_knee_validation.csv` and `.md` — the full held-out
  test check on 765,877 labeled records, cited in the first revision.
- `outputs/revision1/subsample_representativeness.csv` and `.md` — the comparison of
  the 40,000-row working sample against the full test partition.
- `outputs/revision2/` — every supporting table behind the second revision.

## 3. Each manuscript number is mapped to the artifact that produces it

`RESULTS_MAP.md` lists, for every table and for every figure quoted in the text, the
artifact that generates it. `src/revision2/a6_verify_manuscript_tables.py` re-reads
the submitted manuscript and checks its table cells against those artifacts: 176
cells checked, 0 mismatches.

## Layout

```
src/00b_derive_targets.py            target derivation and verification
src/revision2/                       the second-revision analyses and verification scripts
outputs/target_derivation/           target derivation verification report
outputs/revision1/                   outputs cited in the first response letter
outputs/revision2/                   supporting tables for the second revision
RESULTS_MAP.md                       manuscript table to artifact map
```

The `src/revision2/` scripts import the pipeline's own criterion construction,
normalization, and aggregation code from `src/04_auto_mcdm.py`, so the supporting
analyses run through exactly the same code path as the reported results rather than
through a re-implementation.
