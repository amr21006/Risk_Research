# Data availability

## Why the data files are not in this repository

The study uses 3,830,910 construction tender-lot records. The source extract and the intermediate
modeling files are between 46 MB and 4.9 GB each, which exceeds GitHub's 100 MB per-file limit:

| File | Size | Stage |
|---|---|---|
| `All_Construction_Data.csv` | 4.9 GB | source extract |
| `data/processed/tender_lot_dataset.csv` | 936 MB | cleaned tender-lot dataset |
| `data/modeling/tender_lot_audit_priority.parquet` | 142 MB | supplementary feature set |
| `data/modeling/tender_lot_strict_ex_ante_enhanced.parquet` | 105 MB | headline modeling table |
| `data/modeling/tender_lot_strict_ex_ante.parquet` | 46 MB | baseline feature set |

They are excluded by `.gitignore`. Everything derived from them — the split inventory, the
calibrated prediction files for the working samples, the Pareto archive, the ranking files, and
every result table — is committed here, so the reported numbers can be checked without the raw
extract.

## Source of the data

This study did not collect procurement records. It uses the construction subset of an existing,
publicly released dataset:

> Fazekas, M., Tóth, B., Abdou, A., and Al-Shaibani, A. (2024). "Global contract-level public
> procurement dataset." *Data in Brief*, 54, 110412. https://doi.org/10.1016/j.dib.2024.110412

The Global Contract-level Public Procurement Dataset is compiled by the Government Transparency
Institute from Tenders Electronic Daily and national procurement portals, and is distributed through
Mendeley Data (https://doi.org/10.17632/w9mzf4vswh.3) under the terms stated by its publishers.
Please cite the dataset paper above, not this repository, when using the underlying records.

The extract used here is restricted to Common Procurement Vocabulary division 45 (construction
works), spans award years 2000 to 2021, and covers 67 buyer countries. Monetary values are reported
in United States dollars.

### Corruption Risk Index fields

The Corruption Risk Index and its component indicators are **distributed as fields of the source
dataset** and are used here as published; this study does not recompute them. In the source schema
they appear as `cri`, `corr_proc`, `corr_buyer_concentration`, `corr_singleb`, and `corr_nocft`, and
they are carried into the modeling tables as `label_cri`, `label_corr_proc`,
`label_corr_buyer_concentration`, `label_corr_singleb`, and `label_corr_nocft`. Their construction
follows Fazekas, M., Tóth, I. J., and King, L. P. (2016), "An objective corruption risk index using
public procurement data," *European Journal on Criminal Policy and Research*, 22(3), 369–397,
https://doi.org/10.1007/s10610-016-9308-z. The proxy outcomes modeled in this study therefore inherit
the construction, coverage, and documented limitations of the source dataset.

## Obtaining the data

Obtain the source dataset from its publishers at https://doi.org/10.17632/w9mzf4vswh.3 and apply the
construction filter (Common Procurement Vocabulary division 45) to reproduce the extract used here.

For questions about the specific extract, contact Amr A. Mohy, Construction and Building Engineering
Department, Arab Academy for Science, Technology and Maritime Transport, Abu Qir, Alexandria, Egypt.

## Regenerating the modeling files

With the source extract placed at the repository root as `All_Construction_Data.csv`:

```bash
python src/01_data_audit.py
python src/02_preprocess_aggregate.py     # writes data/processed/ and data/modeling/
python src/02b_feature_engineering.py     # writes the enhanced modeling parquet
python src/00b_derive_targets.py          # verifies the five binary targets
```

`02_preprocess_aggregate.py` assigns the stratified 60/20/20 train, validation, and test split with a
fixed seed and writes it as a column on the modeling parquet, so the partition is identical on every
regeneration and every downstream stage reads the same rows.

## Target derivation

The five binary targets are derived from the Corruption Risk Index component scores by
`src/00b_derive_targets.py`:

| Target | Rule |
|---|---|
| `y_cri_high` | `label_cri >= 0.50` |
| `y_proc_high` | `label_corr_proc >= 0.50` |
| `y_buyer_concentration_high` | `label_corr_buyer_concentration >= 0.087196`, the 75th percentile of that score in the modeling data |
| `y_single_bid` | `label_corr_singleb > 0` |
| `y_no_call_for_tender` | `label_corr_nocft > 0` |

The script recomputes each target, compares it against the stored column row by row, and recomputes
the buyer-concentration cutoff as an empirical quantile rather than accepting it as a constant. The
verification report is committed at `outputs/target_derivation/`.

Four targets reproduce exactly. For `y_cri_high`, one row of 3,830,910 differs, and that row holds a
component score of exactly 0.5; component scores are stored in single precision, so a value
marginally below the threshold at full precision can round to the threshold on write. The script
classifies threshold-boundary disagreements separately and fails on any other kind, of which there
are none.

## Interpretation

The five outcomes are rule-based Corruption Risk Index proxy indicators, not verified corruption,
collusion, or misconduct. Reported average precision and F1 measure recovery of these proxies.
