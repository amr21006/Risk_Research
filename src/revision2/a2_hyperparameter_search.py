"""A2 - Validation-set hyperparameter search for the primary target.

Addresses Reviewer #2, Comment 4 (follow-up): the absolute adequacy of the base
learner.  The search reuses the submitted pipeline's data preparation, working
samples, preprocessing, class weighting, and isotonic calibration so that the
searched configurations are directly comparable with the fixed configuration
reported in the manuscript.

Selection is on validation calibrated average precision; the held-out test
sample is touched only once, after selection.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import r2_common as C

spec = importlib.util.spec_from_file_location("risk_prediction_module", C.SRC / "03_risk_prediction.py")
RP = importlib.util.module_from_spec(spec)
sys.modules['risk_prediction_module'] = RP
spec.loader.exec_module(RP)

TARGET = "y_cri_high"
SEED = 2026
FEATURE_SET = "strict_ex_ante_enhanced"

LGBM_GRID = {
    "learning_rate": [0.03, 0.05, 0.10],
    "n_estimators": [120, 240, 480],
    "num_leaves": [15, 31, 63],
}
XGB_GRID = {
    "learning_rate": [0.03, 0.05, 0.10],
    "n_estimators": [120, 240, 480],
    "max_depth": [3, 5, 7],
}

INCUMBENT = {"family": "LightGBM", "learning_rate": 0.05, "n_estimators": 240, "num_leaves": 31}


def f1_at_rate(y: np.ndarray, score: np.ndarray, rate: float) -> float:
    return C.f1_at_rate(y, score, rate)


def main() -> None:
    print("loading working samples through the pipeline loader ...")
    data = RP.load_prepared_data(
        data_dir=C.OPTI_ROOT / "data/modeling",
        manifest_path=C.OPTI_ROOT / "outputs/preprocess/tables/table_02_clean_feature_manifest.csv",
        feature_set=FEATURE_SET,
        split_caps={"train": 120_000, "val": 40_000, "test": 40_000},
        seed=SEED,
        batch_size=150_000,
    )
    frame = RP.sanitize_for_sklearn(data.frame, data.features)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame[frame[TARGET].notna()]

    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    pre = RP.build_preprocessor(train, data.features)
    x_train = pre.fit_transform(train[data.features])
    x_val = pre.transform(val[data.features])
    x_test = pre.transform(test[data.features])
    y_train = train[TARGET].astype(int).to_numpy()
    y_val = val[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()
    pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    rows = []

    def evaluate(family: str, params: dict, model) -> dict:
        start = time.time()
        model.fit(x_train, y_train)
        raw_val = RP.predict_probability(model, x_val)
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="isotonic")
        calibrator.fit(x_val, y_val)
        cal_val = calibrator.predict_proba(x_val)[:, 1]
        record = {
            "family": family,
            **params,
            "val uncalibrated AP (%)": round(100 * average_precision_score(y_val, raw_val), 2),
            "val calibrated AP (%)": round(100 * average_precision_score(y_val, cal_val), 2),
            "val calibrated ROC AUC (%)": round(100 * roc_auc_score(y_val, cal_val), 2),
            "val F1 at 20% review (%)": round(100 * f1_at_rate(y_val, cal_val, 0.20), 2),
            "fit seconds": round(time.time() - start, 1),
        }
        record["_model"] = model
        record["_calibrator"] = calibrator
        rows.append(record)
        print(
            f"  {family:9s} {params} -> val cal AP {record['val calibrated AP (%)']:.2f}%"
            f"  ({record['fit seconds']}s)"
        )
        return record

    print("\nLightGBM grid (27 configurations)")
    for lr, n, leaves in itertools.product(*LGBM_GRID.values()):
        evaluate(
            "LightGBM",
            {"learning_rate": lr, "n_estimators": n, "num_leaves": leaves, "max_depth": None},
            LGBMClassifier(
                n_estimators=n,
                learning_rate=lr,
                num_leaves=leaves,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
                deterministic=True,
                force_row_wise=True,
            ),
        )

    print("\nXGBoost grid (27 configurations)")
    for lr, n, depth in itertools.product(*XGB_GRID.values()):
        evaluate(
            "XGBoost",
            {"learning_rate": lr, "n_estimators": n, "num_leaves": None, "max_depth": depth},
            XGBClassifier(
                n_estimators=n,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=SEED,
                n_jobs=-1,
                scale_pos_weight=pos_weight,
            ),
        )

    table = pd.DataFrame(rows).drop(columns=["_model", "_calibrator"])
    table = table.sort_values("val calibrated AP (%)", ascending=False).reset_index(drop=True)
    table.insert(0, "validation rank", np.arange(1, len(table) + 1))
    C.write(table, "t_hyperparameter_search_full_grid")

    # incumbent and searched-best, evaluated once on the held-out test sample
    def is_incumbent(r: dict) -> bool:
        return (
            r["family"] == INCUMBENT["family"]
            and r["learning_rate"] == INCUMBENT["learning_rate"]
            and r["n_estimators"] == INCUMBENT["n_estimators"]
            and r["num_leaves"] == INCUMBENT["num_leaves"]
        )

    incumbent = next(r for r in rows if is_incumbent(r))
    best = max(rows, key=lambda r: r["val calibrated AP (%)"])
    incumbent_rank = int(table.loc[
        (table["family"] == INCUMBENT["family"])
        & (table["learning_rate"] == INCUMBENT["learning_rate"])
        & (table["n_estimators"] == INCUMBENT["n_estimators"])
        & (table["num_leaves"] == INCUMBENT["num_leaves"]),
        "validation rank",
    ].iloc[0])

    summary = []
    for name, rec in [("Fixed configuration (manuscript)", incumbent), ("Best validation configuration", best)]:
        cal_test = rec["_calibrator"].predict_proba(x_test)[:, 1]
        summary.append(
            {
                "configuration": name,
                "family": rec["family"],
                "learning rate": rec["learning_rate"],
                "tree count": rec["n_estimators"],
                "num leaves": rec["num_leaves"],
                "max depth": rec["max_depth"],
                "validation rank of 54": incumbent_rank if name.startswith("Fixed") else 1,
                "val calibrated AP (%)": rec["val calibrated AP (%)"],
                "test calibrated AP (%)": round(100 * average_precision_score(y_test, cal_test), 2),
                "test calibrated ROC AUC (%)": round(100 * roc_auc_score(y_test, cal_test), 2),
                "test F1 at 20% review (%)": round(100 * f1_at_rate(y_test, cal_test, 0.20), 2),
            }
        )
    summary_df = pd.DataFrame(summary)
    C.write(summary_df, "t_hyperparameter_search_summary")
    print()
    print(summary_df.to_string(index=False))

    spread = pd.DataFrame(
        [
            {
                "statistic": "configurations evaluated",
                "value": f"{len(table)}",
            },
            {
                "statistic": "validation calibrated AP range (%)",
                "value": f"{table['val calibrated AP (%)'].min():.2f} to {table['val calibrated AP (%)'].max():.2f}",
            },
            {
                "statistic": "validation calibrated AP interquartile range (pp)",
                "value": f"{table['val calibrated AP (%)'].quantile(0.75) - table['val calibrated AP (%)'].quantile(0.25):.2f}",
            },
            {
                "statistic": "fixed configuration validation rank",
                "value": f"{incumbent_rank} of {len(table)}",
            },
            {
                "statistic": "fixed minus best validation AP (pp)",
                "value": f"{incumbent['val calibrated AP (%)'] - best['val calibrated AP (%)']:+.2f}",
            },
            {
                "statistic": "fixed minus best test AP (pp)",
                "value": f"{summary[0]['test calibrated AP (%)'] - summary[1]['test calibrated AP (%)']:+.2f}",
            },
            {
                "statistic": "fixed minus best test F1 at 20% review (pp)",
                "value": f"{summary[0]['test F1 at 20% review (%)'] - summary[1]['test F1 at 20% review (%)']:+.2f}",
            },
        ]
    )
    C.write(spread, "t_hyperparameter_search_spread")
    print()
    print(spread.to_string(index=False))


if __name__ == "__main__":
    main()
