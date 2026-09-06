"""Shared utilities imported from models copy.ipynb (XGBoost-focused)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42
CV_FOLDS = 5
OPTUNA_BEST_PATH = "optuna_best_extended.json"
OPTUNA_RANDOM_STATE = RANDOM_STATE
OPTUNA_CV_FOLDS = CV_FOLDS

NEW_ENGINEERED_COLS = [
    "recent_delay_avg",
    "older_delay_avg",
    "delay_deterioration",
    "weighted_delay",
    *[f"pay_bill_ratio_{i}" for i in range(1, 7)],
    "mean_pay_bill_ratio",
    "min_pay_bill_ratio",
    "num_low_pay_ratio",
]

FEATURE_BLOCKS = {
    "demographics": ["SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL"],
    "delay_engineered": [
        "num_severe_delays",
        "num_months_delayed",
        "ever_delayed",
        "max_delay",
        "mean_pay_status",
    ],
    "pay_status": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    "pay_amounts": [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "total_pay",
    ],
    "bill_amounts": [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "total_bill",
    ],
    "credit_util": [f"credit_util_{i}" for i in range(1, 7)],
    "bill_trends": [
        "bill_slope",
        "bill_abs_change_1_2",
        "bill_pct_change_1_2",
        "bill_abs_change_2_3",
        "bill_pct_change_2_3",
        "bill_abs_change_3_4",
        "bill_pct_change_3_4",
        "bill_abs_change_4_5",
        "bill_pct_change_4_5",
        "bill_abs_change_5_6",
        "bill_pct_change_5_6",
        "bill_abs_change_1_6",
        "bill_pct_change_1_6",
    ],
    "models_copy_new_engineered": NEW_ENGINEERED_COLS,
}


def load_data(file: str) -> pd.DataFrame:
    return pd.read_csv(file)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df["total_pay"] = df[pay_amt_cols].sum(axis=1)
    df["total_bill"] = df[bill_cols].sum(axis=1)

    for col in bill_cols:
        df[f"credit_util_{col[-1]}"] = df[col] / df["LIMIT_BAL"]

    months = np.arange(1, 7)
    x_centered = months - months.mean()
    bill_values = df[bill_cols].to_numpy()
    df["bill_slope"] = (bill_values * x_centered).sum(axis=1) / (x_centered**2).sum()

    for i in range(1, 6):
        prev_col = f"BILL_AMT{i}"
        next_col = f"BILL_AMT{i + 1}"
        df[f"bill_abs_change_{i}_{i + 1}"] = df[next_col] - df[prev_col]
        df[f"bill_pct_change_{i}_{i + 1}"] = (
            (df[next_col] - df[prev_col]) / df[prev_col].replace(0, np.nan)
        )

    df["bill_abs_change_1_6"] = df["BILL_AMT6"] - df["BILL_AMT1"]
    df["bill_pct_change_1_6"] = (
        (df["BILL_AMT6"] - df["BILL_AMT1"]) / df["BILL_AMT1"].replace(0, np.nan)
    )

    df["max_delay"] = df[pay_cols].max(axis=1)
    df["num_months_delayed"] = (df[pay_cols] > 0).sum(axis=1)
    df["num_severe_delays"] = (df[pay_cols] >= 2).sum(axis=1)
    df["ever_delayed"] = (df[pay_cols] > 0).any(axis=1).astype(int)
    df["mean_pay_status"] = df[pay_cols].mean(axis=1)

    df["recent_delay_avg"] = df[["PAY_0", "PAY_2"]].mean(axis=1)
    df["older_delay_avg"] = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    df["delay_deterioration"] = df["recent_delay_avg"] - df["older_delay_avg"]

    delay = df[pay_cols].clip(lower=0)
    df["weighted_delay"] = (
        6 * delay["PAY_0"]
        + 5 * delay["PAY_2"]
        + 4 * delay["PAY_3"]
        + 3 * delay["PAY_4"]
        + 2 * delay["PAY_5"]
        + 1 * delay["PAY_6"]
    )

    for i in range(1, 7):
        df[f"pay_bill_ratio_{i}"] = (
            df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].abs() + 100)
        )

    ratio_cols = [f"pay_bill_ratio_{i}" for i in range(1, 7)]
    df["mean_pay_bill_ratio"] = df[ratio_cols].mean(axis=1)
    df["min_pay_bill_ratio"] = df[ratio_cols].min(axis=1)
    df["num_low_pay_ratio"] = (df[ratio_cols] < 0.1).sum(axis=1)
    return df


def get_feature_sets(all_feature_cols: list[str]) -> dict[str, list[str]]:
    original_engineered_cols = [
        col for col in all_feature_cols if col not in NEW_ENGINEERED_COLS
    ]
    return {
        "all_engineered": original_engineered_cols,
        "all_engineered_extended": all_feature_cols,
        "selected_major": [
            "num_severe_delays",
            "num_months_delayed",
            "ever_delayed",
            "max_delay",
            "PAY_0",
            "mean_pay_status",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "total_pay",
            "LIMIT_BAL",
        ],
    }


def build_xgb_classifier(trial: optuna.Trial) -> xgb.XGBClassifier:
    """Optuna search space copied from models copy.ipynb section 10."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        random_state=OPTUNA_RANDOM_STATE,
        n_jobs=-1,
    )


def build_tuned_xgb(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


MODEL_BUILDERS = {
    "xgboost": build_tuned_xgb,
}


def load_optuna_best(path: str | None = None) -> dict:
    best_path = Path(path or OPTUNA_BEST_PATH)
    if not best_path.exists():
        raise FileNotFoundError(
            f"Missing tuned config: {best_path}. Run models copy.ipynb sections 10-12 first."
        )
    with best_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_xgb_best() -> dict:
    return load_optuna_best()["xgboost"]


def evaluate_xgb_cv(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | None = None,
) -> dict:
    cv = cv or StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"log_loss": "neg_log_loss", "roc_auc": "roc_auc"},
        return_train_score=True,
        n_jobs=-1,
    )
    return {
        "train_log_loss_mean": float(-scores["train_log_loss"].mean()),
        "val_log_loss_mean": float(-scores["test_log_loss"].mean()),
        "val_log_loss_std": float(scores["test_log_loss"].std()),
        "train_roc_auc_mean": float(scores["train_roc_auc"].mean()),
        "val_roc_auc_mean": float(scores["test_roc_auc"].mean()),
    }


def xgb_params_from_trial(trial: optuna.Trial) -> dict:
    return {
        key: trial.params[key]
        for key in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "min_child_weight",
        ]
        if key in trial.params
    }
