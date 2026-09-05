"""Train tuned boosting models on full train data and write submission CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42
CV_FOLDS = 5
OPTUNA_N_TRIALS = 100
OPTUNA_STORAGE = "sqlite:///optuna_boosting.db"
OPTUNA_BEST_PATH = Path("optuna_best.json")

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
}


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
    return df


def select_feature_cols(trial, all_feature_cols, selected_major):
    preset = trial.suggest_categorical(
        "feature_preset",
        ["all_engineered", "selected_major", "custom_blocks"],
    )
    if preset == "all_engineered":
        return all_feature_cols, preset
    if preset == "selected_major":
        return selected_major, preset

    cols = []
    active_blocks = []
    for block_name, block_cols in FEATURE_BLOCKS.items():
        if trial.suggest_int(f"block_{block_name}", 0, 1):
            cols.extend(block_cols)
            active_blocks.append(block_name)

    cols = list(dict.fromkeys(cols))
    if not cols:
        raise optuna.TrialPruned("No feature blocks selected")
    return cols, f"custom_blocks({','.join(active_blocks)})"


def build_xgb_classifier(trial):
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_lgb_classifier(trial):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_cat_classifier(trial):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=trial.suggest_int("iterations", 100, 500, step=50),
        depth=trial.suggest_int("depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        random_state=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def store_optuna_best(study, model_name):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError(f"No completed trials for {model_name}")

    best_trial = min(
        completed,
        key=lambda t: (t.value, -t.user_attrs.get("val_roc_auc_mean", 0.0)),
    )
    param_prefixes = ("feature_", "block_")
    return {
        "model": model_name,
        "trial_number": int(best_trial.number),
        "val_log_loss_mean": float(best_trial.value),
        "val_log_loss_std": float(best_trial.user_attrs["val_log_loss_std"]),
        "val_roc_auc_mean": float(best_trial.user_attrs["val_roc_auc_mean"]),
        "feature_label": best_trial.user_attrs["feature_label"],
        "feature_cols": best_trial.user_attrs["feature_cols"],
        "params": {
            key: best_trial.params[key]
            for key in best_trial.params
            if not key.startswith(param_prefixes)
        },
    }


MODEL_BUILDERS = {
    "xgboost": lambda params: xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    ),
    "lightgbm": lambda params: lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **params,
    ),
    "catboost": lambda params: CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_state=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **params,
    ),
}


def make_submission(model_name, best_config, train_df, test_df, y):
    feature_cols = best_config["feature_cols"]
    model = MODEL_BUILDERS[model_name](best_config["params"])
    model.fit(train_df[feature_cols], y)
    probs = model.predict_proba(test_df[feature_cols])[:, 1]

    submission = pd.DataFrame(
        {"client_id": test_df["client_id"], "default_probability": probs}
    )
    output_path = f"submission_{model_name}.csv"
    submission.to_csv(output_path, index=False)
    print(
        f"{model_name}: saved {output_path} "
        f"({len(submission)} rows, mean prob={probs.mean():.6f})"
    )
    return submission


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df = engineer_features(pd.read_csv("train.csv"))
    test_df = engineer_features(pd.read_csv("test.csv"))
    y = train_df["default"]

    all_feature_cols = [
        col for col in train_df.columns if col not in ["client_id", "default"]
    ]
    selected_major = [
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
    ]

    if OPTUNA_BEST_PATH.exists():
        with OPTUNA_BEST_PATH.open(encoding="utf-8") as f:
            best_configs = json.load(f)
        print(f"Loaded {OPTUNA_BEST_PATH}")
    else:
        def make_objective(model_name, build_model_fn):
            def objective(trial):
                feature_cols, feature_label = select_feature_cols(
                    trial, all_feature_cols, selected_major
                )
                model = build_model_fn(trial)
                cv = StratifiedKFold(
                    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
                )
                cv_scores = cross_validate(
                    model,
                    train_df[feature_cols],
                    y,
                    cv=cv,
                    scoring={"log_loss": "neg_log_loss", "roc_auc": "roc_auc"},
                    n_jobs=-1,
                )
                trial.set_user_attr("feature_label", feature_label)
                trial.set_user_attr("feature_cols", feature_cols)
                trial.set_user_attr("val_log_loss_std", cv_scores["test_log_loss"].std())
                trial.set_user_attr("val_roc_auc_mean", cv_scores["test_roc_auc"].mean())
                return -cv_scores["test_log_loss"].mean()

            return objective

        studies = {
            "xgboost": optuna.create_study(
                study_name="xgb_default_prediction",
                storage=OPTUNA_STORAGE,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            ),
            "lightgbm": optuna.create_study(
                study_name="lgb_default_prediction",
                storage=OPTUNA_STORAGE,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1),
            ),
            "catboost": optuna.create_study(
                study_name="cat_default_prediction",
                storage=OPTUNA_STORAGE,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 2),
            ),
        }
        objectives = {
            "xgboost": make_objective("xgboost", build_xgb_classifier),
            "lightgbm": make_objective("lightgbm", build_lgb_classifier),
            "catboost": make_objective("catboost", build_cat_classifier),
        }

        for model_name, study in studies.items():
            completed = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            )
            if completed < OPTUNA_N_TRIALS:
                print(f"Tuning {model_name} ({completed}/{OPTUNA_N_TRIALS} done)...")
                study.optimize(
                    objectives[model_name],
                    n_trials=max(0, OPTUNA_N_TRIALS - completed),
                    show_progress_bar=True,
                )

        best_configs = {
            model_name: store_optuna_best(study, model_name)
            for model_name, study in studies.items()
        }
        with OPTUNA_BEST_PATH.open("w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=2)
        print(f"Saved {OPTUNA_BEST_PATH}")

    for model_name, config in best_configs.items():
        print(
            f"{model_name}: trial={config['trial_number']}, "
            f"features={config['feature_label']}"
        )
        make_submission(model_name, config, train_df, test_df, y)


if __name__ == "__main__":
    main()
