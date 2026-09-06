"""Feature-engineering experiments: ablation blocks + joint Optuna for tuned XGBoost."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from model_copy_utils import (
    CV_FOLDS,
    FEATURE_BLOCKS,
    OPTUNA_CV_FOLDS,
    OPTUNA_RANDOM_STATE,
    RANDOM_STATE,
    build_tuned_xgb,
    build_xgb_classifier,
    engineer_features,
    evaluate_xgb_cv,
    load_data,
    load_xgb_best,
    xgb_params_from_trial,
)

OPTUNA_STORAGE = "sqlite:///optuna_feature_eng_joint.db"
OPTUNA_BEST_PATH = "feature_eng_joint_best.json"
OPTUNA_N_TRIALS = 200

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
DELAY_WEIGHTS = np.array([6, 5, 4, 3, 2, 1])

BASE_ABLATION_BLOCKS = dict(FEATURE_BLOCKS)

NEW_FEATURE_BLOCK_FUNCS = {
    "delay_trends": lambda df: _add_delay_trends_block(df),
    "pay_amt_stats": lambda df: _add_pay_amt_stats_block(df),
    "payment_change": lambda df: _add_payment_change_block(df),
    "util_stats": lambda df: _add_util_stats_block(df),
    "delay_util_interactions": lambda df: _add_delay_util_interactions_block(df),
}

NEW_ABLATION_BLOCKS = {
    "delay_trends": [
        "recent_delay_mean",
        "old_delay_mean",
        "delay_deterioration_v2",
        "weighted_delay_v2",
    ],
    "pay_amt_stats": [
        "mean_pay_amt",
        "std_pay_amt",
        "max_pay_amt",
        "num_zero_payments",
    ],
    "payment_change": [
        "recent_pay_mean",
        "old_pay_mean",
        "payment_change_recent",
    ],
    "util_stats": [
        "mean_util",
        "max_util",
        "std_util",
        "months_high_util",
        "months_over_limit",
        "recent_util_vs_avg",
    ],
    "delay_util_interactions": [
        "recent_delay_x_util",
        "delay_count_x_util",
        "severe_delay_x_util",
    ],
}

ALL_ABLATION_BLOCKS = {**BASE_ABLATION_BLOCKS, **NEW_ABLATION_BLOCKS}


def _add_delay_trends_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delay_pos = df[PAY_COLS].clip(lower=0)
    df["recent_delay_mean"] = df[["PAY_0", "PAY_2"]].mean(axis=1)
    df["old_delay_mean"] = df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    df["delay_deterioration_v2"] = df["recent_delay_mean"] - df["old_delay_mean"]
    df["weighted_delay_v2"] = (delay_pos.to_numpy() * DELAY_WEIGHTS).sum(axis=1)
    return df


def _add_pay_amt_stats_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mean_pay_amt"] = df[PAY_AMT_COLS].mean(axis=1)
    df["std_pay_amt"] = df[PAY_AMT_COLS].std(axis=1)
    df["max_pay_amt"] = df[PAY_AMT_COLS].max(axis=1)
    df["num_zero_payments"] = (df[PAY_AMT_COLS] == 0).sum(axis=1)
    return df


def _add_payment_change_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recent_pay_mean"] = df[["PAY_AMT1", "PAY_AMT2"]].mean(axis=1)
    df["old_pay_mean"] = df[["PAY_AMT5", "PAY_AMT6"]].mean(axis=1)
    df["payment_change_recent"] = df["recent_pay_mean"] - df["old_pay_mean"]
    return df


def _add_util_stats_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    util_cols = [f"credit_util_{i}" for i in range(1, 7)]
    df["mean_util"] = df[util_cols].mean(axis=1)
    df["max_util"] = df[util_cols].max(axis=1)
    df["std_util"] = df[util_cols].std(axis=1)
    df["months_high_util"] = (df[util_cols] > 0.8).sum(axis=1)
    df["months_over_limit"] = (df[util_cols] > 1.0).sum(axis=1)
    df["recent_util_vs_avg"] = df["credit_util_1"] - df["mean_util"]
    return df


def _add_delay_util_interactions_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    util_cols = [f"credit_util_{i}" for i in range(1, 7)]
    if "mean_util" not in df.columns:
        df["mean_util"] = df[util_cols].mean(axis=1)
    if "max_util" not in df.columns:
        df["max_util"] = df[util_cols].max(axis=1)
    df["recent_delay_x_util"] = df["PAY_0"] * df["credit_util_1"]
    df["delay_count_x_util"] = df["num_months_delayed"] * df["mean_util"]
    df["severe_delay_x_util"] = df["num_severe_delays"] * df["max_util"]
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    for fn in NEW_FEATURE_BLOCK_FUNCS.values():
        df = fn(df)
    return df


def all_blocks_active(block_names: list[str] | None = None) -> dict[str, bool]:
    block_names = block_names or list(ALL_ABLATION_BLOCKS)
    return {name: True for name in block_names}


def build_feature_cols_from_blocks(
    active_blocks: dict[str, bool],
    block_map: dict[str, list[str]] | None = None,
) -> list[str]:
    block_map = block_map or ALL_ABLATION_BLOCKS
    cols: list[str] = []
    for block_name, enabled in active_blocks.items():
        if enabled and block_name in block_map:
            cols.extend(block_map[block_name])
    return list(dict.fromkeys(cols))


def evaluate_block_config(
    train_df: pd.DataFrame,
    y: pd.Series,
    active_blocks: dict[str, bool],
    xgb_params: dict | None = None,
    trial: optuna.Trial | None = None,
) -> dict:
    feature_cols = build_feature_cols_from_blocks(active_blocks)
    if not feature_cols:
        raise ValueError("At least one feature block must be enabled")

    if trial is not None:
        model = build_xgb_classifier(trial)
        params = xgb_params_from_trial(trial)
    else:
        if xgb_params is None:
            raise ValueError("Provide xgb_params when trial is None")
        model = build_tuned_xgb(xgb_params)
        params = xgb_params

    metrics = evaluate_xgb_cv(model, train_df[feature_cols], y)
    metrics["feature_cols"] = feature_cols
    metrics["n_features"] = len(feature_cols)
    metrics["active_blocks"] = active_blocks
    metrics["xgb_params"] = params
    return metrics


def run_fixed_params_ablation(
    train_df: pd.DataFrame,
    y: pd.Series,
    xgb_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_on = all_blocks_active()
    all_on_metrics = evaluate_block_config(train_df, y, all_on, xgb_params=xgb_params)
    all_on_val = all_on_metrics["val_log_loss_mean"]

    base_only = all_blocks_active(list(BASE_ABLATION_BLOCKS))
    for name in NEW_ABLATION_BLOCKS:
        base_only[name] = False
    base_only_metrics = evaluate_block_config(train_df, y, base_only, xgb_params=xgb_params)
    base_only_val = base_only_metrics["val_log_loss_mean"]

    leave_one_rows = [
        {
            "scenario": "all_blocks_on",
            "block_changed": "",
            "val_log_loss_mean": all_on_val,
            "val_roc_auc_mean": all_on_metrics["val_roc_auc_mean"],
            "delta_val_log_loss": 0.0,
            "n_features": all_on_metrics["n_features"],
        }
    ]
    for block_name in ALL_ABLATION_BLOCKS:
        cfg = dict(all_on)
        cfg[block_name] = False
        if not any(cfg.values()):
            continue
        metrics = evaluate_block_config(train_df, y, cfg, xgb_params=xgb_params)
        leave_one_rows.append(
            {
                "scenario": f"leave_out: {block_name}",
                "block_changed": block_name,
                "val_log_loss_mean": metrics["val_log_loss_mean"],
                "val_roc_auc_mean": metrics["val_roc_auc_mean"],
                "delta_val_log_loss": metrics["val_log_loss_mean"] - all_on_val,
                "n_features": metrics["n_features"],
            }
        )

    add_one_rows = [
        {
            "scenario": "base_blocks_only",
            "block_changed": "",
            "val_log_loss_mean": base_only_val,
            "val_roc_auc_mean": base_only_metrics["val_roc_auc_mean"],
            "delta_val_log_loss": 0.0,
            "n_features": base_only_metrics["n_features"],
        }
    ]
    for block_name in NEW_ABLATION_BLOCKS:
        cfg = dict(base_only)
        cfg[block_name] = True
        metrics = evaluate_block_config(train_df, y, cfg, xgb_params=xgb_params)
        add_one_rows.append(
            {
                "scenario": f"base_plus: {block_name}",
                "block_changed": block_name,
                "val_log_loss_mean": metrics["val_log_loss_mean"],
                "val_roc_auc_mean": metrics["val_roc_auc_mean"],
                "delta_val_log_loss": metrics["val_log_loss_mean"] - base_only_val,
                "n_features": metrics["n_features"],
            }
        )

    leave_one_out = pd.DataFrame(leave_one_rows).sort_values("val_log_loss_mean")
    add_one_new = pd.DataFrame(add_one_rows).sort_values("val_log_loss_mean")
    return leave_one_out, add_one_new


def make_joint_optuna_objective(train_df: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        active_blocks = {
            block_name: bool(trial.suggest_int(f"block_{block_name}", 0, 1))
            for block_name in ALL_ABLATION_BLOCKS
        }
        if not any(active_blocks.values()):
            raise optuna.TrialPruned("No feature blocks selected")

        metrics = evaluate_block_config(train_df, y, active_blocks, trial=trial)
        enabled = [name for name, on in active_blocks.items() if on]

        trial.set_user_attr("active_blocks", active_blocks)
        trial.set_user_attr("enabled_blocks", enabled)
        trial.set_user_attr("feature_cols", metrics["feature_cols"])
        trial.set_user_attr("n_features", metrics["n_features"])
        trial.set_user_attr("val_roc_auc_mean", metrics["val_roc_auc_mean"])
        trial.set_user_attr("val_log_loss_std", metrics["val_log_loss_std"])
        trial.set_user_attr("xgb_params", metrics["xgb_params"])
        return metrics["val_log_loss_mean"]

    return objective


def summarize_block_effects(study: optuna.Study, reference_val: float) -> pd.DataFrame:
    rows = []
    for block_name in ALL_ABLATION_BLOCKS:
        on_losses = []
        off_losses = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            active = trial.user_attrs.get("active_blocks", {})
            if block_name not in active:
                continue
            if active[block_name]:
                on_losses.append(trial.value)
            else:
                off_losses.append(trial.value)

        on_mean = float(np.mean(on_losses)) if on_losses else np.nan
        off_mean = float(np.mean(off_losses)) if off_losses else np.nan
        rows.append(
            {
                "block": block_name,
                "block_group": "base"
                if block_name in BASE_ABLATION_BLOCKS
                else "feature_eng",
                "trials_with_block_on": len(on_losses),
                "trials_with_block_off": len(off_losses),
                "mean_val_log_loss_block_on": on_mean,
                "mean_val_log_loss_block_off": off_mean,
                "delta_on_minus_off": on_mean - off_mean,
                "improves_when_on": on_mean < off_mean
                if on_losses and off_losses
                else np.nan,
                "delta_vs_reference_when_on": on_mean - reference_val
                if on_losses
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_on_minus_off")


def load_feature_eng_best(path: str | None = None) -> dict:
    best_path = Path(path or OPTUNA_BEST_PATH)
    if not best_path.exists():
        raise FileNotFoundError(
            f"Missing feature eng best config: {best_path}. Run feature_eng.ipynb section 7 first."
        )
    with best_path.open(encoding="utf-8") as f:
        return json.load(f)


def make_feature_eng_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    best_config: dict | None = None,
    output_path: str = "submission_feature_eng.csv",
) -> pd.DataFrame:
    best_config = best_config or load_feature_eng_best()
    feature_cols = best_config["best_feature_cols"]
    model = build_tuned_xgb(best_config["best_xgb_params"])

    model.fit(train_df[feature_cols], y)
    probs = model.predict_proba(test_df[feature_cols])[:, 1]

    submission = pd.DataFrame(
        {
            "client_id": test_df["client_id"],
            "default_probability": probs,
        }
    )
    submission.to_csv(output_path, index=False)
    print(
        f"Saved {output_path} "
        f"({len(submission)} rows, mean prob={probs.mean():.6f}, "
        f"{len(feature_cols)} features)"
    )
    return submission


def store_feature_eng_best(study: optuna.Study, reference_val: float) -> dict:
    best = study.best_trial
    result = {
        "reference_val_log_loss_mean": float(reference_val),
        "best_val_log_loss_mean": float(best.value),
        "best_val_roc_auc_mean": float(best.user_attrs["val_roc_auc_mean"]),
        "best_active_blocks": best.user_attrs["active_blocks"],
        "best_enabled_blocks": best.user_attrs["enabled_blocks"],
        "best_feature_cols": best.user_attrs["feature_cols"],
        "best_xgb_params": best.user_attrs["xgb_params"],
        "n_features": int(best.user_attrs["n_features"]),
    }
    with Path(OPTUNA_BEST_PATH).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
