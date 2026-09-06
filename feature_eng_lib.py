"""Feature-engineering experiments: ablation blocks + joint Optuna for tuned XGBoost."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

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


BILL_PCT_PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)]
BILL_PCT_CLIP_BOUNDS = (-5.0, 5.0)
PAY_BILL_STAB_DENOM = 1000

TARGETED_BLOCK_FUNCS = {
    "robust_bill_pct": lambda df: _add_robust_bill_pct_block(df),
    "pay_amt_stats": lambda df: _add_pay_amt_stats_block(df),
    "payment_change": lambda df: _add_payment_change_block(df),
    "stabilized_pay_bill": lambda df: _add_stabilized_pay_bill_block(df),
    "low_risk_interactions": lambda df: _add_low_risk_interactions_block(df),
}

TARGETED_BLOCKS = {
    "robust_bill_pct": [
        *[f"bill_pct_change_{i}_{j}_robust" for i, j in BILL_PCT_PAIRS],
        *[f"bill_pct_change_{i}_{j}_clip" for i, j in BILL_PCT_PAIRS],
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
    "stabilized_pay_bill": [
        *[f"pay_bill_ratio_stab_{i}" for i in range(1, 7)],
        "mean_pay_bill_ratio_stab",
        "min_pay_bill_ratio_stab",
        "recent_pay_bill_ratio_stab",
        "num_low_pay_ratio_stab",
    ],
    "low_risk_interactions": [
        "low_util_high_limit",
        "limit_over_mean_util",
        "total_pay_over_limit",
        "low_delinq_x_payment_stress",
        "low_severe_delinq_x_low_pay",
    ],
}

TARGETED_BLOCK_ORDER = list(TARGETED_BLOCKS.keys())


def _add_robust_bill_pct_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, j in BILL_PCT_PAIRS:
        prev_col = f"BILL_AMT{i}"
        next_col = f"BILL_AMT{j}"
        pct_col = f"bill_pct_change_{i}_{j}"
        robust_col = f"bill_pct_change_{i}_{j}_robust"
        clip_col = f"bill_pct_change_{i}_{j}_clip"

        df[robust_col] = (df[next_col] - df[prev_col]) / (
            df[prev_col].abs() + PAY_BILL_STAB_DENOM
        )
        if pct_col in df.columns:
            df[clip_col] = df[pct_col].clip(*BILL_PCT_CLIP_BOUNDS)
        else:
            raw_pct = (df[next_col] - df[prev_col]) / df[prev_col].replace(0, np.nan)
            df[clip_col] = raw_pct.clip(*BILL_PCT_CLIP_BOUNDS)
    return df


def _add_stabilized_pay_bill_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stab_cols = []
    for i in range(1, 7):
        col = f"pay_bill_ratio_stab_{i}"
        df[col] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].abs() + PAY_BILL_STAB_DENOM)
        stab_cols.append(col)

    df["mean_pay_bill_ratio_stab"] = df[stab_cols].mean(axis=1)
    df["min_pay_bill_ratio_stab"] = df[stab_cols].min(axis=1)
    df["recent_pay_bill_ratio_stab"] = df["pay_bill_ratio_stab_1"]
    df["num_low_pay_ratio_stab"] = (df[stab_cols] < 0.1).sum(axis=1)
    return df


def _add_low_risk_interactions_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    util_cols = [f"credit_util_{i}" for i in range(1, 7)]
    if "mean_util" not in df.columns:
        df["mean_util"] = df[util_cols].mean(axis=1)
    if "mean_pay_bill_ratio_stab" not in df.columns:
        df = _add_stabilized_pay_bill_block(df)

    low_util = (df["mean_util"] < 0.3).astype(float)
    df["low_util_high_limit"] = low_util * np.log1p(df["LIMIT_BAL"].clip(lower=0))
    df["limit_over_mean_util"] = df["LIMIT_BAL"] / (df["mean_util"] + 0.1)
    df["total_pay_over_limit"] = df["total_pay"] / (df["LIMIT_BAL"].abs() + 1.0)

    payment_stress = (1.0 - df["mean_pay_bill_ratio_stab"]).clip(0.0, 1.0)
    df["low_delinq_x_payment_stress"] = (df["max_delay"] <= 0).astype(float) * payment_stress
    df["low_severe_delinq_x_low_pay"] = (
        (df["num_severe_delays"] == 0).astype(float) * df["num_low_pay_ratio_stab"]
    )
    return df


def engineer_targeted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Base engineered dataset plus all hard-error targeted feature blocks."""
    df = engineer_all_features(df)
    for fn in TARGETED_BLOCK_FUNCS.values():
        df = fn(df)
    return df


def add_targeted_block(df: pd.DataFrame, block_name: str) -> pd.DataFrame:
    if block_name not in TARGETED_BLOCK_FUNCS:
        raise KeyError(f"Unknown targeted block: {block_name}")
    return TARGETED_BLOCK_FUNCS[block_name](df)


def build_targeted_feature_cols(
    base_feature_cols: list[str],
    enabled_blocks: list[str],
) -> list[str]:
    extra_cols: list[str] = []
    for block_name in enabled_blocks:
        extra_cols.extend(TARGETED_BLOCKS[block_name])
    return list(dict.fromkeys(base_feature_cols + extra_cols))


def prediction_confidence(oof_prob: np.ndarray) -> np.ndarray:
    oof_prob = np.asarray(oof_prob, dtype=float)
    return np.where(oof_prob >= 0.5, oof_prob, 1.0 - oof_prob)


def assign_error_groups(
    y_true: pd.Series | np.ndarray,
    oof_prob: np.ndarray,
    confidence_threshold: float = 0.9,
) -> pd.Series:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(oof_prob) >= 0.5).astype(int)
    confidence = prediction_confidence(oof_prob)
    wrong = pred != y

    groups = np.full(len(y), "other_wrong", dtype=object)
    groups[(~wrong) & (y == 1)] = "correct_default"
    groups[(~wrong) & (y == 0)] = "correct_non_default"
    groups[wrong & (pred == 1) & (confidence > confidence_threshold)] = "confident_fp"
    groups[wrong & (pred == 0) & (confidence > confidence_threshold)] = "confident_fn"
    return pd.Series(groups, name="error_group")


def count_confident_false_negatives(
    y_true: pd.Series | np.ndarray,
    oof_prob: np.ndarray,
    confidence_threshold: float = 0.9,
) -> int:
    groups = assign_error_groups(y_true, oof_prob, confidence_threshold)
    return int((groups == "confident_fn").sum())


def standardized_mean_differences(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    combined = pd.concat([group_a[feature_cols], group_b[feature_cols]], axis=0)
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(combined),
        columns=feature_cols,
        index=combined.index,
    )
    return scaled.loc[group_a.index].mean() - scaled.loc[group_b.index].mean()


def compute_oof_with_folds(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    xgb_params: dict,
) -> tuple[np.ndarray, list[dict]]:
    oof_prob = np.zeros(len(y), dtype=float)
    fold_rows: list[dict] = []

    for train_idx, val_idx in folds:
        model = build_tuned_xgb(xgb_params)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        train_prob = model.predict_proba(X_tr)[:, 1]
        val_prob = model.predict_proba(X_va)[:, 1]
        oof_prob[val_idx] = val_prob

        fold_rows.append(
            {
                "train_log_loss": float(log_loss(y_tr, train_prob)),
                "val_log_loss": float(log_loss(y_va, val_prob)),
                "train_roc_auc": float(roc_auc_score(y_tr, train_prob)),
                "val_roc_auc": float(roc_auc_score(y_va, val_prob)),
            }
        )

    return oof_prob, fold_rows


def summarize_oof_metrics(
    y_true: pd.Series,
    oof_prob: np.ndarray,
    fold_rows: list[dict],
    confidence_threshold: float = 0.9,
    baseline_confident_fn: int | None = None,
) -> dict:
    val_ll = [row["val_log_loss"] for row in fold_rows]
    val_auc = [row["val_roc_auc"] for row in fold_rows]
    train_ll = [row["train_log_loss"] for row in fold_rows]
    train_auc = [row["train_roc_auc"] for row in fold_rows]

    confident_fn = count_confident_false_negatives(y_true, oof_prob, confidence_threshold)
    out = {
        "val_log_loss_mean": float(np.mean(val_ll)),
        "val_log_loss_std": float(np.std(val_ll)),
        "val_roc_auc_mean": float(np.mean(val_auc)),
        "val_roc_auc_std": float(np.std(val_auc)),
        "train_log_loss_mean": float(np.mean(train_ll)),
        "train_roc_auc_mean": float(np.mean(train_auc)),
        "train_val_log_loss_gap": float(np.mean(train_ll) - np.mean(val_ll)),
        "confident_fn_count": confident_fn,
    }
    if baseline_confident_fn is not None:
        out["delta_confident_fn"] = confident_fn - baseline_confident_fn
    return out


def run_sequential_targeted_block_search(
    train_df: pd.DataFrame,
    y: pd.Series,
    base_feature_cols: list[str],
    xgb_params: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    block_order: list[str] | None = None,
    confidence_threshold: float = 0.9,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    block_order = block_order or TARGETED_BLOCK_ORDER

    baseline_oof, baseline_rows = compute_oof_with_folds(
        train_df[base_feature_cols], y, folds, xgb_params
    )
    baseline_metrics = summarize_oof_metrics(
        y, baseline_oof, baseline_rows, confidence_threshold
    )
    baseline_conf_fn = baseline_metrics["confident_fn_count"]

    current_cols = list(base_feature_cols)
    enabled_blocks: list[str] = []
    current_val_ll = baseline_metrics["val_log_loss_mean"]

    comparison_rows = [
        {
            "scenario": "baseline",
            "block_added": "",
            "kept": True,
            "n_features": len(current_cols),
            **baseline_metrics,
            "delta_confident_fn": 0,
        }
    ]

    for block_name in block_order:
        new_block_cols = [
            c for c in TARGETED_BLOCKS[block_name] if c not in current_cols
        ]
        if not new_block_cols:
            continue

        candidate_cols = current_cols + new_block_cols
        candidate_oof, candidate_rows = compute_oof_with_folds(
            train_df[candidate_cols], y, folds, xgb_params
        )
        candidate_metrics = summarize_oof_metrics(
            y,
            candidate_oof,
            candidate_rows,
            confidence_threshold,
            baseline_confident_fn=baseline_conf_fn,
        )

        kept = candidate_metrics["val_log_loss_mean"] < current_val_ll
        if kept:
            current_cols = candidate_cols
            enabled_blocks.append(block_name)
            current_val_ll = candidate_metrics["val_log_loss_mean"]

        comparison_rows.append(
            {
                "scenario": f"+ {block_name}",
                "block_added": block_name,
                "kept": kept,
                "n_features": len(candidate_cols),
                **candidate_metrics,
            }
        )

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values("val_log_loss_mean")
        .reset_index(drop=True)
    )
    comparison_df.insert(0, "rank", comparison_df.index + 1)
    return comparison_df, current_cols, enabled_blocks


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
