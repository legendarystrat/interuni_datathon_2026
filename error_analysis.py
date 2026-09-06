"""Reproduce OOF predictions for the global_search_targeted_best blend and
analyze where the model makes its largest errors.

Mirrors the exact CV/blend mechanics in run_global_search.py:
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- per-model OOF via make_oof_predictions
- blend = sum(weight * oof), clipped to [1e-6, 1-1e-6]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from feature_eng_lib import engineer_all_features
from model_copy_utils import load_data

ID_COL = "client_id"
TARGET_COL = "default"
SEED = 42
FOLDS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_model_from_config(config: dict, seed: int):
    model_name = config["model"]
    params = config["params"]
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1, **params,
        )
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss",
            random_state=seed, n_jobs=-1, verbose=-1, **params,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss", eval_metric="Logloss",
            random_state=seed, verbose=False, allow_writing_files=False,
            thread_count=-1, **params,
        )
    raise ValueError(model_name)


def make_oof_predictions(config, train_df, y, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_values = y.to_numpy()
    oof = np.zeros(len(y_values), dtype=float)
    X = train_df[config["feature_cols"]]
    model = build_model_from_config(config, seed)
    for train_idx, val_idx in cv.split(X, y_values):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def clip_probabilities(probs, eps=1e-6):
    return np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)


def blend_predictions(base_predictions, weights):
    first = next(iter(base_predictions.values()))
    blended = np.zeros(len(first), dtype=float)
    for label, weight in weights.items():
        if weight <= 0:
            continue
        blended += weight * base_predictions[label]
    return clip_probabilities(blended)


def main():
    log("loading data")
    train_df = load_data("train.csv")
    test_df = load_data("test.csv")
    y = train_df[TARGET_COL]

    train_fe = engineer_all_features(train_df)
    test_fe = engineer_all_features(test_df)

    best_cfg = json.load(open("global_search_targeted_best.json"))
    configs = best_cfg["config_by_blend_label"]
    weights = best_cfg["blend"]["weights"]
    assert best_cfg["selected_submission"] == "blend"

    labels = list(weights.keys())
    log(f"{len(labels)} base models in blend, weight sum={sum(weights.values()):.6f}")

    base_oof = {}
    base_test = {}
    per_model_metrics = []
    for i, label in enumerate(labels):
        cfg = configs[label]
        t0 = time.time()
        oof = make_oof_predictions(cfg, train_fe, y, FOLDS, SEED)
        base_oof[label] = oof
        ll = log_loss(y, clip_probabilities(oof))
        auc = roc_auc_score(y, oof)
        # full fit for test predictions
        model = build_model_from_config(cfg, SEED)
        model.fit(train_fe[cfg["feature_cols"]], y)
        test_pred = model.predict_proba(test_fe[cfg["feature_cols"]])[:, 1]
        base_test[label] = test_pred
        dt = time.time() - t0
        log(f"[{i+1}/{len(labels)}] {label}: oof_logloss={ll:.6f} oof_auc={auc:.6f} "
            f"weight={weights[label]:.4f} ({dt:.1f}s)")
        per_model_metrics.append({
            "label": label, "model": cfg["model"], "weight": weights[label],
            "oof_log_loss": ll, "oof_auc": auc,
            "reported_val_log_loss_mean": cfg["val_log_loss_mean"],
            "reported_val_roc_auc_mean": cfg["val_roc_auc_mean"],
        })

    blended_oof = blend_predictions(base_oof, weights)
    blended_test = blend_predictions(base_test, weights)

    blend_ll = log_loss(y, blended_oof)
    blend_auc = roc_auc_score(y, blended_oof)
    log(f"REPRODUCED blended OOF log_loss={blend_ll:.6f} auc={blend_auc:.6f}")
    log(f"Reported blend val_log_loss_mean={best_cfg['blend']['val_log_loss_mean']:.6f} "
        f"(note: reported is mean of per-fold log-loss computed on the SAME oof array "
        f"split by fold, not a single log_loss call on the whole vector)")

    # per-row log loss
    row_ll = -(y.to_numpy() * np.log(blended_oof) + (1 - y.to_numpy()) * np.log(1 - blended_oof))

    out = train_fe.copy()
    out["y_true"] = y.to_numpy()
    out["oof_pred"] = blended_oof
    out["row_log_loss"] = row_ll
    out.to_csv("error_analysis_oof.csv", index=False)

    # compare reproduced test blend vs actual submission file
    sub = pd.read_csv("submission_global_targeted_blend.csv")
    sub_sorted = sub.set_index(ID_COL).loc[test_df[ID_COL]]["default_probability"].to_numpy()
    diff = np.abs(blended_test - sub_sorted)
    log(f"Reproduction check vs submission_global_targeted_blend.csv: "
        f"max_abs_diff={diff.max():.6f} mean_abs_diff={diff.mean():.6f}")

    test_out = test_df[[ID_COL]].copy()
    test_out["reproduced_pred"] = blended_test
    test_out["submitted_pred"] = sub_sorted
    test_out.to_csv("error_analysis_test_preds.csv", index=False)

    summary = {
        "per_model_metrics": per_model_metrics,
        "reproduced_blend_oof_log_loss": blend_ll,
        "reproduced_blend_oof_auc": blend_auc,
        "reported_blend_val_log_loss_mean": best_cfg["blend"]["val_log_loss_mean"],
        "reported_blend_val_roc_auc_mean": best_cfg["blend"]["val_roc_auc_mean"],
        "reproduction_check_max_abs_diff": float(diff.max()),
        "reproduction_check_mean_abs_diff": float(diff.mean()),
        "train_prior_default_rate": float(y.mean()),
        "test_pred_stats": {
            "mean": float(blended_test.mean()),
            "min": float(blended_test.min()),
            "max": float(blended_test.max()),
            "std": float(blended_test.std()),
        },
        "oof_pred_stats": {
            "mean": float(blended_oof.mean()),
            "min": float(blended_oof.min()),
            "max": float(blended_oof.max()),
            "std": float(blended_oof.std()),
        },
    }
    json.dump(summary, open("error_analysis_summary.json", "w"), indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
