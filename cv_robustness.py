"""Check whether the blend's edge over the best single model survives a
different CV fold assignment (repeated-CV robustness check).

For each seed in SEEDS, recompute OOF predictions for all 17 base models
under StratifiedKFold(5, shuffle=True, random_state=seed), then compare:
  - the blend using the ORIGINAL saved weights (fit against seed=42)
  - an equal-weight blend (naive baseline)
  - the single best individual model

If the saved-weight blend's edge over "best single model" shrinks or
flips sign on other seeds, the original blend-weight search overfit to
the seed=42 fold assignment.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from feature_eng_lib import engineer_all_features
from model_copy_utils import load_data

TARGET_COL = "default"
FOLDS = 5
SEEDS = [42, 7, 123]  # in addition to the original seed=42 already computed

OUT_PATH = "cv_robustness_results.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_model_from_config(config, seed):
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


def make_oof_predictions(config, train_df, y, folds, cv_seed, model_seed=42):
    # NOTE: cv_seed controls the fold split (what we're stress-testing);
    # model_seed stays fixed at 42 (the original model's own random_state),
    # matching how make_oof_predictions in run_global_search.py builds the
    # model but letting us vary only the fold assignment.
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    y_values = y.to_numpy()
    oof = np.zeros(len(y_values), dtype=float)
    X = train_df[config["feature_cols"]]
    model = build_model_from_config(config, model_seed)
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
    y = train_df[TARGET_COL]
    train_fe = engineer_all_features(train_df)

    best_cfg = json.load(open("global_search_targeted_best.json"))
    configs = best_cfg["config_by_blend_label"]
    saved_weights = best_cfg["blend"]["weights"]
    labels = list(saved_weights.keys())
    equal_weights = {label: 1.0 / len(labels) for label in labels}

    results = json.load(open(OUT_PATH)) if __import__("os").path.exists(OUT_PATH) else {}

    for seed in SEEDS:
        if str(seed) in results:
            log(f"seed {seed} already done, skipping")
            continue
        log(f"=== seed {seed} ===")
        base_oof = {}
        per_model = {}
        for i, label in enumerate(labels):
            cfg = configs[label]
            t0 = time.time()
            oof = make_oof_predictions(cfg, train_fe, y, FOLDS, cv_seed=seed)
            base_oof[label] = oof
            ll = log_loss(y, clip_probabilities(oof))
            per_model[label] = ll
            log(f"  [{i+1}/{len(labels)}] {label}: logloss={ll:.6f} ({time.time()-t0:.1f}s)")

        saved_blend = blend_predictions(base_oof, saved_weights)
        equal_blend = blend_predictions(base_oof, equal_weights)
        best_single_label = min(per_model, key=per_model.get)

        seed_result = {
            "per_model_log_loss": per_model,
            "saved_weight_blend_log_loss": log_loss(y, saved_blend),
            "equal_weight_blend_log_loss": log_loss(y, equal_blend),
            "best_single_model_label": best_single_label,
            "best_single_model_log_loss": per_model[best_single_label],
        }
        results[str(seed)] = seed_result
        json.dump(results, open(OUT_PATH, "w"), indent=2)
        log(f"seed {seed}: saved_blend={seed_result['saved_weight_blend_log_loss']:.6f} "
            f"equal_blend={seed_result['equal_weight_blend_log_loss']:.6f} "
            f"best_single={seed_result['best_single_model_log_loss']:.6f} ({best_single_label})")

    log("DONE")


if __name__ == "__main__":
    main()
