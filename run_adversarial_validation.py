"""Run train-vs-test adversarial validation and save reusable shift artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_eng_lib import engineer_all_features
from model_copy_utils import RANDOM_STATE, load_data


DEFAULT_OUTPUT_DIR = Path("adversarial_outputs")
EXCLUDE_COLS = ["client_id", "default"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--material-threshold", type=float, default=0.55)
    parser.add_argument("--weight-clip-low", type=float, default=0.25)
    parser.add_argument("--weight-clip-high", type=float, default=4.0)
    return parser.parse_args()


def build_shift_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )


def cross_validated_shift_predictions(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_prob = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = val_prob
        fold_auc = float(roc_auc_score(y.iloc[val_idx], val_prob))
        fold_aucs.append(fold_auc)
        print(f"Fold {fold_idx} ROC-AUC: {fold_auc:.6f}")

    return oof, fold_aucs


def make_importance_weights(
    train_probs: pd.Series,
    n_train: int,
    n_test: int,
    clip_low: float,
    clip_high: float,
) -> pd.Series:
    probs = train_probs.clip(1e-6, 1 - 1e-6)
    odds = probs / (1.0 - probs)

    # Correct the adversarial prior. With 24k train and 6k test rows, a neutral
    # row has P(is_test)=0.2 and receives weight 1 after this adjustment.
    weights = odds * (n_train / n_test)
    weights = weights.clip(clip_low, clip_high)
    return weights / weights.mean()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = engineer_all_features(load_data(args.train))
    test_df = engineer_all_features(load_data(args.test))
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    missing_in_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test set missing engineered columns: {missing_in_test}")

    train_adv = train_df[feature_cols].copy()
    train_adv["is_test"] = 0
    train_adv["client_id"] = train_df["client_id"].values

    test_adv = test_df[feature_cols].copy()
    test_adv["is_test"] = 1
    test_adv["client_id"] = test_df["client_id"].values

    adv_df = pd.concat([train_adv, test_adv], ignore_index=True)
    X_adv = adv_df[feature_cols]
    y_adv = adv_df["is_test"]

    print(f"train rows: {len(train_df):,}")
    print(f"test rows:  {len(test_df):,}")
    print(f"shift features: {len(feature_cols)}")

    shift_model = build_shift_model(args.seed)
    oof_prob, fold_aucs = cross_validated_shift_predictions(
        shift_model,
        X_adv,
        y_adv,
        folds=args.folds,
        seed=args.seed,
    )

    adv_df = adv_df[["client_id", "is_test"]].copy()
    adv_df["oof_test_prob"] = oof_prob
    train_oof = adv_df[adv_df["is_test"] == 0].copy()
    test_oof = adv_df[adv_df["is_test"] == 1].copy()

    shift_auc_mean = float(np.mean(fold_aucs))
    shift_auc_std = float(np.std(fold_aucs))
    material_shift = shift_auc_mean > args.material_threshold

    train_weights = make_importance_weights(
        train_probs=train_oof["oof_test_prob"],
        n_train=len(train_df),
        n_test=len(test_df),
        clip_low=args.weight_clip_low,
        clip_high=args.weight_clip_high,
    )
    train_weight_df = pd.DataFrame(
        {
            "client_id": train_oof["client_id"].values,
            "oof_test_prob": train_oof["oof_test_prob"].values,
            "adversarial_weight": train_weights.values,
        }
    )

    full_model = build_shift_model(args.seed)
    full_model.fit(X_adv, y_adv)
    importance = (
        pd.Series(full_model.feature_importances_, index=feature_cols, name="importance")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    adv_df.to_csv(args.output_dir / "adversarial_oof.csv", index=False)
    train_weight_df.to_csv(args.output_dir / "adversarial_train_weights.csv", index=False)
    importance.to_csv(args.output_dir / "adversarial_feature_importance.csv", index=False)

    summary = {
        "fold_aucs": fold_aucs,
        "shift_auc_mean": shift_auc_mean,
        "shift_auc_std": shift_auc_std,
        "material_threshold": args.material_threshold,
        "material_shift": material_shift,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "train_oof_test_prob_mean": float(train_oof["oof_test_prob"].mean()),
        "test_oof_test_prob_mean": float(test_oof["oof_test_prob"].mean()),
        "weight_clip_low": args.weight_clip_low,
        "weight_clip_high": args.weight_clip_high,
        "weight_mean_after_normalization": float(train_weights.mean()),
        "weight_min": float(train_weights.min()),
        "weight_max": float(train_weights.max()),
        "top_shift_features": importance.head(20).to_dict(orient="records"),
    }
    with (args.output_dir / "adversarial_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"\nMean ROC-AUC: {shift_auc_mean:.6f}")
    print(f"Std  ROC-AUC: {shift_auc_std:.6f}")
    print(f"Material shift (>{args.material_threshold}): {material_shift}")
    print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
