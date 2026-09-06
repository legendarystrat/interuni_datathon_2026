"""Broader model search plus top-trial blend for the default prediction task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_eng_lib import ALL_ABLATION_BLOCKS, build_feature_cols_from_blocks
from feature_eng_lib import engineer_all_features
from model_copy_utils import RANDOM_STATE, load_data


MODEL_NAMES = ["xgboost", "lightgbm", "catboost"]
ID_COL = "client_id"
TARGET_COL = "default"
EXCLUDE_COLS = [ID_COL, TARGET_COL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--storage", default="sqlite:///optuna_global_search.db")
    parser.add_argument("--output", type=Path, default=Path("global_search_best.json"))
    parser.add_argument("--submission", type=Path, default=Path("submission_global_blend.csv"))
    parser.add_argument("--trials-per-model", type=int, default=150)
    parser.add_argument("--blend-trials", type=int, default=500)
    parser.add_argument("--top-k-per-model", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--search-models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--sample-weights", type=Path)
    parser.add_argument("--no-existing-configs", action="store_true")
    parser.add_argument("--skip-model-search", action="store_true")
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_sample_weights(path: Path | None, train_df: pd.DataFrame) -> np.ndarray | None:
    if path is None:
        return None
    weights_df = pd.read_csv(path)
    required = {ID_COL, "adversarial_weight"}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    aligned = train_df[[ID_COL]].merge(weights_df, on=ID_COL, how="left")
    weights = aligned["adversarial_weight"].fillna(1.0).to_numpy(dtype=float)
    return weights / weights.mean()


def model_config_from_saved(
    model_name: str,
    saved_config: dict,
    source: str,
) -> dict:
    return {
        "source": source,
        "model": model_name,
        "trial_number": saved_config.get("trial_number", "saved"),
        "val_log_loss_mean": float(saved_config["val_log_loss_mean"]),
        "val_log_loss_std": float(saved_config.get("val_log_loss_std", np.nan)),
        "val_roc_auc_mean": float(saved_config["val_roc_auc_mean"]),
        "feature_label": saved_config.get("feature_label", source),
        "feature_cols": saved_config["feature_cols"],
        "n_features": len(saved_config["feature_cols"]),
        "params": saved_config["params"],
    }


def feature_eng_config_from_saved(saved_config: dict) -> dict:
    return {
        "source": "feature_eng_joint",
        "model": "xgboost",
        "trial_number": saved_config.get("best_trial_number", "saved"),
        "val_log_loss_mean": float(saved_config["best_val_log_loss_mean"]),
        "val_log_loss_std": float(saved_config.get("best_val_log_loss_std", np.nan)),
        "val_roc_auc_mean": float(saved_config["best_val_roc_auc_mean"]),
        "feature_label": "feature_eng_joint_best",
        "feature_cols": saved_config["best_feature_cols"],
        "n_features": int(saved_config["n_features"]),
        "params": saved_config["best_xgb_params"],
    }


def load_existing_configs(models: list[str]) -> list[dict]:
    configs: list[dict] = []

    feature_eng_best = load_json(Path("feature_eng_joint_best.json"))
    if feature_eng_best is not None and "xgboost" in models:
        configs.append(feature_eng_config_from_saved(feature_eng_best))

    for path, source in [
        (Path("optuna_best_extended.json"), "optuna_extended"),
        (Path("optuna_best.json"), "optuna_original"),
    ]:
        payload = load_json(path)
        if payload is None:
            continue
        for model_name, saved_config in payload.items():
            if model_name in models:
                configs.append(model_config_from_saved(model_name, saved_config, source))

    return configs


def config_label(config: dict) -> str:
    raw = (
        f"{config.get('source', 'global')}_{config['model']}"
        f"_trial_{config['trial_number']}_{config['feature_label'][:32]}"
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def choose_feature_cols(
    trial: optuna.Trial,
    all_feature_cols: list[str],
    exported_best_cols: list[str] | None,
) -> tuple[list[str], str]:
    choices = ["all_features", "custom_blocks"]
    if exported_best_cols:
        choices.insert(1, "exported_feature_eng_best")

    mode = trial.suggest_categorical("feature_mode", choices)
    if mode == "all_features":
        return all_feature_cols, "all_features"
    if mode == "exported_feature_eng_best":
        return exported_best_cols or all_feature_cols, "exported_feature_eng_best"

    active_blocks = {
        block_name: bool(trial.suggest_int(f"block_{block_name}", 0, 1))
        for block_name in ALL_ABLATION_BLOCKS
    }
    if not any(active_blocks.values()):
        raise optuna.TrialPruned("No feature blocks selected")

    cols = build_feature_cols_from_blocks(active_blocks)
    if not cols:
        raise optuna.TrialPruned("No feature columns selected")
    enabled_blocks = [name for name, enabled in active_blocks.items() if enabled]
    return cols, f"custom_blocks({','.join(enabled_blocks)})"


def build_xgb_from_trial(trial: optuna.Trial, seed: int) -> tuple[xgb.XGBClassifier, dict]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 25),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    return model, params


def build_lgb_from_trial(trial: optuna.Trial, seed: int) -> tuple[lgb.LGBMClassifier, dict]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 8, 96),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 250),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 2.0, log=True),
    }
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    return model, params


def build_cat_from_trial(
    trial: optuna.Trial,
    seed: int,
) -> tuple[CatBoostClassifier, dict]:
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2500, step=100),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bootstrap_type": bootstrap_type,
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 5.0
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.55, 1.0)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_state=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **params,
    )
    return model, params


TRIAL_BUILDERS: dict[str, Callable[[optuna.Trial, int], tuple[object, dict]]] = {
    "xgboost": build_xgb_from_trial,
    "lightgbm": build_lgb_from_trial,
    "catboost": build_cat_from_trial,
}


def build_model_from_config(config: dict, seed: int) -> object:
    model_name = config["model"]
    params = config["params"]
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            random_state=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **params,
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_cv(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
    sample_weight: np.ndarray | None,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_values = y.to_numpy()
    train_log_losses: list[float] = []
    val_log_losses: list[float] = []
    train_aucs: list[float] = []
    val_aucs: list[float] = []

    for train_idx, val_idx in cv.split(X, y_values):
        fold_model = clone(model)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_idx]
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)

        train_prob = fold_model.predict_proba(X.iloc[train_idx])[:, 1]
        val_prob = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        train_log_losses.append(log_loss(y_values[train_idx], train_prob))
        val_log_losses.append(log_loss(y_values[val_idx], val_prob))
        train_aucs.append(roc_auc_score(y_values[train_idx], train_prob))
        val_aucs.append(roc_auc_score(y_values[val_idx], val_prob))

    return {
        "train_log_loss_mean": float(np.mean(train_log_losses)),
        "val_log_loss_mean": float(np.mean(val_log_losses)),
        "val_log_loss_std": float(np.std(val_log_losses)),
        "train_roc_auc_mean": float(np.mean(train_aucs)),
        "val_roc_auc_mean": float(np.mean(val_aucs)),
        "val_roc_auc_std": float(np.std(val_aucs)),
    }


def make_objective(
    model_name: str,
    train_df: pd.DataFrame,
    y: pd.Series,
    all_feature_cols: list[str],
    exported_best_cols: list[str] | None,
    folds: int,
    seed: int,
    sample_weight: np.ndarray | None,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        feature_cols, feature_label = choose_feature_cols(
            trial,
            all_feature_cols,
            exported_best_cols,
        )
        model, params = TRIAL_BUILDERS[model_name](trial, seed)
        metrics = evaluate_cv(
            model,
            train_df[feature_cols],
            y,
            folds=folds,
            seed=seed,
            sample_weight=sample_weight,
        )

        trial.set_user_attr("model", model_name)
        trial.set_user_attr("feature_label", feature_label)
        trial.set_user_attr("feature_cols", feature_cols)
        trial.set_user_attr("n_features", len(feature_cols))
        trial.set_user_attr("params", params)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics["val_log_loss_mean"]

    return objective


def config_from_trial(model_name: str, trial: optuna.Trial) -> dict:
    return {
        "source": "global_search",
        "model": model_name,
        "trial_number": int(trial.number),
        "val_log_loss_mean": float(trial.value),
        "val_log_loss_std": float(trial.user_attrs["val_log_loss_std"]),
        "val_roc_auc_mean": float(trial.user_attrs["val_roc_auc_mean"]),
        "feature_label": trial.user_attrs["feature_label"],
        "feature_cols": trial.user_attrs["feature_cols"],
        "n_features": int(trial.user_attrs["n_features"]),
        "params": trial.user_attrs["params"],
    }


def top_trial_configs(study: optuna.Study, model_name: str, top_k: int) -> list[dict]:
    if top_k <= 0:
        return []

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    completed.sort(key=lambda trial: (float(trial.value), -trial.number))
    return [config_from_trial(model_name, trial) for trial in completed[:top_k]]


def make_oof_predictions(
    config: dict,
    train_df: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_values = y.to_numpy()
    oof = np.zeros(len(y_values), dtype=float)
    X = train_df[config["feature_cols"]]
    model = build_model_from_config(config, seed)

    for train_idx, val_idx in cv.split(X, y_values):
        fold_model = clone(model)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_idx]
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        oof[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]

    return oof


def optimize_blend(
    base_oof: dict[str, np.ndarray],
    y: pd.Series,
    folds: int,
    seed: int,
    n_trials: int,
    show_progress: bool,
) -> dict:
    labels = list(base_oof)
    y_values = y.to_numpy()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def score_blend(weights: dict[str, float]) -> tuple[float, float]:
        blended = np.zeros(len(y_values), dtype=float)
        for label, weight in weights.items():
            blended += weight * base_oof[label]
        fold_losses = [
            log_loss(y_values[val_idx], blended[val_idx])
            for _, val_idx in cv.split(blended, y_values)
        ]
        fold_aucs = [
            roc_auc_score(y_values[val_idx], blended[val_idx])
            for _, val_idx in cv.split(blended, y_values)
        ]
        return float(np.mean(fold_losses)), float(np.mean(fold_aucs))

    def objective(trial: optuna.Trial) -> float:
        raw_weights = np.array(
            [trial.suggest_float(f"w_{idx}", 0.0, 1.0) for idx in range(len(labels))]
        )
        if np.all(raw_weights == 0):
            raise optuna.TrialPruned("All blend weights are zero")
        weights_arr = raw_weights / raw_weights.sum()
        weights = {label: float(weight) for label, weight in zip(labels, weights_arr)}
        loss, auc = score_blend(weights)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("val_roc_auc_mean", auc)
        return loss

    study = optuna.create_study(
        study_name="global_top_trial_blend",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + 1000),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    best_weights = study.best_trial.user_attrs["weights"]
    best_loss, best_auc = score_blend(best_weights)
    return {
        "trial_number": int(study.best_trial.number),
        "val_log_loss_mean": best_loss,
        "val_roc_auc_mean": best_auc,
        "weights": best_weights,
    }


def predict_config(
    config: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    seed: int,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    model = build_model_from_config(config, seed)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train_df[config["feature_cols"]], y, **fit_kwargs)
    return model.predict_proba(test_df[config["feature_cols"]])[:, 1]


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_models = args.search_models or args.models
    unexpected_search_models = sorted(set(search_models) - set(args.models))
    if unexpected_search_models:
        raise ValueError(
            "--search-models must be a subset of --models; "
            f"got extra models: {unexpected_search_models}"
        )

    train_df = engineer_all_features(load_data(args.train))
    test_df = engineer_all_features(load_data(args.test))
    y = train_df[TARGET_COL]
    all_feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

    exported_best = load_json(Path("feature_eng_joint_best.json"))
    exported_best_cols = None
    if exported_best is not None:
        exported_best_cols = exported_best.get("best_feature_cols")

    sample_weight = load_sample_weights(args.sample_weights, train_df)
    if sample_weight is not None:
        print(
            "Loaded sample weights: "
            f"min={sample_weight.min():.4f}, "
            f"mean={sample_weight.mean():.4f}, "
            f"max={sample_weight.max():.4f}"
        )

    studies: dict[str, optuna.Study] = {}
    for idx, model_name in enumerate(search_models):
        study = optuna.create_study(
            study_name=f"global_{model_name}",
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed + idx),
        )
        studies[model_name] = study
        if not args.skip_model_search:
            completed = sum(
                1
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            )
            remaining = max(0, args.trials_per_model - completed)
            print(f"{model_name}: {completed}/{args.trials_per_model} complete")
            if remaining:
                study.optimize(
                    make_objective(
                        model_name=model_name,
                        train_df=train_df,
                        y=y,
                        all_feature_cols=all_feature_cols,
                        exported_best_cols=exported_best_cols,
                        folds=args.folds,
                        seed=args.seed,
                        sample_weight=sample_weight,
                    ),
                    n_trials=remaining,
                    show_progress_bar=args.show_progress,
                )

    top_configs: list[dict] = []
    if not args.no_existing_configs:
        existing_configs = load_existing_configs(args.models)
        top_configs.extend(existing_configs)
        print(f"Loaded {len(existing_configs)} existing strong configs")

    best_by_model: dict[str, dict] = {}
    for model_name, study in studies.items():
        configs = top_trial_configs(study, model_name, args.top_k_per_model)
        if not configs:
            print(f"{model_name}: no completed trials")
            continue
        top_configs.extend(configs)
        best_by_model[model_name] = configs[0]
        print(
            f"{model_name}: best trial {configs[0]['trial_number']} "
            f"log_loss={configs[0]['val_log_loss_mean']:.6f} "
            f"auc={configs[0]['val_roc_auc_mean']:.6f} "
            f"features={configs[0]['n_features']}"
        )

    if not top_configs:
        raise RuntimeError("No completed model trials available for blending")

    base_oof: dict[str, np.ndarray] = {}
    for config in top_configs:
        label = config_label(config)
        print(f"Building OOF predictions: {label}")
        base_oof[label] = make_oof_predictions(
            config,
            train_df,
            y,
            folds=args.folds,
            seed=args.seed,
            sample_weight=sample_weight,
        )

    blend = optimize_blend(
        base_oof,
        y,
        folds=args.folds,
        seed=args.seed,
        n_trials=args.blend_trials,
        show_progress=args.show_progress,
    )

    config_by_label = {config_label(config): config for config in top_configs}
    output_payload = {
        "best_by_model": best_by_model,
        "top_configs": top_configs,
        "blend": blend,
        "config_by_blend_label": config_by_label,
        "used_sample_weights": str(args.sample_weights) if args.sample_weights else None,
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
        f.write("\n")

    print(
        f"Best blend log_loss={blend['val_log_loss_mean']:.6f}, "
        f"auc={blend['val_roc_auc_mean']:.6f}"
    )
    print(f"Saved {args.output}")

    if args.make_submission:
        blended_test = np.zeros(len(test_df), dtype=float)
        for label, weight in blend["weights"].items():
            if weight <= 0:
                continue
            blended_test += weight * predict_config(
                config_by_label[label],
                train_df,
                test_df,
                y,
                seed=args.seed,
                sample_weight=sample_weight,
            )
        submission = pd.DataFrame(
            {
                ID_COL: test_df[ID_COL],
                "default_probability": blended_test,
            }
        )
        submission.to_csv(args.submission, index=False)
        print(f"Saved {args.submission} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
