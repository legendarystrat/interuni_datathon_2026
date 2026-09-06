"""NN OOF, classical Optuna, and logistic-stack helpers for feature_eng.ipynb."""

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from model_copy_utils import CV_FOLDS, RANDOM_STATE, build_tuned_xgb
from run_global_search import (
    build_cat_from_trial,
    build_lgb_from_trial,
    build_xgb_from_trial,
    clip_probabilities,
    make_logistic_stack_model,
    make_oof_predictions,
    optimize_logistic_stack,
    stack_feature_matrix,
)

def resolve_device(
    device: str | torch.device | None = None,
    device_id: int = 0,
) -> torch.device:
    """Resolve a PyTorch device for NN training.

    ``device`` may be None / ``\"auto\"`` (CUDA if available), ``\"cpu\"``,
    ``\"cuda\"``, ``\"cuda:N\"``, or a ``torch.device``.
    """
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")
    device_str = str(device).lower()
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device(f"cuda:{device_id}")
    if device_str.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device_str}) but no GPU is available")
        return torch.device(device_str)
    raise ValueError(f"Unknown device: {device!r}")


def describe_device(device: torch.device | None = None) -> str:
    device = device or resolve_device()
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"{device} ({name}, {mem_gb:.1f} GB)"
    return str(device)


DEFAULT_DEVICE = resolve_device()


def _configure_torch_for_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _set_random_seeds(random_state: int, device: torch.device) -> None:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)


class DefaultDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DefaultRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_loss(loss_name: str) -> nn.Module:
    if loss_name == "bce":
        return nn.BCELoss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss_name}")


def _train_nn_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    device: torch.device,
) -> DefaultRegressor:
    hidden_dims = tuple(params["hidden_dims"])
    batch_size = int(params["batch_size"])
    lr = float(params["learning_rate"])
    weight_decay = float(params["weight_decay"])
    max_epochs = int(params["max_epochs"])
    patience = int(params["patience"])
    loss_fn = _build_loss(params["loss_name"])

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_cuda,
    }
    if use_cuda:
        loader_kwargs["num_workers"] = 0

    train_loader = DataLoader(
        DefaultDataset(X_train, y_train),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        DefaultDataset(X_val, y_val),
        shuffle=False,
        **loader_kwargs,
    )

    model = DefaultRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                batch_loss = loss_fn(preds, yb).item()
                val_loss += batch_loss * xb.size(0)
                n_val += xb.size(0)
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_nn(model: DefaultRegressor, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


def compute_nn_oof_cv(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    nn_params: dict,
    device: torch.device | None = None,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    device = resolve_device(device)
    _configure_torch_for_device(device)
    _set_random_seeds(random_state, device)

    oof = np.zeros(len(y), dtype=float)
    y_values = y.astype(float).to_numpy()

    for train_idx, val_idx in folds:
        X_tr = X.iloc[train_idx]
        y_tr = y_values[train_idx]
        X_va = X.iloc[val_idx]

        X_inner_tr, X_inner_va, y_inner_tr, y_inner_va = train_test_split(
            X_tr,
            y_tr,
            test_size=0.15,
            stratify=y_tr,
            random_state=random_state,
        )

        scaler = StandardScaler()
        X_inner_tr_s = scaler.fit_transform(X_inner_tr).astype(np.float32)
        X_inner_va_s = scaler.transform(X_inner_va).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)

        model = _train_nn_fold(
            X_inner_tr_s,
            y_inner_tr.astype(np.float32),
            X_inner_va_s,
            y_inner_va.astype(np.float32),
            nn_params,
            device,
        )
        oof[val_idx] = _predict_nn(model, X_va_s, device)
        if device.type == "cuda":
            del model
            torch.cuda.empty_cache()

    return oof


def default_nn_params() -> dict:
    return {
        "hidden_dims": (64, 32, 16),
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 120,
        "patience": 15,
        "loss_name": "bce",
    }


def make_nn_optuna_objective(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device | None = None,
    random_state: int = RANDOM_STATE,
) -> Callable[[optuna.Trial], float]:
    device = resolve_device(device)
    _configure_torch_for_device(device)

    def objective(trial: optuna.Trial) -> float:
        h1 = trial.suggest_int("hidden_1", 32, 128, step=16)
        h2 = trial.suggest_int("hidden_2", 16, 64, step=8)
        h3 = trial.suggest_int("hidden_3", 8, 32, step=4)
        params = {
            "hidden_dims": (h1, h2, h3),
            "dropout": trial.suggest_float("dropout", 0.05, 0.45),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "max_epochs": 80,
            "patience": 10,
            "loss_name": trial.suggest_categorical("loss_name", ["bce", "mse"]),
        }
        oof = compute_nn_oof_cv(X, y, folds, params, device=device, random_state=random_state)
        loss = log_loss(y.to_numpy(), clip_probabilities(oof))
        trial.set_user_attr("nn_params", params)
        trial.set_user_attr("val_roc_auc_mean", float(roc_auc_score(y, oof)))
        return float(loss)

    return objective


CLASSICAL_BUILDERS = {
    "xgboost": build_xgb_from_trial,
    "lightgbm": build_lgb_from_trial,
    "catboost": build_cat_from_trial,
}


def evaluate_model_oof(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict]:
    y_values = y.to_numpy()
    oof = np.zeros(len(y_values), dtype=float)
    fold_rows: list[dict] = []

    for train_idx, val_idx in folds:
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_prob = fold_model.predict_proba(X.iloc[train_idx])[:, 1]
        val_prob = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = val_prob
        fold_rows.append(
            {
                "train_log_loss": float(log_loss(y_values[train_idx], train_prob)),
                "val_log_loss": float(log_loss(y_values[val_idx], val_prob)),
                "train_roc_auc": float(roc_auc_score(y_values[train_idx], train_prob)),
                "val_roc_auc": float(roc_auc_score(y_values[val_idx], val_prob)),
            }
        )

    metrics = {
        "val_log_loss_mean": float(np.mean([r["val_log_loss"] for r in fold_rows])),
        "val_log_loss_std": float(np.std([r["val_log_loss"] for r in fold_rows])),
        "val_roc_auc_mean": float(np.mean([r["val_roc_auc"] for r in fold_rows])),
        "val_roc_auc_std": float(np.std([r["val_roc_auc"] for r in fold_rows])),
        "train_log_loss_mean": float(np.mean([r["train_log_loss"] for r in fold_rows])),
        "train_roc_auc_mean": float(np.mean([r["train_roc_auc"] for r in fold_rows])),
        "train_val_log_loss_gap": float(
            np.mean([r["train_log_loss"] for r in fold_rows])
            - np.mean([r["val_log_loss"] for r in fold_rows])
        ),
    }
    return oof, metrics


def make_classical_objective(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int = RANDOM_STATE,
) -> Callable[[optuna.Trial], float]:
    builder = CLASSICAL_BUILDERS[model_name]

    def objective(trial: optuna.Trial) -> float:
        model, params = builder(trial, seed)
        _, metrics = evaluate_model_oof(model, X, y, folds)
        trial.set_user_attr("model", model_name)
        trial.set_user_attr("params", params)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics["val_log_loss_mean"]

    return objective


def config_from_classical_trial(model_name: str, trial: optuna.Trial, feature_cols: list[str]) -> dict:
    return {
        "source": "feature_eng_compare",
        "model": model_name,
        "trial_number": int(trial.number),
        "val_log_loss_mean": float(trial.value),
        "val_log_loss_std": float(trial.user_attrs.get("val_log_loss_std", np.nan)),
        "val_roc_auc_mean": float(trial.user_attrs.get("val_roc_auc_mean", np.nan)),
        "feature_label": "feature_eng_joint_best",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "params": trial.user_attrs["params"],
    }


def xgb_config_from_saved(
    feature_cols: list[str],
    xgb_params: dict,
    val_log_loss: float | None = None,
    val_roc_auc: float | None = None,
) -> dict:
    return {
        "source": "feature_eng_joint_best",
        "model": "xgboost",
        "trial_number": "saved",
        "val_log_loss_mean": val_log_loss,
        "val_roc_auc_mean": val_roc_auc,
        "feature_label": "feature_eng_joint_best",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "params": xgb_params,
    }


def build_base_oof_predictions(
    configs: list[dict],
    train_df: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    base_oof: dict[str, np.ndarray] = {}
    for config in configs:
        label = f"{config['model']}_{config.get('trial_number', 'saved')}"
        if config["model"] == "xgboost" and config.get("trial_number") == "saved":
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            fold_list = list(cv.split(train_df[config["feature_cols"]], y))
            oof, _ = evaluate_model_oof(
                build_tuned_xgb(config["params"]),
                train_df[config["feature_cols"]],
                y,
                fold_list,
            )
            base_oof[label] = oof
        else:
            base_oof[label] = make_oof_predictions(
                config,
                train_df,
                y,
                folds=folds,
                seed=seed,
                sample_weight=None,
            )
    return base_oof


def compute_logistic_stack_oof(
    base_oof: dict[str, np.ndarray],
    y: pd.Series,
    stack_config: dict,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    labels = stack_config["labels"]
    input_mode = stack_config["input_mode"]
    y_values = y.to_numpy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed + 3000)
    meta_oof = np.zeros(len(y_values), dtype=float)
    X_meta = stack_feature_matrix(base_oof, labels, input_mode)

    for train_idx, val_idx in cv.split(X_meta, y_values):
        model = make_logistic_stack_model(stack_config["params"], seed)
        model.fit(X_meta[train_idx], y_values[train_idx])
        meta_oof[val_idx] = model.predict_proba(X_meta[val_idx])[:, 1]

    return clip_probabilities(meta_oof)


def summarize_predictions(
    name: str,
    oof: np.ndarray,
    y: pd.Series,
    extra: dict | None = None,
) -> dict:
    y_values = y.to_numpy()
    row = {
        "model": name,
        "val_log_loss_mean": float(log_loss(y_values, clip_probabilities(oof))),
        "val_roc_auc_mean": float(roc_auc_score(y_values, oof)),
    }
    if extra:
        row.update(extra)
    return row


def oof_correlation_matrix(oof_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(oof_dict).corr(method="pearson")
