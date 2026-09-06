"""
PyTorch MLP for predicting probability of credit default.

Uses shared NN code from model_comparison_lib (DefaultRegressor, training loop).
Set NN_DEVICE to "auto", "cuda", "cuda:0", or "cpu".

Requirements:
    pip install torch pandas numpy scikit-learn joblib
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_comparison_lib import (
    _predict_nn,
    _train_nn_fold,
    default_nn_params,
    describe_device,
    resolve_device,
)

RANDOM_STATE = 42
DATA_PATH = "train.csv"
MODEL_DIR = Path("artifacts_file")

# "auto" | "cuda" | "cuda:0" | "cpu" — or set env NN_DEVICE
NN_DEVICE = os.environ.get("NN_DEVICE", "auto")
DEVICE = resolve_device(NN_DEVICE)
print(f"Using device: {describe_device(DEVICE)}")

df = pd.read_csv(DATA_PATH)
target_col = "default"
id_col = "client_id"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    X["AVG_BILL_AMT"] = X[bill_cols].mean(axis=1)
    X["AVG_PAY_AMT"] = X[pay_amt_cols].mean(axis=1)
    X["UTILIZATION"] = X["AVG_BILL_AMT"] / X["LIMIT_BAL"].replace(0, 1)
    X["PAY_TO_BILL_RATIO"] = X["AVG_PAY_AMT"] / (X["AVG_BILL_AMT"].abs() + 1)
    X["MAX_DELAY"] = X[pay_status_cols].max(axis=1)
    X["NUM_MONTHS_LATE"] = (X[pay_status_cols] > 0).sum(axis=1)
    return X


feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
X = engineer_features(df[feature_cols])
y = df[target_col].astype(float)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

nn_params = default_nn_params()
nn_params["loss_name"] = "mse"
nn_params["max_epochs"] = 200
nn_params["patience"] = 20

model = _train_nn_fold(
    X_train_scaled,
    y_train.values.astype(np.float32),
    X_val_scaled,
    y_val.values.astype(np.float32),
    nn_params,
    DEVICE,
)


def evaluate(X_scaled: np.ndarray, y_true: np.ndarray, name: str):
    preds = _predict_nn(model, X_scaled, DEVICE)
    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    auc = roc_auc_score(y_true, preds)
    print(f"\n--- {name} ---")
    print(f"MSE: {mse:.5f} | MAE: {mae:.5f} | ROC-AUC: {auc:.4f}")
    return preds, mse, mae, auc


_ = evaluate(X_train_scaled, y_train.values, "Train")
_, val_mse, val_mae, val_auc = evaluate(X_val_scaled, y_val.values, "Validation")
_, test_mse, test_mae, test_auc = evaluate(X_test_scaled, y_test.values, "Test (held-out)")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "pytorch_model.pt")
joblib.dump(scaler, MODEL_DIR / "pytorch_scaler.joblib")

metadata = {
    "feature_cols": list(X.columns),
    "hidden_dims": list(nn_params["hidden_dims"]),
    "dropout": nn_params["dropout"],
    "loss_fn": nn_params["loss_name"],
    "device": str(DEVICE),
    "test_mse": float(test_mse),
    "test_mae": float(test_mae),
    "test_roc_auc": float(test_auc),
    "input_dim": X_train_scaled.shape[1],
}
with open(MODEL_DIR / "pytorch_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved pytorch_model.pt, pytorch_scaler.joblib, pytorch_metadata.json")
