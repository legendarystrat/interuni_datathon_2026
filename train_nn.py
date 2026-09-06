"""
PyTorch regression model for predicting probability of credit default.

NOTE ON FRAMING:
The raw target ('default') is binary (0 = no default, 1 = default).
This script treats the task as REGRESSION rather than classification:
  - The network has a single output neuron passed through a sigmoid,
    producing a continuous value in [0, 1].
  - Loss is Mean Squared Error (MSE) against the 0/1 label, rather than
    Binary Cross-Entropy (the usual choice for binary targets).
This gives you a continuous "probability-like" score via regression
mechanics rather than a classifier's probability calibration. If you'd
rather use BCELoss (arguably more principled for a 0/1 target), swap
the loss function marked below — the rest of the pipeline is unaffected.

Requirements (install on your end):
    pip install torch pandas numpy scikit-learn joblib
"""
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_PATH = 'train.csv'   # update path as needed
MODEL_DIR = 'artifacts_file'           # where artifacts get saved

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv('train.csv')

target_col = 'default'
id_col = 'client_id'
feature_cols = [c for c in df.columns if c not in [target_col, id_col]]

X = df[feature_cols].copy()
y = df[target_col].astype(float).copy()  # cast to float for regression

# ---------------------------------------------------------------
# 2. Feature engineering (same as the classification version)
# ---------------------------------------------------------------
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['EDUCATION'] = X['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    X['MARRIAGE'] = X['MARRIAGE'].replace({0: 3})

    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    X['AVG_BILL_AMT'] = X[bill_cols].mean(axis=1)
    X['AVG_PAY_AMT'] = X[pay_amt_cols].mean(axis=1)
    X['UTILIZATION'] = X['AVG_BILL_AMT'] / X['LIMIT_BAL'].replace(0, 1)
    X['PAY_TO_BILL_RATIO'] = X['AVG_PAY_AMT'] / (X['AVG_BILL_AMT'].abs() + 1)
    X['MAX_DELAY'] = X[pay_status_cols].max(axis=1)
    X['NUM_MONTHS_LATE'] = (X[pay_status_cols] > 0).sum(axis=1)
    return X

X = engineer_features(X)
feature_cols = list(X.columns)

# ---------------------------------------------------------------
# 3. Train / val / test split
# ---------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 4. Scale features
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_arr = y_train.values.astype(np.float32)
y_val_arr = y_val.values.astype(np.float32)
y_test_arr = y_test.values.astype(np.float32)

# ---------------------------------------------------------------
# 5. PyTorch Dataset / DataLoader
# ---------------------------------------------------------------
class DefaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = DefaultDataset(X_train_scaled, y_train_arr)
val_ds = DefaultDataset(X_val_scaled, y_val_arr)
test_ds = DefaultDataset(X_test_scaled, y_test_arr)

BATCH_SIZE = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------
# 6. Model definition: MLP regressor with sigmoid output
# ---------------------------------------------------------------
class DefaultRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # keeps output in [0, 1] like a probability
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


input_dim = X_train_scaled.shape[1]
model = DefaultRegressor(input_dim).to(DEVICE)
print(model)

# ---------------------------------------------------------------
# 7. Loss & optimizer
#    Using MSELoss for a true "regression" framing.
#    Swap to nn.BCELoss() here if you'd prefer classification-style
#    training (often gives better-calibrated probabilities).
# ---------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ---------------------------------------------------------------
# 8. Training loop with early stopping
# ---------------------------------------------------------------
N_EPOCHS = 200
PATIENCE = 20

best_val_loss = float('inf')
epochs_no_improve = 0
best_state = None

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_ds)

    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train MSE: {train_loss:.5f} | val MSE: {val_loss:.5f}")

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val MSE: {best_val_loss:.5f})")
            break

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state)

# ---------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------
def evaluate(loader, y_true, name):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy().flatten()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds)

    mse = mean_squared_error(y_true, all_preds)
    mae = mean_absolute_error(y_true, all_preds)
    auc = roc_auc_score(y_true, all_preds)  # still meaningful: ranks probabilities

    print(f"\n--- {name} ---")
    print(f"MSE: {mse:.5f} | MAE: {mae:.5f} | ROC-AUC: {auc:.4f}")
    return all_preds, mse, mae, auc


_ = evaluate(train_loader, y_train_arr, "Train")
val_preds, val_mse, val_mae, val_auc = evaluate(val_loader, y_val_arr, "Validation")
test_preds, test_mse, test_mae, test_auc = evaluate(test_loader, y_test_arr, "Test (held-out)")

# ---------------------------------------------------------------
# 10. Save artifacts
# ---------------------------------------------------------------
torch.save(model.state_dict(), f'{MODEL_DIR}/pytorch_model.pt')
joblib.dump(scaler, f'{MODEL_DIR}/pytorch_scaler.joblib')

metadata = {
    'feature_cols': feature_cols,
    'hidden_dims': [64, 32, 16],
    'dropout': 0.2,
    'loss_fn': 'MSELoss (regression to 0/1 target)',
    'test_mse': float(test_mse),
    'test_mae': float(test_mae),
    'test_roc_auc': float(test_auc),
    'input_dim': input_dim,
}
with open(f'{MODEL_DIR}/pytorch_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nSaved pytorch_model.pt, pytorch_scaler.joblib, pytorch_metadata.json")