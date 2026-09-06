import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

OOF_PATH = Path("error_analysis_oof.csv")
TEST_PREDS_PATH = Path("error_analysis_test_preds.csv")

if not OOF_PATH.exists() or not TEST_PREDS_PATH.exists():
    missing = [str(path) for path in [OOF_PATH, TEST_PREDS_PATH] if not path.exists()]
    raise FileNotFoundError(
        "Missing generated error-analysis artifact(s): "
        + ", ".join(missing)
        + ". Run `python error_analysis.py` first."
    )

df = pd.read_csv(OOF_PATH)
y = df["y_true"]
p = df["oof_pred"]
ll = df["row_log_loss"]

print("=== OVERALL ===")
print(f"n={len(df)}  base_rate={y.mean():.4f}  mean_pred={p.mean():.4f}")
print(f"overall log loss (mean row_log_loss) = {ll.mean():.6f}")
print(f"median row log loss = {ll.median():.6f}")
print(f"log loss share from worst 1% of rows: {ll.sort_values(ascending=False).head(int(len(df)*0.01)).sum() / ll.sum():.4f}")
print(f"log loss share from worst 5% of rows: {ll.sort_values(ascending=False).head(int(len(df)*0.05)).sum() / ll.sum():.4f}")
print(f"log loss share from worst 10% of rows: {ll.sort_values(ascending=False).head(int(len(df)*0.10)).sum() / ll.sum():.4f}")

print("\n=== CONFUSION @0.5 threshold ===")
pred_label = (p >= 0.5).astype(int)
tp = ((pred_label==1)&(y==1)).sum(); fp=((pred_label==1)&(y==0)).sum()
fn = ((pred_label==0)&(y==1)).sum(); tn=((pred_label==0)&(y==0)).sum()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"precision={tp/(tp+fp):.4f} recall={tp/(tp+fn):.4f}")

print("\n=== CALIBRATION (deciles of predicted prob) ===")
df["decile"] = pd.qcut(p, 10, labels=False, duplicates="drop")
calib = df.groupby("decile").agg(
    n=("y_true","size"), mean_pred=("oof_pred","mean"), actual_rate=("y_true","mean"),
    mean_logloss=("row_log_loss","mean")
)
calib["gap"] = calib["actual_rate"] - calib["mean_pred"]
print(calib.round(4))

print("\n=== WORST 15 ROWS BY LOG LOSS ===")
worst = df.sort_values("row_log_loss", ascending=False).head(15)
cols = ["client_id","y_true","oof_pred","row_log_loss","LIMIT_BAL","AGE","EDUCATION","MARRIAGE",
        "PAY_0","PAY_2","PAY_3","num_severe_delays","credit_util_1","credit_util_6","max_delay"]
cols = [c for c in cols if c in df.columns]
print(worst[cols].to_string(index=False))

print("\n=== ERROR BY EDUCATION ===")
print(df.groupby("EDUCATION").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean")).round(4))

print("\n=== ERROR BY MARRIAGE ===")
print(df.groupby("MARRIAGE").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean")).round(4))

print("\n=== ERROR BY SEX ===")
print(df.groupby("SEX").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean")).round(4))

print("\n=== ERROR BY AGE BUCKET ===")
df["age_bucket"] = pd.cut(df["AGE"], [0,25,30,35,40,50,60,100])
print(df.groupby("age_bucket", observed=True).agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean")).round(4))

print("\n=== ERROR BY PAY_0 (most recent repayment status) ===")
print(df.groupby("PAY_0").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean")).round(4))

print("\n=== ERROR BY num_severe_delays ===")
print(df.groupby("num_severe_delays").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean")).round(4))

print("\n=== ERROR BY credit_util_1 decile ===")
df["util_decile"] = pd.qcut(df["credit_util_1"], 10, labels=False, duplicates="drop")
print(df.groupby("util_decile").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean"),
      mean_util=("credit_util_1","mean")).round(4))

print("\n=== ERROR BY LIMIT_BAL decile ===")
df["limit_decile"] = pd.qcut(df["LIMIT_BAL"], 10, labels=False, duplicates="drop")
print(df.groupby("limit_decile").agg(n=("y_true","size"), base_rate=("y_true","mean"),
      mean_pred=("oof_pred","mean"), logloss=("row_log_loss","mean"),
      mean_limit=("LIMIT_BAL","mean")).round(4))

# high confidence wrong: predicted <0.1 but defaulted, or predicted>0.7 but did not default
print("\n=== HIGH-CONFIDENCE MISSES ===")
conf_fn = df[(p < 0.10) & (y == 1)]
conf_fp = df[(p > 0.70) & (y == 0)]
print(f"Confident-safe-but-defaulted (pred<0.10, y=1): n={len(conf_fn)} ({len(conf_fn)/len(df)*100:.2f}% of all rows, "
      f"{len(conf_fn)/y.sum()*100:.2f}% of all defaulters)")
print(f"Confident-risky-but-repaid (pred>0.70, y=0): n={len(conf_fp)} ({len(conf_fp)/len(df)*100:.2f}% of all rows)")

print("\nProfile of confident-safe-but-defaulted:")
prof_cols = ["LIMIT_BAL","AGE","PAY_0","PAY_2","num_severe_delays","credit_util_1","max_delay"]
prof_cols = [c for c in prof_cols if c in df.columns]
print(conf_fn[prof_cols].describe().round(2))
print("\nOverall population same cols:")
print(df[prof_cols].describe().round(2))

# test set prediction ceiling check
test_df = pd.read_csv(TEST_PREDS_PATH)
print("\n=== TEST SET PREDICTION DISTRIBUTION ===")
print(test_df["reproduced_pred"].describe().round(4))
print(f"fraction of test predictions > 0.5: {(test_df['reproduced_pred']>0.5).mean():.4f}")
print(f"fraction of OOF predictions > 0.5: {(p>0.5).mean():.4f}")
print(f"actual train default rate: {y.mean():.4f}")
