# InterUni Datathon 2026

Credit-default prediction work for the InterUni Datathon 2026 competition.

The target is `default`, and the submitted file must contain one probability
per `client_id` in the `default_probability` column.

## Metric

The primary objective is **log loss**. Lower is better.

ROC AUC is tracked as a secondary diagnostic because it is useful for checking
ranking quality, but model selection in this repo is driven by validation log
loss.

## Current Best Result

The strongest saved artifact is the targeted global blend:

| Item | Value |
| --- | --- |
| Config | `global_search_targeted_best.json` |
| Submission | `submission_global_targeted_blend.csv` |
| Local CV log loss | `0.42286059146172406` |
| Local CV ROC AUC | `0.7900760428929055` |
| Public leaderboard log loss | `0.41217` |
| Rows | `6000` |

The targeted blend combines 17 saved model candidates across XGBoost,
LightGBM, and CatBoost. The largest weights are on CatBoost and XGBoost
variants using engineered repayment-status, bill-trend, utilization, and
payment-ratio features.

A calibrated alternate is also saved as
`submission_global_targeted_calibrated_blend.csv`. It applies the saved
logit-scale/intercept calibration from `global_search_targeted_best.json`, but
its local CV log loss was slightly worse than the raw targeted blend:

| Candidate | Local CV log loss |
| --- | ---: |
| Raw targeted blend | `0.42286059146172406` |
| Calibrated targeted blend | `0.4229396410998545` |

## Repository Map

| Path | Purpose |
| --- | --- |
| `final.ipynb` | Clean final narrative notebook: EDA, correlation analysis, feature engineering, modelling path, calibration checks, and submission validation. |
| `clean.ipynb` | Early data checks and first-pass feature engineering. |
| `models.ipynb`, `models copy.ipynb` | Initial model tuning, Optuna experiments, blending, and calibration experiments. |
| `feature_eng.ipynb` | Feature-engineering experiments and block ablations. |
| `adversarial_validation.ipynb` | Train-vs-test shift analysis. |
| `feature_eng_lib.py` | Shared feature-engineering and feature-selection utilities. |
| `model_copy_utils.py` | Shared model helpers originally extracted from modelling notebooks. |
| `run_global_search.py` | Global model search, blend optimization, optional calibration, optional stacking, and submission creation. |
| `make_calibrated_submission.py` | Rebuilds the saved calibrated targeted-blend submission without rerunning Optuna. |
| `export_best_feature_eng.py` | Exports the best local Optuna feature-engineering trial to JSON after the local study has been run. |
| `error_analysis.py`, `error_segments.py` | Reproduce OOF predictions and inspect high-loss model segments. |
| `cv_robustness.py` | Rechecks blend robustness across alternate CV seeds. |
| `*_best.json`, `global_search_*.json` | Tracked model/config summaries needed to reproduce submissions without keeping local Optuna DBs in git. |
| `submission_*.csv` | Saved competition submission files. |

## Setup

Create and activate the repo-local Python environment:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The repo expects `train.csv`, `test.csv`, and `sample_submission.csv` to be
present in the project root.

## Quick Checks

Validate that the final notebook runs from top to bottom:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace final.ipynb --ExecutePreprocessor.timeout=1800
```

Validate the current best submission directly:

```bash
.venv/bin/python - <<'PY'
import pandas as pd

sample = pd.read_csv("sample_submission.csv")
submission = pd.read_csv("submission_global_targeted_blend.csv")

assert submission.shape == sample.shape
assert submission["client_id"].equals(sample["client_id"])
assert submission["client_id"].is_unique
assert submission["default_probability"].notna().all()
assert submission["default_probability"].between(0, 1).all()

print(submission["default_probability"].describe())
PY
```

## Rebuild Key Artifacts

Regenerate the calibrated targeted submission from the saved config:

```bash
.venv/bin/python make_calibrated_submission.py
```

Run train-vs-test adversarial validation:

```bash
.venv/bin/python run_adversarial_validation.py
```

Run the targeted global search used for the current best submission:

```bash
.venv/bin/python -u run_global_search.py \
  --trials-per-model 50 \
  --models xgboost lightgbm catboost \
  --search-models xgboost lightgbm \
  --top-k-per-model 5 \
  --blend-trials 1500 \
  --calibration-trials 800 \
  --disable-stack \
  --output global_search_targeted_best.json \
  --submission submission_global_targeted_blend.csv \
  --make-submission
```

If you run feature-engineering Optuna studies locally, export the best completed
trial afterwards:

```bash
.venv/bin/python export_best_feature_eng.py
```

## Modelling Story

The modelling path is intentionally ordered around the metric:

1. Start with raw EDA and data-quality checks.
2. Use correlation and normalized mutual information to identify the strongest
   target signals.
3. Build engineered features around the strongest signals: repayment status,
   delay frequency/severity, bill trends, credit utilization, and
   payment-to-bill ratios.
4. Compare a baseline, regularized logistic regression, and boosted trees using
   5-fold stratified CV.
5. Tune XGBoost, LightGBM, and CatBoost variants with Optuna.
6. Blend strong saved model candidates and select the candidate with the lowest
   validation log loss.
7. Test post-hoc probability calibration, but keep the raw blend because it
   scores better locally.

## Feature Engineering

The final model pool uses raw variables plus engineered feature families:

- Raw demographics and account state: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`,
  `LIMIT_BAL`
- Repayment status: `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`
- Bill and payment history: `BILL_AMT1-6`, `PAY_AMT1-6`
- Delay summaries: `max_delay`, `num_months_delayed`, `num_severe_delays`,
  `ever_delayed`, `mean_pay_status`, recent/old delay means, weighted delay
- Exposure and repayment aggregates: `total_bill`, `total_pay`, payment amount
  statistics
- Utilization: monthly utilization, mean/max/std utilization, high-utilization
  counts
- Trends and ratios: bill slopes, bill absolute/percentage changes,
  payment-to-bill ratios
- Interaction terms tested in some model candidates, especially delay by
  utilization

## Generated Files And Git Hygiene

The repo tracks the compact artifacts needed to explain and reproduce the
submission: notebooks, scripts, data, selected submissions, and JSON config
exports.

The repo ignores local runtime outputs such as:

- `.venv/`
- `__pycache__/`
- `catboost_info/`
- `*.log`
- local Optuna SQLite databases (`*.db`)
- bulky regenerated row-level analysis CSVs
- local model artifacts under `artifacts/`

Regenerate ignored artifacts from the scripts above when needed.

## Notes

See `EXPERIMENT_LOG.md` for the chronological experiment history and
`ERROR_ANALYSIS.md` for the main failure modes of the current best blend.
