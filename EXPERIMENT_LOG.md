# Experiment Log

This document summarizes the model-improvement work done after the initial repo review.

## Objective

The goal was to improve the competition model by minimizing validation log loss. AUC ROC was tracked only as a secondary sanity check.

## Environment Setup

Created a repo-local Python virtual environment using Homebrew Python 3.13:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Added `requirements.txt` so the modeling environment can be recreated. Added `.gitignore` to keep `.venv`, `__pycache__`, `.pyc`, and notebook checkpoints out of future commits.

## Exported The Actual Best Feature-Engineering Trial

The Optuna database `optuna_feature_eng_joint.db` contained a better completed trial than the JSON export.

Updated `feature_eng_joint_best.json` from the true best completed trial:

- Trial: `183`
- CV log loss: `0.4234399428011601`
- CV AUC ROC: `0.7896441578207326`
- Features: `57`

Then regenerated:

- `submission_feature_eng.csv`

## Added Reproducible Scripts

Added `export_best_feature_eng.py`.

This script reads `optuna_feature_eng_joint.db` directly with the Python standard library and exports the best completed trial to `feature_eng_joint_best.json`.

Added `run_adversarial_validation.py`.

This script trains a train-vs-test classifier using the same engineered feature pipeline. It saves:

- `adversarial_outputs/adversarial_summary.json`
- `adversarial_outputs/adversarial_feature_importance.csv`
- `adversarial_outputs/adversarial_oof.csv`
- `adversarial_outputs/adversarial_train_weights.csv`

Added `run_global_search.py`.

This script runs a broader Optuna search and blends strong saved models with newly searched models. It supports:

- XGBoost, LightGBM, and CatBoost searches
- selecting which model families to search
- including previously saved best configs as blend candidates
- top-k trial blending
- optional adversarial sample weights
- final submission generation

## Adversarial Validation Result

Train/test shift was not meaningful:

- Mean adversarial ROC-AUC: `0.5074991319444444`
- Material shift threshold: `0.55`
- Material shift detected: `false`

Because the shift signal was weak, adversarial weighting was not used for the winning model run.

## Global Search Result

Ran a blend-first search over existing strong configs, then a deeper XGBoost/LightGBM search while keeping the saved CatBoost winners in the blend pool.

Best final local result:

- Config: `global_search_best.json`
- Submission: `submission_global_blend.csv`
- CV log loss: `0.4229845178218504`
- CV AUC ROC: `0.7898902064728368`
- Submission rows: `6000`
- Mean predicted probability: `0.218777`
- Min predicted probability: `0.02774979701714026`
- Max predicted probability: `0.8514409422048209`

This improved over:

- Previous extended blend: `0.4234613293399816`
- Feature-engineered XGBoost trial 183: `0.4234399428011601`

## Final Blend Composition

The final blend combined saved and newly searched models. The largest weights were:

- `optuna_extended_catboost_trial_28_all_engineered_extended`: `0.21556848828250438`
- `global_search_xgboost_trial_27_exported_feature_eng_best`: `0.21483965959264448`
- `optuna_original_catboost_trial_17_all_engineered`: `0.19951103464807013`
- `global_search_lightgbm_trial_31_all_features`: `0.06220244718138475`
- `global_search_xgboost_trial_30_exported_feature_eng_best`: `0.04832853810473954`
- `feature_eng_joint_xgboost_trial_183_feature_eng_joint_best`: `0.046436661813059`

The full weight dictionary is stored in `global_search_best.json`.

## Main Pull And Follow-Up Blend

After pulling `origin/main`, the incoming feature-engineering notebook state updated `feature_eng_joint_best.json` and `submission_feature_eng.csv`.

The notebook source cells were effectively unchanged, but the Optuna study had been run further:

- Previous local feature-eng study: `189` completed trials plus `1` running trial
- Incoming main feature-eng study: `300` completed trials
- Incoming best CV log loss: `0.4232287030876286`
- Incoming best CV AUC ROC: `0.7895951433786396`
- Incoming selected features: `54`

The incoming best feature set dropped the `delay_util_interactions` block, removing:

- `recent_delay_x_util`
- `delay_count_x_util`
- `severe_delay_x_util`

Then `run_global_search.py` was extended with two post-processing candidates:

- calibrated constrained blend using logit scale/intercept plus optional prior shrinkage
- logistic stack over out-of-fold base-model predictions

The first rebuild reused cached model candidates and included the incoming feature-eng winner:

- Config: `global_search_calibrated_best.json`
- Submission: `submission_global_calibrated_blend.csv`
- Best raw blend CV log loss: `0.422952369254`
- Best calibrated blend CV log loss: `0.423005440117`
- Best logistic stack CV log loss: `0.423037479743`
- Selected candidate: raw blend

Then a targeted XGBoost/LightGBM search was run up to `50` completed trials per searched model family, with the new 54-feature export available to the search:

- Config: `global_search_targeted_best.json`
- Submission: `submission_global_targeted_blend.csv`
- Best raw blend CV log loss: `0.422860591462`
- Best raw blend CV AUC ROC: `0.790076042893`
- Best calibrated blend CV log loss: `0.422939641100`
- Selected candidate: raw blend
- Submission rows: `6000`
- Mean predicted probability: `0.218828`
- Min predicted probability: `0.028064608`
- Max predicted probability: `0.852471507`

This is the best local CV log-loss result so far.

## Commands To Continue

Sync the best feature-engineering trial:

```bash
.venv/bin/python export_best_feature_eng.py
```

Run adversarial validation:

```bash
.venv/bin/python run_adversarial_validation.py
```

Continue the broader XGBoost/LightGBM search:

```bash
.venv/bin/python -u run_global_search.py \
  --models xgboost lightgbm catboost \
  --search-models xgboost lightgbm \
  --trials-per-model 80 \
  --top-k-per-model 5 \
  --blend-trials 2000 \
  --calibration-trials 800 \
  --disable-stack \
  --make-submission \
  --output global_search_targeted_best.json \
  --submission submission_global_targeted_blend.csv
```

## Git Push

The experiment commit was pushed to:

```text
origin/harsh-experiment
```

Commit:

```text
7bada31 added a new submission global_blend.csv
```

Pull request URL:

```text
https://github.com/legendarystrat/interuni_datathon_2026/pull/new/harsh-experiment
```
