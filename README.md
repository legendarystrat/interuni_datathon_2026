# interuni_datathon_2026

Repository for the Interuni Datathon 2026 credit default prediction task.

Primary metric: validation log loss, minimized with 5-fold stratified CV.
AUC ROC is tracked as a secondary sanity check only.

## Current Best Local Result

The strongest local artifact is now:

- Config: `global_search_best.json`
- Submission: `submission_global_blend.csv`
- CV log loss: `0.4229845178218504`
- CV AUC ROC: `0.7898902064728368`

This blend combines the saved original/extended winners with fresh XGBoost and
LightGBM trials from `run_global_search.py`.

## Environment

Create and use a repo-local virtual environment:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Useful Commands

Sync the exported feature-engineering best JSON with the best completed Optuna
trial in `optuna_feature_eng_joint.db`:

```bash
.venv/bin/python export_best_feature_eng.py
```

Regenerate the feature-engineered XGBoost submission after syncing:

```bash
.venv/bin/python -c "from feature_eng_lib import engineer_all_features, load_data, make_feature_eng_submission; train_df=engineer_all_features(load_data('train.csv')); test_df=engineer_all_features(load_data('test.csv')); make_feature_eng_submission(train_df, test_df, train_df['default'])"
```

Run train-vs-test adversarial validation:

```bash
.venv/bin/python run_adversarial_validation.py
```

Run a broader search over XGBoost and LightGBM while still blending against the
saved CatBoost winners:

```bash
.venv/bin/python -u run_global_search.py \
  --models xgboost lightgbm catboost \
  --search-models xgboost lightgbm \
  --trials-per-model 40 \
  --top-k-per-model 5 \
  --blend-trials 1500 \
  --make-submission \
  --output global_search_best.json \
  --submission submission_global_blend.csv
```
