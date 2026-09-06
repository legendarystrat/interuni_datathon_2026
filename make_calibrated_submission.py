"""Create a calibrated submission from a saved global-search config.

The targeted global search stores both the raw blend and an optional
post-hoc calibrated blend. This script materializes the calibrated candidate
without re-running the expensive Optuna search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from feature_eng_lib import engineer_all_features
from model_copy_utils import load_data
from run_global_search import (
    apply_logit_calibration,
    blend_predictions,
    predict_config,
)

ID_COL = "client_id"
TARGET_COL = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a calibrated blend submission from a saved config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("global_search_targeted_best.json"),
        help="Saved global-search JSON containing a calibrated_blend section.",
    )
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_global_targeted_calibrated_blend.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not config.get("calibrated_blend"):
        raise ValueError(f"{path} does not contain a calibrated_blend candidate")
    return config


def main() -> None:
    args = parse_args()
    saved = load_config(args.config)
    calibrated = saved["calibrated_blend"]
    config_by_label = saved["config_by_blend_label"]

    train_df = engineer_all_features(load_data(args.train))
    test_df = engineer_all_features(load_data(args.test))
    y = train_df[TARGET_COL]

    required_labels = [
        label for label, weight in calibrated["weights"].items() if weight > 0
    ]
    base_test = {}
    for index, label in enumerate(required_labels, start=1):
        print(f"[{index}/{len(required_labels)}] fitting {label}", flush=True)
        base_test[label] = predict_config(
            config_by_label[label],
            train_df,
            test_df,
            y,
            seed=args.seed,
            sample_weight=None,
        )

    raw_probs = blend_predictions(base_test, calibrated["weights"])
    calibrated_probs = apply_logit_calibration(
        raw_probs,
        calibrated["calibration"],
        calibrated["base_prior"],
    )

    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL],
            "default_probability": calibrated_probs,
        }
    )
    submission.to_csv(args.output, index=False)

    print(
        f"Saved {args.output} ({len(submission)} rows, "
        f"mean={calibrated_probs.mean():.6f}, "
        f"min={calibrated_probs.min():.6f}, "
        f"max={calibrated_probs.max():.6f})"
    )
    print(
        "OOF comparison from saved config: "
        f"raw={saved['blend']['val_log_loss_mean']:.12f}, "
        f"calibrated={calibrated['val_log_loss_mean']:.12f}"
    )


if __name__ == "__main__":
    main()
