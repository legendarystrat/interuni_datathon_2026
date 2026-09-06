"""Export the best completed feature-engineering Optuna trial from SQLite.

This intentionally uses only the Python standard library so it can run before
the ML environment is installed. It keeps ``feature_eng_joint_best.json`` in
sync with the actual best completed trial in ``optuna_feature_eng_joint.db``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("optuna_feature_eng_joint.db")
DEFAULT_OUTPUT_PATH = Path("feature_eng_joint_best.json")
DEFAULT_REFERENCE_PATH = Path("optuna_best_extended.json")
DEFAULT_STUDY_NAME = "xgb_joint_params_and_blocks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the best feature-engineering Optuna trial to JSON."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    return parser.parse_args()


def json_attr(value: str) -> Any:
    return json.loads(value)


def load_reference_log_loss(path: Path) -> float | None:
    if not path.exists():
        return None

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if "xgboost" in payload:
        return float(payload["xgboost"]["val_log_loss_mean"])
    if "best_val_log_loss_mean" in payload:
        return float(payload["best_val_log_loss_mean"])
    return None


def fetch_best_trial(
    conn: sqlite3.Connection,
    study_name: str,
) -> tuple[int, int, float]:
    row = conn.execute(
        """
        select t.trial_id, t.number, v.value
        from trials t
        join studies s on s.study_id = t.study_id
        join trial_values v on v.trial_id = t.trial_id and v.objective = 0
        where s.study_name = ?
          and t.state = 'COMPLETE'
        order by v.value asc, t.number asc
        limit 1
        """,
        (study_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"No completed trials found for study {study_name!r}")
    return int(row[0]), int(row[1]), float(row[2])


def fetch_user_attrs(conn: sqlite3.Connection, trial_id: int) -> dict[str, Any]:
    rows = conn.execute(
        """
        select key, value_json
        from trial_user_attributes
        where trial_id = ?
        """,
        (trial_id,),
    ).fetchall()
    return {key: json_attr(value_json) for key, value_json in rows}


def build_export_payload(
    trial_number: int,
    trial_value: float,
    attrs: dict[str, Any],
    reference_log_loss: float | None,
) -> dict[str, Any]:
    required = [
        "active_blocks",
        "enabled_blocks",
        "feature_cols",
        "n_features",
        "val_roc_auc_mean",
        "val_log_loss_std",
        "xgb_params",
    ]
    missing = [key for key in required if key not in attrs]
    if missing:
        raise RuntimeError(
            f"Best trial {trial_number} is missing required user attrs: {missing}"
        )

    result: dict[str, Any] = {
        "best_trial_number": trial_number,
        "best_val_log_loss_mean": trial_value,
        "best_val_log_loss_std": float(attrs["val_log_loss_std"]),
        "best_val_roc_auc_mean": float(attrs["val_roc_auc_mean"]),
        "best_active_blocks": attrs["active_blocks"],
        "best_enabled_blocks": attrs["enabled_blocks"],
        "best_feature_cols": attrs["feature_cols"],
        "best_xgb_params": attrs["xgb_params"],
        "n_features": int(attrs["n_features"]),
    }
    if reference_log_loss is not None:
        result = {
            "reference_val_log_loss_mean": reference_log_loss,
            **result,
        }
    return result


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(args.db)

    with sqlite3.connect(args.db) as conn:
        trial_id, trial_number, trial_value = fetch_best_trial(conn, args.study_name)
        attrs = fetch_user_attrs(conn, trial_id)

    payload = build_export_payload(
        trial_number=trial_number,
        trial_value=trial_value,
        attrs=attrs,
        reference_log_loss=load_reference_log_loss(args.reference),
    )

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(
        f"Exported trial {trial_number} to {args.output} "
        f"(val_log_loss={trial_value:.12f}, "
        f"val_auc={payload['best_val_roc_auc_mean']:.12f}, "
        f"features={payload['n_features']})"
    )


if __name__ == "__main__":
    main()
