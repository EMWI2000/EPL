#!/usr/bin/env python3
"""Offline training for the experimental FPL expected-points model.

Version 2 builds every row as it looked at the decision deadline. Player
histories are isolated by season and player, and validation folds are expanding
global deadlines, never adjacent CSV rows. Artifacts remain candidates until a
separate evaluation explicitly promotes them.
"""
from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

import numpy as np
import pandas as pd
import requests

try:
    from fpl_app.evaluation.backtest import DeadlineFold, ExpandingDeadlineSplit
    from fpl_app.ml.v2_dataset import PointInTimeFeatureBuilder, build_point_in_time_dataset
except ImportError:  # ``cd fpl_app && python -m ml.train_model``
    from evaluation.backtest import DeadlineFold, ExpandingDeadlineSplit
    from ml.v2_dataset import PointInTimeFeatureBuilder, build_point_in_time_dataset


MODEL_DIR = Path(__file__).parent / "models"
SEASONS = ["2022-23", "2023-24", "2024-25"]
VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
ARTIFACT_SCHEMA_VERSION = 2
FEATURE_CONTRACT = "point_in_time_history_v2"
CONTEXT_FEATURES = (
    "pos_code", "is_home", "price", "selected_pct", "net_transfers", "opponent_team_id",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_json(path: Path, value: dict) -> None:
    """Write metadata atomically on the same filesystem."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def download_season_data(season: str) -> pd.DataFrame:
    """Download one historical season for an explicitly invoked offline run."""

    url = f"{VAASTAV_BASE}/{season}/gws/merged_gw.csv"
    print(f"  Henter {season}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), encoding="utf-8")
        frame["season"] = season
        return frame
    except Exception as exc:
        print(f"  FEJL ved hentning af {season}: {exc}")
        return pd.DataFrame()


def load_all_seasons() -> pd.DataFrame:
    """Download and combine the configured seasons."""

    frames = [download_season_data(season) for season in SEASONS]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("Ingen data hentet - tjek internetforbindelse")
    return pd.concat(frames, ignore_index=True)


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _position_code(frame: pd.DataFrame) -> pd.Series:
    if "position" in frame:
        return frame["position"].map({"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}).fillna(2).astype(int)
    if "element_type" in frame:
        return pd.to_numeric(frame["element_type"], errors="coerce").fillna(3).astype(int) - 1
    return pd.Series(2, index=frame.index, dtype=int)


def _decision_deadlines(frame: pd.DataFrame) -> pd.Series:
    """Return one globally sortable decision time per season/gameweek."""

    for column in ("deadline_time", "deadline"):
        if column in frame:
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if parsed.notna().all():
                return parsed
    if "kickoff_time" not in frame:
        raise ValueError("Training data needs deadline_time, deadline or kickoff_time")
    kickoff = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    if kickoff.isna().any():
        raise ValueError("kickoff_time contains invalid timestamps")
    return frame.assign(_kickoff=kickoff).groupby(["season", "event"], sort=False)["_kickoff"].transform("min")


def _number(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def prepare_training_frames(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Normalize raw fixture rows into history and decision contracts."""

    frame = raw.copy().rename(columns={
        **({"element": "player_id"} if "element" in raw and "player_id" not in raw else {}),
        **({"GW": "event"} if "GW" in raw and "event" not in raw else {}),
    })
    required_stats = tuple(PointInTimeFeatureBuilder().stats)
    _require(frame, ("season", "player_id", "event", "total_points", *required_stats))
    for column in required_stats:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    frame["event"] = pd.to_numeric(frame["event"], errors="raise").astype(int)

    deadline_source = "deadline_time" if "deadline_time" in frame else "deadline"
    if "deadline_time" not in frame and "deadline" not in frame:
        deadline_source = "earliest_kickoff_proxy"
    frame["deadline"] = _decision_deadlines(frame)
    frame["target"] = pd.to_numeric(frame["total_points"], errors="coerce")
    frame["pos_code"] = _position_code(frame)
    frame["is_home"] = _number(frame, "was_home", 0).astype(int)
    frame["price"] = _number(frame, "value", 50) / 10.0
    frame["selected_pct"] = _number(frame, "selected", 0)
    frame["net_transfers"] = _number(frame, "transfers_in", 0) - _number(frame, "transfers_out", 0)
    frame["opponent_team_id"] = _number(frame, "opponent_team", 0).astype(int)

    history = frame[["season", "player_id", "event", *required_stats]].copy()
    decisions = frame[[
        "season", "player_id", "event", "deadline", "target", *CONTEXT_FEATURES,
    ]].copy()
    return history, decisions, deadline_source


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the point-in-time dataset shared by train and future serving."""

    history, decisions, _ = prepare_training_frames(raw)
    return build_point_in_time_dataset(history, decisions, target_col="target")


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the deterministic v2 feature contract in dataframe order."""

    columns = list(CONTEXT_FEATURES)
    columns.extend(
        column for column in frame.columns
        if column.startswith(("cum_", "roll_"))
        or column in {"history_minutes", "history_events", "sample_weight"}
        or column.endswith("_per90")
    )
    return list(dict.fromkeys(columns))


def deadline_folds(
    dataset: pd.DataFrame, *, min_train_deadlines: int = 5,
) -> Iterator[tuple[DeadlineFold, pd.Series, pd.Series]]:
    """Yield global expanding deadline masks, independent of input row order."""

    splitter = ExpandingDeadlineSplit(min_train_deadlines=min_train_deadlines)
    for fold in splitter.split(dataset, deadline_col="deadline"):
        train_mask, test_mask = splitter.masks(dataset, fold, deadline_col="deadline")
        yield fold, train_mask, test_mask


def train_and_save() -> None:
    """Train a v2 candidate artifact. This is never called on import."""

    import joblib
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    raw = load_all_seasons()
    _, _, deadline_source = prepare_training_frames(raw)
    dataset = engineer_features(raw).dropna(subset=["target"])
    feature_columns = get_feature_columns(dataset)
    X = dataset[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(dataset["target"], errors="coerce")
    if len(pd.unique(pd.to_datetime(dataset["deadline"], utc=True))) < 6:
        raise ValueError("At least six decision deadlines are required for walk-forward validation")

    params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        min_child_samples=20, verbose=-1,
    )
    fold_metrics = []
    for fold, train_mask, test_mask in deadline_folds(dataset):
        model = lgb.LGBMRegressor(**params)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        prediction = model.predict(X.loc[test_mask])
        fold_metrics.append({
            "fold": fold.fold,
            "test_deadline": fold.test_deadline.isoformat(),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "mae": float(mean_absolute_error(y.loc[test_mask], prediction)),
        })

    MODEL_DIR.mkdir(exist_ok=True)
    metadata_path = MODEL_DIR / "model_meta.json"
    # Invalidate any previously promoted bundle before replacing a byte.  A
    # crash during training or publication therefore fails closed at runtime.
    _replace_json(
        metadata_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "feature_contract": FEATURE_CONTRACT,
            "validation_status": "training",
        },
    )
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_contract": FEATURE_CONTRACT,
        "validation_status": "candidate",
        "deadline_source": deadline_source,
        "seasons": SEASONS,
        "n_features": len(feature_columns),
        "n_training_rows": len(dataset),
        "folds": fold_metrics,
        "avg_mae": float(np.mean([item["mae"] for item in fold_metrics])),
        "activation_note": "Requires independent backtest approval and point-in-time runtime history.",
    }
    with tempfile.TemporaryDirectory(dir=MODEL_DIR, prefix=".candidate-") as directory:
        staging = Path(directory)
        model_path = staging / "ep_model.joblib"
        features_path = staging / "feature_columns.joblib"
        joblib.dump(final_model, model_path)
        joblib.dump(feature_columns, features_path)
        metadata["artifact_sha256"] = {
            "model": _sha256(model_path),
            "features": _sha256(features_path),
        }
        os.replace(model_path, MODEL_DIR / "ep_model.joblib")
        os.replace(features_path, MODEL_DIR / "feature_columns.joblib")
        # Metadata is last: readers cannot see candidate/validated status until
        # both artifacts and their checksums are in place.
        _replace_json(metadata_path, metadata)


if __name__ == "__main__":
    train_and_save()
