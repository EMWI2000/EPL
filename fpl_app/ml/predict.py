"""Fail-closed runtime access to the experimental v2 EP model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from fpl_app.ml.v2_dataset import PointInTimeFeatureBuilder
except ImportError:
    from ml.v2_dataset import PointInTimeFeatureBuilder


MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "ep_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
ARTIFACT_SCHEMA_VERSION = 2
FEATURE_CONTRACT = "point_in_time_history_v2"
_model_cache: Dict[str, Any] = {}


class FeatureContractError(RuntimeError):
    """Raised when runtime inputs cannot reproduce the training features."""


def load_meta() -> Dict[str, Any]:
    if not META_PATH.exists():
        return {}
    try:
        value = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_is_validated() -> bool:
    """Require an explicit v2 contract and independent validation marker."""

    meta = load_meta()
    checksums = meta.get("artifact_sha256")
    if not isinstance(checksums, dict):
        return False

    def matches(path: Path, expected: Any) -> bool:
        if not path.is_file() or not isinstance(expected, str):
            return False
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        return digest.hexdigest() == expected

    return (
        meta.get("schema_version") == ARTIFACT_SCHEMA_VERSION
        and meta.get("feature_contract") == FEATURE_CONTRACT
        and meta.get("validation_status") == "validated"
        and matches(MODEL_PATH, checksums.get("model"))
        and matches(FEATURES_PATH, checksums.get("features"))
    )


def model_available(runtime_contract: str | None = None) -> bool:
    """Require artifact, validation and an exactly matching runtime contract."""

    return (
        runtime_contract == FEATURE_CONTRACT
        and MODEL_PATH.exists()
        and FEATURES_PATH.exists()
        and artifact_is_validated()
    )


def load_model(*, runtime_contract: str):
    if not model_available(runtime_contract):
        raise FeatureContractError("The ML artifact is unavailable, unvalidated, or uses another feature contract")
    if "model" not in _model_cache:
        import joblib

        _model_cache["model"] = joblib.load(MODEL_PATH)
        _model_cache["features"] = joblib.load(FEATURES_PATH)
    return _model_cache["model"], _model_cache["features"]


def prepare_point_in_time_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    cutoff: Any,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Build serving features through the exact training-time builder."""

    built = PointInTimeFeatureBuilder().build(history, candidates, cutoff)
    missing = sorted(set(feature_columns).difference(built.columns))
    if missing:
        raise FeatureContractError(f"Runtime data cannot reproduce features: {', '.join(missing)}")
    return built[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def prepare_features_for_player(
    player_row: pd.Series,
    fixtures_data: pd.DataFrame,
    fixture_info: Dict[str, Any],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Reject the legacy aggregate proxy path instead of creating train/serve skew."""

    del player_row, fixtures_data, fixture_info, feature_cols
    raise FeatureContractError(
        "Aggregate FPL API rows do not contain the per-gameweek history required by point_in_time_history_v2"
    )


def predict_point_in_time(
    history: pd.DataFrame, candidates: pd.DataFrame, cutoff: Any,
) -> pd.Series:
    """Predict for callers that can supply the validated v2 runtime contract."""

    model, feature_columns = load_model(runtime_contract=FEATURE_CONTRACT)
    features = prepare_point_in_time_features(history, candidates, cutoff, feature_columns)
    return pd.Series(model.predict(features), index=candidates.index, dtype=float).clip(lower=0.0)


def predict_single_player_multi_gw(
    player_row: pd.Series,
    fixtures_df: pd.DataFrame,
    n: int = 5,
    teams_table: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compatibility fallback for the app's legacy aggregate runtime path."""

    del player_row, fixtures_df, n, teams_table
    return {
        "per_gw": [], "total_next_n": 0.0, "has_dgw": False, "dgw_events": [],
        "unavailable_reason": "point_in_time_history_required",
    }
