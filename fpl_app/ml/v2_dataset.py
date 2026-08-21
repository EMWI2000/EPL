"""Shared point-in-time feature construction for ML training and serving.

Unlike the legacy training script this module keys every history by both
``season`` and ``player_id``.  Features for an event are built only from rows
that were available before its cutoff; neither season resets nor later
backfills can leak into a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # Package import in tests/Vercel.
    from fpl_app.evaluation.backtest import select_snapshot_as_of
except ImportError:  # Streamlit can run with fpl_app as its working directory.
    from evaluation.backtest import select_snapshot_as_of


DEFAULT_STATS = (
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "bonus", "bps", "ict_index", "expected_goals", "expected_assists",
)
IDENTITY = ("season", "player_id")


def _require(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _to_utc(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("cutoff must be a valid timestamp")
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


@dataclass(frozen=True)
class PointInTimeFeatureBuilder:
    """Make model features at a deadline, for train and serve alike."""

    stats: tuple[str, ...] = DEFAULT_STATS
    rolling_windows: tuple[int, ...] = (3, 5)
    prior_minutes: float = 900.0

    def __post_init__(self) -> None:
        if not self.stats or not self.rolling_windows or self.prior_minutes <= 0:
            raise ValueError("stats/windows must be non-empty and prior_minutes positive")
        if any(window < 1 for window in self.rolling_windows):
            raise ValueError("rolling windows must be positive")

    def build(
        self,
        history: pd.DataFrame,
        candidates: pd.DataFrame,
        cutoff: Any,
        *,
        target_col: str | None = None,
    ) -> pd.DataFrame:
        """Build features for candidates with only known pre-cutoff history.

        ``history`` can contain snapshots/revisions. The latest version of a
        season/player/event available at cutoff is selected before any rolling
        aggregate is calculated. ``candidates`` must describe the prediction
        event and is deliberately not treated as historical performance.
        """

        _require(history, (*IDENTITY, "event"))
        _require(candidates, (*IDENTITY, "event"))
        _require(history, self.stats)
        cut = _to_utc(cutoff)
        # Explicit availability means revision-safe snapshots. Without it,
        # plain historical data is accepted but still strictly event-lagged.
        if "effective_at" in history.columns or "observed_at" in history.columns:
            revision_keys = [*IDENTITY, "event"]
            # A double gameweek can contain two legitimate fixture rows for one
            # player/event.  Preserve fixture identity while de-duplicating
            # revisions whenever the source provides it.
            for fixture_key in ("fixture_id", "fixture", "kickoff_time"):
                if fixture_key in history.columns:
                    revision_keys.append(fixture_key)
                    break
            known = select_snapshot_as_of(history, cut, keys=tuple(revision_keys))
        else:
            known = history.copy()
        known = known.copy()
        known["event"] = pd.to_numeric(known["event"], errors="raise")
        candidate_work = candidates.copy()
        candidate_work["event"] = pd.to_numeric(candidate_work["event"], errors="raise")
        # Keep one joined group for *every* candidate.  Filtering rows before
        # grouping used to remove a GW1/cold-start candidate when its only
        # available history was from the same event.  Instead mark valid prior
        # observations and filter inside each candidate group.
        candidate_work["_candidate_row"] = np.arange(len(candidate_work))
        joined = candidate_work[[*IDENTITY, "event", "_candidate_row"]].merge(
            known, on=list(IDENTITY), how="left", suffixes=("_target", "_history"), validate="many_to_many"
        )
        joined["_is_prior_event"] = (
            joined["event_history"].notna()
            & (joined["event_history"] < joined["event_target"])
        )
        rows: list[dict[str, Any]] = []
        for row_id, group in joined.groupby("_candidate_row", sort=False):
            candidate = candidate_work.loc[candidate_work["_candidate_row"] == row_id].iloc[0]
            prior = group.loc[group["_is_prior_event"]].sort_values("event_history", kind="mergesort")
            result = candidate.to_dict()
            minutes = pd.to_numeric(prior.get("minutes", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            result["history_minutes"] = float(minutes.sum())
            result["history_events"] = int(len(prior))
            for stat in self.stats:
                values = pd.to_numeric(prior.get(stat, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                result[f"cum_{stat}"] = float(values.sum())
                for window in self.rolling_windows:
                    result[f"roll_{stat}_{window}"] = float(values.tail(window).mean()) if len(values) else 0.0
            # Player rates are shrinkage-ready features, not a magic small-
            # sample switch. The weight is exposed so a trainer can learn its
            # own blend with a position/league prior.
            result["sample_weight"] = float(result["history_minutes"] / (result["history_minutes"] + self.prior_minutes))
            for stat in ("goals_scored", "assists", "expected_goals", "expected_assists", "total_points"):
                result[f"{stat}_per90"] = float(result[f"cum_{stat}"] / max(result["history_minutes"], 1.0) * 90.0)
            rows.append(result)
        output = pd.DataFrame(rows).drop(columns="_candidate_row", errors="ignore")
        # The outcome is optional at serving time.  When supplied by a training
        # snapshot it is copied through with the candidate row, never read from
        # history, so target presence cannot alter a feature.
        return output


def build_point_in_time_dataset(
    history: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    deadline_col: str = "deadline",
    builder: PointInTimeFeatureBuilder | None = None,
    target_col: str | None = "total_points",
) -> pd.DataFrame:
    """Create a trainable dataset by invoking the same builder per deadline.

    This is intentionally slower but auditable.  A production training job can
    optimise it after equivalence tests prove it produces identical features.
    """

    _require(decisions, (*IDENTITY, "event", deadline_col))
    chosen = builder or PointInTimeFeatureBuilder()
    deadlines = pd.to_datetime(decisions[deadline_col], utc=True, errors="coerce")
    if deadlines.isna().any():
        raise ValueError(f"{deadline_col} contains invalid timestamps")
    work = decisions.copy()
    work[deadline_col] = deadlines
    pieces = []
    for deadline, batch in work.groupby(deadline_col, sort=True):
        pieces.append(chosen.build(history, batch, deadline, target_col=target_col))
    return pd.concat(pieces, ignore_index=True) if pieces else work.iloc[0:0].copy()
