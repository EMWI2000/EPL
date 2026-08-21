"""Point-in-time backtesting utilities.

The functions here deliberately operate on plain :class:`pandas.DataFrame`
objects.  They make no assumptions about a model or a data source, which lets
us use exactly the same evaluation harness for the heuristic, external
projections and a future ML model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY = ("season", "player_id")
DECISION_IDENTITY = ("season", "event")


def _require(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _as_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("cutoff must be a valid timestamp")
    return stamp


def _time_column(frame: pd.DataFrame) -> str:
    """Return the availability time column, preferring the effective time.

    ``effective_at`` is the time consumers could actually see the value.  A
    source's observation time is used only when an effective time is absent.
    This rule prevents evaluations accidentally selecting a backfilled value.
    """

    if "effective_at" in frame.columns:
        return "effective_at"
    if "observed_at" in frame.columns:
        return "observed_at"
    raise ValueError("Snapshot data needs effective_at or observed_at")


def select_snapshot_as_of(
    snapshots: pd.DataFrame,
    cutoff: Any,
    *,
    keys: Sequence[str] = ("season", "player_id", "event"),
) -> pd.DataFrame:
    """Select the latest *available* snapshot per key at ``cutoff``.

    Values published after a gameweek deadline are excluded even if their
    recorded gameweek is earlier.  Ties are resolved deterministically by
    original row order, making repeated offline backtests reproducible.
    """

    _require(snapshots, keys)
    time_col = _time_column(snapshots)
    cut = _as_timestamp(cutoff)
    work = snapshots.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    if work[time_col].isna().any():
        raise ValueError(f"{time_col} contains invalid timestamps")
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    else:
        cut = cut.tz_convert("UTC")
    work = work.loc[work[time_col] <= cut].copy()
    if work.empty:
        return work.drop(columns=[], errors="ignore")
    work["_source_order"] = np.arange(len(work))
    work = work.sort_values([*keys, time_col, "_source_order"], kind="mergesort")
    return work.drop_duplicates(list(keys), keep="last").drop(columns="_source_order")


@dataclass(frozen=True)
class DeadlineFold:
    """One expanding walk-forward split, represented by actual deadlines."""

    fold: int
    cutoff: pd.Timestamp
    train_cutoff: pd.Timestamp
    train_deadlines: tuple[pd.Timestamp, ...]
    test_deadline: pd.Timestamp
    test_deadlines: tuple[pd.Timestamp, ...]


class ExpandingDeadlineSplit:
    """Deadline-based expanding folds with a strict before-deadline boundary.

    A fold trains on all earlier deadlines and tests on one later deadline. It
    never uses row order, so different players/seasons cannot bleed into one
    another merely because their records happen to be adjacent in a CSV.
    """

    def __init__(self, *, min_train_deadlines: int = 3, test_size: int = 1) -> None:
        if min_train_deadlines < 1 or test_size < 1:
            raise ValueError("min_train_deadlines and test_size must be positive")
        self.min_train_deadlines = min_train_deadlines
        self.test_size = test_size

    def split(self, decisions: pd.DataFrame, *, deadline_col: str = "deadline") -> Iterator[DeadlineFold]:
        _require(decisions, ("season", "event", deadline_col))
        dates = pd.to_datetime(decisions[deadline_col], utc=True, errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{deadline_col} contains invalid timestamps")
        # Multiple fixtures in a DGW share one decision deadline.  We use a
        # global date ordering because an inter-season, chronologically earlier
        # result is valid historical data; player aggregates remain season-keyed.
        unique = tuple(sorted(pd.unique(dates)))
        start = self.min_train_deadlines
        number = 0
        for pos in range(start, len(unique), self.test_size):
            test = unique[pos : pos + self.test_size]
            if len(test) < self.test_size:
                break
            number += 1
            yield DeadlineFold(
                fold=number,
                cutoff=test[0],
                train_cutoff=unique[pos - 1],
                train_deadlines=tuple(unique[:pos]),
                test_deadline=test[0],
                test_deadlines=tuple(test),
            )

    def masks(self, decisions: pd.DataFrame, fold: DeadlineFold, *, deadline_col: str = "deadline") -> tuple[pd.Series, pd.Series]:
        """Return train/test masks. Train is strictly before test cutoff."""

        dates = pd.to_datetime(decisions[deadline_col], utc=True, errors="coerce")
        return dates < fold.cutoff, dates.isin(fold.test_deadlines)


def _safe_float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _calibration(frame: pd.DataFrame, *, prediction_col: str, actual_col: str, bins: int) -> list[dict[str, Any]]:
    data = frame[[prediction_col, actual_col]].copy().dropna()
    if data.empty:
        return []
    # Fixed-width bins keep folds comparable; qcut would alter the definition
    # of a bin whenever a model's distribution changes.
    lo, hi = float(data[prediction_col].min()), float(data[prediction_col].max())
    if np.isclose(lo, hi):
        edges = np.array([lo - 0.5, hi + 0.5])
    else:
        edges = np.linspace(lo, hi, bins + 1)
    bucket = pd.cut(data[prediction_col], edges, include_lowest=True, duplicates="drop")
    output: list[dict[str, Any]] = []
    for interval, part in data.groupby(bucket, observed=True):
        output.append({
            "lower": float(interval.left),
            "upper": float(interval.right),
            "count": int(len(part)),
            "mean_prediction": float(part[prediction_col].mean()),
            "mean_actual": float(part[actual_col].mean()),
        })
    return output


def _group_regret(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    actual_col: str,
    group_cols: Sequence[str],
    choice_col: str | None,
    top_n: int,
    eligibility_col: str | None,
) -> tuple[float, float, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        part = group.copy()
        if eligibility_col is not None and eligibility_col in part:
            part = part.loc[part[eligibility_col].fillna(False).astype(bool)]
        part = part.dropna(subset=[prediction_col, actual_col])
        if part.empty:
            continue
        if choice_col is not None and choice_col in part:
            chosen = part.loc[part[choice_col].fillna(False).astype(bool)]
            expected_choices = min(top_n, len(part))
            if len(chosen) != expected_choices:
                raise ValueError(
                    f"{choice_col} must mark exactly {expected_choices} row(s) "
                    "per decision"
                )
        else:
            chosen = part.sort_values([prediction_col, "player_id"], ascending=[False, True], kind="mergesort").head(top_n)
        # A missing chosen player should not silently inflate the result.
        if chosen.empty:
            continue
        oracle = part.nlargest(min(top_n, len(part)), actual_col)
        selected_utility = float(chosen[actual_col].sum())
        oracle_utility = float(oracle[actual_col].sum())
        records.append({
            "decision": key if isinstance(key, tuple) else (key,),
            "utility": selected_utility,
            "oracle_utility": oracle_utility,
            "regret": oracle_utility - selected_utility,
        })
    if not records:
        return float("nan"), float("nan"), records
    return (
        float(np.mean([r["utility"] for r in records])),
        float(np.mean([r["regret"] for r in records])),
        records,
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    actual_col: str = "actual_points",
    minutes_col: str = "minutes",
    start_60_probability_col: str = "p_start_60",
    start_60_actual_col: str | None = None,
    calibration_bins: int = 10,
    decision_top_n: int = 11,
) -> dict[str, Any]:
    """Calculate player, availability and decision metrics from one forecast run.

    Captain regret is the points forfeited by assigning captaincy to the
    highest-projected eligible player rather than the highest-scoring eligible
    player.  ``decision_*`` uses the top ``decision_top_n`` forecasts per
    season/gameweek unless callers provide an ``is_selected`` column.
    """

    _require(predictions, (*IDENTITY, "event", prediction_col, actual_col))
    if calibration_bins < 1 or decision_top_n < 1:
        raise ValueError("calibration_bins and decision_top_n must be positive")
    data = predictions.copy()
    data[prediction_col] = _safe_float_series(data, prediction_col)
    data[actual_col] = _safe_float_series(data, actual_col)
    scored = data.dropna(subset=[prediction_col, actual_col])
    if scored.empty:
        raise ValueError("No rows with both prediction and actual outcome")
    error = scored[prediction_col] - scored[actual_col]
    metrics: dict[str, Any] = {
        "n": int(len(scored)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
        "calibration": _calibration(scored, prediction_col=prediction_col, actual_col=actual_col, bins=calibration_bins),
    }

    if start_60_probability_col in data.columns:
        probability = _safe_float_series(data, start_60_probability_col)
        if start_60_actual_col and start_60_actual_col in data.columns:
            outcome = _safe_float_series(data, start_60_actual_col)
        elif minutes_col in data.columns:
            raw_minutes = _safe_float_series(data, minutes_col)
            outcome = (raw_minutes >= 60).astype(float).where(raw_minutes.notna())
        else:
            raise ValueError("start-60 Brier score needs minutes or an explicit actual column")
        valid = probability.notna() & outcome.notna()
        if valid.any():
            p = probability.loc[valid]
            if ((p < 0) | (p > 1)).any():
                raise ValueError(f"{start_60_probability_col} must be in [0, 1]")
            metrics["start_60_brier"] = float(np.mean(np.square(p - outcome.loc[valid])))
            metrics["start_60_n"] = int(valid.sum())

    captain_utility, captain_regret, captain_details = _group_regret(
        scored,
        prediction_col=prediction_col,
        actual_col=actual_col,
        group_cols=DECISION_IDENTITY,
        choice_col="is_captain" if "is_captain" in scored else None,
        top_n=1,
        eligibility_col="captain_eligible" if "captain_eligible" in scored else None,
    )
    metrics.update({
        "captain_utility": captain_utility,
        "captain_regret": captain_regret,
        "captain_decisions": captain_details,
    })
    utility, regret, details = _group_regret(
        scored,
        prediction_col=prediction_col,
        actual_col=actual_col,
        group_cols=DECISION_IDENTITY,
        choice_col="is_selected" if "is_selected" in scored else None,
        top_n=decision_top_n,
        eligibility_col="selection_eligible" if "selection_eligible" in scored else None,
    )
    metrics.update({"decision_utility": utility, "decision_regret": regret, "decision_details": details})
    return metrics


def paired_gw_bootstrap_ci(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    actual_col: str = "actual_points",
    metric: str = "mae",
    n_bootstrap: int = 2_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Paired season/gameweek bootstrap confidence interval for model deltas.

    The returned ``delta`` is candidate minus baseline, so a negative MAE/RMSE
    delta favours the candidate.  Resampling gameweeks, rather than player
    rows, retains within-GW dependence such as weather and fixture difficulty.
    """

    if metric not in {"mae", "rmse", "bias"}:
        raise ValueError("metric must be one of mae, rmse or bias")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence in (0, 1)")
    keys = [*IDENTITY, "event"]
    _require(candidate, (*keys, prediction_col, actual_col))
    _require(baseline, (*keys, prediction_col, actual_col))
    left = candidate[[*keys, prediction_col, actual_col]].rename(columns={prediction_col: "candidate_prediction", actual_col: "candidate_actual"})
    right = baseline[[*keys, prediction_col, actual_col]].rename(columns={prediction_col: "baseline_prediction", actual_col: "baseline_actual"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("Candidate and baseline have no shared player-gameweeks")
    if not np.allclose(paired["candidate_actual"], paired["baseline_actual"], equal_nan=True):
        raise ValueError("Candidate and baseline outcomes differ on paired rows")
    paired = paired.dropna()
    if paired.empty:
        raise ValueError("No complete paired rows")
    actual = paired["candidate_actual"]
    paired["candidate_error"] = paired["candidate_prediction"] - actual
    paired["baseline_error"] = paired["baseline_prediction"] - actual
    # Calculate each loss at the gameweek level before resampling.  That gives
    # an RMSE delta in points (rather than an opaque delta in squared points)
    # and preserves correlation between player outcomes in the same GW.
    group_losses: list[float] = []
    for _, group in paired.groupby(list(DECISION_IDENTITY), sort=True):
        candidate_error = group["candidate_error"]
        baseline_error = group["baseline_error"]
        if metric == "mae":
            candidate_loss = candidate_error.abs().mean()
            baseline_loss = baseline_error.abs().mean()
        elif metric == "rmse":
            candidate_loss = np.sqrt(candidate_error.pow(2).mean())
            baseline_loss = np.sqrt(baseline_error.pow(2).mean())
        else:
            candidate_loss = candidate_error.mean()
            baseline_loss = baseline_error.mean()
        group_losses.append(float(candidate_loss - baseline_loss))
    gw = np.asarray(group_losses, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(gw), size=(n_bootstrap, len(gw)))
    boot = gw[samples].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "metric": metric,
        "n_gameweeks": int(len(gw)),
        "delta": float(gw.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1 - alpha)),
        "seed": int(seed),
    }
