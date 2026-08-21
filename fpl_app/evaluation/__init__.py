"""Leakage-safe, offline evaluation primitives for FPL forecasts."""

from .backtest import (
    ExpandingDeadlineSplit,
    evaluate_predictions,
    paired_gw_bootstrap_ci,
    select_snapshot_as_of,
)

__all__ = [
    "ExpandingDeadlineSplit",
    "evaluate_predictions",
    "paired_gw_bootstrap_ci",
    "select_snapshot_as_of",
]
