#!/usr/bin/env python3
"""
Offline træningsscript for FPL EP-model (LightGBM).

Bruger vaastav/Fantasy-Premier-League GitHub dataset.
Kør dette script lokalt for at generere ml/models/ep_model.joblib.

Usage:
    cd fpl_app
    python -m ml.train_model

Kræver: lightgbm, scikit-learn, pandas, numpy, joblib, requests
"""
from __future__ import annotations
import os
import io
import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Sæsoner med xG data (tilgængelige i vaastav dataset)
SEASONS = ["2022-23", "2023-24", "2024-25"]
VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def download_season_data(season: str) -> pd.DataFrame:
    """Download merged_gw.csv for en sæson fra vaastav dataset."""
    url = f"{VAASTAV_BASE}/{season}/gws/merged_gw.csv"
    print(f"  Henter {season}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding="utf-8")
        df["season"] = season
        return df
    except Exception as e:
        print(f"  FEJL ved hentning af {season}: {e}")
        return pd.DataFrame()


def load_all_seasons() -> pd.DataFrame:
    """Henter og samler alle sæsoner."""
    print("Henter data fra vaastav/Fantasy-Premier-League...")
    frames = []
    for season in SEASONS:
        df = download_season_data(season)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("Ingen data hentet – tjek internetforbindelse")
    return pd.concat(frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: rolling stats, per-90 stats, position encoding."""
    # Sortér efter spiller + sæson + runde
    df = df.sort_values(["element", "season", "GW"]).reset_index(drop=True)

    # Konverter numeriske kolonner
    num_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps", "influence", "creativity", "threat",
        "ict_index", "value", "selected", "transfers_in", "transfers_out",
    ]
    # xG kolonner (kan mangle i ældre data)
    xg_cols = [
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded",
    ]

    for c in num_cols + xg_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Position encoding
    if "position" in df.columns:
        pos_map = {"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        df["pos_code"] = df["position"].map(pos_map).fillna(2).astype(int)
    elif "element_type" in df.columns:
        df["pos_code"] = df["element_type"].astype(int) - 1
    else:
        df["pos_code"] = 2

    # is_home
    if "was_home" in df.columns:
        df["is_home"] = df["was_home"].astype(int)
    else:
        df["is_home"] = 0

    # opponent_fdr (hvis tilgængelig)
    if "opponent_team" in df.columns:
        df["opponent_team_id"] = pd.to_numeric(df["opponent_team"], errors="coerce").fillna(0).astype(int)
    else:
        df["opponent_team_id"] = 0

    # Rolling features per spiller
    group = df.groupby("element")

    for window in [3, 5]:
        w = f"_{window}gw"
        df[f"roll_pts{w}"] = group["total_points"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f"roll_min{w}"] = group["minutes"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f"roll_bps{w}"] = group["bps"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f"roll_ict{w}"] = group["ict_index"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df[f"roll_bonus{w}"] = group["bonus"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        if "expected_goals" in df.columns:
            df[f"roll_xG{w}"] = group["expected_goals"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f"roll_xA{w}"] = group["expected_assists"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

    # Kumulativ per-90 stats (op til forrige runde)
    df["cum_minutes"] = group["minutes"].transform(lambda x: x.shift(1).cumsum())
    df["cum_goals"] = group["goals_scored"].transform(lambda x: x.shift(1).cumsum())
    df["cum_assists"] = group["assists"].transform(lambda x: x.shift(1).cumsum())
    df["cum_cs"] = group["clean_sheets"].transform(lambda x: x.shift(1).cumsum())

    safe_90s = (df["cum_minutes"] / 90.0).clip(lower=0.5)
    df["goals_per_90"] = df["cum_goals"] / safe_90s
    df["assists_per_90"] = df["cum_assists"] / safe_90s
    df["cs_rate"] = df["cum_cs"] / safe_90s

    if "expected_goals" in df.columns:
        df["cum_xG"] = group["expected_goals"].transform(lambda x: x.shift(1).cumsum())
        df["cum_xA"] = group["expected_assists"].transform(lambda x: x.shift(1).cumsum())
        df["xG_per_90"] = df["cum_xG"] / safe_90s
        df["xA_per_90"] = df["cum_xA"] / safe_90s

    # Net transfers
    df["net_transfers"] = df["transfers_in"] - df["transfers_out"]

    # Value (pris)
    if "value" in df.columns:
        df["price"] = df["value"].astype(float) / 10.0
    else:
        df["price"] = 5.0

    # Selected by
    if "selected" in df.columns:
        df["selected_pct"] = pd.to_numeric(df["selected"], errors="coerce").fillna(0)
    else:
        df["selected_pct"] = 0

    # Target: næste GWs point (shift -1 inden for samme spiller+sæson)
    df["target"] = group["total_points"].shift(-1)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Returnerer liste af feature-kolonner."""
    features = [
        "pos_code", "is_home", "price", "selected_pct", "net_transfers",
        "cum_minutes", "goals_per_90", "assists_per_90", "cs_rate",
    ]

    # Rolling features
    for window in [3, 5]:
        w = f"_{window}gw"
        features += [f"roll_pts{w}", f"roll_min{w}", f"roll_bps{w}", f"roll_ict{w}", f"roll_bonus{w}"]
        if f"roll_xG{w}" in df.columns:
            features += [f"roll_xG{w}", f"roll_xA{w}"]

    # Per-90 xG features
    if "xG_per_90" in df.columns:
        features += ["xG_per_90", "xA_per_90"]

    if "opponent_team_id" in df.columns:
        features.append("opponent_team_id")

    # Returnér kun kolonner der faktisk findes
    return [f for f in features if f in df.columns]


def train_and_save():
    """Hovedfunktion: hent data, træn model, gem."""
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    import joblib

    # 1. Hent data
    raw = load_all_seasons()
    print(f"Rå data: {len(raw)} rækker, {len(raw.columns)} kolonner")

    # 2. Feature engineering
    df = engineer_features(raw)

    # 3. Fjern rækker uden target eller med for lidt data
    df = df.dropna(subset=["target"])
    df = df[df["cum_minutes"] > 90]  # Mindst 1 kamp spillet
    print(f"Efter filtrering: {len(df)} rækker")

    # 4. Features
    feature_cols = get_feature_columns(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].fillna(0)
    y = df["target"]

    # 5. Train/val split (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        print(f"  Fold {fold}: MAE = {mae:.3f}")

    avg_mae = np.mean(maes)
    print(f"\nGennemsnitlig MAE: {avg_mae:.3f}")

    # 6. Træn final model på al data
    print("Træner final model...")
    final_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        verbose=-1,
    )
    final_model.fit(X, y)

    # 7. Feature importance
    importances = dict(zip(feature_cols, final_model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature importance:")
    for name, imp in sorted_imp[:10]:
        print(f"  {name}: {imp}")

    # 8. Gem
    model_path = MODEL_DIR / "ep_model.joblib"
    features_path = MODEL_DIR / "feature_columns.joblib"
    meta_path = MODEL_DIR / "model_meta.json"

    joblib.dump(final_model, model_path)
    joblib.dump(feature_cols, features_path)

    meta = {
        "seasons": SEASONS,
        "n_features": len(feature_cols),
        "n_training_rows": len(X),
        "avg_mae": round(avg_mae, 3),
        "feature_importance": {k: int(v) for k, v in sorted_imp},
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nModel gemt: {model_path}")
    print(f"Features gemt: {features_path}")
    print(f"Metadata gemt: {meta_path}")
    print(f"Model størrelse: {model_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    train_and_save()
