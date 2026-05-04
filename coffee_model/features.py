"""Feature engineering, commodity-specific feature filtering, target builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMMODITIES, HORIZONS

ARABICA_ONLY_PATTERNS = (
    "rain_90d_mg", "dry_streak_mg", "temp_min_mg",
    "brazil_drought_risk", "brazil_frost_season", "brazil_harvest",
    "arabica_stocks",
)

ROBUSTA_ONLY_PATTERNS = (
    "rain_90d_dl", "dry_streak_dl",
    "vietnam_drought_risk", "vietnam_critical_rain", "vietnam_harvest",
    "robusta_stocks",
)

EXCLUDE_COLS = ("date", "arabica_price", "robusta_price")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all engineered features in-place and return df."""
    df = df.copy()

    if "COT_Net_Spec" in df.columns:
        df["COT_Signal"] = df["COT_Net_Spec"].rolling(4).mean()
        df["COT_zscore"] = (
            df["COT_Net_Spec"] - df["COT_Net_Spec"].rolling(252).mean()
        ) / df["COT_Net_Spec"].rolling(252).std()
        df["COT_momentum_4w"] = df["COT_Net_Spec"].diff(20)

    if "rain_90d_mg" in df.columns:
        df["rain_90d_mg_lag90"] = df["rain_90d_mg"].shift(90)
        df["rain_90d_mg_lag180"] = df["rain_90d_mg"].shift(180)
        df["dry_streak_mg_lag30"] = df["dry_streak_mg"].shift(30)

    if "rain_90d_dl" in df.columns:
        df["rain_90d_dl_lag90"] = df["rain_90d_dl"].shift(90)
        df["rain_90d_dl_lag180"] = df["rain_90d_dl"].shift(180)
        df["dry_streak_dl_lag30"] = df["dry_streak_dl"].shift(30)

    if "arabica_stocks" in df.columns:
        df["arabica_stocks_change_30d"] = df["arabica_stocks"].pct_change(30)
        df["arabica_stocks_change_90d"] = df["arabica_stocks"].pct_change(90)
        df["arabica_stocks_zscore"] = (
            df["arabica_stocks"] - df["arabica_stocks"].rolling(252).mean()
        ) / df["arabica_stocks"].rolling(252).std()
        df["arabica_stocks_trend"] = (
            df["arabica_stocks"].rolling(30).mean()
            / df["arabica_stocks"].rolling(90).mean() - 1
        )

    if "robusta_stocks" in df.columns:
        df["robusta_stocks_change_30d"] = df["robusta_stocks"].pct_change(30)
        df["robusta_stocks_change_90d"] = df["robusta_stocks"].pct_change(90)
        df["robusta_stocks_zscore"] = (
            df["robusta_stocks"] - df["robusta_stocks"].rolling(252).mean()
        ) / df["robusta_stocks"].rolling(252).std()
        df["robusta_stocks_trend"] = (
            df["robusta_stocks"].rolling(30).mean()
            / df["robusta_stocks"].rolling(90).mean() - 1
        )

    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["brazil_harvest"] = month.isin([4, 5, 6, 7, 8, 9]).astype(int)
    df["vietnam_harvest"] = month.isin([10, 11, 12, 1]).astype(int)

    if "arabica_price" in df.columns and "robusta_price" in df.columns:
        df["arabica_robusta_ratio"] = df["arabica_price"] / (df["robusta_price"] * 0.0453)
        df["spread_trend_30d"] = df["arabica_robusta_ratio"].pct_change(30) * 100
        df["spread_zscore"] = (
            df["arabica_robusta_ratio"] - df["arabica_robusta_ratio"].rolling(252).mean()
        ) / df["arabica_robusta_ratio"].rolling(252).std()

    if "rain_90d_dl" in df.columns:
        df["vietnam_critical_rain"] = df["rain_90d_dl_lag180"]
        df["vietnam_drought_risk"] = (df["dry_streak_dl"] > 14).astype(int)

    if "rain_90d_mg" in df.columns:
        df["brazil_frost_season"] = month.isin([5, 6, 7, 8]).astype(int)
        df["brazil_drought_risk"] = (df["dry_streak_mg"] > 21).astype(int)

    # ENSO (ONI) — global feature for both commodities. ENSO peaks lead
    # Brazil/Vietnam precipitation anomalies by 6–9 months.
    if "oni" in df.columns:
        df["oni_lag6m"] = df["oni"].shift(180)
        df["oni_lag9m"] = df["oni"].shift(270)
        if "brazil_drought_risk" in df.columns:
            df["oni_x_brazil_drought"] = df["oni"] * df["brazil_drought_risk"]
        if "vietnam_drought_risk" in df.columns:
            df["oni_x_vietnam_drought"] = df["oni"] * df["vietnam_drought_risk"]

    essential_cols = ["arabica_price", "robusta_price", "month_sin", "month_cos"]
    df = df.dropna(subset=essential_cols).reset_index(drop=True)
    return df


def fill_remaining_nan_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Sklearn engine cannot handle NaN; LightGBM can. Caller decides."""
    df = df.copy()
    feature_cols_all = [c for c in df.columns if c not in EXCLUDE_COLS]
    df[feature_cols_all] = df[feature_cols_all].fillna(0)
    return df


def get_commodity_features(all_features: list[str], commodity: str) -> list[str]:
    """Drop features that belong to the *other* commodity's region/stocks."""
    if "arabica" in commodity:
        return [f for f in all_features
                if not any(pat in f for pat in ROBUSTA_ONLY_PATTERNS)]
    return [f for f in all_features
            if not any(pat in f for pat in ARABICA_ONLY_PATTERNS)]


def all_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and "target" not in c]


def create_xy_log_returns(
    data: pd.DataFrame,
    commodity: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build (X, y, feature_cols, target_cols) using log-returns as targets."""
    X_d = data.copy()
    all_fcols = all_feature_columns(X_d)
    feature_cols = get_commodity_features(all_fcols, commodity) if commodity else all_fcols
    target_cols = []
    for comm in COMMODITIES:
        for h in HORIZONS:
            t_col = f"target_ret_{comm}_{h}d"
            X_d[t_col] = np.log(X_d[comm].shift(-h) / X_d[comm])
            target_cols.append(t_col)
    X_d = X_d.dropna(subset=target_cols + [commodity] if commodity else target_cols)
    return X_d[feature_cols], X_d[target_cols], feature_cols, target_cols
