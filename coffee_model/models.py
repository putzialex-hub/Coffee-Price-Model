"""Quantile model training and prediction. Supports sklearn and LightGBM."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .config import COMMODITIES, HORIZONS, MODEL_CONFIGS, QUANTILES
from .features import create_xy_log_returns, fill_remaining_nan_with_zero


def _build_estimator(commodity: str, alpha: float, engine: str):
    cfg = MODEL_CONFIGS[commodity][engine].copy()
    if engine == "sklearn":
        return GradientBoostingRegressor(loss="quantile", alpha=alpha, **cfg)
    if engine == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("lightgbm not installed; pip install lightgbm")
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **cfg)
    raise ValueError(f"unknown engine: {engine}")


def _prepare_xy(df: pd.DataFrame, commodity: str, engine: str):
    if engine == "sklearn":
        df = fill_remaining_nan_with_zero(df)
    X, y, fcols, tcols = create_xy_log_returns(df, commodity=commodity)
    return X, y, fcols, tcols


def train_models(
    df: pd.DataFrame,
    engine: str,
    commodities: list[str] = COMMODITIES,
    horizons: list[int] = HORIZONS,
    quantiles: list[float] = QUANTILES,
) -> dict[tuple, Any]:
    """Train one model per (commodity, horizon, quantile)."""
    models: dict[tuple, Any] = {}
    for comm in commodities:
        X, y, fcols, _ = _prepare_xy(df, comm, engine)
        for h in horizons:
            target = f"target_ret_{comm}_{h}d"
            y_h = y[target]
            for q in quantiles:
                est = _build_estimator(comm, q, engine)
                est.fit(X, y_h)
                models[(comm, h, q)] = est
        models[(comm, "feature_cols")] = fcols
    return models


def _last_row_for_predict(df: pd.DataFrame, fcols: list[str], engine: str) -> pd.DataFrame:
    """Pick the most recent row that has all feature_cols populated."""
    if engine == "sklearn":
        df = fill_remaining_nan_with_zero(df)
    return df.iloc[[-1]][fcols]


def predict_prices(
    models: dict[tuple, Any],
    df: pd.DataFrame,
    engine: str,
    commodities: list[str] = COMMODITIES,
    horizons: list[int] = HORIZONS,
    quantiles: list[float] = QUANTILES,
) -> dict[tuple, float]:
    """Return {(commodity, horizon, quantile): predicted_price}.

    Quantile crossing (lower>base or base>upper) is corrected by sorting the
    three quantile predictions per (commodity, horizon).
    """
    preds: dict[tuple, float] = {}
    last_prices = df.iloc[-1][commodities]
    for comm in commodities:
        fcols = models[(comm, "feature_cols")]
        last_row = _last_row_for_predict(df, fcols, engine)
        start = last_prices[comm]
        for h in horizons:
            q_preds = {q: float(models[(comm, h, q)].predict(last_row)[0]) for q in quantiles}
            sorted_q = sorted(quantiles)
            sorted_logs = sorted(q_preds[q] for q in sorted_q)
            for q, log_ret in zip(sorted_q, sorted_logs):
                preds[(comm, h, q)] = start * np.exp(log_ret)
    return preds
