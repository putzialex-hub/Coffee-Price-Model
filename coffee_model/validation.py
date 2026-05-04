"""Walk-forward backtesting with full quantile fit and pinball/coverage metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMMODITIES, HORIZONS, QUANTILES
from .features import all_feature_columns, get_commodity_features, fill_remaining_nan_with_zero
from .models import _build_estimator


def _pinball_loss(actual: float, pred: float, alpha: float) -> float:
    diff = actual - pred
    return alpha * diff if diff >= 0 else (alpha - 1) * diff


def walk_forward(
    df: pd.DataFrame,
    engine: str,
    val_dates: pd.DatetimeIndex | None = None,
    commodities: list[str] = COMMODITIES,
    horizons: list[int] = HORIZONS,
    quantiles: list[float] = QUANTILES,
    min_history_days: int = 365,
    min_train_rows: int = 500,
) -> pd.DataFrame:
    """Quarterly walk-forward across the full history.

    For each cutoff date and each (commodity, horizon) we fit *all three*
    quantile models — same configuration as the live model — and compare to
    the actual price `horizon` days later. Returns a long-format DataFrame
    with one row per (cutoff, commodity, horizon).
    """
    if val_dates is None:
        val_dates = pd.date_range("2022-01-01", "2025-07-01", freq="3ME")

    rows = []
    for val_cutoff in val_dates:
        val_cutoff = pd.Timestamp(val_cutoff)
        if val_cutoff < df["date"].min() + pd.Timedelta(days=min_history_days):
            continue
        if val_cutoff + pd.Timedelta(days=max(horizons)) > df["date"].max():
            continue

        train_data = df[df["date"] <= val_cutoff].copy()
        if len(train_data) < min_train_rows:
            continue

        # Build per-commodity targets up-front
        for h in horizons:
            for comm in commodities:
                train_data[f"target_{comm}_{h}d"] = np.log(
                    train_data[comm].shift(-h) / train_data[comm]
                )

        if engine == "sklearn":
            train_data = fill_remaining_nan_with_zero(train_data)

        feature_cols_val = all_feature_columns(train_data)

        for comm in commodities:
            fcols = get_commodity_features(feature_cols_val, comm)
            for h in horizons:
                target = f"target_{comm}_{h}d"
                clean = train_data.dropna(subset=[target, comm])
                if len(clean) < 100:
                    continue
                X_val = clean[fcols]
                y_val = clean[target]

                # Need a row for prediction with valid features
                last_row_val = clean.iloc[[-1]][fcols]
                start_price = clean.iloc[-1][comm]

                target_date = val_cutoff + pd.Timedelta(days=h)
                actual_df = df[df["date"] >= target_date]
                if len(actual_df) == 0:
                    continue
                actual_price = actual_df.iloc[0][comm]

                q_preds_log = {}
                for q in quantiles:
                    est = _build_estimator(comm, q, engine)
                    est.fit(X_val, y_val)
                    q_preds_log[q] = float(est.predict(last_row_val)[0])

                # Sort to enforce monotone quantiles
                sorted_qs = sorted(quantiles)
                sorted_logs = sorted(q_preds_log[q] for q in sorted_qs)
                price_by_q = {q: start_price * np.exp(lp)
                              for q, lp in zip(sorted_qs, sorted_logs)}

                base_q = 0.5 if 0.5 in quantiles else sorted_qs[len(sorted_qs) // 2]
                pred_price = price_by_q[base_q]
                error_pct = (pred_price - actual_price) / actual_price * 100
                pinball_base = _pinball_loss(actual_price, pred_price, base_q)
                pinball_low = _pinball_loss(actual_price, price_by_q[min(quantiles)], min(quantiles))
                pinball_high = _pinball_loss(actual_price, price_by_q[max(quantiles)], max(quantiles))
                in_interval = price_by_q[min(quantiles)] <= actual_price <= price_by_q[max(quantiles)]

                rows.append({
                    "cutoff": val_cutoff,
                    "commodity": comm,
                    "horizon": h,
                    "predicted": pred_price,
                    "predicted_low": price_by_q[min(quantiles)],
                    "predicted_high": price_by_q[max(quantiles)],
                    "actual": actual_price,
                    "error_pct": error_pct,
                    "pinball_base": pinball_base,
                    "pinball_low": pinball_low,
                    "pinball_high": pinball_high,
                    "in_interval": int(in_interval),
                })

    return pd.DataFrame(rows)


def summarise(val_df: pd.DataFrame) -> dict[tuple, dict]:
    """Return per (commodity, horizon) summary metrics."""
    out: dict[tuple, dict] = {}
    if val_df.empty:
        return out
    for comm in val_df["commodity"].unique():
        for h in sorted(val_df["horizon"].unique()):
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)]
            if sub.empty:
                continue
            out[(comm, h)] = {
                "mae": sub["error_pct"].abs().mean(),
                "bias": sub["error_pct"].mean(),
                "std": sub["error_pct"].std(),
                "hit_rate_15": (sub["error_pct"].abs() < 15).mean() * 100,
                "coverage_90": sub["in_interval"].mean() * 100,
                "pinball_base": sub["pinball_base"].mean(),
                "pinball_low": sub["pinball_low"].mean(),
                "pinball_high": sub["pinball_high"].mean(),
                "n": len(sub),
            }
    return out


def bias_corrections(val_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Mean log-return error per (commodity, horizon), used as a live offset.

    We compute the log-space bias (predicted/actual) and apply it as a
    multiplicative correction. The 6.4% Robusta bias documented in v5 motivates
    this; the correction is small and is logged in the output CSV.
    """
    out: dict[tuple[str, int], float] = {}
    if val_df.empty:
        return out
    for comm in val_df["commodity"].unique():
        for h in sorted(val_df["horizon"].unique()):
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)]
            if sub.empty:
                continue
            log_bias = np.mean(np.log(sub["predicted"] / sub["actual"]))
            out[(comm, int(h))] = float(log_bias)
    return out
