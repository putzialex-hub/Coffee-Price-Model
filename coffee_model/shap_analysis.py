"""SHAP-based driver decomposition for the live forecast.

Produces a per-(commodity, horizon=180d) bar chart showing which features
pushed the 6-month price forecast above or below the current price,
expressed as their contribution to the log-return prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# Human-readable labels for the most common feature names
FEATURE_LABELS: dict[str, str] = {
    "COT_Signal": "COT Sentiment (4w avg)",
    "COT_zscore": "COT z-score vs 1y",
    "COT_momentum_4w": "COT momentum (4w)",
    "rain_90d_mg_lag90": "Brazil rain (lag 90d)",
    "rain_90d_mg_lag180": "Brazil rain (lag 180d)",
    "dry_streak_mg_lag30": "Brazil dry streak (lag 30d)",
    "rain_90d_dl_lag90": "Vietnam rain (lag 90d)",
    "rain_90d_dl_lag180": "Vietnam rain (lag 180d)",
    "dry_streak_dl_lag30": "Vietnam dry streak (lag 30d)",
    "arabica_stocks_change_30d": "Arabica stocks Δ30d",
    "arabica_stocks_change_90d": "Arabica stocks Δ90d",
    "arabica_stocks_zscore": "Arabica stocks z-score",
    "robusta_stocks_change_30d": "Robusta stocks Δ30d",
    "robusta_stocks_change_90d": "Robusta stocks Δ90d",
    "robusta_stocks_zscore": "Robusta stocks z-score",
    "arabica_robusta_ratio": "Arabica/Robusta spread",
    "spread_trend_30d": "Spread trend 30d",
    "spread_zscore": "Spread z-score",
    "month_sin": "Seasonality (sin)",
    "month_cos": "Seasonality (cos)",
    "brazil_harvest": "Brazil harvest season",
    "vietnam_harvest": "Vietnam harvest season",
    "brazil_drought_risk": "Brazil drought risk",
    "brazil_frost_season": "Brazil frost season",
    "vietnam_drought_risk": "Vietnam drought risk",
    "oni": "ONI / ENSO",
    "oni_3m_ma": "ONI 3-month avg",
    "oni_lag6m": "ONI lag 6m",
    "oni_lag9m": "ONI lag 9m",
    "el_nino": "El Niño flag",
    "la_nina": "La Niña flag",
    "oni_x_brazil_drought": "ENSO × Brazil drought",
    "oni_x_vietnam_drought": "ENSO × Vietnam drought",
    "usd_brl_change_30d": "USD/BRL change 30d (%)",
    "usd_brl_change_90d": "USD/BRL change 90d (%)",
    "usd_brl_zscore": "USD/BRL z-score",
    "dxy_change_30d": "DXY change 30d (%)",
    "dxy_zscore": "DXY z-score",
    "USD_BRL": "USD/BRL spot",
    "DXY": "US Dollar Index",
}


def compute_shap(
    model,
    last_row: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 8,
) -> pd.DataFrame | None:
    """Return a DataFrame with top_n features by |SHAP value| for the last row."""
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(last_row)
        if hasattr(shap_values, "__iter__") and not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        df = pd.DataFrame({
            "feature": feature_names,
            "shap": shap_values,
            "abs_shap": np.abs(shap_values),
        }).sort_values("abs_shap", ascending=False).head(top_n)
        df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
        return df
    except Exception:
        return None


def shap_for_forecast(
    models_final: dict,
    df: pd.DataFrame,
    engine: str,
    horizon: int = 180,
    top_n: int = 8,
) -> dict[str, pd.DataFrame]:
    """Return per-commodity SHAP DataFrames for the 180d base-case model."""
    from .config import COMMODITIES
    from .features import fill_remaining_nan_with_zero

    if not SHAP_AVAILABLE:
        return {}

    results: dict[str, pd.DataFrame] = {}
    for comm in COMMODITIES:
        model = models_final.get((comm, horizon, 0.5))
        if model is None:
            continue
        fcols = models_final.get((comm, "feature_cols"), [])
        df_work = fill_remaining_nan_with_zero(df) if engine == "sklearn" else df.copy()
        last_row = df_work.iloc[[-1]][fcols]
        shap_df = compute_shap(model, last_row, fcols, top_n=top_n)
        if shap_df is not None:
            results[comm] = shap_df
    return results


def plot_shap_drivers(
    shap_results: dict[str, pd.DataFrame],
    current_prices: dict[str, float],
    horizon: int = 180,
    out_path: str | None = None,
) -> str | None:
    """Horizontal bar chart: SHAP contributions to the 180d log-return forecast."""
    if not MPL_AVAILABLE or not shap_results:
        return None
    from .config import OUTPUT_DIR

    out_path = out_path or f"{OUTPUT_DIR}/shap_drivers_chart.png"
    n_plots = len(shap_results)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, (comm, shap_df) in zip(axes, shap_results.items()):
        comm_name = comm.replace("_price", "").capitalize()
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_df["shap"]]
        bars = ax.barh(
            range(len(shap_df)), shap_df["shap"].values[::-1],
            color=colors[::-1], edgecolor="none", height=0.7,
        )
        ax.set_yticks(range(len(shap_df)))
        ax.set_yticklabels(shap_df["label"].values[::-1], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP log-return contribution (push to ↑ / ↓ price)", fontsize=9)
        ax.set_title(
            f"{comm_name} — Top drivers of {horizon}d forecast\n"
            f"(red = bullish, blue = bearish)",
            fontsize=11,
        )
        ax.grid(axis="x", alpha=0.3)
        # Legend patches
        bull = mpatches.Patch(color="#d62728", label="Bullish driver (↑ price)")
        bear = mpatches.Patch(color="#1f77b4", label="Bearish driver (↓ price)")
        ax.legend(handles=[bull, bear], fontsize=8, loc="lower right")

    plt.suptitle(
        f"Model Driver Decomposition — {horizon}-day forecast",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
