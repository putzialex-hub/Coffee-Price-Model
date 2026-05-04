"""Forecast and backtest charts."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from .config import COMMODITIES, HORIZONS, OUTPUT_DIR


def _safe_drought_mg(df: pd.DataFrame) -> str:
    if "dry_streak_mg" not in df.columns or "rain_90d_mg" not in df.columns:
        return ""
    return f" (Drought: {df.iloc[-1]['dry_streak_mg']:.0f}d, 3M-Rain: {df.iloc[-1]['rain_90d_mg']:.1f}mm)"


def plot_forecast(
    df: pd.DataFrame,
    preds_today: dict,
    val_df: pd.DataFrame,
    news_sentiment: dict | None,
    out_path: str = None,
) -> str:
    out_path = out_path or os.path.join(OUTPUT_DIR, "forecast_pro_chart.png")
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    today = df["date"].iloc[-1]

    for i, comm in enumerate(COMMODITIES):
        ax = axes[i]
        mask_hist = df["date"] > (df["date"].max() - pd.Timedelta(days=365))
        ax.plot(df.loc[mask_hist, "date"], df.loc[mask_hist, comm],
                label="Price History", color="black", lw=1.5)

        future_dates = [today + pd.Timedelta(days=h) for h in HORIZONS]
        y_base = [preds_today[(comm, h, 0.5)] for h in HORIZONS]
        y_lower = [preds_today[(comm, h, 0.05)] for h in HORIZONS]
        y_upper = [preds_today[(comm, h, 0.95)] for h in HORIZONS]
        ax.plot(future_dates, y_base, "o--", color="blue",
                label="Forecast (Base Case)", lw=2)
        ax.fill_between(future_dates, y_lower, y_upper, color="blue", alpha=0.15,
                        label="90% Confidence Interval")

        if not val_df.empty:
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == 90)].sort_values("cutoff")
            for _, row in sub.iterrows():
                pred_date = row["cutoff"] + pd.Timedelta(days=90)
                if pred_date > (df["date"].max() - pd.Timedelta(days=365)):
                    ax.plot(pred_date, row["predicted"], "x", color="orange",
                            markersize=8, alpha=0.7)
                    ax.plot(pred_date, row["actual"], "o", color="green",
                            markersize=5, alpha=0.5)
            ax.plot([], [], "x", color="orange", markersize=8,
                    label="Backtest Predicted (90d)")
            ax.plot([], [], "o", color="green", markersize=5,
                    label="Backtest Actual")

        if news_sentiment and news_sentiment.get("total_score", 0) != 0:
            score = news_sentiment["total_score"]
            signal = news_sentiment["signal"]
            badge_color = "green" if "BULLISH" in signal else "red" if "BEARISH" in signal else "gray"
            ax.text(0.98, 0.98, f"News: {signal}\n({score:+.1f})",
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round", facecolor=badge_color, alpha=0.3))

        suffix = _safe_drought_mg(df) if comm == "arabica_price" else ""
        ax.set_title(f"{comm.replace('_price', '').capitalize()} PRO Forecast{suffix}\n"
                     "(Log-Return Model + Quantile Uncertainty)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_backtest(val_df: pd.DataFrame, out_path: str = None) -> str | None:
    if val_df.empty:
        return None
    out_path = out_path or os.path.join(OUTPUT_DIR, "backtest_performance_chart.png")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, comm in enumerate(COMMODITIES):
        comm_name = comm.replace("_price", "").capitalize()
        ax1 = axes[i, 0]
        for h, color, marker in [(30, "blue", "o"), (90, "orange", "s"), (180, "green", "^")]:
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)].copy()
            sub = sub.sort_values("cutoff")
            sub["target_date"] = sub["cutoff"] + pd.to_timedelta(sub["horizon"], unit="D")
            ax1.plot(sub["target_date"], sub["actual"], "-", color=color, alpha=0.3, linewidth=2)
            ax1.plot(sub["target_date"], sub["predicted"], "--", color=color, alpha=0.8, linewidth=1)
            ax1.scatter(sub["target_date"], sub["predicted"], color=color, marker=marker,
                        s=50, label=f"{h}d Predicted", alpha=0.8)
            ax1.scatter(sub["target_date"], sub["actual"], color=color, marker=marker,
                        s=20, facecolors="none", edgecolors=color,
                        label=f"{h}d Actual", alpha=0.5)
        ax1.set_title(f"{comm_name}: Predicted vs Actual")
        ax1.set_xlabel("Datum")
        ax1.set_ylabel("Preis")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[i, 1]
        for h, color in [(30, "blue"), (90, "orange"), (180, "green")]:
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)].copy()
            sub = sub.sort_values("cutoff")
            ax2.bar(sub["cutoff"], sub["error_pct"], width=20, alpha=0.6, color=color, label=f"{h}d Fehler")
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.axhline(y=sub["error_pct"].mean(), color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_title(f"{comm_name}: Prognose-Fehler über Zeit")
        ax2.set_xlabel("Cutoff Datum")
        ax2.set_ylabel("Fehler (%)")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhspan(-15, 15, alpha=0.1, color="green")
        ax2.set_ylim(-50, 50)

    plt.suptitle("Walk-Forward Backtest Performance (2022-2025)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
