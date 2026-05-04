"""Entry point: orchestrates COT update, data load, train, backtest, and output."""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from .config import COMMODITIES, DATA_DIR, DEFAULT_ENGINE, HORIZONS, OUTPUT_DIR, QUANTILES
from .data_loading import load_data, update_cot_data
from .features import add_features
from .models import train_models, predict_prices
from .validation import bias_corrections, summarise, walk_forward
from .plotting import plot_backtest, plot_forecast

warnings.filterwarnings("ignore")

# Optional news monitor (lives at the repo root, not in the package)
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from news_monitor import get_coffee_news_sentiment, check_news_model_alignment
    NEWS_MONITOR_AVAILABLE = True
except ImportError:
    NEWS_MONITOR_AVAILABLE = False


def _print_banner(engine: str) -> None:
    print("🚀 Starte Kaffeepreis-Prognose (Fundamental-Fokus, Commodity-spezifisch)")
    print("=========================================================================")
    print(f"   Engine: {engine}")


def _print_summary(summary: dict) -> None:
    if not summary:
        return
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION ERGEBNISSE")
    print("=" * 60)
    last_comm = None
    for (comm, h), m in summary.items():
        if comm != last_comm:
            print(f"\n{comm.upper()}:")
            last_comm = comm
        print(f"   {h:3d}d: MAE={m['mae']:5.1f}%, Bias={m['bias']:+5.1f}%, "
              f"Hit-Rate(±15%)={m['hit_rate_15']:4.1f}%, Coverage(90%)={m['coverage_90']:4.1f}%, "
              f"Pinball={m['pinball_base']:.4f} (n={m['n']})")


def _apply_bias_correction(preds: dict, bias: dict[tuple[str, int], float]) -> dict:
    """Subtract mean log-bias from each (commodity, horizon) prediction."""
    if not bias:
        return preds
    corrected = {}
    for (comm, h, q), price in preds.items():
        log_bias = bias.get((comm, int(h)), 0.0)
        corrected[(comm, h, q)] = price * np.exp(-log_bias)
    return corrected


def run(engine: str = DEFAULT_ENGINE,
        skip_cot_update: bool = False,
        skip_backtest: bool = False) -> None:
    _print_banner(engine)

    if not skip_cot_update:
        update_cot_data(DATA_DIR)

    df_raw = load_data(DATA_DIR)
    df = add_features(df_raw)
    print(f"   ✅ Feature Engineering abgeschlossen ({len(df)} Zeilen)")

    val_df = pd.DataFrame()
    summary = {}
    bias = {}
    if not skip_backtest:
        print("\n📊 Walk-Forward Validation (2022-2025)...")
        val_df = walk_forward(df, engine=engine)
        summary = summarise(val_df)
        _print_summary(summary)
        bias = bias_corrections(val_df)
        if not val_df.empty:
            val_df.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)

    cutoff_date = df["date"].max() - pd.Timedelta(days=180)
    train_cutoff = df[df["date"] <= cutoff_date]
    print(f"\n   Trainiere Modell bis {cutoff_date.date()} (held-out, commodity-spezifisch)...")
    _ = train_models(train_cutoff, engine=engine)  # held-out for parity with v5

    print("   Trainiere Finales Modell (Full Data)...")
    models_final = train_models(df, engine=engine)
    preds_today = predict_prices(models_final, df, engine=engine)

    if bias:
        preds_today = _apply_bias_correction(preds_today, bias)

    news_sentiment = None
    if NEWS_MONITOR_AVAILABLE:
        print("\n📰 Prüfe News-Sentiment...")
        try:
            news_sentiment = get_coffee_news_sentiment(days_back=7, verbose=True)
            current_arabica = df.iloc[-1]["arabica_price"]
            arabica_dir = "BULLISH" if preds_today[("arabica_price", 90, 0.5)] > current_arabica else "BEARISH"
            alignment = check_news_model_alignment(news_sentiment, arabica_dir)
            if not alignment["aligned"]:
                print(f"\n🚨 {alignment['warning']}")
            else:
                print("\n✅ News und Modell sind aligned")
        except Exception as e:
            print(f"\n⚠️ News-Check Fehler: {e}")

    print("\n📈 Erstelle Charts...")
    chart = plot_forecast(df, preds_today, val_df, news_sentiment)
    print(f"✅ Forecast-Chart: {chart}")
    bt = plot_backtest(val_df)
    if bt:
        print(f"✅ Backtest-Chart: {bt}")

    today = df["date"].iloc[-1]
    rows = []
    for h in HORIZONS:
        row = {"Horizon_Days": h, "Date": (today + pd.Timedelta(days=h)).date()}
        for comm in COMMODITIES:
            row[f"{comm}_Low"] = preds_today[(comm, h, 0.05)]
            row[f"{comm}_Base"] = preds_today[(comm, h, 0.50)]
            row[f"{comm}_High"] = preds_today[(comm, h, 0.95)]
            row[f"{comm}_BiasCorrection"] = bias.get((comm, h), 0.0)
        rows.append(row)
    out_csv = os.path.join(OUTPUT_DIR, "coffee_forecast_pro.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ CSV exportiert: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coffee price quantile forecast")
    parser.add_argument("--engine", default=DEFAULT_ENGINE,
                        choices=["sklearn", "lightgbm"],
                        help="Model engine (default: lightgbm)")
    parser.add_argument("--skip-cot-update", action="store_true",
                        help="Skip the CFTC COT data refresh")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip the walk-forward backtest")
    args = parser.parse_args()
    run(engine=args.engine,
        skip_cot_update=args.skip_cot_update,
        skip_backtest=args.skip_backtest)


if __name__ == "__main__":
    main()
