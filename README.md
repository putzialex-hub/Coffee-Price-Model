# Coffee Price Model

Forecasts Arabica and Robusta coffee prices for 30 / 90 / 180 days using a
quantile gradient-boosting model. See `MODEL_DOCUMENTATION.md` for the full
methodology.

## Setup

```bash
pip install -r requirements.txt
# optional, only if you have an LSEG Workspace:
pip install -r requirements-optional.txt
```

Python 3.11 is recommended.

## Run

```bash
python -m coffee_model.main
# or, equivalently, using the legacy entry point:
python main_forecast_model.py
```

Outputs (written next to the script):

- `coffee_forecast_pro.csv` — Low / Base / High forecasts for 30, 90, 180 days
- `validation_results.csv` — walk-forward backtest metrics
- `forecast_pro_chart.png`, `backtest_performance_chart.png`

## Data sources

| File | Source | Update script |
| --- | --- | --- |
| `arabica_clean.csv`, `robusta_clean.csv` | ICE futures (manual) | — |
| `weather_minas_gerais.csv`, `weather_dak_lak.csv` | Open-Meteo | `update_weather_data.py` |
| `arabica_stocks.csv`, `robusta_stocks.csv` | ICE certified stocks (manual) | — |
| `cot_data.csv` | CFTC Commitments of Traders | `fetch_cot_data.py` |
| `ONI dataset.csv` | NOAA Oceanic Niño Index | — |
