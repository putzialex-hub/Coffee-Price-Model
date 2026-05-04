"""Data loading and merging from the various CSV sources."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import DATA_DIR, REPO_DIR


def should_update_cot(data_dir: str = DATA_DIR) -> bool:
    cot_file = os.path.join(data_dir, "cot_data.csv")
    if not os.path.exists(cot_file):
        return True
    file_time = datetime.fromtimestamp(os.path.getmtime(cot_file))
    if datetime.now() - file_time > timedelta(days=3):
        return True
    try:
        df_check = pd.read_csv(cot_file)
        last_date = pd.to_datetime(df_check["date"]).max()
        if datetime.now() - last_date > timedelta(days=7):
            return True
    except Exception:
        return True
    return False


def update_cot_data(data_dir: str = DATA_DIR) -> None:
    print("🔄 Prüfe COT-Daten...")
    if not should_update_cot(data_dir):
        print("   ✅ COT-Daten sind aktuell (< 7 Tage alt)")
        return
    print("   ⬇️ Lade aktuelle COT-Daten von CFTC...")
    try:
        cot_script_path = os.path.join(REPO_DIR, "fetch_cot_data.py")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, cot_script_path],
            cwd=REPO_DIR,
            capture_output=True, text=True,
            timeout=120, encoding="utf-8", env=env,
        )
        if result.returncode == 0:
            print("   ✅ COT-Daten aktualisiert!")
        else:
            print(f"   ❌ Fehler beim Update. Code: {result.returncode}")
            print(f"   Fehlermeldung: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Kritischer Fehler: {e}")


def _load_oni(data_dir: str) -> pd.DataFrame | None:
    """Load NOAA Oceanic Niño Index, monthly → daily forward-fill.

    The CSV in the repo has years as rows and months (DJF, JFM, ..., NDJ) as
    columns. Each label is the centred 3-month season, so DJF for year Y
    refers to Dec(Y-1)/Jan(Y)/Feb(Y); we anchor the value at the middle month.
    """
    path = os.path.join(data_dir, "ONI dataset.csv")
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path)
    year_col = raw.columns[0]
    season_cols = [c for c in raw.columns if c != year_col]
    season_to_month = {
        "DJF": 1,  "JFM": 2,  "FMA": 3,  "MAM": 4,
        "AMJ": 5,  "MJJ": 6,  "JJA": 7,  "JAS": 8,
        "ASO": 9,  "SON": 10, "OND": 11, "NDJ": 12,
    }
    records = []
    for _, row in raw.iterrows():
        try:
            year = int(row[year_col])
        except (ValueError, TypeError):
            continue
        for season in season_cols:
            month = season_to_month.get(season.upper())
            if month is None:
                continue
            value = pd.to_numeric(row[season], errors="coerce")
            if pd.isna(value):
                continue
            records.append({
                "date": pd.Timestamp(year=year, month=month, day=15),
                "oni": float(value),
            })
    if not records:
        return None
    monthly = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    daily_index = pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="D")
    daily = monthly.set_index("date").reindex(daily_index).ffill().rename_axis("date").reset_index()
    daily["oni_3m_ma"] = daily["oni"].rolling(90).mean()
    daily["el_nino"] = (daily["oni"] > 0.5).astype(int)
    daily["la_nina"] = (daily["oni"] < -0.5).astype(int)
    return daily


def load_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    arabica = pd.read_csv(os.path.join(data_dir, "arabica_clean.csv"))
    robusta = pd.read_csv(os.path.join(data_dir, "robusta_clean.csv"))
    arabica["date"] = pd.to_datetime(arabica["date"])
    robusta["date"] = pd.to_datetime(robusta["date"])

    df = pd.merge(
        arabica[["date", "price"]].rename(columns={"price": "arabica_price"}),
        robusta[["date", "price"]].rename(columns={"price": "robusta_price"}),
        on="date", how="outer",
    )

    path_macro = os.path.join(data_dir, "coffee_data.csv")
    if os.path.exists(path_macro):
        macro = pd.read_csv(path_macro)
        macro["date"] = pd.to_datetime(macro["Date"])
        df = pd.merge(df, macro[["date", "USD_BRL", "DXY"]], on="date", how="left")

    path_stocks_a = os.path.join(data_dir, "arabica_stocks.csv")
    if os.path.exists(path_stocks_a):
        st_a = pd.read_csv(path_stocks_a, sep=";", skiprows=[1])
        st_a["date"] = pd.to_datetime(st_a["Date"], format="%d.%m.%Y", errors="coerce")
        st_a["Certified_Stocks"] = pd.to_numeric(st_a["Certified_Stocks"], errors="coerce")
        df = pd.merge(df, st_a[["date", "Certified_Stocks"]].rename(
            columns={"Certified_Stocks": "arabica_stocks"}), on="date", how="left")

    path_stocks_r = os.path.join(data_dir, "robusta_stocks.csv")
    if os.path.exists(path_stocks_r):
        st_r = pd.read_csv(path_stocks_r, sep=";", skiprows=[1])
        st_r["date"] = pd.to_datetime(st_r["Date"], format="%d.%m.%Y", errors="coerce")
        col_s = [c for c in st_r.columns if c != "Date"][0]
        st_r[col_s] = pd.to_numeric(st_r[col_s], errors="coerce")
        df = pd.merge(df, st_r[["date", col_s]].rename(
            columns={col_s: "robusta_stocks"}), on="date", how="left")

    path_cot = os.path.join(data_dir, "cot_data.csv")
    if os.path.exists(path_cot):
        cot = pd.read_csv(path_cot)
        cot["date"] = pd.to_datetime(cot["date"])
        df = pd.merge(df, cot, on="date", how="left")

    path_w_mg = os.path.join(data_dir, "weather_minas_gerais.csv")
    if os.path.exists(path_w_mg):
        w_mg = pd.read_csv(path_w_mg)
        w_mg["date"] = pd.to_datetime(w_mg["date"])
        w_mg["rain_90d_mg"] = w_mg["precipitation_sum"].rolling(90).sum()
        is_rainy = w_mg["precipitation_sum"] >= 1.0
        streak_id = is_rainy.cumsum()
        w_mg["dry_streak_mg"] = w_mg.groupby(streak_id).cumcount()
        w_mg.loc[is_rainy, "dry_streak_mg"] = 0
        df = pd.merge(df, w_mg[["date", "temperature_2m_min", "rain_90d_mg", "dry_streak_mg"]].rename(
            columns={"temperature_2m_min": "temp_min_mg"}), on="date", how="left")
        print("   ✅ Wetter Minas Gerais geladen!")

    path_w_dl = os.path.join(data_dir, "weather_dak_lak.csv")
    if os.path.exists(path_w_dl):
        w_dl = pd.read_csv(path_w_dl)
        w_dl["date"] = pd.to_datetime(w_dl["date"])
        w_dl["rain_90d_dl"] = w_dl["precipitation_sum"].rolling(90).sum()
        is_rainy_dl = w_dl["precipitation_sum"] >= 1.0
        streak_id_dl = is_rainy_dl.cumsum()
        w_dl["dry_streak_dl"] = w_dl.groupby(streak_id_dl).cumcount()
        w_dl.loc[is_rainy_dl, "dry_streak_dl"] = 0
        df = pd.merge(df, w_dl[["date", "rain_90d_dl", "dry_streak_dl"]], on="date", how="left")
        print("   ✅ Wetter Dak Lak geladen!")

    oni = _load_oni(data_dir)
    if oni is not None:
        df = pd.merge(df, oni, on="date", how="left")
        print("   ✅ ONI/ENSO geladen!")

    df = df.sort_values("date").reset_index(drop=True)

    # Bounded forward-fill: prices 5d, weekly series 7d, weather 3d, monthly
    # ONI 35d. NO backfill — never pull future values backwards.
    price_cols = ["arabica_price", "robusta_price"]
    cot_stock_cols = [c for c in df.columns if any(x in c for x in ["COT_", "stocks", "Stocks"])]
    weather_cols = [c for c in df.columns if any(x in c for x in ["rain_", "dry_streak", "temp_min"])]
    oni_cols = [c for c in df.columns if c.startswith("oni") or c in ("el_nino", "la_nina")]
    other_cols = [c for c in df.columns
                  if c != "date"
                  and c not in price_cols + cot_stock_cols + weather_cols + oni_cols]

    df[price_cols] = df[price_cols].ffill(limit=5)
    df[cot_stock_cols] = df[cot_stock_cols].ffill(limit=7)
    df[weather_cols] = df[weather_cols].ffill(limit=3)
    if oni_cols:
        df[oni_cols] = df[oni_cols].ffill(limit=35)
    df[other_cols] = df[other_cols].ffill(limit=7)

    df = df[df["date"] >= "2016-01-01"].reset_index(drop=True)
    df = df.dropna(subset=["arabica_price", "robusta_price"]).reset_index(drop=True)

    print(f"   📅 Daten ab 2016, {len(df)} Zeilen")
    nan_pct = df.isnull().mean().sort_values(ascending=False)
    top_nan = nan_pct[nan_pct > 0].head(5)
    if len(top_nan) > 0:
        print("   ⚠️ Verbleibende NaN (Top 5):")
        for col, pct in top_nan.items():
            print(f"      {col}: {pct*100:.1f}%")

    return df
