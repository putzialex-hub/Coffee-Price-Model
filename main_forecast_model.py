
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
import sys
import os

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))
# Konfiguration
FIG_SIZE = (15, 8)
HORIZONS = [30, 90, 180] # Tage

print("🚀 Starte Kaffeepreis-Prognose (inkl. Dürre-Indikator & COT)")
print("=========================================================")


# ------------------------------------------------------------------------------
# 0. DATEN-UPDATES (COT automatisch aktualisieren)
# ------------------------------------------------------------------------------
import subprocess
import os
from datetime import datetime, timedelta

def should_update_cot():
    """Prüft ob COT-Daten aktualisiert werden müssen (älter als 3 Tage)."""
    cot_file = 'cot_data.csv'
    
    if not os.path.exists(cot_file):
        return True
    
    # Prüfe Alter der Datei
    file_time = datetime.fromtimestamp(os.path.getmtime(cot_file))
    if datetime.now() - file_time > timedelta(days=3):
        return True
    
    # Prüfe ob Daten aktuell sind (lese letzte Zeile)
    try:
        df_check = pd.read_csv(cot_file)
        last_date = pd.to_datetime(df_check['date']).max()
        # COT wird freitags veröffentlicht, Daten sind von Dienstag
        # Wenn letzte Daten älter als 7 Tage → Update nötig
        if datetime.now() - last_date > timedelta(days=7):
            return True
    except:
        return True
    
    return False

def update_cot_data():
    """Führt fetch_cot_data.py aus um aktuelle COT-Daten zu laden."""
    print("🔄 Prüfe COT-Daten...")
    
    if should_update_cot():
        print("   ⬇️ Lade aktuelle COT-Daten von CFTC...")
        try:
            # Pfad zum Skript
            cot_script_path = r"C:\Users\WZHALP3\OneDrive - Raiffeisen Bank International Group\Agriculture\Coffee\Coffee Price-20260202T064517Z-3-001\Coffee Price\fetch_cot_data.py"
            working_dir = os.path.dirname(cot_script_path)

            # NEU: Wir kopieren die Umgebungsvariablen und erzwingen UTF-8
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"

            # Ausführen mit erzwungenem UTF-8
            result = subprocess.run(
                [sys.executable, cot_script_path], 
                cwd=working_dir,
                capture_output=True, 
                text=True, 
                timeout=120,
                encoding='utf-8',  # Zwingt Python, die Antwort als UTF-8 zu lesen
                env=my_env         # Zwingt das Unterskript, UTF-8 zu schreiben
            )

            if result.returncode == 0:
                print("   ✅ COT-Daten aktualisiert!")
            else:
                print(f"   ❌ Fehler beim Update. Code: {result.returncode}")
                # Falls immer noch Fehler kommen, zeigen wir sie hier an
                print(f"   Fehlermeldung: {result.stderr}") 

        except Exception as e:
            print(f"   ❌ Kritischer Fehler: {e}")
    else:
        print("   ✅ COT-Daten sind aktuell (< 7 Tage alt)")

# Automatisch COT aktualisieren beim Start
update_cot_data()


# ------------------------------------------------------------------------------
# 1. DATEN LADEN & MERGEN
# ------------------------------------------------------------------------------

def load_data():
    # Preise
    arabica = pd.read_csv(r'Coffee\Coffee Price-20260202T064517Z-3-001\Coffee Price\arabica_clean.csv')
    robusta = pd.read_csv(r'Coffee\Coffee Price-20260202T064517Z-3-001\Coffee Price\robusta_clean.csv')
    arabica['date'] = pd.to_datetime(arabica['date'])
    robusta['date'] = pd.to_datetime(robusta['date'])
    df = pd.merge(arabica[['date', 'price']].rename(columns={'price': 'arabica_price'}),
                  robusta[['date', 'price']].rename(columns={'price': 'robusta_price'}),
                  on='date', how='outer')

    # Makro
    if pd.io.common.file_exists('coffee_data.csv'):
        macro = pd.read_csv('coffee_data.csv')
        macro['date'] = pd.to_datetime(macro['Date'])
        # Clean USD_BRL (manchmal String/Komma Problem in Rohdaten)
        # Wir nehmen an sie sind sauber, wenn nicht, kurz checken
        df = pd.merge(df, macro[['date', 'USD_BRL', 'DXY']], on='date', how='left')

    # Stocks
    if pd.io.common.file_exists('arabica_stocks.csv'):
        st_a = pd.read_csv('arabica_stocks.csv', sep=';')
        st_a['date'] = pd.to_datetime(st_a['Date'], format='%d.%m.%Y', errors='coerce')
        df = pd.merge(df, st_a[['date', 'Certified_Stocks']].rename(columns={'Certified_Stocks': 'arabica_stocks'}), on='date', how='left')
    
    if pd.io.common.file_exists('robusta_stocks.csv'):
        st_r = pd.read_csv('robusta_stocks.csv', sep=';')
        st_r['date'] = pd.to_datetime(st_r['Date'], format='%d.%m.%Y', errors='coerce')
        col_s = st_r.columns[1]
        df = pd.merge(df, st_r[['date', col_s]].rename(columns={col_s: 'robusta_stocks'}), on='date', how='left')

    # COT DATEN
    if pd.io.common.file_exists('cot_data.csv'):
        cot = pd.read_csv('cot_data.csv')
        cot['date'] = pd.to_datetime(cot['date'])
        df = pd.merge(df, cot, on='date', how='left')

    # WETTER & DÜRRE-INDIKATOR (NEU!)
    if pd.io.common.file_exists('weather_minas_gerais.csv'):
        w_mg = pd.read_csv('weather_minas_gerais.csv')
        w_mg['date'] = pd.to_datetime(w_mg['date'])
        
        # Feature Engineering VOR dem Merge (auf täglichen Wetterdaten)
        # 1. Rolling Rain 90d
        w_mg['rain_90d_mg'] = w_mg['precipitation_sum'].rolling(90).sum()
        # 2. Dry Streak (Consecutive days < 1mm)
        # Trick: Gruppe bilden bei jedem Regen > 1mm
        is_rainy = w_mg['precipitation_sum'] >= 1.0
        streak_id = is_rainy.cumsum()
        # Zähle Tage innerhalb der Streak-Gruppe (für Non-Rainy)
        w_mg['dry_streak_mg'] = w_mg.groupby(streak_id).cumcount()
        # Wenn es regnet, ist streak 0
        w_mg.loc[is_rainy, 'dry_streak_mg'] = 0
        
        df = pd.merge(df, w_mg[['date', 'temperature_2m_min', 'rain_90d_mg', 'dry_streak_mg']].rename(columns={'temperature_2m_min': 'temp_min_mg'}), on='date', how='left')
        print("   ✅ Wetter Minas Gerais (mit Dürre-Index) geladen!")

    if pd.io.common.file_exists('weather_dak_lak.csv'):
        w_dl = pd.read_csv('weather_dak_lak.csv')
        w_dl['date'] = pd.to_datetime(w_dl['date'])
        
        # Feature Engineering Dak Lak
        w_dl['rain_90d_dl'] = w_dl['precipitation_sum'].rolling(90).sum()
        is_rainy_dl = w_dl['precipitation_sum'] >= 1.0
        streak_id_dl = is_rainy_dl.cumsum()
        w_dl['dry_streak_dl'] = w_dl.groupby(streak_id_dl).cumcount()
        w_dl.loc[is_rainy_dl, 'dry_streak_dl'] = 0
        
        df = pd.merge(df, w_dl[['date', 'rain_90d_dl', 'dry_streak_dl']], on='date', how='left')
        print("   ✅ Wetter Dak Lak geladen!")

    df = df.sort_values('date').ffill().bfill()
    df = df[df['date'] >= '2000-01-01'].reset_index(drop=True)
    return df

df = load_data()

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Fundamental-Fokus, kein Price Leakage)
# ------------------------------------------------------------------------------

# A. TREND-INDIKATOREN (um Mean-Reversion Bias zu korrigieren)
for col in ['arabica_price', 'robusta_price']:
    # 6-Monats Trend: Prozentuale Veränderung (positiv = Aufwärtstrend)
    df[f'{col}_trend_180d'] = (df[col] / df[col].shift(180) - 1) * 100
    # 3-Monats Trend: Kurzfristiger Trend
    df[f'{col}_trend_90d'] = (df[col] / df[col].shift(90) - 1) * 100
    # Position relativ zum 1-Jahres-Durchschnitt
    df[f'{col}_vs_ma252'] = (df[col] / df[col].rolling(252).mean() - 1) * 100
    # Trend-Beschleunigung (2. Ableitung): Beschleunigt oder verlangsamt sich der Trend?
    df[f'{col}_trend_accel'] = df[f'{col}_trend_90d'] - df[f'{col}_trend_90d'].shift(30)

# B. VOLATILITÄT (erlaubt, da es um Risiko geht, nicht Preis-Level)
for col in ['arabica_price', 'robusta_price']:
    returns = np.log(df[col] / df[col].shift(1))
    df[f'{col}_volatility_30d'] = returns.rolling(30).std() * np.sqrt(252)  # Annualisiert

# B. COT Features (Sentiment)
if 'COT_Net_Spec' in df.columns:
    df['COT_Signal'] = df['COT_Net_Spec'].rolling(4).mean()
    # Z-Score für bessere Vergleichbarkeit
    df['COT_zscore'] = (df['COT_Net_Spec'] - df['COT_Net_Spec'].rolling(252).mean()) / df['COT_Net_Spec'].rolling(252).std()
    # Momentum: Veränderung der Positionierung
    df['COT_momentum_4w'] = df['COT_Net_Spec'].diff(20)

# C. WETTER mit LAG (Wetter heute → Ernte in 3-6 Monaten → Preis später)
if 'rain_90d_mg' in df.columns:
    df['rain_90d_mg_lag90'] = df['rain_90d_mg'].shift(90)
    df['rain_90d_mg_lag180'] = df['rain_90d_mg'].shift(180)
    df['dry_streak_mg_lag30'] = df['dry_streak_mg'].shift(30)
    
if 'rain_90d_dl' in df.columns:
    df['rain_90d_dl_lag90'] = df['rain_90d_dl'].shift(90)
    df['rain_90d_dl_lag180'] = df['rain_90d_dl'].shift(180)
    df['dry_streak_dl_lag30'] = df['dry_streak_dl'].shift(30)

# D. STOCKS Features (Lagerbestände als Supply-Indikator)
if 'arabica_stocks' in df.columns:
    df['arabica_stocks_change_30d'] = df['arabica_stocks'].pct_change(30)
    df['arabica_stocks_change_90d'] = df['arabica_stocks'].pct_change(90)
    df['arabica_stocks_zscore'] = (df['arabica_stocks'] - df['arabica_stocks'].rolling(252).mean()) / df['arabica_stocks'].rolling(252).std()
    # Stocks-Trend: Steigen oder fallen die Bestände?
    df['arabica_stocks_trend'] = df['arabica_stocks'].rolling(30).mean() / df['arabica_stocks'].rolling(90).mean() - 1

if 'robusta_stocks' in df.columns:
    df['robusta_stocks_change_30d'] = df['robusta_stocks'].pct_change(30)
    df['robusta_stocks_change_90d'] = df['robusta_stocks'].pct_change(90)
    df['robusta_stocks_zscore'] = (df['robusta_stocks'] - df['robusta_stocks'].rolling(252).mean()) / df['robusta_stocks'].rolling(252).std()
    df['robusta_stocks_trend'] = df['robusta_stocks'].rolling(30).mean() / df['robusta_stocks'].rolling(90).mean() - 1

# E. SAISONALITÄT (Ernte-Zyklen)
# Brasilien Arabica: Haupternte Apr-Sep (Monate 4-9)
# Vietnam Robusta: Haupternte Oct-Jan (Monate 10-12, 1)
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)  # Zyklische Kodierung
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Ernte-Phasen als Dummy (Brasilien)
df['brazil_harvest'] = df['month'].isin([4, 5, 6, 7, 8, 9]).astype(int)
# Ernte-Phasen (Vietnam)  
df['vietnam_harvest'] = df['month'].isin([10, 11, 12, 1]).astype(int)

# Entferne 'month' wieder (nicht als Feature, nur für Berechnung)
df = df.drop(columns=['month'])

# D. Entferne Preis-Spalten aus Features (KRITISCH!)
price_cols_to_exclude = ['arabica_price', 'robusta_price']

# F. CROSS-COMMODITY FEATURES (Arabica-Robusta Spread)
# Der Spread zwischen Arabica und Robusta ist ein wichtiger Indikator
if 'arabica_price' in df.columns and 'robusta_price' in df.columns:
    # Robusta in USD/t, Arabica in cents/lb → Normalisieren für Ratio
    # 1 USD/t ≈ 0.0453 cents/lb (ungefähre Umrechnung)
    df['arabica_robusta_ratio'] = df['arabica_price'] / (df['robusta_price'] * 0.0453)
    # Spread-Trend: Weitet sich der Spread aus oder verengt er sich?
    df['spread_trend_30d'] = df['arabica_robusta_ratio'].pct_change(30) * 100
    # Spread vs historischem Durchschnitt
    df['spread_zscore'] = (df['arabica_robusta_ratio'] - df['arabica_robusta_ratio'].rolling(252).mean()) / df['arabica_robusta_ratio'].rolling(252).std()

# G. VIETNAM-SPEZIFISCHE FEATURES (Robusta)
# Vietnam Ernte ist Oct-Jan, aber Wetter-Impact ist 3-6 Monate vorher
if 'rain_90d_dl' in df.columns:
    # Kritische Periode für Vietnam: April-September (Blüte & Fruchtentwicklung)
    df['vietnam_critical_rain'] = df['rain_90d_dl_lag180']  # Regen 6 Monate vorher
    # Extremwetter-Indikator
    df['vietnam_drought_risk'] = (df['dry_streak_dl'] > 14).astype(int)  # Mehr als 2 Wochen trocken

# H. BRASILIEN-SPEZIFISCHE FEATURES (Arabica)
if 'rain_90d_mg' in df.columns:
    # Frost-Risiko Periode: Mai-August (brasilianischer Winter)
    df['brazil_frost_season'] = df['date'].dt.month.isin([5, 6, 7, 8]).astype(int)
    # Kritische Trockenheit
    df['brazil_drought_risk'] = (df['dry_streak_mg'] > 21).astype(int)  # Mehr als 3 Wochen trocken

# I. MOMENTUM-FEATURES (um Trend-Fortsetzung besser zu erfassen)
for col in ['arabica_price', 'robusta_price']:
    # Kurzfristiges Momentum (10 Tage)
    df[f'{col}_momentum_10d'] = df[col].pct_change(10) * 100
    # Mittelfristiges Momentum (30 Tage)
    df[f'{col}_momentum_30d'] = df[col].pct_change(30) * 100
    # Momentum-Beschleunigung: Wird der Trend stärker oder schwächer?
    df[f'{col}_momentum_accel'] = df[f'{col}_momentum_10d'] - df[f'{col}_momentum_10d'].shift(10)
    # Momentum-Divergenz: Kurzfristig vs Mittelfristig
    df[f'{col}_momentum_div'] = df[f'{col}_momentum_10d'] - df[f'{col}_momentum_30d']

df = df.dropna().reset_index(drop=True)
print(f"   ✅ Feature Engineering abgeschlossen (Fundamental-Fokus)")

# ------------------------------------------------------------------------------
# 3. TRAINING & VALIDATION (LOG-RETURNS & QUANTILES)
# ------------------------------------------------------------------------------
# Wir stellen um auf Log-Returns, um "All Time Highs" vorhersagen zu können.
# Target = ln(Price_Future / Price_Now)

cutoff_date = df['date'].max() - pd.Timedelta(days=180)
train_cutoff = df[df['date'] <= cutoff_date]
valid_real = df[df['date'] > cutoff_date]

print(f"   Trainiere Modell bis {cutoff_date.date()} (Log-Returns & Quantiles)...")

commodities = ['arabica_price', 'robusta_price']
horizons = [30, 90, 180]

def create_xy_log_returns(data):
    X_d = data.copy()
    # KRITISCH: Schließe Preis-Spalten explizit aus!
    exclude_cols = ['date', 'arabica_price', 'robusta_price']
    feature_cols = [c for c in X_d.columns if c not in exclude_cols and 'target' not in c]
    target_cols = []
    
    for comm in commodities:
        for h in horizons:
            t_col = f'target_ret_{comm}_{h}d'
            future_price = X_d[comm].shift(-h)
            X_d[t_col] = np.log(future_price / X_d[comm])
            target_cols.append(t_col)
            
    X_d = X_d.dropna()
    return X_d[feature_cols], X_d[target_cols], feature_cols, target_cols

X_train, y_train, f_cols, t_cols = create_xy_log_returns(train_cutoff)

# Wir trainieren 3 Modelle für Unsicherheit (Quantile Regression)
# Da MultiOutputRegressor "quantile" loss nicht nativ für alle Sklearn Versionen sauber unterstützt in Kombination,
# loopen wir hier sauberer.

models = {} # Key: (commodity, horizon, quantile) -> Model
quantiles = [0.05, 0.50, 0.95] # Worst, Base, Best Case

print("      Trainiere Quantile Models (kann etwas dauern)...")

for comm in commodities:
    for h in horizons:
        col_target = f'target_ret_{comm}_{h}d'
        y_single = y_train[col_target]
        
        for q in quantiles:
            # Gradient Boosting mit Quantile Loss
            # Robusta braucht mehr Regularisierung wegen kürzerer Historie
            if 'robusta' in comm:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q,
                    n_estimators=250, learning_rate=0.03, max_depth=4,  # Konservativer
                    min_samples_leaf=20, random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q,
                    n_estimators=200, learning_rate=0.05, max_depth=5,
                    random_state=42
                )
            model.fit(X_train, y_single)
            models[(comm, h, q)] = model

# --- VALIDATION PROGNOSE (RECONSTRUCTION) ---
# Wir nehmen die letzte Zeile VOR dem Cutoff
last_row_cutoff = train_cutoff.iloc[[-1]][f_cols]
current_prices_cutoff = train_cutoff.iloc[-1][commodities] # Um Log-Ret zurückzurechnen

preds_valid = {} # (comm, h, q) -> Price

for comm in commodities:
    start_price = current_prices_cutoff[comm]
    for h in horizons:
        for q in quantiles:
            pred_log_ret = models[(comm, h, q)].predict(last_row_cutoff)[0]
            # Rückrechnung: Price_Future = Price_Current * exp(Log_Ret)
            pred_price = start_price * np.exp(pred_log_ret)
            preds_valid[(comm, h, q)] = pred_price

# ------------------------------------------------------------------------------
# 4. LIVE PROGNOSE (HEUTE)
# ------------------------------------------------------------------------------
print("   Trainiere Finales Modell (Full Data)...")
X_full, y_full, _, _ = create_xy_log_returns(df)

models_final = {}
for comm in commodities:
    for h in horizons:
        col_target = f'target_ret_{comm}_{h}d'
        y_single = y_full[col_target]
        for q in quantiles:
            # Robusta braucht mehr Regularisierung wegen kürzerer Historie
            if 'robusta' in comm:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q,
                    n_estimators=250, learning_rate=0.03, max_depth=4,  # Konservativer
                    min_samples_leaf=20, random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q,
                    n_estimators=200, learning_rate=0.05, max_depth=5,
                    random_state=42
                )
            model.fit(X_full, y_single)
            models_final[(comm, h, q)] = model

last_row_today = df.iloc[[-1]][f_cols]
current_prices_today = df.iloc[-1][commodities]
preds_today = {}

for comm in commodities:
    start_price = current_prices_today[comm]
    for h in horizons:
        for q in quantiles:
            pred_log_ret = models_final[(comm, h, q)].predict(last_row_today)[0]
            pred_price = start_price * np.exp(pred_log_ret)
            preds_today[(comm, h, q)] = pred_price
            
# ------------------------------------------------------------------------------
# 4b. WALK-FORWARD VALIDATION (Robuste Performance-Messung)
# ------------------------------------------------------------------------------
print("\n📊 Walk-Forward Validation (2022-2025)...")

validation_results = []
validation_dates = pd.date_range('2022-01-01', '2025-07-01', freq='3ME')

for val_cutoff in validation_dates:
    val_cutoff = pd.Timestamp(val_cutoff)
    
    # Skip wenn nicht genug Daten
    if val_cutoff < df['date'].min() + pd.Timedelta(days=365):
        continue
    if val_cutoff + pd.Timedelta(days=180) > df['date'].max():
        continue
    
    # Train bis Cutoff
    train_mask = df['date'] <= val_cutoff
    train_data = df[train_mask].copy()
    
    if len(train_data) < 500:  # Mindestens 500 Tage Training
        continue
    
    # Features erstellen
    exclude_cols = ['date', 'arabica_price', 'robusta_price']
    feature_cols_val = [c for c in train_data.columns if c not in exclude_cols and 'target' not in c]
    
    for comm in commodities:
        for h in horizons:
            # Target berechnen
            future_price = train_data[comm].shift(-h)
            train_data[f'target_{h}d'] = np.log(future_price / train_data[comm])
        
        train_data_clean = train_data.dropna()
        if len(train_data_clean) < 100:
            continue
            
        X_val = train_data_clean[feature_cols_val]
        
        for h in horizons:
            y_val = train_data_clean[f'target_{h}d']
            
            # Trainiere Modell
            model_val = GradientBoostingRegressor(
                loss='quantile', alpha=0.5,
                n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
            )
            model_val.fit(X_val, y_val)
            
            # Prediction für Cutoff-Datum
            last_row_val = train_data_clean.iloc[[-1]][feature_cols_val]
            start_price_val = train_data_clean.iloc[-1][comm]
            pred_log_ret = model_val.predict(last_row_val)[0]
            pred_price = start_price_val * np.exp(pred_log_ret)
            
            # Actual Price
            target_date = val_cutoff + pd.Timedelta(days=h)
            actual_df = df[df['date'] >= target_date]
            if len(actual_df) > 0:
                actual_price = actual_df.iloc[0][comm]
                error_pct = (pred_price - actual_price) / actual_price * 100
                
                validation_results.append({
                    'cutoff': val_cutoff,
                    'commodity': comm,
                    'horizon': h,
                    'predicted': pred_price,
                    'actual': actual_price,
                    'error_pct': error_pct
                })
val_df = pd.DataFrame()  # Initialisiere leer

if validation_results:
    val_df = pd.DataFrame(validation_results)
# Ergebnisse zusammenfassen
if validation_results:
    val_df = pd.DataFrame(validation_results)
    
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION ERGEBNISSE")
    print("="*60)
    
    for comm in commodities:
        print(f"\n{comm.upper()}:")
        for h in horizons:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)]
            if len(subset) > 0:
                mae = subset['error_pct'].abs().mean()
                bias = subset['error_pct'].mean()
                hit_rate = (subset['error_pct'].abs() < 15).mean() * 100  # Innerhalb ±15%
                print(f"   {h:3d}d: MAE={mae:5.1f}%, Bias={bias:+5.1f}%, Hit-Rate(±15%)={hit_rate:4.1f}% (n={len(subset)})")
    
    val_csv_path = os.path.join(script_dir, 'validation_results.csv')
    val_df.to_csv(val_csv_path, index=False)
    print(f"✅ Validation Details: {val_csv_path}")
# ------------------------------------------------------------------------------
# 5. VISUALISIERUNG (PROFI-CHART MIT FAN CHART / CONE)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

for i, comm in enumerate(commodities):
    ax = axes[i]
    col_price = comm
    
    # History
    mask_hist = df['date'] > (df['date'].max() - pd.Timedelta(days=365))
    ax.plot(df.loc[mask_hist, 'date'], df.loc[mask_hist, col_price], label='Price History', color='black', lw=1.5)
    
    # --- Live Forecast Cone ---
    today = df['date'].iloc[-1]
    future_dates = [today + pd.Timedelta(days=h) for h in horizons]
    
    # Base Case (0.5)
    y_base = [preds_today[(comm, h, 0.5)] for h in horizons]
    ax.plot(future_dates, y_base, 'o--', color='blue', label='Forecast (Base Case)', lw=2)
    
    # Confidence Interval (0.05 - 0.95)
    y_lower = [preds_today[(comm, h, 0.05)] for h in horizons]
    y_upper = [preds_today[(comm, h, 0.95)] for h in horizons]
    
    ax.fill_between(future_dates, y_lower, y_upper, color='blue', alpha=0.15, label='90% Confidence Interval')
    
    # Backtest Points aus Walk-Forward Validation (letztes Jahr)
    if 'val_df' in dir() and not val_df.empty:
        val_subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == 90)]  # 90d Horizont
        val_subset = val_subset.sort_values('cutoff')
        
        # Zeige Predicted vs Actual für jeden Backtest-Punkt
        for _, row in val_subset.iterrows():
            pred_date = row['cutoff'] + pd.Timedelta(days=90)
            # Nur Punkte im sichtbaren Bereich (letztes Jahr)
            if pred_date > (df['date'].max() - pd.Timedelta(days=365)):
                ax.plot(pred_date, row['predicted'], 'x', color='orange', markersize=8, alpha=0.7)
                ax.plot(pred_date, row['actual'], 'o', color='green', markersize=5, alpha=0.5)
        
        # Legende-Einträge hinzufügen
        ax.plot([], [], 'x', color='orange', markersize=8, label='Backtest Predicted (90d)')
        ax.plot([], [], 'o', color='green', markersize=5, label='Backtest Actual')
    else:
        # Fallback: Alter Code
        valid_dates = [cutoff_date + pd.Timedelta(days=h) for h in horizons]
        y_valid_base = [preds_valid[(comm, h, 0.5)] for h in horizons]
        ax.plot(valid_dates, y_valid_base, 'x', color='orange', label='Backtest Check (6m ago)', markersize=8)
    
    # Titel & Info
    title_suffix = ""
    if comm == 'arabica_price' and 'dry_streak_mg' in df.columns:
        last_dry = df.iloc[-1]['dry_streak_mg']
        rain90 = df.iloc[-1]['rain_90d_mg']
        title_suffix = f" (Drought: {last_dry:.0f} days, 3M-Rain: {rain90:.1f}mm)"
        
    ax.set_title(f"{comm.replace('_price', '').capitalize()} PRO Forecast {title_suffix}\n(Log-Return Model + Quantile Uncertainty)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

plt.tight_layout()

# Chart speichern mit absolutem Pfad
import os
chart_path = os.path.join(script_dir, 'forecast_pro_chart.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()  # Wichtig: Figure schließen um Speicher freizugeben

print(f"✅ Professional Chart: {chart_path}")

# Prüfe ob Datei erstellt wurde
if os.path.exists(chart_path):
    file_size = os.path.getsize(chart_path) / 1024
    print(f"   Dateigröße: {file_size:.1f} KB")
else:
    print("   ⚠️ WARNUNG: Chart-Datei wurde nicht erstellt!")

# ------------------------------------------------------------------------------
# 5b. BACKTEST PERFORMANCE CHART (Historische Genauigkeit)
# ------------------------------------------------------------------------------
if not val_df.empty:
    print("\n📊 Erstelle Backtest Performance Chart...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, comm in enumerate(commodities):
        comm_name = comm.replace('_price', '').capitalize()
        
        # --- Chart 1: Predicted vs Actual über Zeit ---
        ax1 = axes2[i, 0]
        
        for h, color, marker in [(30, 'blue', 'o'), (90, 'orange', 's'), (180, 'green', '^')]:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)].copy()
            subset = subset.sort_values('cutoff')
            subset['target_date'] = subset['cutoff'] + pd.to_timedelta(subset['horizon'], unit='D')
            
            ax1.plot(subset['target_date'], subset['actual'], '-', color=color, alpha=0.3, linewidth=2)
            ax1.plot(subset['target_date'], subset['predicted'], '--', color=color, alpha=0.8, linewidth=1)
            ax1.scatter(subset['target_date'], subset['predicted'], color=color, marker=marker, 
                       s=50, label=f'{h}d Predicted', alpha=0.8)
            ax1.scatter(subset['target_date'], subset['actual'], color=color, marker=marker,
                       s=20, facecolors='none', edgecolors=color, label=f'{h}d Actual', alpha=0.5)
        
        ax1.set_title(f'{comm_name}: Predicted vs Actual')
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('Preis')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # --- Chart 2: Fehler (%) über Zeit ---
        ax2 = axes2[i, 1]
        
        for h, color in [(30, 'blue'), (90, 'orange'), (180, 'green')]:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)].copy()
            subset = subset.sort_values('cutoff')
            
            ax2.bar(subset['cutoff'], subset['error_pct'], width=20, alpha=0.6, 
                   color=color, label=f'{h}d Fehler')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Mittelwert-Linie (Bias)
            mean_err = subset['error_pct'].mean()
            ax2.axhline(y=mean_err, color=color, linestyle='--', linewidth=1, alpha=0.7)
        
        ax2.set_title(f'{comm_name}: Prognose-Fehler über Zeit')
        ax2.set_xlabel('Cutoff Datum')
        ax2.set_ylabel('Fehler (%)')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Markiere ±15% Zone
        ax2.axhspan(-15, 15, alpha=0.1, color='green', label='±15% Zone')
        ax2.set_ylim(-50, 50)
    
    plt.suptitle('Walk-Forward Backtest Performance (2022-2025)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Speichern
    backtest_chart_path = os.path.join(script_dir, 'backtest_performance_chart.png')
    plt.savefig(backtest_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Backtest Chart: {backtest_chart_path}")
    
    # --- Zusätzlich: Performance-Tabelle ausgeben ---
    print("\n📋 Performance-Zusammenfassung nach Horizont:")
    print("-" * 70)
    
    for comm in commodities:
        comm_name = comm.replace('_price', '').capitalize()
        print(f"\n{comm_name}:")
        
        for h in horizons:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)]
            if len(subset) > 0:
                mae = subset['error_pct'].abs().mean()
                bias = subset['error_pct'].mean()
                std = subset['error_pct'].std()
                hit_rate = (subset['error_pct'].abs() < 15).mean() * 100
                
                print(f"   {h:3d}d: MAE={mae:5.1f}%, Bias={bias:+5.1f}%, Std={std:5.1f}%, Hit-Rate(±15%)={hit_rate:4.1f}%")

else:
    print("⚠️ Keine Validation-Daten für Backtest-Chart verfügbar")

# CSV Export (Detailed)
rows = []
for h in horizons:
    row = {'Horizon_Days': h, 'Date': (today + pd.Timedelta(days=h)).date()}
    for comm in commodities:
        row[f'{comm}_Low'] = preds_today[(comm, h, 0.05)]
        row[f'{comm}_Base'] = preds_today[(comm, h, 0.50)]
        row[f'{comm}_High'] = preds_today[(comm, h, 0.95)]
    rows.append(row)

csv_path = os.path.join(script_dir, 'coffee_forecast_pro.csv')
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"✅ CSV: {csv_path}")
