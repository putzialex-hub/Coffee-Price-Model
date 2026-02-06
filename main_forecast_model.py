import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import sys
import os

# News Monitor Import
try:
    from news_monitor import get_coffee_news_sentiment, check_news_model_alignment
    NEWS_MONITOR_AVAILABLE = True
except ImportError:
    NEWS_MONITOR_AVAILABLE = False

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Konfiguration
FIG_SIZE = (15, 8)
HORIZONS = [30, 90, 180]
DATA_DIR = '.'

print("🚀 Starte Kaffeepreis-Prognose (Fundamental-Fokus, Commodity-spezifisch)")
print("=========================================================================")

# ------------------------------------------------------------------------------
# 0. DATEN-UPDATES (COT automatisch aktualisieren)
# ------------------------------------------------------------------------------
import subprocess
from datetime import datetime, timedelta

def should_update_cot():
    cot_file = 'cot_data.csv'
    if not os.path.exists(cot_file):
        return True
    file_time = datetime.fromtimestamp(os.path.getmtime(cot_file))
    if datetime.now() - file_time > timedelta(days=3):
        return True
    try:
        df_check = pd.read_csv(cot_file)
        last_date = pd.to_datetime(df_check['date']).max()
        if datetime.now() - last_date > timedelta(days=7):
            return True
    except:
        return True
    return False

def update_cot_data():
    print("🔄 Prüfe COT-Daten...")
    if should_update_cot():
        print("   ⬇️ Lade aktuelle COT-Daten von CFTC...")
        try:
            cot_script_path = r"C:\Users\WZHALP3\OneDrive - Raiffeisen Bank International Group\Agriculture\Coffee\Coffee Price-20260202T064517Z-3-001\Coffee Price\fetch_cot_data.py"
            working_dir = os.path.dirname(cot_script_path)
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, cot_script_path],
                cwd=working_dir, capture_output=True, text=True,
                timeout=120, encoding='utf-8', env=my_env
            )
            if result.returncode == 0:
                print("   ✅ COT-Daten aktualisiert!")
            else:
                print(f"   ❌ Fehler beim Update. Code: {result.returncode}")
                print(f"   Fehlermeldung: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Kritischer Fehler: {e}")
    else:
        print("   ✅ COT-Daten sind aktuell (< 7 Tage alt)")

update_cot_data()

# ------------------------------------------------------------------------------
# 1. DATEN LADEN & MERGEN
# ------------------------------------------------------------------------------

def load_data(data_dir=DATA_DIR):
    arabica = pd.read_csv(os.path.join(data_dir, 'arabica_clean.csv'))
    robusta = pd.read_csv(os.path.join(data_dir, 'robusta_clean.csv'))
    arabica['date'] = pd.to_datetime(arabica['date'])
    robusta['date'] = pd.to_datetime(robusta['date'])
    df = pd.merge(arabica[['date', 'price']].rename(columns={'price': 'arabica_price'}),
                  robusta[['date', 'price']].rename(columns={'price': 'robusta_price'}),
                  on='date', how='outer')

    if pd.io.common.file_exists('coffee_data.csv'):
        macro = pd.read_csv('coffee_data.csv')
        macro['date'] = pd.to_datetime(macro['Date'])
        df = pd.merge(df, macro[['date', 'USD_BRL', 'DXY']], on='date', how='left')

    # Stocks (mit skiprows Fix)
    path_stocks_a = os.path.join(data_dir, 'arabica_stocks.csv')
    if pd.io.common.file_exists(path_stocks_a):
        st_a = pd.read_csv(path_stocks_a, sep=';', skiprows=[1])
        st_a['date'] = pd.to_datetime(st_a['Date'], format='%d.%m.%Y', errors='coerce')
        st_a['Certified_Stocks'] = pd.to_numeric(st_a['Certified_Stocks'], errors='coerce')
        df = pd.merge(df, st_a[['date', 'Certified_Stocks']].rename(
            columns={'Certified_Stocks': 'arabica_stocks'}), on='date', how='left')

    path_stocks_r = os.path.join(data_dir, 'robusta_stocks.csv')
    if pd.io.common.file_exists(path_stocks_r):
        st_r = pd.read_csv(path_stocks_r, sep=';', skiprows=[1])
        st_r['date'] = pd.to_datetime(st_r['Date'], format='%d.%m.%Y', errors='coerce')
        col_s = [c for c in st_r.columns if c != 'Date'][0]
        st_r[col_s] = pd.to_numeric(st_r[col_s], errors='coerce')
        df = pd.merge(df, st_r[['date', col_s]].rename(
            columns={col_s: 'robusta_stocks'}), on='date', how='left')

    if pd.io.common.file_exists('cot_data.csv'):
        cot = pd.read_csv('cot_data.csv')
        cot['date'] = pd.to_datetime(cot['date'])
        df = pd.merge(df, cot, on='date', how='left')

    if pd.io.common.file_exists('weather_minas_gerais.csv'):
        w_mg = pd.read_csv('weather_minas_gerais.csv')
        w_mg['date'] = pd.to_datetime(w_mg['date'])
        w_mg['rain_90d_mg'] = w_mg['precipitation_sum'].rolling(90).sum()
        is_rainy = w_mg['precipitation_sum'] >= 1.0
        streak_id = is_rainy.cumsum()
        w_mg['dry_streak_mg'] = w_mg.groupby(streak_id).cumcount()
        w_mg.loc[is_rainy, 'dry_streak_mg'] = 0
        df = pd.merge(df, w_mg[['date', 'temperature_2m_min', 'rain_90d_mg', 'dry_streak_mg']].rename(
            columns={'temperature_2m_min': 'temp_min_mg'}), on='date', how='left')
        print("   ✅ Wetter Minas Gerais geladen!")

    if pd.io.common.file_exists('weather_dak_lak.csv'):
        w_dl = pd.read_csv('weather_dak_lak.csv')
        w_dl['date'] = pd.to_datetime(w_dl['date'])
        w_dl['rain_90d_dl'] = w_dl['precipitation_sum'].rolling(90).sum()
        is_rainy_dl = w_dl['precipitation_sum'] >= 1.0
        streak_id_dl = is_rainy_dl.cumsum()
        w_dl['dry_streak_dl'] = w_dl.groupby(streak_id_dl).cumcount()
        w_dl.loc[is_rainy_dl, 'dry_streak_dl'] = 0
        df = pd.merge(df, w_dl[['date', 'rain_90d_dl', 'dry_streak_dl']], on='date', how='left')
        print("   ✅ Wetter Dak Lak geladen!")

    df = df.sort_values('date').reset_index(drop=True)
    
    # Begrenztes Forward-Fill: Preise max 5 Tage (Wochenenden/Feiertage),
    # COT/Stocks max 7 Tage (wöchentliche Daten), Wetter max 3 Tage
    price_cols = ['arabica_price', 'robusta_price']
    cot_stock_cols = [c for c in df.columns if any(x in c for x in ['COT_', 'stocks', 'Stocks'])]
    weather_cols = [c for c in df.columns if any(x in c for x in ['rain_', 'dry_streak', 'temp_min'])]
    other_cols = [c for c in df.columns if c != 'date' and c not in price_cols 
                  and c not in cot_stock_cols and c not in weather_cols]
    
    df[price_cols] = df[price_cols].ffill(limit=5)
    df[cot_stock_cols] = df[cot_stock_cols].ffill(limit=7)
    df[weather_cols] = df[weather_cols].ffill(limit=3)
    df[other_cols] = df[other_cols].ffill(limit=7)
    
    # KEIN bfill! Keine Daten aus der Zukunft zurückfüllen.
    
    # Starte erst ab 2016 (Robusta ab 2015 + 1 Jahr Warmup)
    df = df[df['date'] >= '2016-01-01'].reset_index(drop=True)
    
    # Nur Zeilen wo beide Preise vorhanden
    df = df.dropna(subset=['arabica_price', 'robusta_price']).reset_index(drop=True)
    
    print(f"   📅 Daten ab 2016, {len(df)} Zeilen")
    nan_pct = df.isnull().mean().sort_values(ascending=False)
    top_nan = nan_pct[nan_pct > 0].head(5)
    if len(top_nan) > 0:
        print(f"   ⚠️ Verbleibende NaN (Top 5):")
        for col, pct in top_nan.items():
            print(f"      {col}: {pct*100:.1f}%")
    
    return df

df = load_data(DATA_DIR)

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Fundamental-Fokus, kein Price Leakage)
# ------------------------------------------------------------------------------

if 'COT_Net_Spec' in df.columns:
    df['COT_Signal'] = df['COT_Net_Spec'].rolling(4).mean()
    df['COT_zscore'] = (df['COT_Net_Spec'] - df['COT_Net_Spec'].rolling(252).mean()) / df['COT_Net_Spec'].rolling(252).std()
    df['COT_momentum_4w'] = df['COT_Net_Spec'].diff(20)

if 'rain_90d_mg' in df.columns:
    df['rain_90d_mg_lag90'] = df['rain_90d_mg'].shift(90)
    df['rain_90d_mg_lag180'] = df['rain_90d_mg'].shift(180)
    df['dry_streak_mg_lag30'] = df['dry_streak_mg'].shift(30)

if 'rain_90d_dl' in df.columns:
    df['rain_90d_dl_lag90'] = df['rain_90d_dl'].shift(90)
    df['rain_90d_dl_lag180'] = df['rain_90d_dl'].shift(180)
    df['dry_streak_dl_lag30'] = df['dry_streak_dl'].shift(30)

if 'arabica_stocks' in df.columns:
    df['arabica_stocks_change_30d'] = df['arabica_stocks'].pct_change(30)
    df['arabica_stocks_change_90d'] = df['arabica_stocks'].pct_change(90)
    df['arabica_stocks_zscore'] = (df['arabica_stocks'] - df['arabica_stocks'].rolling(252).mean()) / df['arabica_stocks'].rolling(252).std()
    df['arabica_stocks_trend'] = df['arabica_stocks'].rolling(30).mean() / df['arabica_stocks'].rolling(90).mean() - 1

if 'robusta_stocks' in df.columns:
    df['robusta_stocks_change_30d'] = df['robusta_stocks'].pct_change(30)
    df['robusta_stocks_change_90d'] = df['robusta_stocks'].pct_change(90)
    df['robusta_stocks_zscore'] = (df['robusta_stocks'] - df['robusta_stocks'].rolling(252).mean()) / df['robusta_stocks'].rolling(252).std()
    df['robusta_stocks_trend'] = df['robusta_stocks'].rolling(30).mean() / df['robusta_stocks'].rolling(90).mean() - 1

df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['brazil_harvest'] = df['month'].isin([4, 5, 6, 7, 8, 9]).astype(int)
df['vietnam_harvest'] = df['month'].isin([10, 11, 12, 1]).astype(int)
df = df.drop(columns=['month'])

if 'arabica_price' in df.columns and 'robusta_price' in df.columns:
    df['arabica_robusta_ratio'] = df['arabica_price'] / (df['robusta_price'] * 0.0453)
    df['spread_trend_30d'] = df['arabica_robusta_ratio'].pct_change(30) * 100
    df['spread_zscore'] = (df['arabica_robusta_ratio'] - df['arabica_robusta_ratio'].rolling(252).mean()) / df['arabica_robusta_ratio'].rolling(252).std()

if 'rain_90d_dl' in df.columns:
    df['vietnam_critical_rain'] = df['rain_90d_dl_lag180']
    df['vietnam_drought_risk'] = (df['dry_streak_dl'] > 14).astype(int)

if 'rain_90d_mg' in df.columns:
    df['brazil_frost_season'] = df['date'].dt.month.isin([5, 6, 7, 8]).astype(int)
    df['brazil_drought_risk'] = (df['dry_streak_mg'] > 21).astype(int)

# Nur Zeilen droppen wo die wichtigsten Features fehlen
essential_cols = ['arabica_price', 'robusta_price', 'month_sin', 'month_cos']
df = df.dropna(subset=essential_cols).reset_index(drop=True)
print(f"   ✅ Feature Engineering abgeschlossen ({len(df)} Zeilen)")

# Verbleibende NaN mit 0 füllen (für Spalten wie stocks_zscore die erst nach 252d verfügbar sind)
feature_cols_all = [c for c in df.columns if c not in ['date', 'arabica_price', 'robusta_price']]
df[feature_cols_all] = df[feature_cols_all].fillna(0)
print(f"   ✅ Verbleibende NaN mit 0 gefüllt")

# ------------------------------------------------------------------------------
# 2b. COMMODITY-SPEZIFISCHE FEATURE-SETS (Punkt 2)
# ------------------------------------------------------------------------------

ARABICA_ONLY_PATTERNS = [
    'rain_90d_mg', 'dry_streak_mg', 'temp_min_mg',
    'brazil_drought_risk', 'brazil_frost_season', 'brazil_harvest',
    'arabica_stocks',
]

ROBUSTA_ONLY_PATTERNS = [
    'rain_90d_dl', 'dry_streak_dl',
    'vietnam_drought_risk', 'vietnam_critical_rain', 'vietnam_harvest',
    'robusta_stocks',
]

def get_commodity_features(all_features, commodity):
    if 'arabica' in commodity:
        return [f for f in all_features
                if not any(pat in f for pat in ROBUSTA_ONLY_PATTERNS)]
    else:
        return [f for f in all_features
                if not any(pat in f for pat in ARABICA_ONLY_PATTERNS)]

commodities = ['arabica_price', 'robusta_price']
horizons = [30, 90, 180]

exclude_cols = ['date', 'arabica_price', 'robusta_price']
all_features_list = [c for c in df.columns if c not in exclude_cols and 'target' not in c]
for comm in commodities:
    comm_feats = get_commodity_features(all_features_list, comm)
    print(f"   {comm.replace('_price','').upper()}: {len(comm_feats)} Features")

def create_xy_log_returns(data, commodity=None):
    X_d = data.copy()
    exclude = ['date', 'arabica_price', 'robusta_price']
    all_fcols = [c for c in X_d.columns if c not in exclude and 'target' not in c]
    if commodity:
        feature_cols = get_commodity_features(all_fcols, commodity)
    else:
        feature_cols = all_fcols
    target_cols = []
    for comm in commodities:
        for h in horizons:
            t_col = f'target_ret_{comm}_{h}d'
            X_d[t_col] = np.log(X_d[comm].shift(-h) / X_d[comm])
            target_cols.append(t_col)
    X_d = X_d.dropna()
    return X_d[feature_cols], X_d[target_cols], feature_cols, target_cols

# ------------------------------------------------------------------------------
# 3. TRAINING (commodity-spezifisch)
# ------------------------------------------------------------------------------

cutoff_date = df['date'].max() - pd.Timedelta(days=180)
train_cutoff = df[df['date'] <= cutoff_date]

print(f"\n   Trainiere Modell bis {cutoff_date.date()} (commodity-spezifisch)...")

models = {}
quantiles = [0.05, 0.50, 0.95]

for comm in commodities:
    X_train, y_train, f_cols_comm, t_cols = create_xy_log_returns(train_cutoff, commodity=comm)
    for h in horizons:
        col_target = f'target_ret_{comm}_{h}d'
        y_single = y_train[col_target]
        for q in quantiles:
            if 'robusta' in comm:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q, n_estimators=250,
                    learning_rate=0.03, max_depth=4, min_samples_leaf=20, random_state=42)
            else:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q, n_estimators=200,
                    learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_train, y_single)
            models[(comm, h, q)] = model
    models[(comm, 'feature_cols')] = f_cols_comm

current_prices_cutoff = train_cutoff.iloc[-1][commodities]
preds_valid = {}
for comm in commodities:
    f_cols_comm = models[(comm, 'feature_cols')]
    last_row = train_cutoff.iloc[[-1]][f_cols_comm]
    start_price = current_prices_cutoff[comm]
    for h in horizons:
        for q in quantiles:
            pred_log_ret = models[(comm, h, q)].predict(last_row)[0]
            preds_valid[(comm, h, q)] = start_price * np.exp(pred_log_ret)

# ------------------------------------------------------------------------------
# 4. LIVE PROGNOSE (commodity-spezifisch)
# ------------------------------------------------------------------------------
print("   Trainiere Finales Modell (Full Data)...")

models_final = {}
for comm in commodities:
    X_full, y_full, f_cols_comm, _ = create_xy_log_returns(df, commodity=comm)
    for h in horizons:
        col_target = f'target_ret_{comm}_{h}d'
        y_single = y_full[col_target]
        for q in quantiles:
            if 'robusta' in comm:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q, n_estimators=250,
                    learning_rate=0.03, max_depth=4, min_samples_leaf=20, random_state=42)
            else:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q, n_estimators=200,
                    learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_full, y_single)
            models_final[(comm, h, q)] = model
    models_final[(comm, 'feature_cols')] = f_cols_comm

current_prices_today = df.iloc[-1][commodities]
preds_today = {}
for comm in commodities:
    f_cols_comm = models_final[(comm, 'feature_cols')]
    last_row = df.iloc[[-1]][f_cols_comm]
    start_price = current_prices_today[comm]
    for h in horizons:
        for q in quantiles:
            pred_log_ret = models_final[(comm, h, q)].predict(last_row)[0]
            preds_today[(comm, h, q)] = start_price * np.exp(pred_log_ret)

# ------------------------------------------------------------------------------
# 4b. WALK-FORWARD VALIDATION (commodity-spezifisch)
# ------------------------------------------------------------------------------
print("\n📊 Walk-Forward Validation (2022-2025)...")

validation_results = []
validation_dates = pd.date_range('2022-01-01', '2025-07-01', freq='3ME')

for val_cutoff in validation_dates:
    val_cutoff = pd.Timestamp(val_cutoff)
    if val_cutoff < df['date'].min() + pd.Timedelta(days=365):
        continue
    if val_cutoff + pd.Timedelta(days=180) > df['date'].max():
        continue
    train_data = df[df['date'] <= val_cutoff].copy()
    if len(train_data) < 500:
        continue
    feature_cols_val = [c for c in train_data.columns
                        if c not in ['date', 'arabica_price', 'robusta_price'] and 'target' not in c]

    for comm in commodities:
        feature_cols_comm = get_commodity_features(feature_cols_val, comm)
        for h in horizons:
            train_data[f'target_{h}d'] = np.log(train_data[comm].shift(-h) / train_data[comm])
        train_data_clean = train_data.dropna(subset=[f'target_{h}d' for h in horizons] + [comm])
        train_data_clean[feature_cols_comm] = train_data_clean[feature_cols_comm].fillna(0)
        if len(train_data_clean) < 100:
            continue
        X_val = train_data_clean[feature_cols_comm]
        for h in horizons:
            y_val = train_data_clean[f'target_{h}d']
            if 'robusta' in comm:
                model_val = GradientBoostingRegressor(
                    loss='quantile', alpha=0.5, n_estimators=250,
                    learning_rate=0.03, max_depth=4, min_samples_leaf=20, random_state=42)
            else:
                model_val = GradientBoostingRegressor(
                    loss='quantile', alpha=0.5, n_estimators=200,
                    learning_rate=0.05, max_depth=5, random_state=42)
            model_val.fit(X_val, y_val)
            last_row_val = train_data_clean.iloc[[-1]][feature_cols_comm]
            start_price_val = train_data_clean.iloc[-1][comm]
            pred_log_ret = model_val.predict(last_row_val)[0]
            pred_price = start_price_val * np.exp(pred_log_ret)
            target_date = val_cutoff + pd.Timedelta(days=h)
            actual_df = df[df['date'] >= target_date]
            if len(actual_df) > 0:
                actual_price = actual_df.iloc[0][comm]
                error_pct = (pred_price - actual_price) / actual_price * 100
                validation_results.append({
                    'cutoff': val_cutoff, 'commodity': comm, 'horizon': h,
                    'predicted': pred_price, 'actual': actual_price, 'error_pct': error_pct
                })

val_df = pd.DataFrame()
if validation_results:
    val_df = pd.DataFrame(validation_results)
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION ERGEBNISSE")
    print("=" * 60)
    for comm in commodities:
        print(f"\n{comm.upper()}:")
        for h in horizons:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)]
            if len(subset) > 0:
                mae = subset['error_pct'].abs().mean()
                bias = subset['error_pct'].mean()
                hit_rate = (subset['error_pct'].abs() < 15).mean() * 100
                print(f"   {h:3d}d: MAE={mae:5.1f}%, Bias={bias:+5.1f}%, Hit-Rate(±15%)={hit_rate:4.1f}% (n={len(subset)})")
    val_df.to_csv(os.path.join(script_dir, 'validation_results.csv'), index=False)

# ------------------------------------------------------------------------------
# 4c. NEWS SENTIMENT CHECK
# ------------------------------------------------------------------------------
news_sentiment = None
if NEWS_MONITOR_AVAILABLE:
    print("\n📰 Prüfe News-Sentiment...")
    try:
        news_sentiment = get_coffee_news_sentiment(days_back=7, verbose=True)
        arabica_direction = "BULLISH" if preds_today[('arabica_price', 90, 0.5)] > current_prices_today['arabica_price'] else "BEARISH"
        alignment = check_news_model_alignment(news_sentiment, arabica_direction)
        if not alignment['aligned']:
            print(f"\n🚨 {alignment['warning']}")
        else:
            print("\n✅ News und Modell sind aligned")
    except Exception as e:
        print(f"\n⚠️ News-Check Fehler: {e}")

# ------------------------------------------------------------------------------
# 5. VISUALISIERUNG
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
for i, comm in enumerate(commodities):
    ax = axes[i]
    mask_hist = df['date'] > (df['date'].max() - pd.Timedelta(days=365))
    ax.plot(df.loc[mask_hist, 'date'], df.loc[mask_hist, comm], label='Price History', color='black', lw=1.5)
    today = df['date'].iloc[-1]
    future_dates = [today + pd.Timedelta(days=h) for h in horizons]
    y_base = [preds_today[(comm, h, 0.5)] for h in horizons]
    y_lower = [preds_today[(comm, h, 0.05)] for h in horizons]
    y_upper = [preds_today[(comm, h, 0.95)] for h in horizons]
    ax.plot(future_dates, y_base, 'o--', color='blue', label='Forecast (Base Case)', lw=2)
    ax.fill_between(future_dates, y_lower, y_upper, color='blue', alpha=0.15, label='90% Confidence Interval')
    if not val_df.empty:
        val_subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == 90)].sort_values('cutoff')
        for _, row in val_subset.iterrows():
            pred_date = row['cutoff'] + pd.Timedelta(days=90)
            if pred_date > (df['date'].max() - pd.Timedelta(days=365)):
                ax.plot(pred_date, row['predicted'], 'x', color='orange', markersize=8, alpha=0.7)
                ax.plot(pred_date, row['actual'], 'o', color='green', markersize=5, alpha=0.5)
        ax.plot([], [], 'x', color='orange', markersize=8, label='Backtest Predicted (90d)')
        ax.plot([], [], 'o', color='green', markersize=5, label='Backtest Actual')
    if news_sentiment and news_sentiment.get('total_score', 0) != 0:
        score = news_sentiment['total_score']
        signal = news_sentiment['signal']
        badge_color = 'green' if 'BULLISH' in signal else 'red' if 'BEARISH' in signal else 'gray'
        ax.text(0.98, 0.98, f"News: {signal}\n({score:+.1f})",
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor=badge_color, alpha=0.3))
    title_suffix = ""
    if comm == 'arabica_price' and 'dry_streak_mg' in df.columns:
        title_suffix = f" (Drought: {df.iloc[-1]['dry_streak_mg']:.0f}d, 3M-Rain: {df.iloc[-1]['rain_90d_mg']:.1f}mm)"
    ax.set_title(f"{comm.replace('_price', '').capitalize()} PRO Forecast{title_suffix}\n(Log-Return Model + Quantile Uncertainty)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

plt.tight_layout()
chart_path = os.path.join(script_dir, 'forecast_pro_chart.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Professional Chart: {chart_path}")

# ------------------------------------------------------------------------------
# 5b. BACKTEST PERFORMANCE CHART
# ------------------------------------------------------------------------------
if not val_df.empty:
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    for i, comm in enumerate(commodities):
        comm_name = comm.replace('_price', '').capitalize()
        ax1 = axes2[i, 0]
        for h, color, marker in [(30, 'blue', 'o'), (90, 'orange', 's'), (180, 'green', '^')]:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)].copy()
            subset = subset.sort_values('cutoff')
            subset['target_date'] = subset['cutoff'] + pd.to_timedelta(subset['horizon'], unit='D')
            ax1.plot(subset['target_date'], subset['actual'], '-', color=color, alpha=0.3, linewidth=2)
            ax1.plot(subset['target_date'], subset['predicted'], '--', color=color, alpha=0.8, linewidth=1)
            ax1.scatter(subset['target_date'], subset['predicted'], color=color, marker=marker, s=50, label=f'{h}d Predicted', alpha=0.8)
            ax1.scatter(subset['target_date'], subset['actual'], color=color, marker=marker, s=20, facecolors='none', edgecolors=color, label=f'{h}d Actual', alpha=0.5)
        ax1.set_title(f'{comm_name}: Predicted vs Actual')
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('Preis')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes2[i, 1]
        for h, color in [(30, 'blue'), (90, 'orange'), (180, 'green')]:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)].copy()
            subset = subset.sort_values('cutoff')
            ax2.bar(subset['cutoff'], subset['error_pct'], width=20, alpha=0.6, color=color, label=f'{h}d Fehler')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.axhline(y=subset['error_pct'].mean(), color=color, linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_title(f'{comm_name}: Prognose-Fehler über Zeit')
        ax2.set_xlabel('Cutoff Datum')
        ax2.set_ylabel('Fehler (%)')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhspan(-15, 15, alpha=0.1, color='green')
        ax2.set_ylim(-50, 50)

    plt.suptitle('Walk-Forward Backtest Performance (2022-2025)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'backtest_performance_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n📋 Performance-Zusammenfassung:")
    print("-" * 70)
    for comm in commodities:
        print(f"\n{comm.replace('_price', '').capitalize()}:")
        for h in horizons:
            subset = val_df[(val_df['commodity'] == comm) & (val_df['horizon'] == h)]
            if len(subset) > 0:
                mae = subset['error_pct'].abs().mean()
                bias = subset['error_pct'].mean()
                std = subset['error_pct'].std()
                hit_rate = (subset['error_pct'].abs() < 15).mean() * 100
                print(f"   {h:3d}d: MAE={mae:5.1f}%, Bias={bias:+5.1f}%, Std={std:5.1f}%, Hit-Rate(±15%)={hit_rate:4.1f}%")

# CSV Export
rows = []
for h in horizons:
    row = {'Horizon_Days': h, 'Date': (today + pd.Timedelta(days=h)).date()}
    for comm in commodities:
        row[f'{comm}_Low'] = preds_today[(comm, h, 0.05)]
        row[f'{comm}_Base'] = preds_today[(comm, h, 0.50)]
        row[f'{comm}_High'] = preds_today[(comm, h, 0.95)]
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(script_dir, 'coffee_forecast_pro.csv'), index=False)
print(f"✅ CSV exportiert")