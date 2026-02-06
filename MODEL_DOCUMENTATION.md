# ☕ Coffee Price Forecasting Model - Dokumentation (v5 Production)

Dieses Dokument beschreibt die Architektur, Datenquellen und Funktionsweise des Profi-Prognosemodells für Arabica- und Robusta-Preise (6 Monate Horizont).

**Version:** 5.0  
**Letzte Aktualisierung:** Februar 2026  
**Status:** Production-Ready

---

## 1. Performance-Übersicht

Das Modell wurde durch mehrere Iterationen signifikant verbessert:

| Metrik | Original (v3) | Aktuell (v5) | Verbesserung |
|--------|---------------|--------------|--------------|
| **Arabica 180d MAE** | 26.5% | **19.0%** | -7.5% |
| **Arabica 180d Bias** | -2.6% | **-3.5%** | ~ |
| **Arabica 180d Hit-Rate** | 21.4% | **42.9%** | +21.5% |
| **Robusta 180d MAE** | 31.4% | **20.0%** | -11.4% |
| **Robusta 180d Bias** | -15.4% | **-6.4%** | +9.0% |
| **Robusta 180d Hit-Rate** | 28.6% | **28.6%** | ~ |

**Metriken erklärt:**
- **MAE (Mean Absolute Error):** Durchschnittlicher Prognosefehler in %
- **Bias:** Systematische Über-/Unterschätzung (negativ = unterschätzt)
- **Hit-Rate:** Anteil der Prognosen innerhalb ±15% des tatsächlichen Preises

---

## 2. Datenquellen & Inventar

Das System nutzt einen hybriden Ansatz aus harten Daten (CSV) und Live-Signalen (News).

| Dateiname | Inhalt | Verwendung im Modell |
|:----------|:-------|:---------------------|
| **`arabica_clean.csv`** | Historische Preise für Arabica (Cent/lb). Ab 2000. | Basis für Log-Returns & Targets |
| **`robusta_clean.csv`** | Historische Preise für Robusta (USD/t). Ab 2015. | Basis für Log-Returns & Targets |
| **`cot_data.csv`** | Commitment of Traders (Net Speculator Position) | Sentiment-Indikator |
| **`weather_minas_gerais.csv`** | Wetterdaten Brasilien (Regen/Dürre) | Fundamental-Indikator (Arabica) |
| **`weather_dak_lak.csv`** | Wetterdaten Vietnam (Regen/Dürre) | Fundamental-Indikator (Robusta) |
| **`arabica_stocks.csv`** | Zertifizierte Lagerbestände Arabica | Supply-Indikator |
| **`robusta_stocks.csv`** | Zertifizierte Lagerbestände Robusta | Supply-Indikator |
| **`news_monitor.py`** | Live News-Scanner via LSEG | Circuit Breaker & Sentiment |

---

## 3. Modellarchitektur

### A. Kern-Algorithmus: Log-Linear Quantile Boosting

**Warum Log-Returns?**
- **Problem (v3):** Modell lernte absolute Preise ("300 USD") → blind bei All-Time-Highs
- **Lösung (v5):** Modell lernt relative Veränderungen (ln(P_future / P_now)) → kann endlose Trends vorhersagen

**Quantile Regression:**
Statt einer Linie prognostiziert das Modell einen Wahrscheinlichkeits-Trichter:
- **5% Quantil (Lower Bound):** Worst Case - nur 5% Risiko dass es darunter fällt
- **50% Quantil (Base Case):** Erwartungswert - die Hauptprognose
- **95% Quantil (Upper Bound):** Best Case - Luft nach oben

### B. Feature-Kategorien

Das Modell nutzt **keine Preis-Features** direkt (Data Leakage vermieden), sondern:

**1. Trend-Indikatoren:**
```
- {commodity}_trend_180d    → 6-Monats Trend (%)
- {commodity}_trend_90d     → 3-Monats Trend (%)
- {commodity}_vs_ma252      → Position vs 1-Jahres-Durchschnitt
- {commodity}_trend_accel   → Trend-Beschleunigung
```

**2. Volatilität:**
```
- {commodity}_volatility_30d → Annualisierte 30-Tage Volatilität
```

**3. Momentum:**
```
- {commodity}_momentum_10d   → 10-Tage Momentum (%)
- {commodity}_momentum_30d   → 30-Tage Momentum (%)
- {commodity}_momentum_accel → Momentum-Beschleunigung
- {commodity}_momentum_div   → Kurzfristig vs Mittelfristig
```

**4. COT (Sentiment):**
```
- COT_Net_Spec      → Net Speculator Position (Roh)
- COT_Signal        → 4-Wochen geglättet
- COT_zscore        → Normalisiert (Z-Score)
- COT_momentum_4w   → Positionsveränderung
```

**5. Wetter (mit Lag):**
```
- rain_90d_mg_lag90/180     → Brasilien Regen (verzögert)
- rain_90d_dl_lag90/180     → Vietnam Regen (verzögert)
- dry_streak_mg/dl_lag30    → Dürre-Stress (verzögert)
- brazil_drought_risk       → Binär: >21 Tage trocken
- vietnam_drought_risk      → Binär: >14 Tage trocken
- brazil_rain_anomaly       → Regen vs Normal (Z-Score)
- vietnam_rain_anomaly      → Regen vs Normal (Z-Score)
```

**6. Saisonalität:**
```
- month_sin / month_cos     → Zyklische Kodierung
- brazil_harvest            → Binär: Apr-Sep
- vietnam_harvest           → Binär: Oct-Jan
- brazil_frost_season       → Binär: Mai-Aug
```

**7. Cross-Commodity:**
```
- arabica_robusta_ratio     → Spread (normalisiert)
- spread_trend_30d          → Spread-Veränderung
- spread_zscore             → Spread vs historisch
```

**8. Stocks:**
```
- {commodity}_stocks_change_30d/90d → Bestandsveränderung
- {commodity}_stocks_zscore         → Bestände vs Normal
- {commodity}_stocks_trend          → Kurz- vs Mittelfristig
```

### C. Separate Hyperparameter

Robusta hat kürzere Historie und höhere Volatilität, daher konservativere Parameter:

| Parameter | Arabica | Robusta |
|-----------|---------|---------|
| n_estimators | 200 | 250 |
| learning_rate | 0.05 | 0.03 |
| max_depth | 5 | 4 |
| min_samples_leaf | default | 20 |

---

## 4. News Monitor (Circuit Breaker)

Der News Monitor scannt LSEG/Refinitiv nach Kaffee-relevanten Nachrichten und berechnet ein Sentiment-Score.

### Funktionsweise:
1. **Queries:** Coffee Brazil, Coffee Vietnam, Arabica, Robusta, Coffee Weather, etc.
2. **Sentiment-Keywords:**
   - **Bullish:** drought, frost, shortage, crop damage, prices surge (+1.0 bis +2.5)
   - **Bearish:** favorable weather, bumper crop, surplus, prices fall (-1.0 bis -2.0)
3. **Region-Boost:** News aus Hauptanbaugebieten (Brazil, Vietnam) +30% Gewicht
4. **Signal-Klassifikation:**
   - STRONG_BULLISH (Score ≥ 5)
   - BULLISH (Score ≥ 2)
   - NEUTRAL (-2 < Score < 2)
   - BEARISH (Score ≤ -2)
   - STRONG_BEARISH (Score ≤ -5)

### Circuit Breaker Regel:
```
WENN Modell = BULLISH UND News = BEARISH → WARNUNG!
WENN Modell = BEARISH UND News = BULLISH → WARNUNG!
```
Bei Konflikt: Nicht blind dem Modell folgen, News manuell prüfen!

---

## 5. Output-Dateien

| Datei | Beschreibung |
|-------|--------------|
| **`forecast_pro_chart.png`** | Haupt-Prognose mit Confidence Interval |
| **`backtest_performance_chart.png`** | Historische Genauigkeit (Predicted vs Actual) |
| **`coffee_forecast_pro.csv`** | Prognose-Werte (Low/Base/High für 30/90/180d) |
| **`validation_results.csv`** | Walk-Forward Backtest Details |
| **`news_sentiment.csv`** | News-Analyse (falls News Monitor aktiv) |

---

## 6. Workflow & Bedienung

### A. Einmalige Einrichtung
```bash
# Python 3.11 Environment (3.13 hat Kompatibilitätsprobleme!)
python -m venv venv311
venv311\Scripts\activate  # Windows
source venv311/bin/activate  # Mac/Linux

# Packages installieren
pip install pandas numpy scikit-learn matplotlib requests
pip install refinitiv-data  # Für News Monitor (optional)
```

### B. Täglicher/Wöchentlicher Betrieb

**Option 1: Alles manuell**
```bash
# 1. COT-Daten aktualisieren (wöchentlich, nach Freitag)
python fetch_cot_data_v2.py

# 2. Wetterdaten aktualisieren (falls Skript vorhanden)
python update_weather_data.py

# 3. Modell ausführen
python main_forecast_model.py

# 4. News prüfen (optional, LSEG muss laufen)
python news_monitor.py
```

**Option 2: Master-Skript (empfohlen)**
```bash
python run_forecast.py  # Macht alles automatisch
```

### C. Interpretation der Ergebnisse

**Forecast Pro Chart:**
- **Schwarze Linie:** Historische Preise
- **Blaue Punkte + Linie:** Base Case Prognose
- **Blaue Fläche:** 90% Confidence Interval
- **Orange X:** Backtest Predicted (90d)
- **Grüner Punkt:** Backtest Actual

**Backtest Performance Chart:**
- **Linke Spalte:** Predicted vs Actual über Zeit
- **Rechte Spalte:** Prognose-Fehler (%) mit ±15% Zone

**Entscheidungsregeln:**
- Preis am unteren Band des Confidence Intervals → Potenzielles Kauf-Signal
- Preis am oberen Band → Potenzielles Verkaufs-Signal
- News-Signal widerspricht Modell → VORSICHT, manuell prüfen!

---

## 7. Bekannte Limitierungen

1. **Black Swan Events:** Das Modell kann plötzliche Ereignisse (Frost über Nacht, politische Krisen) nicht vorhersagen. Daher der News Monitor als Circuit Breaker.

2. **Kurzfristige Prognosen (30d):** Höherer Fehler und Bias als langfristige. Für kurzfristiges Trading weniger geeignet.

3. **Robusta-Bias:** Trotz Verbesserungen noch leicht negativer Bias (-6.4%). Modell tendiert dazu, Robusta zu unterschätzen.

4. **COT-Daten Lag:** COT wird freitags für Dienstag veröffentlicht → 3 Tage Verzögerung.

5. **Python Version:** Erfordert Python 3.11. Version 3.13 hat Kompatibilitätsprobleme mit refinitiv-data.

---

## 8. Änderungshistorie

| Version | Datum | Änderungen |
|---------|-------|------------|
| v3 | 2025-Q4 | Ursprüngliches Modell mit Log-Returns |
| v4 | 2026-01 | Data Leakage Fix, Walk-Forward Validation |
| v5 | 2026-02 | Trend/Momentum Features, COT Fix, News Monitor, Saisonalität, Robusta-Optimierung |

---

## 9. Datei-Übersicht
```
Coffee Price/
├── main_forecast_model.py    # Haupt-Engine (Forecast + Charts)
├── fetch_cot_data.py         # COT-Daten von CFTC laden
├── news_monitor.py           # LSEG News Sentiment Scanner
├── run_forecast.py           # Master-Skript (optional)
├── update_weather_data.py    # Wetterdaten aktualisieren
│
├── arabica_clean.csv         # Preisdaten Arabica
├── robusta_clean.csv         # Preisdaten Robusta
├── cot_data.csv              # COT Speculator Positions
├── weather_minas_gerais.csv  # Wetter Brasilien
├── weather_dak_lak.csv       # Wetter Vietnam
├── arabica_stocks.csv        # Lagerbestände Arabica
├── robusta_stocks.csv        # Lagerbestände Robusta
│
├── forecast_pro_chart.png    # OUTPUT: Prognose-Chart
├── backtest_performance_chart.png  # OUTPUT: Backtest-Chart
├── coffee_forecast_pro.csv   # OUTPUT: Prognose-Werte
├── validation_results.csv    # OUTPUT: Backtest-Details
│
└── MODEL_DOCUMENTATION.md    # Diese Datei
```

---

## 10. Kontakt & Support

Bei Fragen oder Problemen:
1. Prüfe ob alle Datenquellen aktuell sind
2. Prüfe Python-Version (muss 3.11 sein)
3. Prüfe ob LSEG Workspace läuft (für News Monitor)

**Fazit:** Das System ist kein "Orakel", sondern ein **Wahrscheinlichkeits-Rechner**, der durch menschliche Überprüfung (News-Scanner) abgesichert wird. Nutze die Prognosen als einen Input neben anderen Analysen, nicht als alleinige Trading-Grundlage.