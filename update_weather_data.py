import os
import time
import pandas as pd
import requests
import certifi
from requests.exceptions import SSLError

print("🌤️ Open-Meteo Wetterdaten Downloader")
print("====================================")

# Konfiguration
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\WZHALP3\OneDrive - Raiffeisen Bank International Group\Agriculture"
    r"\Coffee\Coffee Price-20260202T064517Z-3-001\Coffee Price"
)
OUTPUT_DIR = os.getenv("OPEN_METEO_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

REGIONS = {
    'Minas Gerais (Arabica)': {'lat': -18.5122, 'lon': -44.5550, 'filename': 'weather_minas_gerais.csv'},
    'Dak Lak (Robusta)':      {'lat': 12.6667,  'lon': 108.0500, 'filename': 'weather_dak_lak.csv'}
}

START_DATE = "2000-01-01"
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')

for name, info in REGIONS.items():
    print(f"   Lade Daten für {name}...")
    
    # Open-Meteo Historical Weather API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": info['lat'],
        "longitude": info['lon'],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_hours"],
        "timezone": "auto"
    }
    
    verify_ssl = os.getenv("OPEN_METEO_VERIFY_SSL", "true").lower() in {"1", "true", "yes"}

    try:
        r = requests.get(
            url,
            params=params,
            timeout=30,
            verify=certifi.where() if verify_ssl else False,
        )
        r.raise_for_status()
        data = r.json()
    except SSLError:
        if verify_ssl:
            print("⚠️ SSL-Überprüfung fehlgeschlagen, erneuter Versuch ohne Verifikation...")
            r = requests.get(url, params=params, timeout=30, verify=False)
            r.raise_for_status()
            data = r.json()
        else:
            raise

    try:    
        if 'daily' in data:
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.rename(columns={'time': 'date'})
            latest_date = df['date'].max().date()
            requested_end = pd.to_datetime(END_DATE).date()

            # Speichern
            output_path = os.path.join(OUTPUT_DIR, info['filename'])
            df.to_csv(output_path, index=False)
            print(f"✅ Gespeichert: {output_path} ({len(df)} Zeilen)")
            if latest_date < requested_end:
                print(
                    "⚠️ Hinweis: Open-Meteo liefert aktuell nur Daten bis "
                    f"{latest_date}. Angefordert war {requested_end}."
                )
        else:
            print(f"❌ Keine Daten erhalten: {data}")
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
    
    time.sleep(2) # Höflich sein zur API

print("\nFertig.")
