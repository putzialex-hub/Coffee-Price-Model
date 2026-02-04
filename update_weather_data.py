
import requests
import pandas as pd
import time

print("🌤️ Open-Meteo Wetterdaten Downloader")
print("====================================")

# Konfiguration
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
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        if 'daily' in data:
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.rename(columns={'time': 'date'})
            
            # Speichern
            df.to_csv(info['filename'], index=False)
            print(f"✅ Gespeichert: {info['filename']} ({len(df)} Zeilen)")
        else:
            print(f"❌ Keine Daten erhalten: {data}")
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
    
    time.sleep(2) # Höflich sein zur API

print("\nFertig.")
