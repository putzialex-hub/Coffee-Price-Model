"""
COT Daten Downloader (CFTC) - Version 2
========================================
Lädt Legacy Futures-Only COT Reports für Kaffee (Coffee C = Arabica)

Zwei Methoden:
1. Direkt von CFTC ZIP-Files (Legacy Format)
2. Fallback: cot-reports Library (pip install cot-reports)
"""

import pandas as pd
import requests
import zipfile
import io
import os
from datetime import datetime

def get_requests_verify():
    """
    Liefert den Verify-Parameter für requests.
    Nutzt certifi falls verfügbar und erlaubt optionales Deaktivieren via ENV.
    """

    if os.getenv('COT_SSL_NO_VERIFY') == '1':
        print("   ⚠️ SSL-Verify deaktiviert (COT_SSL_NO_VERIFY=1)")
        return False

    custom_bundle = os.getenv('COT_CA_BUNDLE')
    if custom_bundle:
        return custom_bundle

    try:
        import certifi

        return certifi.where()
    except Exception:
        return True


def configure_ca_bundle_env():
    """Setzt REQUESTS_CA_BUNDLE, damit Libraries certifi nutzen."""

    custom_bundle = os.getenv('COT_CA_BUNDLE')
    if custom_bundle:
        os.environ.setdefault('REQUESTS_CA_BUNDLE', custom_bundle)
        os.environ.setdefault('SSL_CERT_FILE', custom_bundle)
        return

    try:
        import certifi

        ca_bundle = certifi.where()
        os.environ.setdefault('REQUESTS_CA_BUNDLE', ca_bundle)
        os.environ.setdefault('SSL_CERT_FILE', ca_bundle)
    except Exception:
        return


def enforce_requests_no_verify():
    """Erzwingt verify=False für requests, wenn COT_SSL_NO_VERIFY=1 gesetzt ist."""

    if os.getenv('COT_SSL_NO_VERIFY') != '1':
        return

    if getattr(requests.sessions.Session.request, '_cot_patched', False):
        return

    original_request = requests.sessions.Session.request

    def patched_request(self, method, url, **kwargs):
        kwargs.setdefault('verify', False)
        return original_request(self, method, url, **kwargs)

    patched_request._cot_patched = True  # type: ignore[attr-defined]
    requests.sessions.Session.request = patched_request

print("🕵️‍♂️ COT Daten Downloader (CFTC) v2")
print("=" * 50)

# =============================================================================
# METHODE 1: Direkt von CFTC (Legacy Futures Only)
# =============================================================================

def download_cftc_legacy():
    """
    Lädt Legacy Futures-Only Reports direkt von CFTC.
    URL-Format: https://www.cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip (Financial)
                https://www.cftc.gov/files/dea/history/deacot{YYYY}.zip (Legacy)
    """
    
    YEARS = range(2015, 2027)  # 2015 bis 2026
    all_cot = []
    
    for year in YEARS:
        # Legacy Report URL
        url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
        print(f"   📥 {year}...", end=" ")
        
        try:
            verify_setting = get_requests_verify()
            r = requests.get(url, timeout=30, verify=verify_setting)
            
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                
                # Finde die richtige Datei im ZIP
                txt_files = [f for f in z.namelist() if f.endswith('.txt')]
                if not txt_files:
                    print("❌ Keine TXT-Datei im ZIP")
                    continue
                    
                file_name = txt_files[0]
                
                with z.open(file_name) as f:
                    # Legacy COT hat spezifische Spalten
                    df_year = pd.read_csv(f, low_memory=False)
                    
                    # Debug: Zeige verfügbare Spalten für erstes Jahr
                    if year == 2015:
                        print(f"\n   📋 Verfügbare Spalten: {list(df_year.columns)[:10]}...")
                    
                    # Suche nach der Market-Spalte (Name variiert)
                    market_col = None
                    for col in df_year.columns:
                        if 'market' in col.lower() and 'name' in col.lower():
                            market_col = col
                            break
                    
                    if market_col is None:
                        # Fallback: Erste Spalte ist oft der Market Name
                        market_col = df_year.columns[0]
                    
                    # Filtere nach Coffee C (Arabica)
                    coffee_mask = df_year[market_col].str.contains('COFFEE C', na=False, case=False)
                    coffee_df = df_year[coffee_mask].copy()
                    
                    if not coffee_df.empty:
                        all_cot.append(coffee_df)
                        print(f"✅ {len(coffee_df)} Zeilen")
                    else:
                        # Versuche breitere Suche
                        coffee_mask = df_year[market_col].str.contains('COFFEE', na=False, case=False)
                        coffee_df = df_year[coffee_mask].copy()
                        if not coffee_df.empty:
                            all_cot.append(coffee_df)
                            print(f"✅ {len(coffee_df)} Zeilen (broad match)")
                        else:
                            print("⚠️ Keine Coffee-Daten")
                            
            elif r.status_code == 404:
                print("⚠️ Noch nicht verfügbar")
            else:
                print(f"❌ HTTP {r.status_code}")
        except requests.exceptions.SSLError:
            print("⚠️ SSL-Fehler, versuche ohne Verifikation...", end=" ")
            try:
                r = requests.get(url, timeout=30, verify=False)
                if r.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    txt_files = [f for f in z.namelist() if f.endswith('.txt')]
                    if not txt_files:
                        print("❌ Keine TXT-Datei im ZIP")
                        continue
                    file_name = txt_files[0]
                    with z.open(file_name) as f:
                        df_year = pd.read_csv(f, low_memory=False)
                        if year == 2015:
                            print(f"\n   📋 Verfügbare Spalten: {list(df_year.columns)[:10]}...")
                        market_col = None
                        for col in df_year.columns:
                            if 'market' in col.lower() and 'name' in col.lower():
                                market_col = col
                                break
                        if market_col is None:
                            market_col = df_year.columns[0]
                        coffee_mask = df_year[market_col].str.contains('COFFEE C', na=False, case=False)
                        coffee_df = df_year[coffee_mask].copy()
                        if not coffee_df.empty:
                            all_cot.append(coffee_df)
                            print(f"✅ {len(coffee_df)} Zeilen (no-verify)")
                        else:
                            coffee_mask = df_year[market_col].str.contains('COFFEE', na=False, case=False)
                            coffee_df = df_year[coffee_mask].copy()
                            if not coffee_df.empty:
                                all_cot.append(coffee_df)
                                print(f"✅ {len(coffee_df)} Zeilen (broad match, no-verify)")
                            else:
                                print("⚠️ Keine Coffee-Daten (no-verify)")
                elif r.status_code == 404:
                    print("⚠️ Noch nicht verfügbar")
                else:
                    print(f"❌ HTTP {r.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Netzwerk-Fehler: {type(e).__name__}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Netzwerk-Fehler: {type(e).__name__}")
        except Exception as e:
            print(f"❌ Fehler: {e}")
    
    return all_cot


def process_cot_data(all_cot):
    """Verarbeitet die rohen COT-Daten in das finale Format."""
    
    if not all_cot:
        return None
        
    full_cot = pd.concat(all_cot, ignore_index=True)
    print(f"\n   📊 Gesamte Rohdaten: {len(full_cot)} Zeilen")
    
    # Finde die richtigen Spaltennamen (CFTC ändert diese manchmal)
    col_mapping = {
        'date': None,
        'noncomm_long': None,
        'noncomm_short': None,
        'comm_long': None,
        'comm_short': None
    }
    
    for col in full_cot.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        
        # Datum - suche nach "as of date" mit YYYY-MM-DD Format bevorzugt
        if 'as_of_date' in col_lower:
            if 'yyyy_mm_dd' in col_lower:
                col_mapping['date'] = col  # Bevorzugtes Format
            elif col_mapping['date'] is None:
                col_mapping['date'] = col
        elif 'report' in col_lower and 'date' in col_lower and col_mapping['date'] is None:
            col_mapping['date'] = col
            
        # Non-Commercial Positions (LEGACY REPORT - für Commodities!)
        # WICHTIG: Nutze "Noncommercial Positions" NICHT "Traders-Noncommercial"
        if 'noncommercial_positions' in col_lower or 'noncommercial positions' in col.lower():
            if 'long' in col_lower and 'all' in col_lower and 'change' not in col_lower and '%' not in col:
                col_mapping['noncomm_long'] = col
            elif 'short' in col_lower and 'all' in col_lower and 'change' not in col_lower and '%' not in col:
                col_mapping['noncomm_short'] = col
                
        # Commercial Positions (LEGACY REPORT)
        if 'commercial_positions' in col_lower or 'commercial positions' in col.lower():
            if 'noncommercial' not in col_lower:
                if 'long' in col_lower and 'all' in col_lower and 'change' not in col_lower and '%' not in col:
                    col_mapping['comm_long'] = col
                elif 'short' in col_lower and 'all' in col_lower and 'change' not in col_lower and '%' not in col:
                    col_mapping['comm_short'] = col
    
    print(f"   🔍 Gefundene Spalten:")
    for key, val in col_mapping.items():
        print(f"      {key}: {val}")
    
    # Prüfe ob alle nötigen Spalten gefunden wurden
    if col_mapping['date'] is None:
        print("   ❌ Datum-Spalte nicht gefunden!")
        print(f"   Verfügbare Spalten: {list(full_cot.columns)}")
        return None
        
    if col_mapping['noncomm_long'] is None or col_mapping['noncomm_short'] is None:
        print("   ❌ Non-Commercial Spalten nicht gefunden!")
        return None
    
    # Erstelle finalen DataFrame
    df_final = pd.DataFrame()
    
    # Datum parsen (verschiedene Formate möglich)
    date_col = col_mapping['date']
    try:
        df_final['date'] = pd.to_datetime(full_cot[date_col])
    except:
        try:
            df_final['date'] = pd.to_datetime(full_cot[date_col], format='%Y%m%d')
        except:
            df_final['date'] = pd.to_datetime(full_cot[date_col], format='%m/%d/%Y')
    
    # Net Speculator Position berechnen
    long_col = col_mapping['noncomm_long']
    short_col = col_mapping['noncomm_short']
    
    df_final['noncomm_long'] = pd.to_numeric(full_cot[long_col], errors='coerce').fillna(0)
    df_final['noncomm_short'] = pd.to_numeric(full_cot[short_col], errors='coerce').fillna(0)
    df_final['COT_Net_Spec'] = df_final['noncomm_long'] - df_final['noncomm_short']
    
    # Optional: Commercial Positions
    if col_mapping['comm_long'] and col_mapping['comm_short']:
        df_final['comm_long'] = pd.to_numeric(full_cot[col_mapping['comm_long']], errors='coerce').fillna(0)
        df_final['comm_short'] = pd.to_numeric(full_cot[col_mapping['comm_short']], errors='coerce').fillna(0)
        df_final['COT_Net_Comm'] = df_final['comm_long'] - df_final['comm_short']
    
    # Finalisieren
    df_final = df_final[['date', 'COT_Net_Spec']].sort_values('date').drop_duplicates(subset='date')
    
    return df_final


def expand_to_daily(df_weekly):
    """Expandiert wöchentliche COT-Daten auf tägliche Frequenz."""
    
    if df_weekly is None or df_weekly.empty:
        return None
        
    date_range = pd.date_range(
        df_weekly['date'].min(), 
        df_weekly['date'].max(), 
        freq='D'
    )
    
    df_daily = pd.DataFrame({'date': date_range})
    df_daily = df_daily.merge(df_weekly, on='date', how='left')
    df_daily['COT_Net_Spec'] = df_daily['COT_Net_Spec'].ffill()
    
    return df_daily


# =============================================================================
# METHODE 2: Fallback mit cot-reports Library
# =============================================================================

def download_via_library():
    """Fallback: Nutze cot-reports Library wenn CFTC-Zugriff nicht funktioniert."""
    
    try:
        from cot_reports import cot_reports
        
        print("   📚 Nutze cot-reports Library...")
        
        # Legacy Futures Only Report
        configure_ca_bundle_env()
        enforce_requests_no_verify()
        try:
            df = cot_reports.cot_year(
                year=2024,  # Aktuelles Jahr
                cot_report_type='legacy_fut'
            )
        except requests.exceptions.SSLError:
            print("   ⚠️ SSL-Fehler in Library, retry ohne Verifikation...")
            os.environ['COT_SSL_NO_VERIFY'] = '1'
            enforce_requests_no_verify()
            df = cot_reports.cot_year(
                year=2024,  # Aktuelles Jahr
                cot_report_type='legacy_fut'
            )
        
        # Filter für Coffee
        coffee_df = df[df['Market and Exchange Names'].str.contains('COFFEE', na=False, case=False)]
        
        if not coffee_df.empty:
            print(f"   ✅ {len(coffee_df)} Zeilen via Library")
            return coffee_df
        else:
            print("   ⚠️ Keine Coffee-Daten in Library")
            return None
            
    except ImportError:
        print("   ⚠️ cot-reports nicht installiert (pip install cot-reports)")
        return None
    except Exception as e:
        print(f"   ❌ Library-Fehler: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    configure_ca_bundle_env()
    enforce_requests_no_verify()

    # Versuche Methode 1: Direkt von CFTC
    print("\n📡 Methode 1: Direkter CFTC Download")
    print("-" * 50)
    all_cot = download_cftc_legacy()
    
    if all_cot:
        df_weekly = process_cot_data(all_cot)
        
        if df_weekly is not None:
            df_daily = expand_to_daily(df_weekly)
            
            if df_daily is not None:
                # Speichern
                df_daily.to_csv('cot_data.csv', index=False)
                
                print("\n" + "=" * 50)
                print("✅ COT Daten erfolgreich gespeichert!")
                print("=" * 50)
                print(f"   Datei: cot_data.csv")
                print(f"   Zeitraum: {df_daily['date'].min().date()} bis {df_daily['date'].max().date()}")
                print(f"   Datenpunkte: {len(df_daily)} (täglich)")
                print(f"\n   Letzte 5 Werte:")
                print(df_daily.tail().to_string(index=False))
            else:
                print("\n❌ Fehler bei der täglichen Expansion")
        else:
            print("\n❌ Fehler bei der Datenverarbeitung")
    else:
        print("\n⚠️ CFTC-Download fehlgeschlagen, versuche Library...")
        download_via_library()