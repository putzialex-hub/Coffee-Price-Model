"""
Coffee News Monitor - LSEG/Refinitiv Integration
=================================================
Überwacht Kaffee-relevante Nachrichten und berechnet ein Sentiment-Score.

Verwendung:
    1. Standalone: python news_monitor.py
    2. Als Modul: from news_monitor import get_coffee_news_sentiment

Voraussetzungen:␊
    - LSEG Workspace oder Eikon muss laufen (optional)
    - pip install refinitiv-data (optional)
    
Autor: Coffee Forecast System
Version: 1.0
"""

import pandas as pd
from datetime import datetime, timedelta
import re
import os
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Keywords für Kaffee-News (LSEG News Codes)
COFFEE_TOPICS = [
    "CFFE",      # Coffee commodity
    "KC",        # Coffee C Futures (Arabica)
    "RC",        # Robusta Coffee
    "COFFEE",    # General coffee news
]

# Länder/Regionen für Supply-News
COFFEE_REGIONS = {
    "brazil": ["brazil", "brasil", "minas gerais", "sao paulo", "espirito santo"],
    "vietnam": ["vietnam", "dak lak", "central highlands"],
    "colombia": ["colombia", "colombian"],
    "indonesia": ["indonesia", "sumatra", "java"],
    "ethiopia": ["ethiopia", "ethiopian"],
}

# Sentiment-Keywords (für Preis-Impact)
BULLISH_KEYWORDS = {
    # Wetter-Risiken (schlecht für Ernte = gut für Preis)
    "drought": 2.0,
    "frost": 2.5,
    "freeze": 2.5,
    "dry weather": 1.5,
    "dry spell": 1.5,
    "la nina": 1.5,
    "el nino": 1.0,
    "heat wave": 1.5,
    "flooding": 1.0,
    
    # Supply-Probleme
    "shortage": 2.0,
    "deficit": 2.0,
    "tight supply": 1.5,
    "low stocks": 1.5,
    "stock decline": 1.5,
    "crop damage": 2.0,
    "crop failure": 2.5,
    "lower output": 1.5,
    "reduced harvest": 1.5,
    "smaller crop": 1.5,
    "production drop": 1.5,
    "production decline": 1.5,
    "export restrictions": 1.5,
    "export ban": 2.0,
    
    # Nachfrage
    "strong demand": 1.0,
    "demand surge": 1.5,
    "consumption rise": 1.0,
    
    # Preis-Bewegungen
    "prices surge": 1.5,
    "prices rally": 1.5,
    "prices jump": 1.5,
    "prices spike": 2.0,
    "all-time high": 2.0,
    "record high": 2.0,
    "multi-year high": 1.5,
}

BEARISH_KEYWORDS = {
    # Gutes Wetter (gut für Ernte = schlecht für Preis)
    "favorable weather": -1.5,
    "good rains": -1.5,
    "above average rain": -1.5,
    "normal rainfall": -1.0,
    "ideal conditions": -1.5,
    "weather improves": -1.0,
    
    # Supply-Überschuss
    "bumper crop": -2.0,
    "record harvest": -2.0,
    "record crop": -2.0,
    "surplus": -1.5,
    "oversupply": -2.0,
    "ample supply": -1.5,
    "stock build": -1.0,
    "stock increase": -1.0,
    "higher output": -1.5,
    "larger crop": -1.5,
    "production increase": -1.5,
    "output rise": -1.5,
    
    # Nachfrage-Schwäche
    "weak demand": -1.0,
    "demand slump": -1.5,
    "consumption drop": -1.0,
    
    # Preis-Bewegungen
    "prices fall": -1.5,
    "prices drop": -1.5,
    "prices decline": -1.5,
    "prices slump": -1.5,
    "prices tumble": -2.0,
    "prices plunge": -2.0,
    "multi-year low": -1.5,
}

# ==============================================================================
# LSEG/REFINITIV INTEGRATION
# ==============================================================================

def init_lseg_session():
    """Initialisiert die LSEG Session (Workspace/Eikon muss laufen)."""
    try:
        import refinitiv.data as rd
        
        rd.open_session()
        print("   ✅ LSEG Session geöffnet (Desktop)")
        return rd
        
    except ImportError:
        print("   ❌ refinitiv-data nicht installiert!")
        print("      Installiere mit: pip install refinitiv-data")
        return None
        
    except Exception as e:
        print(f"   ❌ LSEG Session Fehler: {e}")
        print("      Stelle sicher dass Workspace/Eikon läuft!")
        return None


def fetch_news_lseg(rd, query="Coffee", count=50, days_back=7):
    """Holt News von LSEG/Refinitiv."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        headlines = rd.news.get_headlines(
            query=query,
            count=count,
            date_from=start_date.strftime("%Y-%m-%d"),
            date_to=end_date.strftime("%Y-%m-%d")
        )
        
        if headlines is not None and not headlines.empty:
            return headlines
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"   ⚠️ News-Abruf Fehler: {e}")
        return pd.DataFrame()


# ==============================================================================
# PUBLIC RSS FALLBACK
# ==============================================================================

def fetch_news_rss(query="Coffee", days_back=7, timeout=10):
    """Holt News über öffentliches RSS (Google News)."""
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote(query)}%20when%3A{days_back}d&hl=en-US&gl=US&ceid=US:en"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read()
    except Exception as exc:
        print(f"   ⚠️ RSS-Abruf Fehler ({query}): {exc}")
        return []

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        print(f"   ⚠️ RSS-Parsing Fehler ({query}): {exc}")
        return []

    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title", default="").strip()
        pub_date = item.findtext("pubDate", default="").strip()
        if not title:
            continue
        parsed_date = None
        if pub_date:
            try:
                parsed_date = parsedate_to_datetime(pub_date)
            except (TypeError, ValueError):
                parsed_date = None
        items.append(
            {
                "headline": title,
                "date": parsed_date or datetime.now(),
                "source": "Google News RSS",
                "query": query,
            }
        )
    return items


def get_public_news_sentiment(days_back=7, verbose=True):
    """Fallback: Nutzt öffentliche RSS-News für Sentiment."""
    queries = [
        "Coffee Brazil",
        "Coffee Vietnam",
        "Arabica",
        "Robusta Coffee",
        "Coffee Harvest",
        "Coffee Weather",
        "Coffee Prices",
    ]

    if verbose:
        print("\n🌐 Lade News via RSS (public fallback)...")

    all_items = []
    for query in queries:
        items = fetch_news_rss(query=query, days_back=days_back)
        if items:
            all_items.extend(items)
            if verbose:
                print(f"   {query}: {len(items)} Artikel")

    if not all_items:
        if verbose:
            print("\n   ⚠️ Keine RSS-News gefunden")
        return {
            "total_score": 0,
            "signal": "NEUTRAL",
            "news_count": 0,
            "news_df": pd.DataFrame(),
            "by_type": {},
            "note": "RSS nicht verfügbar - Fallback ohne News",
        }

    results = []
    for item in all_items:
        headline = item.get("headline", "")
        sentiment = analyze_sentiment(headline)
        news_type = classify_news_type(headline)
        results.append(
            {
                "date": item.get("date", datetime.now()),
                "headline": headline,
                "score": sentiment["score"],
                "type": news_type,
                "bullish_hits": sentiment["bullish_hits"],
                "bearish_hits": sentiment["bearish_hits"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.drop_duplicates(subset="headline")

    total_score = results_df["score"].sum()
    by_type = results_df.groupby("type")["score"].agg(["sum", "count"]).to_dict("index")

    if total_score >= 5:
        signal = "STRONG_BULLISH"
    elif total_score >= 2:
        signal = "BULLISH"
    elif total_score <= -5:
        signal = "STRONG_BEARISH"
    elif total_score <= -2:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "total_score": total_score,
        "signal": signal,
        "news_count": len(results_df),
        "news_df": results_df,
        "by_type": by_type,
        "note": "RSS fallback (öffentlich) genutzt",
    }


# ==============================================================================
# SENTIMENT ANALYSE
# ==============================================================================

def analyze_sentiment(text):
    """Analysiert den Sentiment eines News-Textes."""
    if not text:
        return {"score": 0, "bullish_hits": [], "bearish_hits": []}
    
    text_lower = text.lower()
    
    bullish_hits = []
    bearish_hits = []
    total_score = 0
    
    for keyword, weight in BULLISH_KEYWORDS.items():
        if keyword in text_lower:
            bullish_hits.append((keyword, weight))
            total_score += weight
    
    for keyword, weight in BEARISH_KEYWORDS.items():
        if keyword in text_lower:
            bearish_hits.append((keyword, weight))
            total_score += weight
    
    # Region-Boost
    region_boost = 1.0
    for region, keywords in COFFEE_REGIONS.items():
        for kw in keywords:
            if kw in text_lower:
                region_boost = 1.3
                break
    
    return {
        "score": total_score * region_boost,
        "bullish_hits": bullish_hits,
        "bearish_hits": bearish_hits,
        "region_boost": region_boost
    }


def classify_news_type(text):
    """Klassifiziert die News-Art."""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["weather", "rain", "drought", "frost", "temperature"]):
        return "WEATHER"
    elif any(w in text_lower for w in ["harvest", "crop", "production", "output", "yield"]):
        return "SUPPLY"
    elif any(w in text_lower for w in ["demand", "consumption", "import", "export"]):
        return "DEMAND"
    elif any(w in text_lower for w in ["price", "futures", "contract", "trading"]):
        return "MARKET"
    elif any(w in text_lower for w in ["stock", "inventory", "warehouse", "certified"]):
        return "STOCKS"
    else:
        return "OTHER"


# ==============================================================================
# HAUPT-FUNKTIONEN
# ==============================================================================

def get_coffee_news_sentiment(days_back=7, verbose=True):
    """
    Hauptfunktion: Holt Kaffee-News und berechnet Gesamt-Sentiment.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("☕ COFFEE NEWS MONITOR")
        print(f"   Zeitraum: Letzte {days_back} Tage")
        print("=" * 60)
    
    rd = init_lseg_session()
    
    if rd is None:
        if verbose:
            print("\n   ⚠️ LSEG nicht verfügbar - nutze RSS-Fallback")
        return get_public_news_sentiment(days_back=days_back, verbose=verbose)
    
    all_news = []
    queries = [
        "Coffee Brazil",
        "Coffee Vietnam", 
        "Arabica",
        "Robusta Coffee",
        "Coffee Harvest",
        "Coffee Weather",
        "Coffee Prices"
    ]
    
    if verbose:
        print("\n📡 Lade News von LSEG...")
    
    for query in queries:
        news_df = fetch_news_lseg(rd, query=query, count=30, days_back=days_back)
        if not news_df.empty:
            news_df['query'] = query
            all_news.append(news_df)
            if verbose:
                print(f"   {query}: {len(news_df)} Artikel")
    
    try:
        rd.close_session()
    except:
        pass
    
    if not all_news:
        if verbose:
            print("\n   ⚠️ Keine News gefunden")
        return {
            "total_score": 0,
            "signal": "NEUTRAL",
            "news_count": 0,
            "news_df": pd.DataFrame(),
            "by_type": {}
        }
    
    combined_df = pd.concat(all_news, ignore_index=True)
    if 'storyId' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset='storyId')
    elif 'headline' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset='headline')
    
    if verbose:
        print(f"\n📊 Analysiere {len(combined_df)} einzigartige Artikel...")
    
    results = []
    for idx, row in combined_df.iterrows():
        headline = row.get('headline', row.get('title', ''))
        
        sentiment = analyze_sentiment(headline)
        news_type = classify_news_type(headline)
        
        results.append({
            'date': row.get('versionCreated', row.get('date', datetime.now())),
            'headline': headline,
            'score': sentiment['score'],
            'type': news_type,
            'bullish_hits': sentiment['bullish_hits'],
            'bearish_hits': sentiment['bearish_hits']
        })
    
    results_df = pd.DataFrame(results)
    
    total_score = results_df['score'].sum()
    by_type = results_df.groupby('type')['score'].agg(['sum', 'count']).to_dict('index')
    
    if total_score >= 5:
        signal = "STRONG_BULLISH"
    elif total_score >= 2:
        signal = "BULLISH"
    elif total_score <= -5:
        signal = "STRONG_BEARISH"
    elif total_score <= -2:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"
    
    if verbose:
        print("\n" + "-" * 60)
        print("📰 TOP NEWS (nach Impact):")
        print("-" * 60)
        
        top_news = results_df.nlargest(5, 'score', keep='first')
        bottom_news = results_df.nsmallest(5, 'score', keep='first')
        
        if not top_news[top_news['score'] > 0].empty:
            print("\n🐂 BULLISH:")
            for _, row in top_news[top_news['score'] > 0].iterrows():
                print(f"   +{row['score']:.1f} | {row['headline'][:70]}...")
        
        if not bottom_news[bottom_news['score'] < 0].empty:
            print("\n🐻 BEARISH:")
            for _, row in bottom_news[bottom_news['score'] < 0].iterrows():
                print(f"   {row['score']:.1f} | {row['headline'][:70]}...")
        
        print("\n" + "=" * 60)
        print(f"📊 GESAMT-SENTIMENT: {total_score:+.1f}")
        print(f"📢 SIGNAL: {signal}")
        print("=" * 60)
        
        if "STRONG" in signal:
            print(f"\n🚨 WARNUNG: Starkes {signal.replace('STRONG_', '')}-Signal!")
            print("   Überprüfe Prognose-Modell vor Trading-Entscheidungen!")
    
    return {
        "total_score": total_score,
        "signal": signal,
        "news_count": len(results_df),
        "news_df": results_df,
        "by_type": by_type
    }


def get_fallback_sentiment():
    """Fallback wenn LSEG nicht verfügbar."""
    return {
        "total_score": 0,
        "signal": "NEUTRAL",
        "news_count": 0,
        "news_df": pd.DataFrame(),
        "by_type": {},
        "note": "LSEG nicht verfügbar - Fallback-Modus"
    }


def check_news_model_alignment(news_sentiment, model_direction):
    """Prüft ob News-Sentiment und Modell-Prognose übereinstimmen."""
    news_signal = news_sentiment.get('signal', 'NEUTRAL')
    
    news_is_bullish = 'BULLISH' in news_signal
    news_is_bearish = 'BEARISH' in news_signal
    
    model_is_bullish = model_direction == "BULLISH"
    model_is_bearish = model_direction == "BEARISH"
    
    if news_is_bullish and model_is_bearish:
        return {
            "aligned": False,
            "warning": "⚠️ KONFLIKT: News sind BULLISH aber Modell prognostiziert BEARISH!"
        }
    elif news_is_bearish and model_is_bullish:
        return {
            "aligned": False,
            "warning": "⚠️ KONFLIKT: News sind BEARISH aber Modell prognostiziert BULLISH!"
        }
    else:
        return {
            "aligned": True,
            "warning": None
        }


def get_news_adjustment_factor(news_sentiment):
    """Berechnet einen Adjustment-Faktor basierend auf News-Sentiment."""
    score = news_sentiment.get('total_score', 0)
    score = max(-10, min(10, score))
    factor = 1.0 + (score / 200)
    return factor


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    result = get_coffee_news_sentiment(days_back=7, verbose=True)
    
    if not result['news_df'].empty:
        output_path = os.path.join(os.path.dirname(__file__), 'news_sentiment.csv')
        result['news_df'].to_csv(output_path, index=False)
        print(f"\n✅ News-Daten gespeichert: {output_path}")