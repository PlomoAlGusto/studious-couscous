import yfinance as yf
import pandas as pd
import requests
import feedparser
import numpy as np
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from config import config

# --- IMPORTACIÓN SEGURA DE LIBRERÍA TÉCNICA ---
try:
    import pandas_ta_classic as ta
except ImportError:
    try:
        import pandas_ta as ta
    except ImportError:
        pass
# ----------------------------------------------

# --- CORRECCIÓN AUTOMÁTICA DE NLTK ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
# -------------------------------------

class DataManager:
    def __init__(self):
        # Yahoo Finance no necesita inicialización compleja
        pass

    @st.cache_data(ttl=15)
    def fetch_market_data(_self, symbol, timeframe, limit=100):
        """
        Descarga datos usando Yahoo Finance (Resistente a bloqueos de IP en la nube)
        """
        try:
            # 1. Adaptar el símbolo (Binance: BTC/USDT -> Yahoo: BTC-USD)
            yahoo_symbol = symbol.replace("/", "-").replace("USDT", "USD")
            if "USD" not in yahoo_symbol: 
                yahoo_symbol += "-USD"

            # 2. Mapeo de Timeframes para Yahoo
            period = "5d" 
            if timeframe == "15m": period = "5d"
            elif timeframe == "1h": period = "1mo"
            elif timeframe == "4h": period = "3mo"
            elif timeframe == "1d": period = "1y"

            # 3. Descargar
            df = yf.download(tickers=yahoo_symbol, period=period, interval=timeframe, progress=False)
            
            if df.empty:
                print(f"❌ Yahoo devolvió vacío para {yahoo_symbol}")
                return None

            # 4. Limpieza de formato (Yahoo a veces devuelve MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            df.reset_index(inplace=True)
            
            # 5. Renombrar columnas al estándar del Bot
            # Yahoo: Date/Datetime, Open, High, Low, Close, Volume
            # Bot: timestamp, open, high, low, close, volume
            rename_map = {
                'Date': 'timestamp', 
                'Datetime': 'timestamp',
                'Open': 'open', 
                'High': 'high', 
                'Low': 'low', 
                'Close': 'close', 
                'Volume': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)

            # 6. Asegurar tipos numéricos
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].astype(float)

            return df

        except Exception as e:
            print(f"❌ ERROR YAHOO: {e}")
            return None

    @st.cache_data(ttl=60)
    def get_funding_rate(_self, symbol):
        # Yahoo no tiene Funding Rate de futuros, devolvemos neutro
        return 0.01, 0

    @st.cache_data(ttl=300)
    def fetch_news(_self, symbol):
        """Descarga las últimas 10 noticias y analiza sentimiento"""
        news = []
        try:
            # Inicialización tardía para evitar errores de carga
            sia = SentimentIntensityAnalyzer()
            
            feed = feedparser.parse("https://cointelegraph.com/rss")
            
            # --- AHORA TRAEMOS 10 NOTICIAS ---
            for entry in feed.entries[:10]:
                s = sia.polarity_scores(entry.title)['compound']
                news.append({
                    "title": entry.title, 
                    "link": entry.link, 
                    "sentiment": s,
                    "source": "CoinTelegraph"
                })
        except Exception: 
            pass
        return news
    
    @st.cache_data(ttl=3600)
    def fetch_fear_greed(_self):
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
            d = r['data'][0]
            return int(d['value']), d['value_classification']
        except: return 50, "Neutral"
