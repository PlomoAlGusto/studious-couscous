import ccxt
import pandas as pd
import feedparser
import logging
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from config import config

# Configuración NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

class DataManager:
    def __init__(self):
        # RESTAURANDO TU LÓGICA ORIGINAL DE CONEXIÓN
        self.exchange, self.source_name = self._init_exchange()

    def _init_exchange(self):
        """
        Lógica original del usuario:
        1. Binance API Key (si existe)
        2. Binance Público
        3. Kraken (Fallback)
        """
        # 1. Intentar Binance Privado
        try:
            if config.BINANCE_API_KEY and config.BINANCE_SECRET:
                ex = ccxt.binance({
                    'apiKey': config.BINANCE_API_KEY,
                    'secret': config.BINANCE_SECRET,
                    'options': {'defaultType': 'spot'}
                })
                ex.load_markets()
                print("✅ CONECTADO: Binance (Privado)")
                return ex, "Binance"
        except: pass

        # 2. Intentar Binance Público
        try:
            ex = ccxt.binance()
            ex.load_markets()
            print("✅ CONECTADO: Binance (Público)")
            return ex, "Binance"
        except: pass

        # 3. Intentar Kraken (Tu Fallback original)
        try:
            ex = ccxt.kraken()
            ex.load_markets()
            print("⚠️ CONECTADO: Kraken (Respaldo)")
            return ex, "Kraken"
        except:
            print("❌ ERROR: Todos los exchanges fallaron.")
            return None, "None"

    @st.cache_data(ttl=15)
    def fetch_market_data(_self, symbol, timeframe, limit=100):
        if not _self.exchange: return None
        
        try:
            # Ajuste de símbolo según el exchange conectado
            ticker = symbol
            
            # Binance prefiere formato con barra BTC/USDT
            if _self.source_name == "Binance" and "/" not in ticker:
                ticker = ticker.replace("USDT", "/USDT")
            
            # Kraken a veces requiere códigos específicos, pero probamos estándar primero
            
            # TU LLAMADA ORIGINAL
            ohlcv = _self.exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
            
            if not ohlcv: return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error obteniendo velas: {e}")
            return None

    @st.cache_data(ttl=60)
    def get_funding_rate(_self, symbol):
        # Simplificado para no provocar errores de API extras
        return 0.01, 0

    @st.cache_data(ttl=300)
    def fetch_news(_self, symbol):
        news = []
        try:
            feed = feedparser.parse("https://cointelegraph.com/rss")
            for entry in feed.entries[:5]:
                s = sia.polarity_scores(entry.title)['compound']
                news.append({"title": entry.title, "link": entry.link, "sentiment": s})
        except: pass
        return news
    
    @st.cache_data(ttl=3600)
    def fetch_fear_greed(_self):
        # Valor por defecto para no bloquear la app con más peticiones HTTP
        return 50, "Neutral"