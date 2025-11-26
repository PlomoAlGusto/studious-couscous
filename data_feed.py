import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import feedparser
import numpy as np
import time
import logging
import streamlit as st
from datetime import datetime, timezone
from nltk.sentiment import SentimentIntensityAnalyzer
from config import config

sia = SentimentIntensityAnalyzer()

class DataManager:
    def __init__(self):
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        try:
            if config.BINANCE_API_KEY and config.BINANCE_SECRET:
                ex = ccxt.binance({'apiKey': config.BINANCE_API_KEY, 'secret': config.BINANCE_SECRET, 'options': {'defaultType': 'spot'}})
                ex.load_markets()
                return ex
        except: pass
        return ccxt.binance()

    @st.cache_data(ttl=15)
    def fetch_market_data(_self, symbol, timeframe, limit=100):
        try:
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            ohlcv = _self.exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
            if not ohlcv: return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except: return None

    @st.cache_data(ttl=60)
    def get_funding_rate(_self, symbol):
        base = symbol.split('/')[0].replace('USDT','')
        try:
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={base}USDT"
            r = requests.get(url, timeout=2).json()
            return float(r['lastFundingRate']) * 100, 0
        except: return 0.01, 0

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
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
            d = r['data'][0]
            return int(d['value']), d['value_classification']
        except: return 50, "Neutral"