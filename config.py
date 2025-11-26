import streamlit as st
import os

class Config:
    """
    Gestor centralizado de configuración.
    Lee de los secretos de Streamlit (seguro para GitHub público).
    """
    def __init__(self):
        # Configuración General
        self.PAGE_TITLE = "Quimera Pro v18.0 Refactored"
        self.INITIAL_CAPITAL = 10000.0
        self.DB_NAME = 'trades.db'
        
        # Carga segura usando st.secrets
        # NO poner claves reales aquí
        self.BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", None)
        self.BINANCE_SECRET = st.secrets.get("BINANCE_SECRET", None)
        self.COINGLASS_KEY = st.secrets.get("COINGLASS_API_KEY", None)
        self.TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", None)
        self.TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", None)
        
        # Manejo de CryptoPanic (a veces es un diccionario)
        cpanic = st.secrets.get("CRYPTOPANIC", None)
        if isinstance(cpanic, dict):
            self.CRYPTOPANIC_KEY = cpanic.get("API_KEY")
        else:
            self.CRYPTOPANIC_KEY = None

    def validate(self):
        warnings = []
        if not self.BINANCE_API_KEY:
            warnings.append("⚠️ API Key no detectada. Modo Observador.")
        return warnings

config = Config()
