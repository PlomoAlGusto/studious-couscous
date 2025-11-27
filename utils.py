import logging
import nltk
import streamlit as st
import requests # Necesario para Telegram
from config import config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    logging.getLogger('ccxt').setLevel(logging.WARNING)

@st.cache_resource
def init_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

# --- FUNCIÓN TELEGRAM ---
def send_telegram_alert(symbol, signal, price, sl, tp1, leverage):
    """Envía notificación al móvil"""
    token = config.TELEGRAM_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    
    if not token or not chat_id:
        return False # No configurado
        
    emoji = "💎" if "STRONG" in signal else "🚀"
    direction = "LONG 🟢" if "LONG" in signal else "SHORT 🔴"
    
    msg = (
        f"{emoji} **QUIMERA SIGNAL** {emoji}\n"
        f"--------------------------\n"
        f"🦁 **Par:** {symbol}\n"
        f"📡 **Orden:** {direction}\n"
        f"💵 **Entrada:** ${price:,.2f}\n"
        f"🛑 **Stop Loss:** ${sl:,.2f}\n"
        f"🎯 **Take Profit:** ${tp1:,.2f}\n"
        f"⚖️ **Apalancamiento:** {leverage}x\n"
        f"--------------------------\n"
        f"🤖 *Trade ejecutado desde Streamlit*"
    )
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        requests.get(url, params=params)
        return True
    except Exception as e:
        print(f"Error Telegram: {e}")
        return False
