# Conexión Segura usando la Caja Fuerte de Streamlit
try:
    exchange = ccxt.binance({
        'apiKey': st.secrets["BINANCE_API_KEY"],
        'secret': st.secrets["BINANCE_SECRET"],
        'enableRateLimit': True,
    })
except:
    exchange = None

# Telegram Seguro
bot_token = st.secrets.get("TELEGRAM_TOKEN", "")
chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
