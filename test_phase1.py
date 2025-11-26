import streamlit as st
from config import config
from database import TradeManager
from utils import setup_logging

# 1. Probar Utils
setup_logging()
st.title("✅ Test de Fase 1 Completado")

# 2. Probar Config
st.write(f"Conexión a Secretos: {'EXITOSA' if config.BINANCE_API_KEY else 'NO DETECTADA (Revisar Secrets)'}")

# 3. Probar DB
db = TradeManager()
if st.button("Probar Base de Datos"):
    try:
        # Intentar insertar un dato dummy
        dummy = {"symbol": "BTC/TEST", "type": "LONG", "entry": 100, "status": "OPEN"}
        db.add_trade(dummy)
        st.success("Base de Datos SQLite funcionando correctamente.")
        st.dataframe(db.load_trades())
    except Exception as e:
        st.error(f"Error en DB: {e}")
