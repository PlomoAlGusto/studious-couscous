import streamlit as st  # <--- ESTO ES LO QUE FALTABA AL PRINCIPIO
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import requests
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro (Testnet)", layout="wide", page_icon="🦁")

# Estilos CSS
st.markdown("""
<style>
    .metric-card {background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stAlert {background-color: #2b2b2b; color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONEXIÓN SEGURA Y TELEGRAM
# -----------------------------------------------------------------------------
def init_exchange():
    """Inicializa la conexión con Binance Testnet (SPOT)"""
    try:
        if "BINANCE_API_KEY" in st.secrets:
            exchange = ccxt.binance({
                'apiKey': st.secrets["BINANCE_API_KEY"],
                'secret': st.secrets["BINANCE_SECRET"],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # <--- AQUÍ ESTABA EL PROBLEMA (Antes ponía 'future')
                    'adjustForTimeDifference': True
                }
            })
            exchange.set_sandbox_mode(True) # Modo Testnet
            return exchange
        else:
            return None
    except Exception as e:
        st.error(f"Error conectando a Binance: {e}")
        return None
        
def send_telegram_msg(msg):
    """Envía mensajes a Telegram"""
    token = st.secrets.get("TELEGRAM_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": msg}
        try:
            requests.get(url, params=params)
        except:
            pass

exchange = init_exchange()

# -----------------------------------------------------------------------------
# 3. BARRA LATERAL (CONFIGURACIÓN)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🦁 QUIMERA CONTROL")
    st.info("Modo: Binance Testnet (Dinero Ficticio)")
    
    symbol = st.text_input("Símbolo", "BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=0)
    
    st.subheader("Filtros de Estrategia")
    use_ema_cross = st.checkbox("Cruce EMAs (20/50)", True)
    use_rsi = st.checkbox("Filtro RSI (30/70)", True)

# -----------------------------------------------------------------------------
# 4. DESCARGA DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_data(symbol, timeframe, limit=500):
    if not exchange:
        return None, None
    
    try:
        # Datos principales
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Datos 4H para tendencia
        ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=100)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_200'] = ta.ema(df_4h['close'], length=200)
        last_ema_4h = df_4h['EMA_200'].iloc[-1]
        
        return df, last_ema_4h
    except Exception as e:
        st.error(f"Error bajando datos: {e}")
        return None, None

# -----------------------------------------------------------------------------
# 5. CÁLCULOS
# -----------------------------------------------------------------------------
def calculate_indicators(df):
    if df is None: return None
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    return df

def check_signal(row, prev_row, use_ema, use_rsi):
    signal = "NEUTRO"
    if use_ema:
        if row['EMA_20'] > row['EMA_50'] and prev_row['EMA_20'] <= prev_row['EMA_50']:
            signal = "LONG"
        elif row['EMA_20'] < row['EMA_50'] and prev_row['EMA_20'] >= prev_row['EMA_50']:
            signal = "SHORT"
    
    if use_rsi:
        if signal == "LONG" and row['RSI'] > 70: signal = "NEUTRO"
        if signal == "SHORT" and row['RSI'] < 30: signal = "NEUTRO"
            
    return signal

# -----------------------------------------------------------------------------
# 6. INTERFAZ PRINCIPAL
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 PANEL DE CONTROL", "🔮 INTELIGENCIA ARTIFICIAL"])

df, trend_4h = get_data(symbol, timeframe)

if df is not None:
    df = calculate_indicators(df)
    current = df.iloc[-1]
    prev = df.iloc[-2]
    signal = check_signal(current, prev, use_ema_cross, use_rsi)
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Precio", f"${current['close']:.2f}")
        col2.metric("RSI", f"{current['RSI']:.1f}")
        
        if signal == "LONG": col3.success("SEÑAL DE COMPRA")
        elif signal == "SHORT": col3.error("SEÑAL DE VENTA")
        else: col3.info("NEUTRO")
        
        # Gráfico
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='orange'), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], line=dict(color='blue'), name='EMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔔 TEST TELEGRAM"):
            send_telegram_msg(f"🦁 Prueba desde Quimera Bot.\nPrecio {symbol}: {current['close']}")
            st.success("Enviado!")

    with tab2:
        st.subheader("Predicción con Machine Learning (Prophet)")
        if st.button("Ejecutar Predicción"):
            with st.spinner("Analizando mercado..."):
                m = Prophet()
                df_p = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
                m.fit(df_p)
                future = m.make_future_dataframe(periods=24, freq='H')
                forecast = m.predict(future)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Real'))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicción', line=dict(color='cyan')))
                st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("⚠️ No se pudieron cargar datos. Revisa las claves en 'Secrets' y que sean de TESTNET.")
