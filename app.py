import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import requests
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Spot (Testnet)", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stAlert {background-color: #2b2b2b; color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONEXIÓN (CORREGIDA PARA SPOT)
# -----------------------------------------------------------------------------
def init_exchange():
    """Conexión específica para Binance SPOT Testnet"""
    try:
        if "BINANCE_API_KEY" in st.secrets:
            exchange = ccxt.binance({
                'apiKey': st.secrets["BINANCE_API_KEY"],
                'secret': st.secrets["BINANCE_SECRET"],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # <--- CAMBIO CLAVE: AHORA ES SPOT
                }
            })
            exchange.set_sandbox_mode(True) # MODO TESTNET
            # Hacemos una llamada de prueba para ver si conecta
            exchange.load_markets()
            return exchange
        else:
            return None
    except Exception as e:
        # ESTO ES EL CHIVATO DE ERRORES:
        st.error(f"❌ ERROR DE CONEXIÓN: {str(e)}")
        return None

def send_telegram_msg(msg):
    token = st.secrets.get("TELEGRAM_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.get(url, params={"chat_id": chat_id, "text": msg})
        except: pass

exchange = init_exchange()

# -----------------------------------------------------------------------------
# 3. INTERFAZ
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🦁 QUIMERA SPOT")
    st.caption("Modo: Testnet (Dinero Ficticio)")
    
    symbol = st.text_input("Símbolo", "BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=0)
    
    st.subheader("Filtros")
    use_ema = st.checkbox("Cruce EMAs", True)
    use_rsi = st.checkbox("RSI", True)

# -----------------------------------------------------------------------------
# 4. LÓGICA
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10) # Cache corto para pruebas
def get_data(symbol, timeframe, limit=200):
    if not exchange: return None
    try:
        # Descargamos velas
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"❌ ERROR DESCARGANDO DATOS: {str(e)}")
        return None

def calculate_indicators(df):
    if df is None: return None
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    return df

# -----------------------------------------------------------------------------
# 5. DASHBOARD
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 DASHBOARD", "🔮 IA"])

df = get_data(symbol, timeframe)

if df is not None:
    df = calculate_indicators(df)
    current = df.iloc[-1]
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Actual", f"${current['close']:.2f}")
        c2.metric("RSI", f"{current['RSI']:.1f}")
        
        # Gráfico
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='orange'), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], line=dict(color='blue'), name='EMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔔 Probar Telegram"):
            send_telegram_msg(f"Test Quimera Spot: Precio {symbol} es {current['close']}")
            st.success("Mensaje enviado")

    with tab2:
        if st.button("Ejecutar IA (Prophet)"):
            with st.spinner("Calculando..."):
                m = Prophet()
                d_p = df[['timestamp', 'close']].rename(columns={'timestamp':'ds', 'close':'y'})
                m.fit(d_p)
                fut = m.make_future_dataframe(periods=12, freq='H')
                fcst = m.predict(fut)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=d_p['ds'], y=d_p['y'], name='Real'))
                fig2.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='Predicción', line=dict(color='cyan')))
                st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("⚠️ Esperando conexión... Si ves un error arriba en rojo, cópialo y pégalo en el chat.")
