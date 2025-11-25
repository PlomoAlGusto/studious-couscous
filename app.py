import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import requests
from datetime import datetime
import time
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Ultimate AI", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 28px; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.2); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.2); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
</style>
""", unsafe_allow_html=True)

# Variables de Estado
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = []
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. CONEXIÓN Y UTILIDADES
# -----------------------------------------------------------------------------
def send_telegram_msg(msg):
    token = st.secrets.get("TELEGRAM_TOKEN", "")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        try:
            requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg})
        except: pass

def init_exchange():
    """Bypass Inteligente: Intenta Binance, si falla usa Kraken"""
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({
                'apiKey': st.secrets["BINANCE_API_KEY"], 
                'secret': st.secrets["BINANCE_SECRET"], 
                'options': {'defaultType': 'spot'}
            })
            ex.load_markets()
            return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Visual)"

exchange, source_info = init_exchange()

# -----------------------------------------------------------------------------
# 3. MOTOR DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=15)
def get_market_data(symbol, tf, limit=200):
    if not exchange: return None, 0
    
    ticker_fix = symbol if "Binance" in source_info else "BTC/USDT"
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0

    # OBI (Order Book Imbalance)
    try:
        orderbook = exchange.fetch_order_book(ticker_fix, limit=20)
        bids = sum([x[1] for x in orderbook['bids']]) 
        asks = sum([x[1] for x in orderbook['asks']]) 
        obi = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
    except: obi = 0
    
    return df, obi

# -----------------------------------------------------------------------------
# 4. CÁLCULO DE INDICADORES (CORREGIDO)
# -----------------------------------------------------------------------------
def calculate_complex_indicators(df):
    if df is None: return None
    
    # 1. EMAs Básicas
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    
    # 2. RSI
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # 3. Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1) # BBL, BBM, BBU
    
    # 4. ATR (Volatilidad)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # 5. Ichimoku Cloud
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1) # ISA_9, ISB_26
    
    # 6. VWAP MANUAL (Aquí estaba el error, lo hacemos manual para evitar fallos)
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vp = typical_price * df['volume']
        total_vp = vp.cumsum()
        total_vol = df['volume'].cumsum()
        df['VWAP'] = total_vp / total_vol
    except:
        df['VWAP'] = df['EMA_50'] # Si falla por división por cero, usamos EMA como respaldo
    
    # Limpiamos datos vacíos (NaN) del principio para que no den error
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# -----------------------------------------------------------------------------
# 5. MOTOR DE ML (PROPHET)
# -----------------------------------------------------------------------------
def get_ml_bias(df):
    """Consulta rápida a Prophet"""
    try:
        m = Prophet(daily_seasonality=True, yearly_seasonality=False)
        df_p = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'}).tail(100)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=6, freq='H')
        forecast = m.predict(future)
        
        last_real = df_p['y'].iloc[-1]
        last_pred = forecast['yhat'].iloc[-1]
        
        if last_pred > last_real * 1.001: return "BULLISH"
        elif last_pred < last_real * 0.999: return "BEARISH"
        else: return "NEUTRO"
    except: return "NEUTRO"

# -----------------------------------------------------------------------------
# 6. EL EMBUDO DE DECISIÓN
# -----------------------------------------------------------------------------
def check_funnel_signal(df, obi, filters, ml_bias):
    row = df.iloc[-1]
    score = 0
    max_score = 0
    reasons = []
    
    # FILTRO 1: TENDENCIA EMA
    if filters['use_ema']:
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; reasons.append("EMA Alcista")
        elif row['EMA_20'] < row['EMA_50']: score -= 1; reasons.append("EMA Bajista")
            
    # FILTRO 2: VWAP
    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1
        else: score -= 1
        
    # FILTRO 3: ICHIMOKU
    if filters['use_ichimoku']:
        max_score += 1
        if row['ISA_9'] > row['ISB_26'] and row['close'] > row['ISA_9']: score += 1; reasons.append("Sobre Nube")
        elif row['ISA_9'] < row['ISB_26'] and row['close'] < row['ISA_9']: score -= 1
            
    # FILTRO 4: OBI
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1; reasons.append("Presión Compras")
        elif obi < -0.05: score -= 1; reasons.append("Presión Ventas")

    # FILTRO 5: ML
    if filters['use_ml']:
        max_score += 1
        if ml_bias == "BULLISH": score += 1; reasons.append("IA Bullish")
        elif ml_bias == "BEARISH": score -= 1; reasons.append("IA Bearish")

    # DECISIÓN
    threshold = max_score * 0.5 
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    # Filtro RSI
    if filters['use_rsi']:
        if signal == "LONG" and row['RSI'] > 70: signal = "NEUTRO"
        if signal == "SHORT" and row['RSI'] < 30: signal = "NEUTRO"
        
    return signal, reasons, row['ATR']

# -----------------------------------------------------------------------------
# 7. INTERFAZ Y SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🦁 QUIMERA ULTIMATE")
    
    symbol = st.text_input("Símbolo", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)
    
    with st.expander("🛠️ FILTROS ACTIVOS", expanded=True):
        f_ema = st.checkbox("Tendencia (EMAs)", True)
        f_vwap = st.checkbox("Institucional (VWAP)", True)
        f_ichi = st.checkbox("Nube Ichimoku", False)
        f_rsi = st.checkbox("RSI", True)
        f_obi = st.checkbox("Order Book (OBI)", False)
        f_ml = st.checkbox("🤖 Confirmación IA (ML)", True)
        
    with st.expander("💰 GESTIÓN SALIDAS"):
        rr_ratio = st.number_input("Ratio Beneficio", 1.0, 5.0, 2.0)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
df, obi_val = get_market_data(symbol, tf)

if df is not None:
    df = calculate_complex_indicators(df)
    
    # ML Bias
    ml_bias = "NEUTRO"
    if f_ml: ml_bias = get_ml_bias(df)
    
    # Señal
    filters = {'use_ema': f_ema, 'use_vwap': f_vwap, 'use_ichimoku': f_ichi, 
               'use_rsi': f_rsi, 'use_obi': f_obi, 'use_ml': f_ml}
    signal, reasons, atr = check_funnel_signal(df, obi_val, filters, ml_bias)
    current_price = df['close'].iloc[-1]
    
    # Alertas
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        send_telegram_msg(f"🦁 ULTIMATE: {signal} en {symbol}\nPrecio: ${current_price}\n{', '.join(reasons)}")
        st.session_state.last_alert = signal
    elif signal == "NEUTRO":
        st.session_state.last_alert = "NEUTRO"

    # UI
    tab1, tab2, tab3 = st.tabs(["⚡ DASHBOARD", "📈 GRÁFICO", "💼 PAPER TRADES"])
    
    with tab1:
        css_class = "neutral"
        if signal == "LONG": css_class = "bullish"
        elif signal == "SHORT": css_class = "bearish"
        
        st.markdown(f"""
        <div class='big-signal {css_class}'>
            SEÑAL: {signal}
            <div style='font-size: 14px; margin-top:10px;'>
            {' + '.join(reasons) if reasons else 'Esperando confirmación...'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("OBI", f"{obi_val:.2%}")
        c3.metric("IA", ml_bias)
        c4.metric("ATR", f"${atr:.2f}")

        if signal != "NEUTRO":
            sl_dist = atr * 1.5
            sl = current_price - sl_dist if signal == "LONG" else current_price + sl_dist
            risk = abs(current_price - sl)
            tp = current_price + (risk * rr_ratio) if signal == "LONG" else current_price - (risk * rr_ratio)
            
            st.info(f"🎯 ENTRADA: ${current_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}")
            
            if st.button(f"EJECUTAR {signal} (PAPER)"):
                trade = {"time": datetime.now(), "symbol": symbol, "type": signal, "entry": current_price, "sl": sl, "tp": tp, "status": "OPEN"}
                st.session_state.positions.append(trade)
                st.success("Orden simulada enviada.")

    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        
        if f_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', width=1, dash='dot'), name='VWAP'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if st.session_state.positions:
            st.dataframe(pd.DataFrame(st.session_state.positions))
        else:
            st.info("Sin operaciones abiertas.")

else:
    st.warning("Cargando datos... (Si tarda mucho, cambia de timeframe)")

if auto_refresh:
    time.sleep(60)
    st.rerun()
