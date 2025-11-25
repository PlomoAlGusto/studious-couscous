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
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            ex.load_markets()
            return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Visual)"

exchange, source_info = init_exchange()

# -----------------------------------------------------------------------------
# 3. MOTOR DE DATOS COMPLEJO (OHLCV + OBI)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=15)
def get_market_data(symbol, tf, limit=200):
    """Descarga Velas y Datos de Order Book"""
    if not exchange: return None, 0
    
    # 1. Velas OHLCV
    ticker_fix = symbol if "Binance" in source_info else "BTC/USDT"
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0

    # 2. OBI (Order Book Imbalance) - Presión de Compra/Venta real
    try:
        orderbook = exchange.fetch_order_book(ticker_fix, limit=20)
        bids = sum([x[1] for x in orderbook['bids']]) # Volumen de compra
        asks = sum([x[1] for x in orderbook['asks']]) # Volumen de venta
        obi = (bids - asks) / (bids + asks) # Ratio entre -1 (Venta) y 1 (Compra)
    except: obi = 0
    
    return df, obi

# -----------------------------------------------------------------------------
# 4. CÁLCULO DE INDICADORES (EL CEREBRO MATEMÁTICO)
# -----------------------------------------------------------------------------
def calculate_complex_indicators(df):
    if df is None: return None
    
    # 1. EMAs Básicas
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)
    
    # 2. RSI & StochRSI
    df['RSI'] = ta.rsi(df['close'], length=14)
    stoch = ta.stochrsi(df['close'], length=14)
    df = pd.concat([df, stoch], axis=1) # STOCHRSIk_14...
    
    # 3. Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1) # BBL, BBM, BBU
    
    # 4. ATR (Volatilidad) & ADX (Fuerza)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1) # ADX_14
    
    # 5. Ichimoku Cloud
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1) # ISA, ISB (Span A, Span B)
    
    # 6. VWAP (Aproximación para dataframes)
    df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    return df

# -----------------------------------------------------------------------------
# 5. MOTOR DE ML (PROPHET)
# -----------------------------------------------------------------------------
def get_ml_bias(df):
    """Consulta rápida a Prophet para ver tendencia futura"""
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
# 6. EL EMBUDO DE DECISIÓN (LÓGICA DEL PDF)
# -----------------------------------------------------------------------------
def check_funnel_signal(df, obi, filters, ml_bias):
    row = df.iloc[-1]
    
    score = 0
    max_score = 0
    reasons = []
    
    # --- FILTRO 1: TENDENCIA EMA (Base) ---
    if filters['use_ema']:
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: 
            score += 1; reasons.append("EMA Alcista")
        elif row['EMA_20'] < row['EMA_50']: 
            score -= 1; reasons.append("EMA Bajista")
            
    # --- FILTRO 2: VWAP (Institucional) ---
    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1
        else: score -= 1
        
    # --- FILTRO 3: ICHIMOKU (Nube) ---
    if filters['use_ichimoku']:
        max_score += 1
        # Span A > Span B es nube verde (alcista)
        if row['ISA_9'] > row['ISB_26'] and row['close'] > row['ISA_9']: 
            score += 1; reasons.append("Sobre Nube")
        elif row['ISA_9'] < row['ISB_26'] and row['close'] < row['ISA_9']: 
            score -= 1
            
    # --- FILTRO 4: OBI (Order Book) ---
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1; reasons.append("Presión Compras")
        elif obi < -0.05: score -= 1; reasons.append("Presión Ventas")

    # --- FILTRO 5: ML (IA) ---
    if filters['use_ml']:
        max_score += 1
        if ml_bias == "BULLISH": score += 1; reasons.append("IA Bullish")
        elif ml_bias == "BEARISH": score -= 1; reasons.append("IA Bearish")

    # --- DECISIÓN FINAL ---
    # Necesitamos al menos el 60% de consenso
    threshold = max_score * 0.5 
    
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    # Filtro RSI (Anti-Atrapados)
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
    
    with st.expander("🛠️ GRUPO A: FILTROS ACTIVOS", expanded=True):
        f_ema = st.checkbox("Tendencia (EMAs)", True)
        f_vwap = st.checkbox("Institucional (VWAP)", True)
        f_ichi = st.checkbox("Nube Ichimoku", False)
        f_rsi = st.checkbox("RSI (Sobrecompra/Venta)", True)
        f_obi = st.checkbox("Order Book (OBI)", False)
        f_ml = st.checkbox("🤖 Confirmación IA (ML)", True)
        
    with st.expander("💰 GRUPO B: GESTIÓN SALIDAS"):
        risk_per_trade = st.slider("Riesgo %", 1.0, 5.0, 1.0)
        rr_ratio = st.number_input("Ratio Beneficio", 1.0, 5.0, 2.0)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
# Carga de Datos
df, obi_val = get_market_data(symbol, tf)

if df is not None:
    # Cálculos
    df = calculate_complex_indicators(df)
    
    # IA Bias (Solo si está activo para no gastar recursos)
    ml_bias = "NEUTRO"
    if f_ml: 
        ml_bias = get_ml_bias(df)
    
    # Embudo de Señal
    filters = {'use_ema': f_ema, 'use_vwap': f_vwap, 'use_ichimoku': f_ichi, 
               'use_rsi': f_rsi, 'use_obi': f_obi, 'use_ml': f_ml}
    
    signal, reasons, atr = check_funnel_signal(df, obi_val, filters, ml_bias)
    current_price = df['close'].iloc[-1]
    
    # Gestión de Alertas
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        send_telegram_msg(f"🦁 ULTIMATE SIGNAL: {signal} en {symbol}\nPrecio: ${current_price}\nMotivos: {', '.join(reasons)}")
        st.session_state.last_alert = signal
        st.toast(f"Alerta enviada: {signal}")
    elif signal == "NEUTRO":
        st.session_state.last_alert = "NEUTRO"

    # --- UI DASHBOARD ---
    tab1, tab2, tab3 = st.tabs(["⚡ LIVE COMMAND", "📈 SUPER CHART", "💼 PAPER TRADES"])
    
    with tab1:
        # TARJETA PRINCIPAL
        css_class = "neutral"
        if signal == "LONG": css_class = "bullish"
        elif signal == "SHORT": css_class = "bearish"
        
        st.markdown(f"""
        <div class='big-signal {css_class}'>
            SEÑAL ACTUAL: {signal}
            <div style='font-size: 14px; font-weight: normal; margin-top:10px;'>
            {' + '.join(reasons) if reasons else 'Sin confluencia suficiente'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # MÉTRICAS CLAVE
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("OBI (Presión)", f"{obi_val:.2%}", delta_color="off")
        c3.metric("Predicción IA", ml_bias)
        c4.metric("Volatilidad (ATR)", f"${atr:.2f}")

        # SL / TP CALCULATOR
        if signal != "NEUTRO":
            sl_dist = atr * 1.5
            sl = current_price - sl_dist if signal == "LONG" else current_price + sl_dist
            risk = abs(current_price - sl)
            tp = current_price + (risk * rr_ratio) if signal == "LONG" else current_price - (risk * rr_ratio)
            
            st.info(f"🎯 SETUP SUGERIDO: Entrada ${current_price:.2f} | SL ${sl:.2f} | TP ${tp:.2f}")
            
            if st.button(f"EJECUTAR {signal} (PAPER)"):
                trade = {"time": datetime.now(), "symbol": symbol, "type": signal, "entry": current_price, "sl": sl, "tp": tp, "status": "OPEN"}
                st.session_state.positions.append(trade)
                st.success("Orden simulada enviada.")

    with tab2:
        # GRÁFICO COMPLEJO
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        
        # Velas
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        
        # Indicadores Visuales
        if f_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', width=1, dash='dot'), name='VWAP'), row=1, col=1)
        if f_bb: 
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BBU_20_2.0'], line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BBL_20_2.0'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='BB'), row=1, col=1)
        if f_ichi:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ISA_9'], line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ISB_26'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,255,0,0.1)', name='Cloud'), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Cartera Simulada")
        if st.session_state.positions:
            st.dataframe(pd.DataFrame(st.session_state.positions))
        else:
            st.info("Sin operaciones abiertas.")

else:
    st.warning("Cargando datos... (Si tarda, Kraken puede estar lento)")

if auto_refresh:
    time.sleep(60)
    st.rerun()
