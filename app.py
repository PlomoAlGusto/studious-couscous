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
# 1. CONFIGURACIÓN DEL SISTEMA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v5.1 Intelligence", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.1); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
    .prob-box {text-align: center; font-size: 18px; margin-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# VARIABLES DE ESTADO
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = []
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v5.1")
    st.caption("Intelligence Core")
    
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)

    with st.expander("🛡️ FILTROS (CONFLUENCIA)", expanded=True):
        use_ema = st.checkbox("Tendencia (EMAs)", True)
        use_vwap = st.checkbox("VWAP Institucional", True)
        use_ichi = st.checkbox("Nube Ichimoku", False)
        use_regime = st.checkbox("Anti-Rango (ADX)", True)
        use_rsi = st.checkbox("RSI & Stoch", True)
        use_obi = st.checkbox("Order Book", True)

    with st.expander("💰 GESTIÓN DE RIESGO"):
        use_trailing = st.checkbox("Trailing Stop", True)
        risk_per_trade = st.slider("Riesgo %", 0.5, 5.0, 1.0)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 3. DATOS & INDICADORES
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            ex.load_markets(); return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Visual)"

exchange, source_name = init_exchange()

@st.cache_data(ttl=15)
def get_data_engine(symbol, tf):
    if not exchange: return None, 0
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0

    obi = 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids, asks = sum([x[1] for x in book['bids']]), sum([x[1] for x in book['asks']])
        if (bids + asks) > 0: obi = (bids - asks) / (bids + asks)
    except: pass
    return df, obi

def calculate_indicators(df):
    if df is None: return None
    # Tendencia
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    
    # Volumen Relativo
    df['SMA_VOL'] = ta.sma(df['volume'], length=20)
    
    # VWAP
    try:
        vp = ((df['high'] + df['low'] + df['close'])/3) * df['volume']
        df['VWAP'] = vp.cumsum() / df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    
    # Ichimoku & BB
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1)
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    
    # Osciladores
    df['RSI'] = ta.rsi(df['close'], length=14)
    stoch = ta.stochrsi(df['close'], length=14)
    df = pd.concat([df, stoch], axis=1)
    
    # Régimen
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# -----------------------------------------------------------------------------
# 4. LÓGICA Y CÁLCULO DE PROBABILIDAD
# -----------------------------------------------------------------------------
def run_strategy_funnel(df, obi):
    row = df.iloc[-1]
    reasons = []
    
    # Puntuación de Confluencia (Para la probabilidad)
    score = 0
    max_score = 0
    
    # 1. EMAs
    if use_ema:
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; reasons.append("EMAs")
        else: score -= 1
        
    # 2. VWAP
    if use_vwap:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; reasons.append("VWAP")
        else: score -= 1
        
    # 3. ICHIMOKU
    if use_ichi:
        max_score += 1
        cloud_top = max(row['ISA_9'], row['ISB_26'])
        if row['close'] > cloud_top: score += 1
        elif row['close'] < min(row['ISA_9'], row['ISB_26']): score -= 1
        
    # 4. OBI
    if use_obi:
        max_score += 1
        if obi > 0.05: score += 1; reasons.append("OrderBook")
        elif obi < -0.05: score -= 1

    # 5. OSCILADORES
    if use_rsi:
        max_score += 1
        # Si RSI sube y no está sobrecomprado, suma punto
        if row['RSI'] > 50 and row['RSI'] < 70: score += 1
        elif row['RSI'] < 50 and row['RSI'] > 30: score -= 1

    # CÁLCULO DE SEÑAL
    threshold = max_score * 0.4 # Umbral de activación
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    # Filtros de Veto (Regimen)
    if use_regime and row['ADX_14'] < 20: 
        signal = "NEUTRO"
        reasons = ["Mercado Lateral (ADX Bajo)"]
    
    # CÁLCULO DE PROBABILIDAD (%)
    # Normalizamos el score. Si score es igual a max_score -> 99%
    if max_score > 0:
        raw_prob = (abs(score) / max_score)
        probability = 50 + (raw_prob * 45) # Base 50%, Max 95%
    else:
        probability = 50.0
        
    return signal, reasons, row['ATR'], probability

# -----------------------------------------------------------------------------
# 5. GESTIÓN DE POSICIONES
# -----------------------------------------------------------------------------
def manage_positions(current_price):
    for pos in st.session_state.positions[:]:
        close_reason = ""
        pnl = 0
        
        # Trailing Stop Simulado
        if use_trailing:
            if pos['type'] == "LONG":
                new_sl = current_price - (pos['atr_entry'] * 1.5)
                if new_sl > pos['sl']: pos['sl'] = new_sl
            elif pos['type'] == "SHORT":
                new_sl = current_price + (pos['atr_entry'] * 1.5)
                if new_sl < pos['sl']: pos['sl'] = new_sl

        # TP / SL Check
        if pos['type'] == "LONG":
            if current_price >= pos['tp']: close_reason = "TP 🎯"; pnl = (pos['tp']-pos['entry'])*pos['size']
            elif current_price <= pos['sl']: close_reason = "SL 🛑"; pnl = (pos['sl']-pos['entry'])*pos['size']
        elif pos['type'] == "SHORT":
            if current_price <= pos['tp']: close_reason = "TP 🎯"; pnl = (pos['entry']-pos['tp'])*pos['size']
            elif current_price >= pos['sl']: close_reason = "SL 🛑"; pnl = (pos['entry']-pos['sl'])*pos['size']
                
        if close_reason:
            st.session_state.balance += pnl
            pos['status'] = "CLOSED"; pos['exit'] = current_price; pos['pnl'] = pnl; pos['reason'] = close_reason
            st.session_state.trade_history.insert(0, pos)
            st.session_state.positions.remove(pos)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

# -----------------------------------------------------------------------------
# 6. DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
df, obi = get_data_engine(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    signal, reasons, atr, prob = run_strategy_funnel(df, obi)
    current_price = df['close'].iloc[-1]
    last_row = df.iloc[-1]
    
    # Alertas
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        send_telegram_msg(f"🦁 SIGNAL v5.1: {signal}\nActivo: {symbol}\nProb: {prob:.1f}%")
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_positions(current_price)
    
    tab1, tab2, tab3 = st.tabs(["📊 COMANDO & PROBABILIDAD", "🧪 PAPER LAB", "🔮 AI FORECAST"])
    
    with tab1:
        # 1. SEÑAL GRANDE
        css = "neutral"
        if signal == "LONG": css = "bullish"
        elif signal == "SHORT": css = "bearish"
        
        st.markdown(f"""
        <div class='big-signal {css}'>
            {signal}
            <div style='font-size:14px; margin-top:5px; color:#ddd;'>
                Estrategia: {' + '.join(reasons) if reasons else 'Sin confluencia clara'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. BARRA DE PROBABILIDAD (NUEVO)
        col_prob1, col_prob2 = st.columns([1, 4])
        with col_prob1:
            st.markdown(f"<div class='prob-box'>Probabilidad<br><b>{prob:.1f}%</b></div>", unsafe_allow_html=True)
        with col_prob2:
            st.write("") # Espaciador
            bar_color = "green" if prob > 70 else "yellow" if prob > 50 else "gray"
            st.progress(int(prob))
            st.caption("Confluencia Técnica (Basado en filtros activos)")

        st.divider()

        # 3. COCKPIT DE DATOS DE INTERÉS (NUEVO)
        st.subheader("🔍 Contexto de Mercado")
        d1, d2, d3, d4 = st.columns(4)
        
        # A. Salud del Volumen
        vol_state = "Normal"
        if last_row['volume'] > last_row['SMA_VOL'] * 1.5: vol_state = "🔥 ALTO (Interés)"
        elif last_row['volume'] < last_row['SMA_VOL'] * 0.7: vol_state = "💤 BAJO (Sin fuerza)"
        d1.metric("Volumen", vol_state, f"{(last_row['volume']/last_row['SMA_VOL']-1)*100:.1f}% vs Avg")
        
        # B. Régimen de Mercado (ADX)
        regime = "Lateral / Rango"
        if last_row['ADX_14'] > 25: regime = "Tendencia Fuerte 🚀"
        d2.metric("Régimen (ADX)", regime, f"{last_row['ADX_14']:.1f}")
        
        # C. Distancia a VWAP (Valor)
        dist_vwap = ((current_price - last_row['VWAP']) / last_row['VWAP']) * 100
        val_state = "Caro (Sobre VWAP)" if dist_vwap > 0 else "Barato (Bajo VWAP)"
        d3.metric("Valor Justo", val_state, f"{dist_vwap:.2f}% dist")
        
        # D. Presión Order Book
        obi_txt = "Neutro"
        if obi > 0.05: obi_txt = "Toros dominan 🐂"
        elif obi < -0.05: obi_txt = "Osos dominan 🐻"
        d4.metric("Libro Órdenes", obi_txt, f"{obi:.2%}")

        st.divider()
        
        # 4. GRÁFICO Y BOTONES
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        if signal != "NEUTRO":
            sl = current_price - (atr * 1.5) if signal == "LONG" else current_price + (atr * 1.5)
            tp = current_price + (abs(current_price-sl)*2) if signal == "LONG" else current_price - (abs(current_price-sl)*2)
            if st.button(f"🚀 EJECUTAR {signal} (CONFIRMADO)"):
                trade = {"time": datetime.now(), "symbol": symbol, "type": signal, "entry": current_price, "sl": sl, "tp": tp, "size": 100, "atr_entry": atr, "status": "OPEN"}
                st.session_state.positions.append(trade)
                st.success("Orden enviada.")

    with tab2:
        st.metric("Balance Virtual", f"${st.session_state.balance:,.2f}")
        if st.session_state.positions:
            st.dataframe(pd.DataFrame(st.session_state.positions))
        else: st.info("Mesa de operaciones limpia.")

    with tab3:
        if st.button("🔮 Consultar Prophet AI"):
            with st.spinner("Procesando..."):
                m = Prophet()
                d_p = df[['timestamp', 'close']].rename(columns={'timestamp':'ds', 'close':'y'})
                m.fit(d_p)
                fut = m.make_future_dataframe(periods=12, freq='H')
                fcst = m.predict(fut)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=d_p['ds'], y=d_p['y'], name='Real'))
                fig2.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='AI Trend', line=dict(color='cyan')))
                st.plotly_chart(fig2, use_container_width=True)

else: st.warning("Cargando datos...")

if auto_refresh: time.sleep(60); st.rerun()
