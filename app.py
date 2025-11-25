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
st.set_page_config(page_title="Quimera Pro v5.2 Tactical", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.1); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
    .prob-box {text-align: center; font-size: 18px; margin-bottom: 5px;}
    
    /* ESTILOS NUEVOS PARA LA TARJETA TÁCTICA */
    .trade-setup {
        background-color: #1E1E1E; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #555;
        margin-top: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    .label-mini { font-size: 12px; color: #aaa; }
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
    st.title("🦁 QUIMERA TACTICAL")
    st.caption("v5.2 Multi-Target")
    
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
    
    # Puntuación de Confluencia
    score = 0
    max_score = 0
    
    if use_ema:
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; reasons.append("EMAs")
        else: score -= 1
        
    if use_vwap:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; reasons.append("VWAP")
        else: score -= 1
        
    if use_ichi:
        max_score += 1
        cloud_top = max(row['ISA_9'], row['ISB_26'])
        if row['close'] > cloud_top: score += 1
        elif row['close'] < min(row['ISA_9'], row['ISB_26']): score -= 1
        
    if use_obi:
        max_score += 1
        if obi > 0.05: score += 1; reasons.append("OrderBook")
        elif obi < -0.05: score -= 1

    if use_rsi:
        max_score += 1
        if row['RSI'] > 50 and row['RSI'] < 70: score += 1
        elif row['RSI'] < 50 and row['RSI'] > 30: score -= 1

    # CÁLCULO DE SEÑAL
    threshold = max_score * 0.4 
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    if use_regime and row['ADX_14'] < 20: 
        signal = "NEUTRO"
        reasons = ["Mercado Lateral (ADX Bajo)"]
    
    # Probabilidad
    probability = 50.0
    if max_score > 0:
        raw_prob = (abs(score) / max_score)
        probability = 50 + (raw_prob * 45) # Max 95%
        
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
            if current_price >= pos['tp3']: close_reason = "TP3 (Moonbag) 🚀"; pnl = (pos['tp3']-pos['entry'])*pos['size']
            elif current_price <= pos['sl']: close_reason = "SL 🛑"; pnl = (pos['sl']-pos['entry'])*pos['size']
        elif pos['type'] == "SHORT":
            if current_price <= pos['tp3']: close_reason = "TP3 (Moonbag) 🚀"; pnl = (pos['entry']-pos['tp3'])*pos['size']
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
    
    # CÁLCULO DE NIVELES TÁCTICOS (TP1, TP2, TP3)
    setup = None
    if signal != "NEUTRO":
        sl_dist = atr * 1.5
        risk = sl_dist
        
        if signal == "LONG":
            sl = current_price - sl_dist
            tp1 = current_price + risk      # Ratio 1:1
            tp2 = current_price + (risk*2)  # Ratio 1:2
            tp3 = current_price + (risk*3.5)# Ratio 1:3.5
        else:
            sl = current_price + sl_dist
            tp1 = current_price - risk
            tp2 = current_price - (risk*2)
            tp3 = current_price - (risk*3.5)
            
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3}

    # GESTIÓN DE ALERTAS (AHORA CON TODOS LOS DATOS)
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        msg = f"""🦁 *QUIMERA SIGNAL: {signal}*
Activo: {symbol}
Probabilidad: {prob:.1f}%

🔵 *ENTRADA:* ${current_price:.2f}

🎯 *TP1:* ${setup['tp1']:.2f}
🎯 *TP2:* ${setup['tp2']:.2f}
🚀 *TP3:* ${setup['tp3']:.2f}

🛑 *SL:* ${setup['sl']:.2f}
"""
        send_telegram_msg(msg)
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_positions(current_price)
    
    tab1, tab2, tab3 = st.tabs(["📊 COMANDO", "🧪 PAPER LAB", "🔮 FORECAST"])
    
    with tab1:
        # 1. SEÑAL Y PROBABILIDAD
        css = "neutral"
        if signal == "LONG": css = "bullish"
        elif signal == "SHORT": css = "bearish"
        
        c_sig, c_prob = st.columns([1, 1])
        with c_sig:
            st.markdown(f"<div class='big-signal {css}'>{signal}</div>", unsafe_allow_html=True)
        with c_prob:
            st.markdown(f"<div class='prob-box'>Prob: <b>{prob:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(int(prob))

        # 2. CONTEXTO DE MERCADO
        st.subheader("🔍 Contexto")
        d1, d2, d3, d4 = st.columns(4)
        
        # Volumen
        vol_state = "Bajo"
        if last_row['volume'] > last_row['SMA_VOL'] * 1.5: vol_state = "🔥 ALTO"
        d1.metric("Volumen", vol_state)
        
        # Régimen
        regime = "Lateral"
        if last_row['ADX_14'] > 25: regime = "Tendencia 🚀"
        d2.metric("Régimen", regime, f"ADX: {last_row['ADX_14']:.1f}")
        
        # Valor
        dist_vwap = ((current_price - last_row['VWAP']) / last_row['VWAP']) * 100
        d3.metric("Valor", f"{dist_vwap:.2f}%", "vs VWAP")
        
        # OBI
        obi_txt = "Neutro"
        if obi > 0.05: obi_txt = "Toros 🐂"
        elif obi < -0.05: obi_txt = "Osos 🐻"
        d4.metric("Libro", obi_txt, f"{obi:.1%}")

        st.divider()

        # 3. TARJETA DE ESTRATEGIA TÁCTICA (NUEVO !!!)
        if signal != "NEUTRO" and setup:
            st.subheader("🎯 Setup Táctico Sugerido")
            st.markdown(f"""
            <div class="trade-setup">
                <div style="display: flex; justify-content: space-around;">
                    <div><span class="label-mini">ENTRADA</span><br><span class="entry-blue">${setup['entry']:.2f}</span></div>
                    <div><span class="label-mini">STOP LOSS</span><br><span class="sl-red">${setup['sl']:.2f}</span></div>
                    <div><span class="label-mini">TP 1 (1:1)</span><br><span class="tp-green">${setup['tp1']:.2f}</span></div>
                    <div><span class="label-mini">TP 2 (1:2)</span><br><span class="tp-green">${setup['tp2']:.2f}</span></div>
                    <div><span class="label-mini">TP 3 (Moon)</span><br><span class="tp-green">${setup['tp3']:.2f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🚀 EJECUTAR {signal} (TP3 Objetivo)"):
                trade = {"time": datetime.now(), "symbol": symbol, "type": signal, "entry": current_price, "sl": setup['sl'], "tp": setup['tp3'], "tp3": setup['tp3'], "size": 100, "atr_entry": atr, "status": "OPEN"}
                st.session_state.positions.append(trade)
                st.success("Orden enviada.")
        else:
            st.info("Esperando señal clara para generar Setup Táctico...")

        # 4. GRÁFICO
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        # PINTAR NIVELES SI HAY SEÑAL
        if signal != "NEUTRO":
            fig.add_hline(y=setup['tp1'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['tp2'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.metric("Balance Virtual", f"${st.session_state.balance:,.2f}")
        if st.session_state.positions:
            st.dataframe(pd.DataFrame(st.session_state.positions))
        
    with tab3:
        if st.button("🔮 Prophet AI"):
            m = Prophet(); d_p = df[['timestamp', 'close']].rename(columns={'timestamp':'ds', 'close':'y'})
            m.fit(d_p); fut = m.make_future_dataframe(periods=12, freq='H'); fcst = m.predict(fut)
            fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=d_p['ds'], y=d_p['y'])); fig2.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], line=dict(color='cyan')))
            st.plotly_chart(fig2)

else: st.warning("Cargando...")

if auto_refresh: time.sleep(60); st.rerun()
