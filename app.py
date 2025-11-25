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
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v6.1", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.1); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
    
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    .label-mini { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    .direction-header-long { color: #00FF00; font-size: 22px; font-weight: 900; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .direction-header-short { color: #FF4444; font-size: 22px; font-weight: 900; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ARCHIVOS (PERSISTENCIA CSV)
CSV_FILE = 'paper_trades.csv'
if not os.path.exists(CSV_FILE):
    df_empty = pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
    df_empty.to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. CAPA DE CONFIGURACIÓN (SIDEBAR MODULAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v6.1")
    st.caption("Institutional Architect")
    
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)

    with st.expander("🛡️ GRUPO A: FILTROS ESTRUCTURA", expanded=True):
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP (Institucional)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)

    with st.expander("🌊 GRUPO B: MOMENTO Y VOLUMEN"):
        use_rsi = st.checkbox("RSI & Stoch", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        
    with st.expander("💰 GRUPO C: SALIDAS AVANZADAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
        risk_per_trade = st.slider("Riesgo %", 0.5, 5.0, 1.0)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 3. CAPA DE DATOS (DATA ENGINE MTF)
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
def get_mtf_data(symbol, tf_lower):
    """Descarga datos del timeframe actual Y del timeframe superior (4h)"""
    if not exchange: return None, 0, None
    
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    
    # 1. Datos Principales (ej: 15m)
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None

    # 2. Datos Macro (4h) para Filtro MTF
    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
        last_4h = df_4h.iloc[-1]
        if last_4h['close'] > last_4h['EMA_50']: trend_4h = "BULLISH"
        else: trend_4h = "BEARISH"
    except: pass

    # 3. OBI Live
    obi = 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids, asks = sum([x[1] for x in book['bids']]), sum([x[1] for x in book['asks']])
        if (bids + asks) > 0: obi = (bids - asks) / (bids + asks)
    except: pass
    
    return df, obi, trend_4h

# -----------------------------------------------------------------------------
# 4. CAPA LÓGICA (INDICADORES Y SEÑAL)
# -----------------------------------------------------------------------------
def calculate_indicators(df):
    if df is None: return None
    # Tendencia
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    
    # VWAP Manual
    try:
        vp = ((df['high'] + df['low'] + df['close'])/3) * df['volume']
        df['VWAP'] = vp.cumsum() / df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    
    # Ichimoku
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1)
    
    # Osciladores
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1)
    
    return df.fillna(method='bfill').fillna(method='ffill')

def get_ai_advice(row, trend_4h, obi):
    advice = []
    if row['ADX_14'] < 20: advice.append("⚠️ Lateral: Riesgo de falsas señales.")
    if trend_4h == "BULLISH": advice.append("✅ Macro 4H Alcista: Prioriza Longs.")
    elif trend_4h == "BEARISH": advice.append("🔻 Macro 4H Bajista: Prioriza Shorts.")
    if abs(obi) > 0.1: advice.append("🔥 Volumen fuerte en libro.")
    return " | ".join(advice)

def run_strategy(df, obi, trend_4h, filters): # AÑADIDO FILTERS AQUÍ
    row = df.iloc[-1]
    score = 0
    max_score = 0
    reasons = []

    # 1. MTF (El Rey de los Filtros)
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2
        elif trend_4h == "BEARISH": score -= 2
        else: max_score -= 2 

    # 2. EMAs
    if filters['use_ema']: # Ahora usa filters['use_ema']
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; reasons.append("Cruce EMA")
        else: score -= 1

    # 3. VWAP
    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; reasons.append("VWAP Support")
        else: score -= 1
        
    # 4. OBI
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1; reasons.append("OrderBook Bull")
        elif obi < -0.05: score -= 1
    
    # Decisión
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    # Veto: RSI y Régimen
    if filters['use_rsi'] and (row['RSI'] > 70 and signal == "LONG"): signal = "NEUTRO"
    if filters['use_rsi'] and (row['RSI'] < 30 and signal == "SHORT"): signal = "NEUTRO"
    if filters['use_regime'] and row['ADX_14'] < 20: signal = "NEUTRO"; reasons = ["Rango (ADX < 20)"]
    
    # Conflictos MTF
    if filters['use_mtf'] and signal == "LONG" and trend_4h == "BEARISH": signal = "NEUTRO"; reasons = ["Contra Tendencia 4H"]
    if filters['use_mtf'] and signal == "SHORT" and trend_4h == "BULLISH": signal = "NEUTRO"; reasons = ["Contra Tendencia 4H"]

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    return signal, reasons, row['ATR'], prob

# -----------------------------------------------------------------------------
# 5. MOTOR DE PAPER TRADING (PERSISTENTE CSV)
# -----------------------------------------------------------------------------
def load_trades():
    if os.path.exists(CSV_FILE): return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])

def save_trades(df):
    df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr):
    df = load_trades()
    new_trade = {
        "id": int(time.time()),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "type": type,
        "entry": entry,
        "size": size,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "status": "OPEN",
        "pnl": 0.0,
        "reason": "Entry",
        "candles_held": 0,
        "atr_entry": atr
    }
    # Concatenar usando pd.concat en lugar de append
    new_row = pd.DataFrame([new_trade])
    df = pd.concat([new_row, df], ignore_index=True)
    save_trades(df)
    return new_trade

def manage_open_positions(current_price):
    df = load_trades()
    if df.empty: return
    
    open_trades_idx = df.index[df['status'] == "OPEN"].tolist()
    if not open_trades_idx: return

    updated = False
    for idx in open_trades_idx:
        row = df.loc[idx]
        close_reason = ""
        pnl = 0
        
        # 1. Time Stop
        if use_time_stop:
            df.at[idx, 'candles_held'] += 1
            if df.at[idx, 'candles_held'] > 12:
                current_pnl = (current_price - row['entry']) if row['type'] == "LONG" else (row['entry'] - current_price)
                if current_pnl <= 0: close_reason = "Time Stop ⏳"
        
        # 2. Logic Updates
        if not close_reason:
            if row['type'] == "LONG":
                # Trailing
                if use_trailing:
                    new_sl = current_price - (row['atr_entry'] * 1.5)
                    if new_sl > row['sl']: df.at[idx, 'sl'] = new_sl
                # Breakeven
                if use_breakeven and current_price > (row['entry'] * 1.015):
                     if row['sl'] < row['entry']: df.at[idx, 'sl'] = row['entry']
                
                # Exits
                if current_price >= row['tp3']: close_reason = "TP3 Moon 🚀"; pnl = (row['tp3']-row['entry'])*row['size']
                elif current_price <= row['sl']: close_reason = "SL 🛑"; pnl = (row['sl']-row['entry'])*row['size']
            
            else: # SHORT
                if use_trailing:
                    new_sl = current_price + (row['atr_entry'] * 1.5)
                    if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
                if use_breakeven and current_price < (row['entry'] * 0.985):
                     if row['sl'] > row['entry']: df.at[idx, 'sl'] = row['entry']

                if current_price <= row['tp3']: close_reason = "TP3 Moon 🚀"; pnl = (row['entry']-row['tp3'])*row['size']
                elif current_price >= row['sl']: close_reason = "SL 🛑"; pnl = (row['entry']-row['sl'])*row['size']

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"
            df.at[idx, 'pnl'] = pnl
            df.at[idx, 'reason'] = close_reason
            st.toast(f"Posición Cerrada: {close_reason}")
            send_telegram_msg(f"💰 CIERRE {symbol}: {close_reason}\nPnL: ${pnl:.2f}")
            updated = True

    if updated or use_time_stop: # Guardar si hubo cambios o si actualizamos velas
        save_trades(df)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

# -----------------------------------------------------------------------------
# 6. DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    
    # Empaquetamos filtros para evitar errores de scope
    filters = {
        'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap,
        'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi,
        'use_obi': use_obi
    }
    
    signal, reasons, atr, prob = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    
    # Copilot Advice
    advice = get_ai_advice(df.iloc[-1], trend_4h, obi)
    
    # Táctica
    setup = None
    if signal != "NEUTRO":
        sl_dist = atr * 1.5
        risk = sl_dist
        if signal == "LONG":
            sl = current_price - sl_dist
            tp1, tp2, tp3 = current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            direction_emoji = "⬆️ COMPRA"
        else:
            sl = current_price + sl_dist
            tp1, tp2, tp3 = current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            direction_emoji = "⬇️ VENTA"
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': direction_emoji}

    # Alertas
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        send_telegram_msg(f"🦁 v6.0: {signal} {symbol}\nMTF 4H: {trend_4h}\nProb: {prob:.1f}%")
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    # Ejecutar Gestión de Posiciones (Loop)
    manage_open_positions(current_price)
    
    # --- UI ---
    tab1, tab2 = st.tabs(["📊 LIVE COMMAND", "🧪 PAPER TRADING (CSV)"])
    
    with tab1:
        # HEADER DE ESTADO
        st.info(f"🤖 AI COPILOT: {advice}")
        
        # METRICAS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Tendencia 4H", trend_4h, delta="Bullish" if trend_4h=="BULLISH" else "Bearish", delta_color="normal")
        c3.metric("OBI", f"{obi:.1%}")
        c4.metric("Probabilidad", f"{prob:.1f}%")

        # TARJETA TÁCTICA
        if signal != "NEUTRO" and setup:
            header_class = "direction-header-long" if signal == "LONG" else "direction-header-short"
            st.markdown(f"""
            <div class="trade-setup">
                <div class="{header_class}">OPERACIÓN: {setup['dir']}</div>
                <div style="display: flex; justify-content: space-around;">
                    <div><span class="label-mini">ENTRADA</span><br><span class="entry-blue">${setup['entry']:.2f}</span></div>
                    <div><span class="label-mini">STOP LOSS</span><br><span class="sl-red">${setup['sl']:.2f}</span></div>
                    <div><span class="label-mini">TP 1</span><br><span class="tp-green">${setup['tp1']:.2f}</span></div>
                    <div><span class="label-mini">TP 3</span><br><span class="tp-green">${setup['tp3']:.2f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🚀 EJECUTAR {signal} (PAPER CSV)"):
                execute_trade(signal, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], 100, atr)
                st.success("Guardado en paper_trades.csv")
        
        # GRÁFICO
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        if signal != "NEUTRO" and setup:
            fig.add_hline(y=setup['tp1'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_trades = load_trades()
        if not df_trades.empty:
            open_trades = df_trades[df_trades['status'] == "OPEN"]
            closed_trades = df_trades[df_trades['status'] == "CLOSED"]
            
            st.subheader("🟢 Posiciones Abiertas")
            if not open_trades.empty:
                st.dataframe(open_trades[['time', 'symbol', 'type', 'entry', 'sl', 'tp3', 'candles_held']])
            else: st.info("No hay operaciones en curso.")
            
            st.subheader("📜 Historial Cerrado")
            st.dataframe(closed_trades[['time', 'symbol', 'type', 'pnl', 'reason']])
            
            total_pnl = closed_trades['pnl'].sum()
            st.metric("PnL Total Acumulado", f"${total_pnl:.2f}")
        else:
            st.info("El historial está vacío. Ejecuta tu primera operación.")

else: st.warning("Cargando Motores MTF...")

if auto_refresh: time.sleep(60); st.rerun()
