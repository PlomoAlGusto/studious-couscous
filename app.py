import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import time
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera v7.3 Lite", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .hunter-log {background-color: #000; color: #0f0; font-family: monospace; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.1); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
</style>
""", unsafe_allow_html=True)

if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'hunter_logs' not in st.session_state: st.session_state.hunter_logs = []
if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

CSV_FILE = 'paper_trades.csv'
# ⚠️ LISTA REDUCIDA PARA EVITAR CUELGUES
HUNTER_ASSETS = ['BTC/USDT', 'XRP/USDT']

# -----------------------------------------------------------------------------
# 2. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v7.3")
    st.caption("Lite Edition ⚡")
    
    st.header("🔫 HUNTER AUTO")
    hunter_active = st.toggle("ACTIVAR CAZA", False)
    if hunter_active: st.success("ON (BTC/XRP)")
    
    st.divider()
    st.header("🔬 MANUAL")
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["15m", "1h"], index=0)

    with st.expander("⚙️ FILTROS"):
        use_mtf = st.checkbox("MTF 4H", True)
        use_vwap = st.checkbox("VWAP", True)
        use_ema = st.checkbox("EMAs", True)
        use_obi = st.checkbox("OBI", True)
        
    auto_refresh = st.checkbox("🔄 Refresco (60s)", False)

# -----------------------------------------------------------------------------
# 3. MOTORES LIGEROS
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            # Timeout muy corto para fallar rápido si se cuelga
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}, 'timeout': 5000})
            return ex, "Binance"
    except: pass
    return ccxt.kraken({'timeout': 5000}), "Kraken"

exchange, source_name = init_exchange()

# CACHÉ CORTO PARA EVITAR BUCLES
@st.cache_data(ttl=15)
def get_data_lite(ticker, tf):
    try:
        # Solo bajamos 100 velas para ir rápido
        ohlcv = exchange.fetch_ohlcv(ticker, tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # MTF 4H Simplificado
        trend4h = "NEUTRO"
        try:
            ohlcv4h = exchange.fetch_ohlcv(ticker, '4h', limit=30)
            df4h = pd.DataFrame(ohlcv4h, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            ema50 = ta.ema(df4h['c'], length=50).iloc[-1] if len(df4h) > 50 else df4h['c'].mean()
            trend4h = "BULLISH" if df4h['c'].iloc[-1] > ema50 else "BEARISH"
        except: pass
        
        # OBI Ultrarrápido (Solo top 5 ordenes)
        obi = 0
        try:
            book = exchange.fetch_order_book(ticker, limit=5)
            b, a = sum([x[1] for x in book['bids']]), sum([x[1] for x in book['asks']])
            obi = (b-a)/(b+a) if (b+a)>0 else 0
        except: pass
        
        return df, obi, trend4h
    except: return None, 0, "NEUTRO"

def calculate_indicators(df):
    if df is None: return None
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    try:
        vp = ((df['high']+df['low']+df['close'])/3)*df['volume']
        df['VWAP'] = vp.cumsum()/df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df.fillna(method='bfill').fillna(method='ffill')

def run_strategy_check(df, obi, trend_4h):
    row = df.iloc[-1]
    score = 0
    if use_mtf: score += 2 if trend_4h == "BULLISH" else -2
    if use_ema: score += 1 if row['EMA_20'] > row['EMA_50'] else -1
    if use_vwap: score += 1 if row['close'] > row['VWAP'] else -1
    if use_obi: score += 1 if obi > 0.05 else (-1 if obi < -0.05 else 0)
    
    signal = "NEUTRO"
    if score >= 3: signal = "LONG"
    elif score <= -3: signal = "SHORT"
    return signal, row['ATR']

# -----------------------------------------------------------------------------
# 4. ANALYTICS
# -----------------------------------------------------------------------------
def render_analytics(df_trades, unique_key):
    if df_trades.empty:
        st.caption("Esperando datos...")
        return

    closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if closed.empty:
        st.caption("Sin operaciones cerradas aún.")
        return

    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = 10000 + closed['cumulative_pnl']
    start = pd.DataFrame([{'time': 'Inicio', 'equity': 10000}])
    curve = pd.concat([start, closed[['time', 'equity']]])

    wins = len(closed[closed['pnl'] > 0])
    total = len(closed)
    win_rate = (wins / total) * 100 if total > 0 else 0
    profit = closed['pnl'].sum()

    c1, c2 = st.columns(2)
    c1.metric("PnL Neto", f"${profit:.2f}", delta_color="normal")
    c2.metric("Win Rate", f"{win_rate:.0f}%")

    fig = px.area(curve, x='time', y='equity', title="Curva de Capital")
    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=30,b=0))
    fig.update_traces(line_color='#00FF00' if profit > 0 else '#FF4444')
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

# -----------------------------------------------------------------------------
# 5. GESTIÓN AUTOMÁTICA (HUNTER)
# -----------------------------------------------------------------------------
def log_event(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.hunter_logs.insert(0, f"[{timestamp}] {msg}")
    if len(st.session_state.hunter_logs) > 20: st.session_state.hunter_logs.pop()

def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
    try: return pd.read_csv(CSV_FILE)
    except: return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])

def execute_trade_db(sym, type, entry, sl, tp1, tp2, tp3, size, atr, reason="Manual"):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": sym, "type": type, "entry": entry, "size": size, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": reason, "candles_held": 0, "atr_entry": atr}
    df = pd.concat([pd.DataFrame([new]), df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c: 
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg}, timeout=1)
        except: pass

def run_hunter_cycle():
    # Ciclo optimizado: No bloquea la UI, solo añade operaciones si las encuentra
    for coin in HUNTER_ASSETS:
        df, obi, trend_4h = get_data_lite(coin, tf) # Usamos la version LITE
        if df is not None:
            df = calculate_indicators(df)
            sig, atr = run_strategy_check(df, obi, trend_4h)
            
            if sig != "NEUTRO":
                # Check si ya existe
                df_t = load_trades()
                is_open = False
                if not df_t.empty:
                    if not df_t[(df_t['status']=='OPEN') & (df_t['symbol']==coin)].empty: is_open = True
                
                if not is_open:
                    p = df['close'].iloc[-1]
                    sl_d = atr * 1.5
                    if sig == "LONG": sl, tp1, tp2, tp3 = p-sl_d, p+sl_d, p+(sl_d*2), p+(sl_d*3.5)
                    else: sl, tp1, tp2, tp3 = p+sl_d, p-sl_d, p-(sl_d*2), p-(sl_d*3.5)
                    
                    execute_trade_db(coin, sig, p, sl, tp1, tp2, tp3, 100, atr, "Auto")
                    msg = f"🔫 AUTO: {sig} {coin} @ ${p}"
                    log_event(msg)
                    send_telegram_msg(msg)

def manage_positions(prices_map):
    df = load_trades()
    if df.empty: return
    open_idx = df.index[df['status'] == "OPEN"].tolist()
    updated = False
    for idx in open_idx:
        row = df.loc[idx]
        curr = prices_map.get(row['symbol'])
        if not curr: continue
        
        close_reason = ""
        pnl = 0
        if row['type'] == "LONG":
            if curr >= row['tp3']: close_reason="TP3"; pnl=(row['tp3']-row['entry'])*row['size']
            elif curr <= row['sl']: close_reason="SL"; pnl=(row['sl']-row['entry'])*row['size']
        else:
            if curr <= row['tp3']: close_reason="TP3"; pnl=(row['entry']-row['tp3'])*row['size']
            elif curr >= row['sl']: close_reason="SL"; pnl=(row['entry']-row['sl'])*row['size']
            
        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            send_telegram_msg(f"💰 CIERRE {row['symbol']}: {close_reason} (${pnl:.2f})")
            updated = True
    if updated: df.to_csv(CSV_FILE, index=False)

# -----------------------------------------------------------------------------
# 6. EJECUCIÓN PRINCIPAL (RENDER PRIMERO, CÁLCULO DESPUÉS)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 MANUAL", "🧪 CARTERA", "🔫 HUNTER"])

# 1. LÓGICA HUNTER (Solo si activo)
prices_map = {}
if hunter_active:
    run_hunter_cycle()
    # Obtener precios rápidos para gestión
    for c in HUNTER_ASSETS:
        df_tmp, _, _ = get_data_lite(c, tf)
        if df_tmp is not None: prices_map[c] = df_tmp['close'].iloc[-1]

# 2. DATOS MANUALES
df_m, obi_m, trend_m = get_data_lite(symbol, tf)
if df_m is not None: prices_map[symbol] = df_m['close'].iloc[-1]

# 3. GESTIÓN POSICIONES
manage_positions(prices_map)

# --- RENDER TABS ---

with tab1:
    if df_m is not None:
        df_m = calculate_indicators(df_m)
        sig, atr = run_strategy_check(df_m, obi_m, trend_m)
        curr = df_m['close'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${curr:,.2f}")
        c2.metric("Tendencia 4H", trend_m)
        c3.metric("OBI", f"{obi_m:.1%}")
        
        if sig != "NEUTRO":
            st.success(f"SEÑAL: {sig}")
            if st.button("EJECUTAR MANUAL"):
                execute_trade_db(symbol, sig, curr, curr-(atr*1.5), curr+atr, curr+(atr*2), curr+(atr*3.5), 100, atr)
                st.toast("Enviada")
        else:
            st.info("Sin señal clara.")
            
        fig = go.Figure(data=[go.Candlestick(x=df_m['timestamp'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'])])
        fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_t = load_trades()
    render_analytics(df_t, "k1")
    st.divider()
    st.caption("Operaciones Abiertas")
    st.dataframe(df_t[df_t['status']=='OPEN'], use_container_width=True)
    st.caption("Historial")
    st.dataframe(df_t[df_t['status']=='CLOSED'], use_container_width=True)

with tab3:
    st.markdown("### 🛰️ Logs")
    st.text_area("", value="\n".join(st.session_state.hunter_logs), height=200)
    render_analytics(load_trades(), "k2")

if auto_refresh or hunter_active:
    time.sleep(60)
    st.rerun()
