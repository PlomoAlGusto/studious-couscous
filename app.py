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
st.set_page_config(page_title="Quimera Pro v7.1 Analytics", layout="wide", page_icon="🦁")

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
HUNTER_ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'PAXG/USDT', 'XAUT/USDT']

# -----------------------------------------------------------------------------
# 2. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v7.1")
    st.caption("Analytics Edition 📊")
    
    st.header("🔫 HUNTER AUTO")
    hunter_active = st.toggle("ACTIVAR CAZA", False)
    if hunter_active: st.success("ON 🟢")
    
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
# 3. MOTORES
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken"

exchange, source_name = init_exchange()

@st.cache_data(ttl=15)
def get_full_analysis_data(ticker, tf):
    try:
        ohlcv = exchange.fetch_ohlcv(ticker, tf, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Trend 4H
        ohlcv4h = exchange.fetch_ohlcv(ticker, '4h', limit=50)
        df4h = pd.DataFrame(ohlcv4h, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        df4h['ema50'] = ta.ema(df4h['c'], length=50)
        trend4h = "BULLISH" if df4h['c'].iloc[-1] > df4h['ema50'].iloc[-1] else "BEARISH"
        
        # OBI
        book = exchange.fetch_order_book(ticker, limit=10)
        b, a = sum([x[1] for x in book['bids']]), sum([x[1] for x in book['asks']])
        obi = (b-a)/(b+a) if (b+a)>0 else 0
        
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
# 4. ANALYTICS & CHARTS (NUEVO)
# -----------------------------------------------------------------------------
def render_analytics(df_trades):
    if df_trades.empty:
        st.info("No hay datos suficientes para generar analíticas.")
        return

    closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if closed.empty:
        st.info("Aún no has cerrado ninguna operación.")
        return

    # 1. CÁLCULO DE LA CURVA DE EQUIDAD
    # Empezamos con 10,000 y vamos sumando/restando el PnL acumulado
    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = 10000 + closed['cumulative_pnl']
    
    # Añadimos el punto inicial (Día 0 = $10,000)
    start_point = pd.DataFrame([{'time': 'Inicio', 'equity': 10000}])
    equity_curve = pd.concat([start_point, closed[['time', 'equity']]])

    # 2. KPIs
    total_trades = len(closed)
    wins = len(closed[closed['pnl'] > 0])
    losses = len(closed[closed['pnl'] <= 0])
    win_rate = (wins / total_trades) * 100
    best_trade = closed['pnl'].max()
    worst_trade = closed['pnl'].min()
    total_profit = closed['pnl'].sum()

    # 3. VISUALIZACIÓN
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Beneficio Neto", f"${total_profit:,.2f}", delta_color="normal")
    k2.metric("Win Rate", f"{win_rate:.1f}%")
    k3.metric("Mejor Trade", f"${best_trade:,.2f}")
    k4.metric("Peor Trade", f"${worst_trade:,.2f}")

    # GRÁFICO DE ÁREA (EQUITY CURVE)
    fig = px.area(equity_curve, x='time', y='equity', title='Curva de Capital (Equity Curve)')
    fig.update_layout(template="plotly_dark", height=350)
    fig.update_traces(line_color='#00FF00' if total_profit > 0 else '#FF4444')
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. GESTIÓN AUTOMÁTICA
# -----------------------------------------------------------------------------
def log_event(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.hunter_logs.insert(0, f"[{timestamp}] {msg}")
    if len(st.session_state.hunter_logs) > 50: st.session_state.hunter_logs.pop()

def execute_auto_trade(ticker, signal, price, atr):
    df_trades = load_trades()
    if not df_trades.empty:
        if not df_trades[(df_trades['status'] == 'OPEN') & (df_trades['symbol'] == ticker)].empty:
            return False

    sl_dist = atr * 1.5
    if signal == "LONG":
        sl = price - sl_dist
        tp1, tp2, tp3 = price+sl_dist, price+(sl_dist*2), price+(sl_dist*3.5)
    else:
        sl = price + sl_dist
        tp1, tp2, tp3 = price-sl_dist, price-(sl_dist*2), price-(sl_dist*3.5)

    execute_trade_db(ticker, signal, price, sl, tp1, tp2, tp3, 100, atr)
    msg = f"🔫 AUTO-SHOT: {signal} {ticker} @ ${price}"
    log_event(msg)
    send_telegram_msg(msg)
    return True

def run_hunter_cycle():
    log_event("🦁 Escaneando mercado...")
    for coin in HUNTER_ASSETS:
        df, obi, trend_4h = get_full_analysis_data(coin, tf)
        if df is not None:
            df = calculate_indicators(df)
            sig, atr = run_strategy_check(df, obi, trend_4h)
            if sig != "NEUTRO":
                executed = execute_auto_trade(coin, sig, df['close'].iloc[-1], atr)
                if executed: log_event(f"✅ Trade abierto: {coin}")

def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
    try: return pd.read_csv(CSV_FILE)
    except: return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])

def execute_trade_db(sym, type, entry, sl, tp1, tp2, tp3, size, atr):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": sym, "type": type, "entry": entry, "size": size, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": "Auto", "candles_held": 0, "atr_entry": atr}
    df = pd.concat([pd.DataFrame([new]), df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

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
            new_sl = curr - (row['atr_entry'] * 1.5)
            if new_sl > row['sl']: df.at[idx, 'sl'] = new_sl
            if curr >= row['tp3']: close_reason="TP3"; pnl=(row['tp3']-row['entry'])*row['size']
            elif curr <= row['sl']: close_reason="SL"; pnl=(row['sl']-row['entry'])*row['size']
        else:
            new_sl = curr + (row['atr_entry'] * 1.5)
            if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
            if curr <= row['tp3']: close_reason="TP3"; pnl=(row['entry']-row['tp3'])*row['size']
            elif curr >= row['sl']: close_reason="SL"; pnl=(row['entry']-row['sl'])*row['size']
            
        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            send_telegram_msg(f"💰 CIERRE {row['symbol']}: {close_reason} (${pnl:.2f})")
            updated = True
            
    if updated: df.to_csv(CSV_FILE, index=False)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------
prices_map = {}
if hunter_active:
    run_hunter_cycle()
    for coin in HUNTER_ASSETS:
        try: prices_map[coin] = exchange.fetch_ticker(coin)['last']
        except: pass
else:
    df_m, _, _ = get_full_analysis_data(symbol, tf)
    if df_m is not None: prices_map[symbol] = df_m['close'].iloc[-1]

manage_positions(prices_map)

# -----------------------------------------------------------------------------
# 7. INTERFAZ
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 MANUAL", "🧪 CARTERA & ANALYTICS", "🔫 HUNTER LOGS"])

with tab1:
    df, obi, trend_4h = get_full_analysis_data(symbol, tf)
    if df is not None:
        df = calculate_indicators(df)
        sig, atr = run_strategy_check(df, obi, trend_4h)
        curr = df['close'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${curr:,.2f}")
        c2.metric("Tendencia 4H", trend_4h)
        c3.metric("OBI", f"{obi:.1%}")
        
        if sig != "NEUTRO":
            st.success(f"SEÑAL MANUAL: {sig}")
            if st.button(f"EJECUTAR {sig}"):
                execute_trade_db(symbol, sig, curr, curr-(atr*1.5), curr+atr, curr+(atr*2), curr+(atr*3.5), 100, atr)
                st.toast("Orden manual enviada")
        
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_trades = load_trades()
    
    # --- AQUÍ ESTÁ LA MAGIA DE LAS ANALÍTICAS ---
    st.subheader("📈 Rendimiento de la Cuenta")
    render_analytics(df_trades) # LLAMADA A LA FUNCIÓN DE GRÁFICOS
    
    st.divider()
    st.subheader("Operaciones Abiertas")
    st.dataframe(df_trades[df_trades['status']=='OPEN'])
    st.subheader("Historial")
    st.dataframe(df_trades[df_trades['status']=='CLOSED'])

with tab3:
    st.markdown("### 🛰️ Centro de Mando Autónomo")
    logs_txt = "\n".join(st.session_state.hunter_logs)
    st.text_area("Terminal:", value=logs_txt, height=300, disabled=True)
    
    # También mostramos el gráfico de rendimiento aquí para el modo Hunter
    st.divider()
    st.subheader("Evolución del Cazador")
    render_analytics(df_trades)

if auto_refresh or hunter_active:
    time.sleep(60)
    st.rerun()
