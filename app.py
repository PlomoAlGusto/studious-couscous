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
st.set_page_config(page_title="Quimera Pro v7.2 Performance", layout="wide", page_icon="🦁")

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
    st.title("🦁 QUIMERA v7.2")
    st.caption("Performance Edition ⚡")
    
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
# 3. MOTORES DE DATOS (OPTIMIZADOS)
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            # Timeout corto para no colgar la app
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}, 'timeout': 10000})
            return ex, "Binance"
    except: pass
    return ccxt.kraken({'timeout': 10000}), "Kraken"

exchange, source_name = init_exchange()

@st.cache_data(ttl=30) # Cache aumentado a 30s para ir más fluido
def get_full_analysis_data(ticker, tf):
    try:
        ohlcv = exchange.fetch_ohlcv(ticker, tf, limit=100) # Bajamos limite a 100 velas
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Trend 4H (Optimizado)
        trend4h = "NEUTRO"
        if use_mtf:
            try:
                ohlcv4h = exchange.fetch_ohlcv(ticker, '4h', limit=30)
                df4h = pd.DataFrame(ohlcv4h, columns=['t', 'o', 'h', 'l', 'c', 'v'])
                ema50_4h = ta.ema(df4h['c'], length=50).iloc[-1] if len(df4h) > 50 else df4h['c'].mean()
                trend4h = "BULLISH" if df4h['c'].iloc[-1] > ema50_4h else "BEARISH"
            except: pass
        
        # OBI (Optimizado)
        obi = 0
        if use_obi:
            try:
                book = exchange.fetch_order_book(ticker, limit=5) # Solo top 5 ordenes
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
# 4. ANALYTICS & CHARTS
# -----------------------------------------------------------------------------
def render_analytics(df_trades, unique_key):
    if df_trades.empty:
        st.info("No hay datos suficientes para generar analíticas.")
        return

    closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if closed.empty:
        st.info("Aún no has cerrado ninguna operación.")
        return

    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = 10000 + closed['cumulative_pnl']
    start_point = pd.DataFrame([{'time': 'Inicio', 'equity': 10000}])
    equity_curve = pd.concat([start_point, closed[['time', 'equity']]])

    total_trades = len(closed)
    wins = len(closed[closed['pnl'] > 0])
    win_rate = (wins / total_trades) * 100
    total_profit = closed['pnl'].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Beneficio Neto", f"${total_profit:,.2f}")
    k2.metric("Win Rate", f"{win_rate:.1f}%")
    k3.metric("Trades", total_trades)

    fig = px.area(equity_curve, x='time', y='equity', title='Curva de Capital')
    fig.update_layout(template="plotly_dark", height=300)
    fig.update_traces(line_color='#00FF00' if total_profit > 0 else '#FF4444')
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

# -----------------------------------------------------------------------------
# 5. GESTIÓN AUTOMÁTICA
# -----------------------------------------------------------------------------
def log_event(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.hunter_logs.insert(0, f"[{timestamp}] {msg}")
    if len(st.session_state.hunter_logs) > 50: st.session_state.hunter_logs.pop()

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
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg}, timeout=2)
        except: pass

# -----------------------------------------------------------------------------
# 6. INTERFAZ GRÁFICA (PRIMERO LA UI, LUEGO LOS DATOS)
# -----------------------------------------------------------------------------
# Dibujamos las pestañas PRIMERO para que no se quede la pantalla negra
tab1, tab2, tab3 = st.tabs(["📊 MANUAL", "🧪 CARTERA & ANALYTICS", "🔫 HUNTER LOGS"])

# --- LÓGICA PRINCIPAL ---
prices_map = {}

# Ejecución HUNTER (Con barra de carga)
if hunter_active:
    with st.status("🦁 Cazando oportunidades...", expanded=True) as status:
        for coin in HUNTER_ASSETS:
            status.write(f"Analizando {coin}...")
            df, obi, trend_4h = get_full_analysis_data(coin, tf)
            
            if df is not None:
                prices_map[coin] = df['close'].iloc[-1] # Guardar precio para gestión
                df = calculate_indicators(df)
                sig, atr = run_strategy_check(df, obi, trend_4h)
                
                # DISPARO AUTOMÁTICO
                if sig != "NEUTRO":
                    df_trades = load_trades()
                    is_open = False
                    if not df_trades.empty:
                        if not df_trades[(df_trades['status'] == 'OPEN') & (df_trades['symbol'] == coin)].empty:
                            is_open = True
                    
                    if not is_open:
                        p = df['close'].iloc[-1]
                        sl_dist = atr * 1.5
                        if sig == "LONG":
                            sl, tp1, tp2, tp3 = p-sl_dist, p+sl_dist, p+(sl_dist*2), p+(sl_dist*3.5)
                        else:
                            sl, tp1, tp2, tp3 = p+sl_dist, p-sl_dist, p-(sl_dist*2), p-(sl_dist*3.5)
                            
                        execute_trade_db(coin, sig, p, sl, tp1, tp2, tp3, 100, atr, "Auto-Hunter")
                        msg = f"🔫 HUNTER: {sig} {coin} @ ${p}"
                        log_event(msg)
                        send_telegram_msg(msg)
                        status.write(f"✅ DISPARO: {coin}")

        status.update(label="Ciclo completado", state="complete", expanded=False)

# Obtener datos MANUALES (si no es hunter mode, necesitamos al menos el precio manual)
df_m, obi_m, trend_m = get_full_analysis_data(symbol, tf)
if df_m is not None: 
    prices_map[symbol] = df_m['close'].iloc[-1]

# GESTIÓN DE POSICIONES (Loop rápido)
df_trades = load_trades()
if not df_trades.empty:
    open_idx = df_trades.index[df_trades['status'] == "OPEN"].tolist()
    updated = False
    for idx in open_idx:
        row = df_trades.loc[idx]
        curr = prices_map.get(row['symbol'])
        if not curr: continue # Si no tenemos precio actualizado, saltamos
        
        close_reason = ""
        pnl = 0
        if row['type'] == "LONG":
            if curr >= row['tp3']: close_reason="TP3"; pnl=(row['tp3']-row['entry'])*row['size']
            elif curr <= row['sl']: close_reason="SL"; pnl=(row['sl']-row['entry'])*row['size']
        else:
            if curr <= row['tp3']: close_reason="TP3"; pnl=(row['entry']-row['tp3'])*row['size']
            elif curr >= row['sl']: close_reason="SL"; pnl=(row['entry']-row['sl'])*row['size']
            
        if close_reason:
            df_trades.at[idx, 'status'] = "CLOSED"; df_trades.at[idx, 'pnl'] = pnl; df_trades.at[idx, 'reason'] = close_reason
            send_telegram_msg(f"💰 CIERRE {row['symbol']}: {close_reason} (${pnl:.2f})")
            updated = True
    if updated: df_trades.to_csv(CSV_FILE, index=False)

# --- RENDERIZADO DE TABS ---

with tab1:
    if df_m is not None:
        df_m = calculate_indicators(df_m)
        sig, atr = run_strategy_check(df_m, obi_m, trend_m)
        curr = df_m['close'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${curr:,.2f}")
        c2.metric("Trend 4H", trend_m)
        c3.metric("OBI", f"{obi_m:.1%}")
        
        if sig != "NEUTRO":
            st.success(f"SEÑAL MANUAL: {sig}")
            if st.button(f"EJECUTAR {sig}"):
                execute_trade_db(symbol, sig, curr, curr-(atr*1.5), curr+atr, curr+(atr*2), curr+(atr*3.5), 100, atr)
                st.toast("Orden manual enviada")
        
        fig = go.Figure(data=[go.Candlestick(x=df_m['timestamp'], open=df_m['open'], high=df_m['high'], low=df_m['low'], close=df_m['close'])])
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Cargando datos manuales...")

with tab2:
    df_view = load_trades()
    st.subheader("📈 Rendimiento")
    render_analytics(df_view, "analytics_key")
    st.divider()
    st.subheader("Abiertas")
    st.dataframe(df_view[df_view['status']=='OPEN'])
    st.subheader("Historial")
    st.dataframe(df_view[df_view['status']=='CLOSED'])

with tab3:
    st.markdown("### 🛰️ Centro de Mando")
    st.write(f"Monitoreando: {', '.join(HUNTER_ASSETS)}")
    logs_txt = "\n".join(st.session_state.hunter_logs)
    st.text_area("Logs:", value=logs_txt, height=300, disabled=True)

if auto_refresh or hunter_active:
    time.sleep(60)
    st.rerun()
