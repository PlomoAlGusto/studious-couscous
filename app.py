import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import os
import feedparser

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v16.4 Visual Fix", layout="wide", page_icon="🦁")

# CSS MÍNIMO (Solo para lo esencial)
st.markdown("""
<style>
    .stMetric { background-color: #111; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between; border: 1px solid #333; }
    .clock-open { background-color: rgba(0, 255, 0, 0.1); border-color: #00FF00; }
    .clock-closed { background-color: rgba(255, 255, 255, 0.05); }
    
    /* Estilo para la caja de IA nativa */
    .ai-container { border: 1px solid #44AAFF; border-left: 5px solid #44AAFF; background-color: #0e1117; padding: 20px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ARCHIVOS
CSV_FILE = 'paper_trades.csv'
COLUMNS_DB = ["id", "time", "symbol", "type", "entry", "size", "leverage", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"]
INITIAL_CAPITAL = 10000.0

if not os.path.exists(CSV_FILE):
    df_empty = pd.DataFrame(columns=COLUMNS_DB)
    df_empty.to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. MOTOR DE DATOS (BYBIT V5 - ROBUSTO)
# -----------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=COLUMNS_DB)
    try:
        df = pd.read_csv(CSV_FILE)
        if 'leverage' not in df.columns: df['leverage'] = 1.0
        return df
    except: return pd.DataFrame(columns=COLUMNS_DB)

def get_current_balance():
    df = load_trades()
    if df.empty: return INITIAL_CAPITAL
    realized_pnl = df[df['status'] == 'CLOSED']['pnl'].sum()
    return INITIAL_CAPITAL + realized_pnl

def reset_account():
    df_empty = pd.DataFrame(columns=COLUMNS_DB)
    df_empty.to_csv(CSV_FILE, index=False)
    st.rerun()

def get_market_sessions():
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("### 🌍 SESIONES")
    for name, (start, end) in sessions.items():
        is_open = start <= hour < end
        css_class = "clock-open" if is_open else "clock-closed"
        icon = "🟢" if is_open else "🔴"
        st.sidebar.markdown(f"<div class='market-clock {css_class}'><span>{name}</span><span>{icon}</span></div>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_deriv_data(symbol):
    """
    Obtiene Funding Rate y Open Interest usando BYBIT V5 (Público y Estable).
    """
    base_coin = symbol.split('/')[0]
    
    # INTENTO 1: BYBIT V5 (Suele funcionar siempre)
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={base_coin}USDT"
        r = requests.get(url, timeout=4).json()
        if r['retCode'] == 0:
            info = r['result']['list'][0]
            fr = float(info['fundingRate']) * 100
            oi_val = float(info['openInterestValue'])
            return fr, oi_val, "Bybit"
    except:
        pass

    # INTENTO 2: BINANCE (Fallback)
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={base_coin}USDT"
        r = requests.get(url, timeout=4).json()
        fr = float(r['lastFundingRate']) * 100
        
        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={base_coin}USDT"
        r_oi = requests.get(url_oi, timeout=4).json()
        oi_val = float(r_oi['openInterest']) * float(r['markPrice'])
        return fr, oi_val, "Binance"
    except:
        pass

    return 0.0, 0.0, "Error"

@st.cache_data(ttl=30)
def get_mtf_trends_analysis(symbol):
    ex = ccxt.binance()
    ticker_fix = symbol.replace("/", "USDT") if "/" not in symbol else symbol
    trends = {}
    score = 0
    for tf in ['15m', '1h', '4h']:
        try:
            ohlcv = ex.fetch_ohlcv(ticker_fix, tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            ema_fast = ta.ema(df['c'], length=20).iloc[-1]
            ema_slow = ta.ema(df['c'], length=50).iloc[-1]
            if ema_fast > ema_slow: trends[tf] = "BULL"; score += 1
            else: trends[tf] = "BEAR"; score -= 1
        except: trends[tf] = "NEUTRO"
    return trends, score

# -----------------------------------------------------------------------------
# 3. INTERFAZ SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v16.4")
    st.success("● SYSTEM ONLINE")
    get_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)
    
    with st.expander("🛡️ FILTROS & CORE", expanded=True):
        use_ema = st.checkbox("Tendencia Base (EMAs)", True)
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP (Institucional)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)
    
    with st.expander("🌊 MOMENTO Y VOLUMEN"):
        use_rsi = st.checkbox("RSI & Stoch", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        use_tsi = st.checkbox("TSI (True Strength)", True)
        
    with st.expander("💰 GESTIÓN DE RIESGO"):
        current_balance = get_current_balance()
        st.metric("Balance Disponible", f"${current_balance:,.2f}")
        risk_per_trade = st.slider("Riesgo por Trade (%)", 0.5, 5.0, 1.0)
        
    with st.expander("⚙️ SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
    if st.button("🔥 RESETEAR CUENTA"): reset_account()

# -----------------------------------------------------------------------------
# 4. CAPA DE DATOS
# -----------------------------------------------------------------------------
def init_exchange():
    try: return ccxt.binance(), "Binance"
    except: return ccxt.kraken(), "Kraken"

exchange, source_name = init_exchange()

@st.cache_data(ttl=300)
def get_crypto_news():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        return [{"title": e.title, "link": e.link, "published": e.published_parsed} for e in feed.entries[:5]]
    except: return []

@st.cache_data(ttl=15)
def get_mtf_data(symbol, tf_lower):
    if not exchange: return None, 0, None
    ticker_fix = symbol
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None

    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
        last_4h = df_4h.iloc[-1]
        trend_4h = "BULLISH" if last_4h['close'] > last_4h['EMA_50'] else "BEARISH"
    except: pass

    obi = 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids = sum([x[1] for x in book['bids']])
        asks = sum([x[1] for x in book['asks']])
        obi = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
    except: pass
    return df, obi, trend_4h

def calculate_indicators(df):
    if df is None: return None
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    try:
        vp = ((df['high'] + df['low'] + df['close'])/3) * df['volume']
        df['VWAP'] = vp.cumsum() / df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1)
    df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    try:
        tsi = ta.tsi(df['close'], fast=13, slow=25)
        df = pd.concat([df, tsi], axis=1)
        tsi_col = [c for c in df.columns if 'TSI' in c][0]
        df['TSI'] = df[tsi_col]
    except: df['TSI'] = 0
    
    high_w, low_w, close_w = df['high'].rolling(20).max(), df['low'].rolling(20).min(), df['close']
    df['PIVOT'] = (high_w + low_w + close_w) / 3
    df['R1'], df['S1'] = (2 * df['PIVOT']) - low_w, (2 * df['PIVOT']) - high_w
    return df.fillna(method='bfill').fillna(method='ffill')

# -----------------------------------------------------------------------------
# 5. IA ANALISTA (NATIVE UI - NO HTML ERRORS)
# -----------------------------------------------------------------------------
def render_native_ai_analysis(row, mtf_trends, mtf_score, obi, fr, open_interest, data_src, signal, prob):
    # LOGICA DE ANALISIS
    t_15m = mtf_trends.get('15m', 'NEUTRO')
    t_1h = mtf_trends.get('1h', 'NEUTRO')
    t_4h = mtf_trends.get('4h', 'NEUTRO')
    
    if mtf_score == 3: context_txt = "ALCISTA FUERTE (Alineación Total)"; context_col = "green"
    elif mtf_score == -3: context_txt = "BAJISTA FUERTE (Alineación Total)"; context_col = "red"
    elif t_4h == "BULL" and t_15m == "BEAR": context_txt = "CORRECCIÓN (Macro Alcista / Micro Bajista)"; context_col = "orange"
    elif t_4h == "BEAR" and t_15m == "BULL": context_txt = "REBOTE (Macro Bajista / Micro Alcista)"; context_col = "orange"
    else: context_txt = "MERCADO MIXTO"; context_col = "grey"
    
    deriv_txt = "Saludable"
    if fr > 0.01: deriv_txt = "Riesgo Long Squeeze"
    elif fr < -0.01: deriv_txt = "Riesgo Short Squeeze"
    
    if open_interest > 1000000000: oi_fmt = f"${open_interest/1000000000:.2f}B"
    elif open_interest > 1000000: oi_fmt = f"${open_interest/1000000:.2f}M"
    else: oi_fmt = f"${open_interest:,.0f}"

    mfi = row['MFI']
    adx = row['ADX_14']
    tsi = row['TSI']
    
    gas_status = "LLENO" if mfi > 60 else "RESERVA" if mfi < 40 else "MEDIO"
    tsi_status = "ALCISTA" if tsi > 0 else "BAJISTA"
    
    pressure = "COMPRADORA" if obi > 0.05 else "VENDEDORA" if obi < -0.05 else "NEUTRA"
    obi_col = "green" if obi > 0.05 else "red" if obi < -0.05 else "grey"

    # RENDERIZADO NATIVO DE STREAMLIT (SIN HTML ROTO)
    with st.container():
        st.markdown(f"### 🤖 QUIMERA COPILOT (Source: {data_src})")
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**📡 ESTRUCTURA:** :{context_col}[{context_txt}]")
        c2.markdown(f"**📊 DERIVADOS:** Funding: `{fr:.4f}%` ({deriv_txt})")
        
        c3, c4 = st.columns(2)
        c3.markdown(f"**⛽ VOLUMEN:** Libro :{obi_col}[{pressure}] ({obi*100:.1f}%)")
        c4.markdown(f"**💸 INTERÉS ABIERTO:** `{oi_fmt}`")
        
        st.markdown(f"**🔥 MOMENTO:** Gasolina: **{gas_status}** | ADX: `{adx:.1f}` | TSI: `{tsi_status}`")
        
        st.divider()
        
        if signal == "LONG":
            st.success(f"🎯 VEREDICTO FINAL: **ENTRADA LONG** (Probabilidad: {prob:.1f}%)")
        elif signal == "SHORT":
            st.error(f"🎯 VEREDICTO FINAL: **ENTRADA SHORT** (Probabilidad: {prob:.1f}%)")
        else:
            st.warning("⏳ VEREDICTO FINAL: **ESPERAR CONFIRMACIÓN**")

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
    score, max_score = 0, 0
    
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2
        elif trend_4h == "BEARISH": score -= 2

    if filters['use_ema']: 
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1
        else: score -= 1

    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1
        else: score -= 1
        
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1
        elif obi < -0.05: score -= 1
    
    if filters.get('use_tsi', False): 
        max_score += 1
        if row['TSI'] > 0: score += 1
        else: score -= 1
    
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    if filters['use_regime'] and row['ADX_14'] < 20: signal = "NEUTRO"
        
    if filters['use_rsi']:
        if row['RSI'] > 70 and signal == "LONG": signal = "NEUTRO"
        if row['RSI'] < 30 and signal == "SHORT": signal = "NEUTRO"

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    return signal, row['ATR'], prob

# -----------------------------------------------------------------------------
# 6. EJECUCIÓN
# -----------------------------------------------------------------------------
def save_trades(df): df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr, leverage):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": symbol, "type": type, "entry": entry, "size": size, "leverage": leverage, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": "Entry", "candles_held": 0, "atr_entry": atr}
    df = pd.concat([pd.DataFrame([new]), df], ignore_index=True)
    save_trades(df)
    return new

def manage_open_positions(current_price):
    df = load_trades()
    if df.empty: return
    open_idx = df.index[df['status'] == "OPEN"].tolist()
    updated = False
    for idx in open_idx:
        row = df.loc[idx]
        close_reason, pnl = "", 0
        if row['type'] == "LONG":
            if use_trailing:
                new_sl = current_price - (row['atr_entry'] * 1.5)
                if new_sl > row['sl']: df.at[idx, 'sl'] = new_sl
            if current_price >= row['tp3']: close_reason, pnl = "TP3 🚀", (row['tp3'] - row['entry']) * row['size']
            elif current_price <= row['sl']: close_reason, pnl = "SL 🛑", (row['sl'] - row['entry']) * row['size']
        else:
            if use_trailing:
                new_sl = current_price + (row['atr_entry'] * 1.5)
                if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
            if current_price <= row['tp3']: close_reason, pnl = "TP3 🚀", (row['entry'] - row['tp3']) * row['size']
            elif current_price >= row['sl']: close_reason, pnl = "SL 🛑", (row['entry'] - row['sl']) * row['size']

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            updated = True
    if updated: save_trades(df)

def render_analytics(df_trades):
    if df_trades.empty:
        st.info("Esperando operaciones para generar gráficos.")
        return
    closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if closed.empty:
        st.info("Aún no has cerrado ninguna operación.")
        return
    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = INITIAL_CAPITAL + closed['cumulative_pnl']
    fig = px.area(closed, x='time', y='equity', title="Curva de Capital (Equity Curve)")
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi, 'use_tsi': use_tsi}
    signal, atr, prob = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    
    # DATOS FIX (BYBIT)
    fr, open_interest, data_src = get_deriv_data(symbol)
    mtf_trends, mtf_score = get_mtf_trends_analysis(symbol)
    
    # UI NATIVA
    tab1, tab2 = st.tabs(["📊 COMANDO CENTRAL", "🧪 PAPER TRADING"])
    
    with tab1:
        render_native_ai_analysis(df.iloc[-1], mtf_trends, mtf_score, obi, fr, open_interest, data_src, signal, prob)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Funding", f"{fr:.4f}%")
        c3.metric("MTF", f"{mtf_trends['15m']} | {mtf_trends['1h']} | {mtf_trends['4h']}")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        setup = None
        if signal != "NEUTRO":
            sl_dist = atr * 1.5
            risk_amount = current_balance * (risk_per_trade / 100)
            qty = risk_amount / sl_dist if sl_dist > 0 else 0
            leverage = max(1.0, (qty * current_price) / current_balance)
            
            if signal == "LONG":
                sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+sl_dist, current_price+(sl_dist*2), current_price+(sl_dist*3.5)
                emoji = "⬆️ LONG"
            else:
                sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-sl_dist, current_price-(sl_dist*2), current_price-(sl_dist*3.5)
                emoji = "⬇️ SHORT"
            
            setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'qty': qty, 'lev': leverage}
            
            st.info(f"🔥 SEÑAL {setup['dir']} DETECTADA | Prob: {prob:.1f}% | Lev: {setup['lev']:.1f}x")
            if st.button("EJECUTAR OPERACIÓN"):
                execute_trade(signal, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr, setup['lev'])
                st.success("Orden Ejecutada")

    with tab2:
        df_trades = load_trades()
        render_analytics(df_trades)
        st.dataframe(df_trades, use_container_width=True)
        
    manage_open_positions(current_price)

else: st.warning("Cargando datos...")

if auto_refresh: time.sleep(60); st.rerun()
