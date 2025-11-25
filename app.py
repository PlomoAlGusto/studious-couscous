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
# 1. CONFIGURACIÓN ESTRUCTURAL & CSS CYBERPUNK
# -----------------------------------------------------------------------------
st.set_page_config(page_title="QUIMERA v14 NEON", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    /* IMPORTAR FUENTES FUTURISTAS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;500;700&display=swap');

    /* FONDO GENERAL */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #111 0%, #000 100%);
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }

    /* TITULOS */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #fff;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }

    /* TARJETAS HUD (GLASSMORPHISM) */
    .hud-card {
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(0, 255, 255, 0.1);
        border-radius: 4px;
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .hud-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 2px; height: 100%;
        background: linear-gradient(180deg, #00ffff, transparent);
    }

    /* METRICAS SUPERIORES */
    .metric-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: 700; color: #fff; font-family: 'Orbitron'; }
    .metric-sub { font-size: 11px; font-weight: bold; }
    .neon-cyan { color: #00ffff; text-shadow: 0 0 5px rgba(0,255,255,0.5); }
    .neon-green { color: #39ff14; text-shadow: 0 0 5px rgba(57,255,20,0.5); }
    .neon-red { color: #ff0055; text-shadow: 0 0 5px rgba(255,0,85,0.5); }

    /* RELOJES DE MERCADO */
    .market-clock {
        font-family: 'Roboto Mono', monospace;
        font-size: 11px;
        padding: 8px;
        margin-bottom: 6px;
        border-radius: 2px;
        display: flex; 
        justify-content: space-between;
        background: #0a0a0a;
        border: 1px solid #222;
    }
    .open-mkt { border-left: 3px solid #39ff14; }
    .closed-mkt { border-left: 3px solid #333; color: #555; }

    /* NOTICIAS */
    .news-container {
        height: 220px; overflow-y: auto;
        scrollbar-width: thin; scrollbar-color: #333 #111;
    }
    .news-item { 
        padding: 10px 0; 
        border-bottom: 1px solid #222; 
        font-size: 12px; 
        transition: background 0.2s;
    }
    .news-item:hover { background: rgba(255,255,255,0.02); }
    .news-link { color: #aaa; text-decoration: none; }
    .news-link:hover { color: #00ffff; }

    /* TARJETA DE SEÑAL */
    .signal-box {
        background: radial-gradient(circle at center, #1a1a1a 0%, #000 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(0,0,0,0.8);
        position: relative;
    }
    .signal-header { font-family: 'Orbitron'; font-size: 22px; font-weight: 900; margin-bottom: 10px; text-transform: uppercase; }
    .border-long { border: 1px solid #39ff14; box-shadow: 0 0 10px rgba(57,255,20,0.2); }
    .border-short { border: 1px solid #ff0055; box-shadow: 0 0 10px rgba(255,0,85,0.2); }
    .border-wait { border: 1px solid #ffff00; box-shadow: 0 0 10px rgba(255,255,0,0.2); }

    /* BADGES INDICADORES */
    .badge { padding: 2px 6px; border-radius: 2px; font-size: 9px; font-weight: bold; margin-right: 3px; border: 1px solid; }
    .b-bull { color: #39ff14; border-color: #39ff14; background: rgba(57,255,20,0.1); }
    .b-bear { color: #ff0055; border-color: #ff0055; background: rgba(255,0,85,0.1); }
    .b-neut { color: #555; border-color: #555; background: rgba(255,255,255,0.05); }

    /* BOTONES */
    div.stButton > button {
        background: transparent !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
        font-family: 'Orbitron';
        border-radius: 0px !important;
        transition: all 0.3s !important;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background: #00ffff !important;
        color: #000 !important;
        box-shadow: 0 0 15px #00ffff;
    }
    
    /* INPUTS */
    div[data-baseweb="input"] {
        background-color: #111 !important;
        border: 1px solid #333 !important;
        color: white !important;
    }
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
# 2. LOGICA (Igual a v13.9)
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
    sessions = {"🇬🇧 LONDON": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("<h3 style='font-size:14px; color:#666;'>SYSTEM STATUS</h3>", unsafe_allow_html=True)
    for name, (start, end) in sessions.items():
        is_open = False
        if start < end: is_open = start <= hour < end
        else: is_open = hour >= start or hour < end
        status_txt = "ONLINE" if is_open else "OFFLINE"
        css_class = "open-mkt" if is_open else "closed-mkt"
        st.sidebar.markdown(f"<div class='market-clock {css_class}'><span>{name}</span><span>{status_txt}</span></div>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("<h1 style='color:#00ffff; text-align:center;'>QUIMERA<br><span style='font-size:12px; color:#fff;'>PRO v14.0</span></h1>", unsafe_allow_html=True)
    st.divider()
    get_market_sessions()
    
    st.markdown("### 📡 DATA FEED")
    symbol = st.text_input("ASSET", "BTC/USDT")
    tf = st.selectbox("TIMEFRAME", ["15m", "1h"], index=0)
    
    with st.expander("🛠️ STRATEGY CORE", expanded=True):
        use_ema = st.checkbox("EMA Trend", True)
        use_mtf = st.checkbox("Macro 4H", True)
        use_vwap = st.checkbox("VWAP Inst.", True)
        use_ichi = st.checkbox("Ichimoku Cloud", False)
        use_regime = st.checkbox("ADX Filter", True)
        use_rsi = st.checkbox("RSI Momentum", True)
        use_obi = st.checkbox("OrderBook Imb.", True)

    with st.expander("💸 RISK MANAGEMENT"):
        current_balance = get_current_balance()
        # Visualización de Balance en Sidebar
        st.markdown(f"""
        <div style='background:#111; padding:10px; border-radius:4px; border:1px solid #333; text-align:center;'>
            <div style='font-size:10px; color:#888;'>NET LIQUIDITY</div>
            <div style='font-family:"Orbitron"; font-size:18px; color:#fff;'>${current_balance:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
        
    with st.expander("⚡ EXECUTION"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Bars)", True)
        
    auto_refresh = st.checkbox("♻️ AUTO-SYNC (60s)", False)
    
    if st.button("🔥 HARD RESET"):
        reset_account()

# -----------------------------------------------------------------------------
# 3. BACKEND (Igual a v13.9)
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            ex.load_markets(); return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Visual)"

exchange, source_name = init_exchange()

@st.cache_data(ttl=3600) 
def get_fear_and_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/")
        data = r.json()['data'][0]
        return int(data['value']), data['value_classification']
    except: return 50, "Neutral"

@st.cache_data(ttl=300)
def get_crypto_news():
    rss_url = "https://cointelegraph.com/rss"
    try:
        feed = feedparser.parse(rss_url)
        return [{"title": e.title, "link": e.link, "published": e.published_parsed} for e in feed.entries[:5]]
    except: return []

@st.cache_data(ttl=15)
def get_mtf_data(symbol, tf_lower):
    if not exchange: return None, 0, None
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None

    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
        last_4h = df_4h.iloc[-1]
        if last_4h['close'] > last_4h['EMA_50']: trend_4h = "BULLISH"
        else: trend_4h = "BEARISH"
    except: pass

    obi = 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids, asks = sum([x[1] for x in book['bids']]), sum([x[1] for x in book['asks']])
        if (bids + asks) > 0: obi = (bids - asks) / (bids + asks)
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
    high_w, low_w, close_w = df['high'].rolling(20).max(), df['low'].rolling(20).min(), df['close']
    df['PIVOT'] = (high_w + low_w + close_w) / 3
    df['R1'] = (2 * df['PIVOT']) - low_w
    df['S1'] = (2 * df['PIVOT']) - high_w
    return df.fillna(method='bfill').fillna(method='ffill')

def detect_candle_patterns(row, prev_row):
    patterns = []
    body_size = abs(row['close'] - row['open'])
    full_range = row['high'] - row['low']
    if full_range > 0 and (body_size / full_range) < 0.1: patterns.append("Doji")
    lower_wick = min(row['close'], row['open']) - row['low']
    upper_wick = row['high'] - max(row['close'], row['open'])
    if lower_wick > (body_size * 2) and upper_wick < body_size: patterns.append("Hammer")
    if row['close'] > row['open'] and prev_row['close'] < prev_row['open']:
        if row['close'] > prev_row['open'] and row['open'] < prev_row['close']: patterns.append("Engulfing")
    return patterns

def generate_ai_analysis(row, prev_row, trend_4h, obi, signal, prob, fng_val, fng_class):
    analysis = []
    if trend_4h == "BULLISH": analysis.append("MACRO: <span class='neon-green'>BULLISH</span>")
    elif trend_4h == "BEARISH": analysis.append("MACRO: <span class='neon-red'>BEARISH</span>")
    
    mfi = row['MFI']
    if mfi > 60: analysis.append("VOL FLOW: POSITIVE")
    elif mfi < 40: analysis.append("VOL FLOW: NEGATIVE")
    
    patterns = detect_candle_patterns(row, prev_row)
    if patterns: analysis.append(f"PATTERN: {', '.join(patterns)}")
    
    return " // ".join(analysis)

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
    score = 0
    max_score = 0
    details = [] 
    
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": 
            score += 2; details.append("<span class='badge b-bull'>MACRO</span>")
        elif trend_4h == "BEARISH": 
            score -= 2; details.append("<span class='badge b-bear'>MACRO</span>")
        else: details.append("<span class='badge b-neut'>MACRO</span>")

    if filters['use_ema']: 
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: 
            score += 1; details.append("<span class='badge b-bull'>EMA</span>")
        else: 
            score -= 1; details.append("<span class='badge b-bear'>EMA</span>")

    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: 
            score += 1; details.append("<span class='badge b-bull'>VWAP</span>")
        else: 
            score -= 1; details.append("<span class='badge b-bear'>VWAP</span>")
        
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: 
            score += 1; details.append("<span class='badge b-bull'>OBI</span>")
        elif obi < -0.05: 
            score -= 1; details.append("<span class='badge b-bear'>OBI</span>")
        else: details.append("<span class='badge b-neut'>OBI</span>")
    
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    if filters['use_rsi'] and (row['RSI'] > 70 and signal == "LONG"): 
        signal = "NEUTRO"; details.append("<span class='badge b-neut'>RSI-MAX</span>")
    if filters['use_rsi'] and (row['RSI'] < 30 and signal == "SHORT"): 
        signal = "NEUTRO"; details.append("<span class='badge b-neut'>RSI-MIN</span>")
    
    if filters['use_mtf'] and signal == "LONG" and trend_4h == "BEARISH": signal = "NEUTRO"
    if filters['use_mtf'] and signal == "SHORT" and trend_4h == "BULLISH": signal = "NEUTRO"

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    thermometer_score = 0
    if max_score > 0: thermometer_score = (score / max_score) * 100
    
    return signal, row['ATR'], prob, thermometer_score, details

def save_trades(df):
    df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr, leverage):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": symbol, "type": type, "entry": entry, "size": size, "leverage": leverage, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": "Entry", "candles_held": 0, "atr_entry": atr}
    df = pd.concat([pd.DataFrame([new]), df], ignore_index=True)
    save_trades(df)
    return new

def manage_open_positions(current_price, current_high, current_low):
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
            if use_breakeven and current_high >= row['tp1'] and row['sl'] < row['entry']:
                df.at[idx, 'sl'] = row['entry'] * 1.001 
            if current_high >= row['tp3']: close_reason, pnl = "TP3 (Final)", (row['tp3'] - row['entry']) * row['size']
            elif current_low <= row['sl']: close_reason, pnl = "SL", (row['sl'] - row['entry']) * row['size']
        else:
            if use_trailing:
                new_sl = current_price + (row['atr_entry'] * 1.5)
                if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
            if use_breakeven and current_low <= row['tp1'] and row['sl'] > row['entry']:
                df.at[idx, 'sl'] = row['entry'] * 0.999 
            if current_low <= row['tp3']: close_reason, pnl = "TP3 (Final)", (row['entry'] - row['tp3']) * row['size']
            elif current_high >= row['sl']: close_reason, pnl = "SL", (row['entry'] - row['sl']) * row['size']

        if not close_reason and use_time_stop:
            df.at[idx, 'candles_held'] += 1
            current_pnl_calc = (current_price - row['entry']) * row['size'] if row['type'] == "LONG" else (row['entry'] - current_price) * row['size']
            if df.at[idx, 'candles_held'] > 12 and current_pnl_calc < 0:
                close_reason, pnl = "Time Stop", current_pnl_calc

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            send_telegram_msg(f"💰 CLOSE {symbol}: {close_reason}\nPnL: ${pnl:.2f}")
            updated = True
    if updated or use_time_stop: save_trades(df)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

# -----------------------------------------------------------------------------
# 4. FRONTEND DASHBOARD (CYBERPUNK UI)
# -----------------------------------------------------------------------------
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi}
    signal, atr, prob, thermo_score, details_list = run_strategy(df, obi, trend_4h, filters)
    current_price, cur_high, cur_low = df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    mfi_val, adx_val = df['MFI'].iloc[-1], df['ADX_14'].iloc[-1]
    
    fng_val, fng_label = get_fear_and_greed()
    news = get_crypto_news()
    ai_narrative = generate_ai_analysis(df.iloc[-1], df.iloc[-2], trend_4h, obi, signal, prob, fng_val, fng_label)
    
    setup = None
    calc_dir = signal 
    setup_type = "CONFIRMED" if signal != "NEUTRO" else "POTENTIAL"
    
    if signal == "NEUTRO":
        if trend_4h == "BULLISH": calc_dir = "LONG"
        elif trend_4h == "BEARISH": calc_dir = "SHORT"
        else: calc_dir = None

    qty, leverage = 0, 1.0
    current_balance = get_current_balance()
    
    if calc_dir:
        sl_dist = atr * 1.5
        risk = sl_dist
        risk_amount = current_balance * (risk_per_trade / 100)
        qty = risk_amount / risk if risk > 0 else 0
        notional_value = qty * current_price
        leverage = notional_value / current_balance
        if leverage < 1: leverage = 1.0
        
        if calc_dir == "LONG":
            sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            emoji = "⬆️ LONG"
            sig_class = "border-long"
        else:
            sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            emoji = "⬇️ SHORT"
            sig_class = "border-short"
            
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'status': setup_type, 'qty': qty, 'lev': leverage, 'class': sig_class}
    else:
        setup = {'class': 'border-wait', 'dir': 'WAIT', 'qty': 0, 'lev': 1.0}

    if signal != "NEUTRO" and signal != st.session_state.last_alert and setup:
        msg = f"""🦁 *QUIMERA SIGNAL*
📉 {symbol} | {setup['dir']}
📊 Prob: {prob:.1f}% | Lev: {setup['lev']:.1f}x
🔵 Entry: ${setup['entry']:.2f}
🛑 SL: ${setup['sl']:.2f}
🎯 TP1: ${setup['tp1']:.2f}
"""
        send_telegram_msg(msg)
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_open_positions(current_price, cur_high, cur_low)
    
    # --- METRICS ROW (HUD STYLE) ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='hud-card'>
            <div class='metric-label'>CURRENT PRICE</div>
            <div class='metric-value neon-cyan'>${current_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        trend_col = "neon-green" if trend_4h == "BULLISH" else "neon-red"
        st.markdown(f"""
        <div class='hud-card'>
            <div class='metric-label'>4H TREND</div>
            <div class='metric-value {trend_col}'>{trend_4h}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        gas_col = "neon-green" if mfi_val > 60 else "neon-red" if mfi_val < 40 else "#888"
        st.markdown(f"""
        <div class='hud-card'>
            <div class='metric-label'>GAS (MFI)</div>
            <div class='metric-value {gas_col}'>{mfi_val:.0f} <span style='font-size:12px'>/ 100</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        adx_col = "neon-cyan" if adx_val > 25 else "#555"
        st.markdown(f"""
        <div class='hud-card'>
            <div class='metric-label'>ADX POWER</div>
            <div class='metric-value {adx_col}'>{adx_val:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    tab1, tab2 = st.tabs(["⚡ LIVE COMMAND", "🧪 PAPER LAB"])
    
    with tab1:
        # GRID LAYOUT: News | Sentiment | AI
        col_left, col_mid, col_right = st.columns([1.2, 1, 1])
        
        with col_left:
            st.markdown("<div class='hud-card' style='height:250px;'><div class='metric-label' style='margin-bottom:10px;'>NEWS FEED</div><div class='news-container'>", unsafe_allow_html=True)
            if news:
                for n in news:
                    t = n.get('published', time.gmtime())
                    st.markdown(f"<div class='news-item'><span style='color:#00ffff'>{t.tm_hour:02}:{t.tm_min:02}</span> <a href='{n['link']}' target='_blank' class='news-link'>{n['title'][:60]}...</a></div>", unsafe_allow_html=True)
            else: st.info("No data.")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with col_mid:
             # GAUGE BOT
            fig_thermo = go.Figure(go.Indicator(
                mode = "gauge+number", value = thermo_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<span style='color:#ccc; font-size:12px;'>SYSTEM CONFIDENCE</span>"},
                gauge = {'axis': {'range': [-100, 100]}, 'bar': {'color': "white"}, 'bgcolor': "rgba(0,0,0,0)",
                    'steps': [{'range': [-100, -40], 'color': "rgba(255,0,85,0.4)"}, {'range': [-40, 40], 'color': "rgba(100,100,100,0.2)"}, {'range': [40, 100], 'color': "rgba(57,255,20,0.4)"}]}
            ))
            fig_thermo.update_layout(height=180, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Orbitron"})
            st.plotly_chart(fig_thermo, use_container_width=True)
            st.markdown(f"<div style='text-align:center; margin-top:-20px;'>{' '.join(details_list)}</div>", unsafe_allow_html=True)

        with col_right:
            # GAUGE FNG
            fig_fng = go.Figure(go.Indicator(
                mode = "gauge+number", value = fng_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<span style='color:#ccc; font-size:12px;'>FEAR & GREED</span>"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'bgcolor': "rgba(0,0,0,0)",
                    'steps': [{'range': [0, 40], 'color': "rgba(255,0,85,0.4)"}, {'range': [40, 60], 'color': "rgba(255,255,0,0.4)"}, {'range': [60, 100], 'color': "rgba(57,255,20,0.4)"}]}
            ))
            fig_fng.update_layout(height=180, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Orbitron"})
            st.plotly_chart(fig_fng, use_container_width=True)

        st.markdown(f"<div class='hud-card' style='margin-top:15px; border-left: 3px solid #00ffff;'><b style='color:#00ffff'>AI COPILOT:</b> {ai_narrative}</div>", unsafe_allow_html=True)

        # SIGNAL CARD
        if setup and calc_dir:
            st.markdown(f"""
            <div class='signal-box {setup['class']}'>
                <div class='signal-header' style='color:{'#39ff14' if calc_dir=='LONG' else '#ff0055'}'>{setup['dir']} SIGNAL DETECTED</div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px; font-size:12px; color:#888;'>
                    <span>CONFIDENCE: <b style='color:#fff'>{prob:.1f}%</b></span>
                    <span>LEVERAGE: <b style='color:#00ffff'>{setup['lev']:.1f}x</b></span>
                </div>
                <div style='background:#111; height:6px; border-radius:3px; overflow:hidden; margin-bottom:15px;'>
                    <div style='width:{prob}%; height:100%; background: {'#39ff14' if prob>60 else '#ff0055'}; box-shadow:0 0 10px {'#39ff14' if prob>60 else '#ff0055'};'></div>
                </div>
                <div style='display:flex; justify-content:space-around; text-align:center;'>
                     <div><div style='font-size:10px; color:#888;'>ENTRY</div><div style='color:#00ffff; font-weight:bold;'>${setup['entry']:.2f}</div></div>
                     <div><div style='font-size:10px; color:#888;'>STOP</div><div style='color:#ff0055; font-weight:bold;'>${setup['sl']:.2f}</div></div>
                     <div><div style='font-size:10px; color:#888;'>TARGET 1</div><div style='color:#39ff14; font-weight:bold;'>${setup['tp1']:.2f}</div></div>
                     <div><div style='font-size:10px; color:#888;'>TARGET 3</div><div style='color:#39ff14; font-weight:bold;'>${setup['tp3']:.2f}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"🚀 EXECUTE {calc_dir} ({setup['qty']:.4f})"):
                 execute_trade(calc_dir, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr, setup['lev'])
                 st.success("ORDER SENT TO EXECUTION ENGINE")

        # CHART
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        # Velas
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        # VWAP
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='#ff00ff', width=1, dash='dot'), name='VWAP'), row=1, col=1)
        # Levels
        fig.add_hline(y=setup['entry'] if setup else current_price, line_color="#555", line_width=1, row=1, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='#00ffff', width=1), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#555", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#555", row=2, col=1)
        
        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            height=600, 
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis_rangeslider_visible=False,
            font=dict(family="Roboto Mono", size=10, color="#888")
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#222')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_stats = load_trades()
        total_pnl = 0.0
        win_rate = 0.0
        
        if not df_stats.empty:
            closed = df_stats[df_stats['status']=='CLOSED']
            if not closed.empty:
                total_pnl = closed['pnl'].sum()
                wins = len(closed[closed['pnl']>0])
                win_rate = (wins/len(closed))*100
        
        # STATS BAR
        st.markdown(f"""
        <div class='stats-bar'>
            <div class='stat-item'><div class='stat-label'>NET PNL</div><div class='stat-value' style='color:{'#39ff14' if total_pnl>=0 else '#ff0055'}'>${total_pnl:.2f}</div></div>
            <div class='stat-item'><div class='stat-label'>WIN RATE</div><div class='stat-value'>{win_rate:.1f}%</div></div>
            <div class='stat-item'><div class='stat-label'>TRADES</div><div class='stat-value'>{len(df_stats)}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        if not df_stats.empty:
            # Open Positions
            open_ops = df_stats[df_stats['status']=='OPEN'].copy()
            if not open_ops.empty:
                st.markdown("### 🟢 ACTIVE POSITIONS")
                open_ops['Floating PnL'] = np.where(open_ops['type'] == 'LONG', 
                                                   (current_price - open_ops['entry']) * open_ops['size'], 
                                                   (open_ops['entry'] - current_price) * open_ops['size'])
                st.dataframe(open_ops[['time','symbol','type','leverage','entry','pnl','Floating PnL']], use_container_width=True)
            
            # History
            st.markdown("### 📜 TRADE LOG")
            closed_ops = df_stats[df_stats['status']=='CLOSED']
            if not closed_ops.empty:
                st.dataframe(closed_ops, use_container_width=True)
        else:
            st.info("NO TRADES RECORDED IN DATABASE")

else: st.warning("INITIALIZING NEURAL LINK...")

if auto_refresh: time.sleep(60); st.rerun()
