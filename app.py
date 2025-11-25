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
import base64

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v14.1 Arsenal", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    .label-mini { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    
    .header-confirmed-long { color: #00FF00; font-size: 20px; font-weight: 900; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .header-confirmed-short { color: #FF4444; font-size: 20px; font-weight: 900; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .header-potential { color: #FFFF00; font-size: 18px; font-weight: bold; border-bottom: 1px dashed #555; padding-bottom: 10px; }
    
    .ai-box { background-color: #223344; border-left: 5px solid #44AAFF; padding: 15px; border-radius: 5px; margin-bottom: 15px; font-family: monospace; }
    .news-box { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 10px; margin-bottom: 15px; height: 200px; overflow-y: auto; }
    .news-item { padding: 8px 0; border-bottom: 1px solid #222; font-size: 13px; }
    .news-link { text-decoration: none; color: #ddd; }
    .news-link:hover { color: #44AAFF; }
    
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
    
    .badge-bull { background-color: #004400; color: #00FF00; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #00FF00; margin-right: 4px; }
    .badge-bear { background-color: #440000; color: #FF4444; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #FF4444; margin-right: 4px; }
    .badge-neutral { background-color: #333; color: #aaa; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #555; margin-right: 4px; }
    
    .stats-bar { background-color: #1E1E1E; border: 1px solid #333; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: flex; justify-content: space-around; align-items: center; }
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
# 2. FUNCIONES DE DATOS AVANZADOS
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
        is_open = False
        if start < end: is_open = start <= hour < end
        else: is_open = hour >= start or hour < end
        status_icon = "🟢 ABIERTO" if is_open else "🔴 CERRADO"
        css_class = "clock-open" if is_open else "clock-closed"
        st.sidebar.markdown(f"<div class='market-clock {css_class}'><span>{name}</span><span>{status_icon}</span></div>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_funding_rate(symbol):
    """Obtiene Funding Rate de Binance Futuros (API Pública)"""
    try:
        clean_symbol = symbol.replace("/", "")
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={clean_symbol}"
        r = requests.get(url, timeout=2).json()
        fr = float(r['lastFundingRate']) * 100
        return fr
    except: return 0.0

@st.cache_data(ttl=300)
def get_correlation_matrix():
    """Obtiene correlación de 4 monedas principales"""
    tickers = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    data = {}
    ex = ccxt.binance()
    for t in tickers:
        try:
            ohlcv = ex.fetch_ohlcv(t, '1h', limit=48)
            df = pd.DataFrame(ohlcv, columns=['time','o','h','l','c','v'])
            data[t.split('/')[0]] = df['c']
        except: pass
    return pd.DataFrame(data).corr() if data else None

@st.cache_data(ttl=15)
def get_mtf_trend(symbol):
    """Semáforo Multi-Timeframe"""
    ex = ccxt.binance()
    trends = {}
    for tf in ['15m', '1h', '4h']:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            ema = ta.ema(df['c'], length=50).iloc[-1]
            close = df['c'].iloc[-1]
            trends[tf] = "🟢" if close > ema else "🔴"
        except: trends[tf] = "⚪"
    return trends

def play_sound():
    """Reproduce un sonido de alerta invisible"""
    sound_url = "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"
    st.markdown(f"""<audio autoplay style="display:none;"><source src="{sound_url}" type="audio/mp3"></audio>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. INTERFAZ SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v14.1")
    st.caption("Full Arsenal Edition ⚔️")
    get_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)
    
    # --- CORRECCIÓN AQUÍ: Agregados use_rsi y use_obi ---
    with st.expander("🛡️ FILTROS & CORE", expanded=True):
        use_ema = st.checkbox("Tendencia Base (EMAs)", True)
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP (Institucional)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)
        use_rsi = st.checkbox("Filtro RSI (Sobrecompra/Venta)", False) # Agregado
        use_obi = st.checkbox("Order Book Imbalance (OBI)", True)      # Agregado
    
    with st.expander("📈 CHART OVERLAYS"):
        show_fib = st.checkbox("Auto Fibonacci", True)
        show_vpvr = st.checkbox("Volume Profile (VPVR)", True)
        
    with st.expander("💰 GESTIÓN DE RIESGO"):
        current_balance = get_current_balance()
        st.metric("Balance Disponible", f"${current_balance:,.2f}", delta=f"{current_balance-INITIAL_CAPITAL:.2f}")
        risk_per_trade = st.slider("Riesgo por Trade (%)", 0.5, 5.0, 1.0)
        
    with st.expander("⚙️ SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
    if st.button("🔥 RESETEAR CUENTA"): reset_account()

# -----------------------------------------------------------------------------
# 4. CAPA DE DATOS & INDICADORES
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
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        return [{"title": e.title, "link": e.link, "published": e.published_parsed} for e in feed.entries[:5]]
    except: return []

@st.cache_data(ttl=15)
def get_mtf_data(symbol, tf_lower):
    if not exchange: return None, 0, None, None
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    
    # Main Data
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None, None

    # 4H Trend
    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
        last_4h = df_4h.iloc[-1]
        trend_4h = "BULLISH" if last_4h['close'] > last_4h['EMA_50'] else "BEARISH"
    except: pass

    # Order Book Info
    bids, asks = 0, 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids = sum([x[1] for x in book['bids']])
        asks = sum([x[1] for x in book['asks']])
        obi = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
    except: obi = 0
    
    return df, obi, trend_4h, (bids, asks)

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
    
    # Pivots
    high_w, low_w, close_w = df['high'].rolling(20).max(), df['low'].rolling(20).min(), df['close']
    df['PIVOT'] = (high_w + low_w + close_w) / 3
    df['R1'], df['S1'] = (2 * df['PIVOT']) - low_w, (2 * df['PIVOT']) - high_w
    return df.fillna(method='bfill').fillna(method='ffill')

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
    score, max_score, details = 0, 0, []
    
    # 1. Macro Trend
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2; details.append("<span class='badge-bull'>MACRO</span>")
        elif trend_4h == "BEARISH": score -= 2; details.append("<span class='badge-bear'>MACRO</span>")
        else: details.append("<span class='badge-neutral'>MACRO</span>")

    # 2. Local Trend (EMA)
    if filters['use_ema']: 
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; details.append("<span class='badge-bull'>EMA</span>")
        else: score -= 1; details.append("<span class='badge-bear'>EMA</span>")

    # 3. VWAP
    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; details.append("<span class='badge-bull'>VWAP</span>")
        else: score -= 1; details.append("<span class='badge-bear'>VWAP</span>")
        
    # 4. Order Book (USANDO VARIABLE RESTAURADA)
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1; details.append("<span class='badge-bull'>OBI</span>")
        elif obi < -0.05: score -= 1; details.append("<span class='badge-bear'>OBI</span>")
        else: details.append("<span class='badge-neutral'>OBI</span>")
    
    # Decision
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    # Invalidations
    if filters['use_regime'] and row['ADX_14'] < 20: 
        signal = "NEUTRO"; details.append("<span class='badge-neutral'>ADX-LOW</span>")
        
    # RSI FILTER (USANDO VARIABLE RESTAURADA)
    if filters['use_rsi']:
        if row['RSI'] > 70 and signal == "LONG": 
            signal = "NEUTRO"; details.append("<span class='badge-neutral'>RSI-MAX</span>")
        if row['RSI'] < 30 and signal == "SHORT": 
            signal = "NEUTRO"; details.append("<span class='badge-neutral'>RSI-MIN</span>")

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    thermo_score = (score / max_score) * 100 if max_score > 0 else 0
    
    return signal, row['ATR'], prob, thermo_score, details

# -----------------------------------------------------------------------------
# 5. LÓGICA DE EJECUCIÓN & VISUALIZACIÓN
# -----------------------------------------------------------------------------
def save_trades(df): df.to_csv(CSV_FILE, index=False)

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
            if use_breakeven and current_high >= row['tp1'] and row['sl'] < row['entry']: df.at[idx, 'sl'] = row['entry'] * 1.001 
            if current_high >= row['tp3']: close_reason, pnl = "TP3 (Final) 🚀", (row['tp3'] - row['entry']) * row['size']
            elif current_low <= row['sl']: close_reason, pnl = "SL 🛑", (row['sl'] - row['entry']) * row['size']
        else:
            if use_trailing:
                new_sl = current_price + (row['atr_entry'] * 1.5)
                if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
            if use_breakeven and current_low <= row['tp1'] and row['sl'] > row['entry']: df.at[idx, 'sl'] = row['entry'] * 0.999 
            if current_low <= row['tp3']: close_reason, pnl = "TP3 (Final) 🚀", (row['entry'] - row['tp3']) * row['size']
            elif current_high >= row['sl']: close_reason, pnl = "SL 🛑", (row['entry'] - row['sl']) * row['size']

        if not close_reason and use_time_stop:
            df.at[idx, 'candles_held'] += 1
            current_pnl_calc = (current_price - row['entry']) * row['size'] if row['type'] == "LONG" else (row['entry'] - current_price) * row['size']
            if df.at[idx, 'candles_held'] > 12 and current_pnl_calc < 0: close_reason, pnl = "Time Stop ⏳", current_pnl_calc

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            updated = True
    if updated or use_time_stop: save_trades(df)

# -----------------------------------------------------------------------------
# 6. DASHBOARD
# -----------------------------------------------------------------------------
df, obi, trend_4h, ob_data = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    # DICCIONARIO DE FILTROS ACTUALIZADO
    filters = {
        'use_mtf': use_mtf, 
        'use_ema': use_ema, 
        'use_vwap': use_vwap, 
        'use_ichi': use_ichi, 
        'use_regime': use_regime, 
        'use_rsi': use_rsi, 
        'use_obi': use_obi
    }
    signal, atr, prob, thermo_score, details_list = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    
    # Extra Data
    fng_val, fng_label = get_fear_and_greed()
    fr = get_funding_rate(symbol)
    corr_matrix = get_correlation_matrix()
    mtf_trends = get_mtf_trend(symbol)
    
    # Seasonality
    df['hour'] = df['timestamp'].dt.hour
    seasonality = df.groupby('hour')['close'].apply(lambda x: x.iloc[-1]/x.iloc[0] - 1).mean()
    
    setup = None
    calc_dir = signal if signal != "NEUTRO" else None
    
    # Auto-Entry Setup
    if signal == "NEUTRO":
        if trend_4h == "BULLISH": calc_dir = "LONG"
        elif trend_4h == "BEARISH": calc_dir = "SHORT"
    
    qty, leverage = 0, 1.0
    current_balance = get_current_balance()
    
    if calc_dir:
        sl_dist = atr * 1.5
        risk = sl_dist
        risk_amount = current_balance * (risk_per_trade / 100)
        qty = risk_amount / risk if risk > 0 else 0
        leverage = max(1.0, (qty * current_price) / current_balance)
        
        if calc_dir == "LONG":
            sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            emoji = "⬆️ LONG"
        else:
            sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            emoji = "⬇️ SHORT"
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'status': "CONFIRMED" if signal!="NEUTRO" else "POTENTIAL", 'qty': qty, 'lev': leverage}

    # Signal Alert
    if signal != "NEUTRO" and signal != st.session_state.last_alert and setup:
        play_sound()
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_open_positions(current_price, df['high'].iloc[-1], df['low'].iloc[-1])
    
    # --- UI LAYOUT ---
    tab1, tab2 = st.tabs(["📊 COMANDO CENTRAL", "🧪 PAPER TRADING"])
    
    with tab1:
        # 1. Market Context Bar
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Funding Rate", f"{fr:.4f}%", delta_color="inverse")
        c3.metric("MTF Trend", f"{mtf_trends['15m']} {mtf_trends['1h']} {mtf_trends['4h']}")
        c4.metric("ADX Fuerza", f"{df['ADX_14'].iloc[-1]:.1f}")
        liq_target = current_price * 1.05 if fr < 0 else current_price * 0.95
        c5.metric("Liq. Target Est.", f"${liq_target:,.0f}")

        # 2. Main Dashboard
        col_chart, col_data = st.columns([2.5, 1])
        
        with col_chart:
            # CHART PRINCIPAL
            fig = make_subplots(rows=2, cols=2, shared_xaxes=True, row_heights=[0.7, 0.3], column_widths=[0.85, 0.15], horizontal_spacing=0.01)
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
            
            # Auto-Fibonacci
            if show_fib:
                max_h, min_l = df['high'][-100:].max(), df['low'][-100:].min()
                diff = max_h - min_l
                levels = [0, 0.382, 0.5, 0.618, 1]
                colors = ['gray', 'red', 'yellow', 'green', 'gray']
                for i, l in enumerate(levels):
                    val = max_h - (diff * l)
                    fig.add_hline(y=val, line_width=1, line_color=colors[i], line_dash='dot', row=1, col=1)

            # TP/SL Visuals
            if setup:
                fig.add_hline(y=setup['tp1'], line_color='green', row=1, col=1)
                fig.add_hline(y=setup['sl'], line_color='red', row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color="gray", line_dash="dot", row=2, col=1)
            fig.add_hline(y=30, line_color="gray", line_dash="dot", row=2, col=1)
            
            # VPVR (Simplificado)
            if show_vpvr:
                price_bins = pd.cut(df['close'], bins=30)
                vol_profile = df.groupby(price_bins)['volume'].sum()
                fig.add_trace(go.Bar(y=[str(i.mid) for i in vol_profile.index], x=vol_profile.values, orientation='h', marker_color='rgba(255,255,255,0.2)', name='VPVR'), row=1, col=2)

            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal Box
            if setup:
                prob_str = f"{prob:.1f}%"
                prob_col = "#00FF00" if prob > 65 else "#FFFF00"
                html_card = f"""
                <div class="trade-setup">
                    <div class="{ 'header-confirmed-long' if calc_dir=='LONG' else 'header-confirmed-short' }">🔥 {setup['dir']} CONFIRMED</div>
                    <div style='display:flex; justify-content:space-between; color:#ccc; font-size:12px;'>
                        <span>Prob: <b style='color:{prob_col}'>{prob_str}</b></span>
                        <span>Lev: <b style='color:#00FFFF'>{setup['lev']:.1f}x</b></span>
                    </div>
                    <div style='width: 100%; background: #333; height: 6px; margin: 5px 0;'><div style='width: {prob}%; background: {prob_col}; height: 100%;'></div></div>
                    <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                        <div><span class="label-mini">ENTRY</span><br><span class="entry-blue">${setup['entry']:.2f}</span></div>
                        <div><span class="label-mini">STOP</span><br><span class="sl-red">${setup['sl']:.2f}</span></div>
                        <div><span class="label-mini">TARGET 1</span><br><span class="tp-green">${setup['tp1']:.2f}</span></div>
                        <div><span class="label-mini">TARGET 3</span><br><span class="tp-green">${setup['tp3']:.2f}</span></div>
                    </div>
                </div>
                """
                st.markdown(html_card, unsafe_allow_html=True)
                if st.button(f"🚀 EJECUTAR {calc_dir} ({setup['qty']:.4f})"):
                    execute_trade(calc_dir, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr, setup['lev'])
                    st.success("ORDEN ENVIADA")

        with col_data:
            # 1. Visual Order Book
            if ob_data:
                bids, asks = ob_data
                fig_ob = go.Figure([go.Bar(x=['Bids', 'Asks'], y=[bids, asks], marker_color=['#00FF00', '#FF4444'])])
                fig_ob.update_layout(title="Order Book Balance", height=200, margin=dict(l=20,r=20,t=30,b=10), template="plotly_dark")
                st.plotly_chart(fig_ob, use_container_width=True)
            
            # 2. Correlation Matrix
            if corr_matrix is not None:
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                fig_corr.update_layout(title="Correlaciones", height=250, margin=dict(l=0,r=0,t=30,b=0), template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)

            # 3. Bot Sentiment Gauge
            fig_thermo = go.Figure(go.Indicator(
                mode = "gauge+number", value = thermo_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Bot Sentiment"},
                gauge = {'axis': {'range': [-100, 100]}, 'bar': {'color': "white"}, 'bgcolor': "rgba(0,0,0,0)",
                    'steps': [{'range': [-100, -40], 'color': "#FF4444"}, {'range': [-40, 40], 'color': "#555"}, {'range': [40, 100], 'color': "#00FF00"}]}
            ))
            fig_thermo.update_layout(height=180, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_thermo, use_container_width=True)
            st.markdown(f"<div style='text-align:center; font-size:10px'>{' '.join(details_list)}</div>", unsafe_allow_html=True)

    with tab2:
        df_stats = load_trades()
        total_pnl_val, win_rate, total_closed = 0.0, 0.0, 0
        
        if not df_stats.empty:
            closed_s = df_stats[df_stats['status'] == 'CLOSED']
            total_closed = len(closed_s)
            if total_closed > 0:
                total_pnl_val = closed_s['pnl'].sum()
                wins = len(closed_s[closed_s['pnl'] > 0])
                win_rate = (wins / total_closed) * 100
        
        # Stats Bar
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("PnL Realizado", f"${total_pnl_val:.2f}", delta_color="normal")
        sc2.metric("Win Rate", f"{win_rate:.1f}%")
        sc3.metric("Trades Cerrados", total_closed)
        
        # Open Trades Table
        st.subheader("🟢 Posiciones Abiertas")
        if not df_stats.empty:
            open_s = df_stats[df_stats['status'] == 'OPEN'].copy()
            if not open_s.empty:
                open_s['Floating PnL'] = np.where(open_s['type'] == 'LONG', (current_price - open_s['entry']) * open_s['size'], (open_s['entry'] - current_price) * open_s['size'])
                def color_float(val): return f'color: {"#00FF00" if val > 0 else "#FF4444"}; font-weight: bold;'
                st.dataframe(open_s[['time', 'symbol', 'type', 'leverage', 'entry', 'sl', 'tp3', 'Floating PnL']].style.applymap(color_float, subset=['Floating PnL']).format({'leverage': '{:.1f}x'}), use_container_width=True)
            else: st.info("Sin operaciones activas.")
        
        # Equity Curve
        if not df_stats.empty and total_closed > 0:
            closed_s = df_stats[df_stats['status'] == 'CLOSED'].copy()
            closed_s['equity'] = INITIAL_CAPITAL + closed_s['pnl'].cumsum()
            fig_eq = px.area(closed_s, x='time', y='equity', title="Curva de Crecimiento")
            fig_eq.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_eq, use_container_width=True)

else: st.warning("Cargando datos del mercado...")

if auto_refresh: time.sleep(60); st.rerun()
