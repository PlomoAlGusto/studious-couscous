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
# 1. CONFIGURACIÓN ESTRUCTURAL & ESTILO NEÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v16.6 Lazarus", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    /* FONDO Y ESTRUCTURA */
    .stApp { background-color: #0e1117; }
    .metric-card { background-color: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 10px; }
    
    /* CAJA IA CYBERPUNK (ESTRUCTURA RECUPERADA) */
    .ai-box {
        background: linear-gradient(90deg, rgba(15,15,20,1) 0%, rgba(25,25,35,1) 100%);
        border-left: 4px solid #00e5ff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        font-family: 'Consolas', 'Roboto Mono', monospace;
        color: #e0e0e0;
        font-size: 13px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    .ai-header { 
        color: #00e5ff; font-weight: 900; font-size: 16px; 
        margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;
        display: block; letter-spacing: 1px; 
    }
    .ai-section-title {
        color: #FFD700; font-weight: bold; margin-top: 12px; margin-bottom: 5px; display: block;
    }
    .ai-line { margin-bottom: 4px; }
    
    /* TARJETA DE SEÑAL */
    .trade-box {
        background-color: #111;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 10px;
    }
    .trade-long { border: 2px solid #00ff00; box-shadow: 0 0 10px rgba(0,255,0,0.2); }
    .trade-short { border: 2px solid #ff0044; box-shadow: 0 0 10px rgba(255,0,68,0.2); }
    
    /* CLASES DE TEXTO */
    .txt-green { color: #00ff00; font-weight: bold; }
    .txt-red { color: #ff0044; font-weight: bold; }
    .txt-blue { color: #00e5ff; font-weight: bold; }
    .txt-warn { color: #ffcc00; font-weight: bold; }
    
    /* RELOJES */
    .market-clock { font-size: 11px; padding: 5px; margin-bottom: 2px; border-radius: 4px; display: flex; justify-content: space-between; background: #111; border: 1px solid #222;}
    .clock-open { border-left: 3px solid #00ff00; }
    .clock-closed { border-left: 3px solid #555; color: #666; }
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

# API KEY
try:
    COINGLASS_API_KEY = st.secrets.get("COINGLASS_KEY", "1d4579f4c59149c6b6a1d83494a4f67c")
except:
    COINGLASS_API_KEY = "1d4579f4c59149c6b6a1d83494a4f67c"

# -----------------------------------------------------------------------------
# 2. MOTOR DE DATOS (KRAKEN PRIORITY FOR STABILITY)
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
        status = "🟢" if is_open else "🔴"
        css = "clock-open" if is_open else "clock-closed"
        st.sidebar.markdown(f"<div class='market-clock {css}'><span>{name}</span><span>{status}</span></div>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_deriv_data(symbol):
    """Obtiene Funding y OI usando BYBIT V5 (Pública y Estable)"""
    base_coin = symbol.split('/')[0]
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={base_coin}USDT"
        r = requests.get(url, timeout=4).json()
        if r['retCode'] == 0:
            info = r['result']['list'][0]
            fr = float(info['fundingRate']) * 100
            oi_val = float(info['openInterestValue'])
            return fr, oi_val, "Bybit"
    except: pass
    
    # Fallback Binance
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={base_coin}USDT"
        r = requests.get(url, timeout=4).json()
        fr = float(r['lastFundingRate']) * 100
        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={base_coin}USDT"
        r_oi = requests.get(url_oi, timeout=4).json()
        oi_val = float(r_oi['openInterest']) * float(r['markPrice'])
        return fr, oi_val, "Binance"
    except: pass

    return 0.0, 0.0, "Error"

@st.cache_data(ttl=30)
def get_mtf_trends_analysis(symbol):
    # Usamos Kraken para MTF también para asegurar carga
    try:
        ex = ccxt.binance() 
    except:
        ex = ccxt.kraken()
        
    ticker_fix = symbol.replace("/", "USDT") if "/" not in symbol else symbol
    trends = {}
    score = 0
    for tf in ['15m', '1h', '4h']:
        try:
            ohlcv = ex.fetch_ohlcv(ticker_fix, tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            ema = ta.ema(df['c'], length=50).iloc[-1]
            close = df['c'].iloc[-1]
            if close > ema: trends[tf] = "BULL"; score += 1
            else: trends[tf] = "BEAR"; score -= 1
        except: trends[tf] = "NEUTRO"
    return trends, score

# -----------------------------------------------------------------------------
# 3. INTERFAZ SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v16.6")
    st.caption("Lazarus Edition 🧬")
    get_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)
    
    with st.expander("🛡️ FILTROS", expanded=True):
        use_ema = st.checkbox("Tendencia Base (EMAs)", True)
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP", True)
        use_ichi = st.checkbox("Filtro Ichimoku", False)
        use_regime = st.checkbox("Filtro ADX", True)
        use_rsi = st.checkbox("Filtro RSI", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        use_tsi = st.checkbox("TSI (True Strength)", True)
        
    with st.expander("💰 RIESGO"):
        current_balance = get_current_balance()
        st.metric("Capital", f"${current_balance:,.2f}")
        risk_per_trade = st.slider("Riesgo %", 0.5, 5.0, 1.0)
        
    with st.expander("⚙️ SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven", True)
        use_time_stop = st.checkbox("Time Stop", True)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN", False)
    if st.button("🔥 RESETEAR CUENTA"): reset_account()

# -----------------------------------------------------------------------------
# 4. DATOS Y LÓGICA
# -----------------------------------------------------------------------------
def init_exchange():
    # Priorizamos Kraken para evitar el "Cargando..." infinito de Binance
    try: return ccxt.kraken(), "Kraken"
    except: return ccxt.binance(), "Binance"

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
    
    # Fix ticker for Kraken (BTC/USDT -> XBT/USDT)
    ticker_fix = symbol
    if source_name == "Kraken" and symbol == "BTC/USDT": ticker_fix = "BTC/USDT" # Kraken maneja esto bien usualmente

    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Order Book
        try:
            book = exchange.fetch_order_book(ticker_fix, limit=20)
            bids = sum([x[1] for x in book['bids']])
            asks = sum([x[1] for x in book['asks']])
            obi = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
        except: obi = 0
        
        # Trend 4H
        try:
            ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
            df_4h = pd.DataFrame(ohlcv_4h, columns=['t','o','h','l','c','v'])
            ema_50 = ta.ema(df_4h['c'], length=50).iloc[-1]
            trend_4h = "BULLISH" if df_4h['c'].iloc[-1] > ema_50 else "BEARISH"
        except: trend_4h = "NEUTRO"
        
        return df, obi, trend_4h
    except Exception as e:
        # Si falla, devolvemos None para que el main lo detecte
        return None, 0, None

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
        df['TSI'] = tsi.iloc[:, 0] if isinstance(tsi, pd.DataFrame) else tsi
    except: df['TSI'] = 0
    
    high_w = df['high'].rolling(20).max()
    low_w = df['low'].rolling(20).min()
    close_w = df['close']
    df['PIVOT'] = (high_w + low_w + close_w) / 3
    df['R1'] = (2 * df['PIVOT']) - low_w
    df['S1'] = (2 * df['PIVOT']) - high_w
    return df.fillna(method='bfill').fillna(method='ffill')

# -----------------------------------------------------------------------------
# 5. RENDERIZADO IA (SECCIONES RESTAURADAS)
# -----------------------------------------------------------------------------
def render_neon_ai_analysis(row, mtf_trends, mtf_score, obi, fr, open_interest, data_src, signal, prob):
    # 1. ANALISIS
    t_15m, t_1h, t_4h = mtf_trends.get('15m','N'), mtf_trends.get('1h','N'), mtf_trends.get('4h','N')
    
    if mtf_score == 3: context = "ALCISTA FUERTE (Full)"; c_col = "txt-green"
    elif mtf_score == -3: context = "BAJISTA FUERTE (Full)"; c_col = "txt-red"
    elif t_4h == "BULL": context = "CORRECCIÓN (Macro Bull)"; c_col = "txt-warn"
    elif t_4h == "BEAR": context = "REBOTE (Macro Bear)"; c_col = "txt-warn"
    else: context = "MERCADO MIXTO"; c_col = ""
    
    # 2. VARIABLES OCULTAS
    dist_ema = ((row['close'] - row['EMA_50']) / row['EMA_50']) * 100
    grav_txt = f"{dist_ema:.2f}% desde EMA50"
    grav_col = "txt-red" if abs(dist_ema) > 3 else "txt-green"
    
    mfi = row['MFI']
    gas_status = "LLENO (Volumen Alto)" if mfi > 60 else "RESERVA (Volumen Bajo)" if mfi < 40 else "MEDIO"
    gas_col = "txt-green" if mfi > 60 else "txt-red" if mfi < 40 else ""

    # 3. RADIOGRAFÍA
    adx = row['ADX_14']
    rsi = row['RSI']
    tsi = row['TSI']
    tsi_txt = "ALCISTA" if tsi > 0 else "BAJISTA"
    
    # 4. DERIVADOS
    fr_col = "txt-red" if abs(fr) > 0.01 else "txt-green"
    oi_fmt = f"${open_interest/1_000_000:.1f}M" if open_interest > 1_000_000 else f"${open_interest:.0f}"

    # 5. VEREDICTO
    if signal == "LONG": verdict = f"🚀 <span class='txt-green'>ENTRADA LONG</span> (Prob: {prob:.1f}%)"
    elif signal == "SHORT": verdict = f"🔻 <span class='txt-red'>ENTRADA SHORT</span> (Prob: {prob:.1f}%)"
    else: verdict = "⏳ <span class='txt-warn'>ESPERAR CONFIRMACIÓN</span>"

    # HTML ESTRUCTURADO
    html = f"""
    <div class='ai-box'>
        <span class='ai-header'>🤖 QUIMERA COPILOT <span style='font-size:10px; color:#666;'>({data_src})</span></span>
        
        <span class='ai-section-title'>🔍 CONTEXTO OCULTO</span>
        <div class='ai-line'>• <b>Gravedad:</b> <span class='{grav_col}'>{grav_txt}</span></div>
        <div class='ai-line'>• <b>Gasolina:</b> <span class='{gas_col}'>{gas_status}</span> (MFI: {mfi:.0f})</div>
        <div class='ai-line'>• <b>Libro:</b> {obi*100:.1f}% Desequilibrio</div>
        
        <span class='ai-section-title'>📈 RADIOGRAFÍA TÉCNICA</span>
        <div class='ai-line'>• <b>Fuerza (ADX):</b> {adx:.1f} | <b>TSI:</b> {tsi_txt} ({tsi:.2f})</div>
        <div class='ai-line'>• <b>Tendencia:</b> <span class='{c_col}'>{context}</span></div>
        <div class='ai-line'>• <b>Derivados:</b> Funding <span class='{fr_col}'>{fr:.4f}%</span> | OI: <span class='txt-blue'>{oi_fmt}</span></div>
        
        <div style='margin-top:15px; border-top:1px solid #333; padding-top:10px; text-align:center; font-size:16px; font-weight:bold;'>
            {verdict}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

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
# 6. EJECUCIÓN Y DASHBOARD
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
    if df_trades.empty: return
    closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if closed.empty: return
    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = INITIAL_CAPITAL + closed['cumulative_pnl']
    fig = px.area(closed, x='time', y='equity', title="Curva de Capital (Equity Curve)")
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

# MAIN
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi, 'use_tsi': use_tsi}
    signal, atr, prob = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    
    fr, open_interest, data_src = get_deriv_data(symbol)
    mtf_trends, mtf_score = get_mtf_trends_analysis(symbol)
    
    tab1, tab2 = st.tabs(["📊 COMANDO CENTRAL", "🧪 PAPER TRADING"])
    
    with tab1:
        render_neon_ai_analysis(df.iloc[-1], mtf_trends, mtf_score, obi, fr, open_interest, data_src, signal, prob)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Funding", f"{fr:.4f}%")
        c3.metric("MTF", f"{mtf_trends['15m']} | {mtf_trends['1h']} | {mtf_trends['4h']}")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=f'{source_name} Data'), row=1, col=1)
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
                emoji = "⬆️ LONG"; cls = "trade-long"
            else:
                sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-sl_dist, current_price-(sl_dist*2), current_price-(sl_dist*3.5)
                emoji = "⬇️ SHORT"; cls = "trade-short"
            
            setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'qty': qty, 'lev': leverage}
            
            st.markdown(f"""
            <div class='trade-box {cls}'>
                <h3>{emoji} DETECTADO ({prob:.1f}%)</h3>
                <p>Entry: ${current_price:.2f} | SL: ${sl:.2f} | TP1: ${tp1:.2f}</p>
                <p>Lev: {leverage:.1f}x | Qty: {qty:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("EJECUTAR OPERACIÓN"):
                execute_trade(signal, current_price, sl, tp1, tp2, tp3, qty, atr, leverage)
                st.success("Orden Enviada")

    with tab2:
        df_trades = load_trades()
        render_analytics(df_trades)
        st.dataframe(df_trades, use_container_width=True)
        
    manage_open_positions(current_price)

else:
    st.error("Error de conexión con el Exchange. Reintentando en 5s...")
    time.sleep(5)
    st.rerun()

if auto_refresh: time.sleep(60); st.rerun()
