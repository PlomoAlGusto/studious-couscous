import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timezone
import time
import numpy as np
import os
import feedparser
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

# --- Configuración Inicial de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v17.0 ML Enhanced", layout="wide", page_icon="🧠")
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
    .ai-box {
        background-color: #0e1117;
        border-left: 4px solid #44AAFF; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 15px; 
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 13px;
        color: #e0e0e0;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .ai-title { 
        color: #44AAFF; 
        font-weight: bold; 
        font-size: 14px; 
        margin-bottom: 8px; 
        display: block; 
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
        font-family: 'Segoe UI', sans-serif;
    }
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
    .status-dot-on { color: #00FF00; font-weight: bold; text-shadow: 0 0 5px #00FF00; }
    .badge-bull { background-color: #004400; color: #00FF00; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #00FF00; margin-right: 4px; }
    .badge-bear { background-color: #440000; color: #FF4444; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #FF4444; margin-right: 4px; }
    .badge-neutral { background-color: #333; color: #aaa; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #555; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ARCHIVOS
CSV_FILE = 'paper_trades.csv'
COLUMNS_DB = ["id", "time", "symbol", "type", "entry", "size", "leverage", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"]
INITIAL_CAPITAL = 10000.0

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=COLUMNS_DB).to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state:
    st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. FUNCIONES BÁSICAS
# -----------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=COLUMNS_DB)
    df = pd.read_csv(CSV_FILE)
    if 'leverage' not in df.columns: df['leverage'] = 1.0
    return df

def get_fear_and_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            item = data['data'][0]
            return int(item['value']), str(item['value_classification'])
    except Exception as e:
        logging.error(f"Error Fear/Greed: {e}")
    return 50, "Neutral"

def get_current_balance():
    df = load_trades()
    return INITIAL_CAPITAL + (df[df['status'] == 'CLOSED']['pnl'].sum() if not df.empty else 0)

def reset_account():
    pd.DataFrame(columns=COLUMNS_DB).to_csv(CSV_FILE, index=False)
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

# =============================================================================
# DERIVADOS: PROMEDIO PONDERADO DE 4 FUENTES (SIN BINANCE FUTURES)
# =============================================================================
@st.cache_data(ttl=60)
def get_deriv_data(symbol):
    base = symbol.split('/')[0]
    quote = symbol.split('/')[1]
    COINGLASS_API_KEY = st.secrets.get("COINGLASS_API_KEY", None)
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Quimera/1.0)'}

    fr_sources, oi_sources = [], []

    # Bybit
    try:
        inst = symbol.replace('/', '')
        r_fr = requests.get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={inst}&limit=1", timeout=6, headers=headers).json()
        r_oi = requests.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={inst}&period=5min", timeout=6, headers=headers).json()
        if r_fr.get('retCode') == 0 and r_fr['result']['list']:
            fr_sources.append(float(r_fr['result']['list'][0]['fundingRate']) * 100)
        if r_oi.get('retCode') == 0 and r_oi['result']['list']:
            oi_sources.append(float(r_oi['result']['list'][0]['openInterest']))
    except: logging.warning("Bybit Fetch Failed")

    # OKX
    try:
        inst = f"{base}-{quote}-SWAP"
        r_fr = requests.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={inst}", timeout=6, headers=headers).json()
        r_oi = requests.get(f"https://www.okx.com/api/v5/public/open-interest?instId={inst}", timeout=6, headers=headers).json()
        if r_fr.get('code') == '0' and r_fr['data']: fr_sources.append(float(r_fr['data'][0]['fundingRate']) * 100)
        if r_oi.get('code') == '0' and r_oi['data']: oi_sources.append(float(r_oi['data'][0]['openInterest']))
    except: logging.warning("OKX Fetch Failed")

    # CoinGlass
    if COINGLASS_API_KEY:
        try:
            h = {"coinglassSecret": COINGLASS_API_KEY, "User-Agent": "Quimera"}
            r_fr = requests.get(f"https://open-api.coinglass.com/public/v2/funding?symbol={base}", headers=h, timeout=6).json()
            r_oi = requests.get(f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base}", headers=h, timeout=6).json()
            if r_fr.get('code') == 200:
                fr_list = [float(ex['uMarginList'][0]['rate']) * 100 for ex in r_fr.get('data', []) if ex.get('uMarginList')]
                if fr_list: fr_sources.append(np.mean(fr_list))
            if r_oi.get('code') == 200:
                oi_total = sum(float(ex.get('openInterestAmount', 0)) * float(ex.get('price', 1)) for ex in r_oi.get('data', []))
                if oi_total > 0: oi_sources.append(oi_total)
        except: logging.warning("CoinGlass Fetch Failed")

    # dYdX
    try:
        symbol_dydx = f"{base}-{quote}" if quote == "USD" else base
        r = requests.get(f"https://indexer.dydx.exchange/v4/historicalFunding/{symbol_dydx}?limit=1", timeout=6, headers=headers).json()
        if r and 'historicalFunding' in r and r['historicalFunding']:
            fr_sources.append(float(r['historicalFunding'][0]['rate']) * 100)
    except: logging.warning("dYdX Fetch Failed")

    fr = np.mean(fr_sources) if fr_sources else None
    oi = np.mean(oi_sources) if oi_sources else None
    src = "Hybrid" if len(fr_sources) >= 2 else "Fallback"
    return fr, oi, src

# =============================================================================
# FUNCIÓN FALTANTE: RESTAURADA
# =============================================================================
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
            if ema_fast > ema_slow: 
                trends[tf] = "BULL"
                score += 1
            else: 
                trends[tf] = "BEAR"
                score -= 1
        except:
            trends[tf] = "NEUTRO"
    return trends, score

# =============================================================================
# BACKTESTING SIMPLE
# =============================================================================
def run_backtest(df, filters, atr_threshold=None):
    if df is None or len(df) < 100: 
        return None, 0.0, 0.0, 0, 0

    df_bt = df.copy()
    df_bt['EMA_20'] = ta.ema(df_bt['close'], length=20)
    df_bt['EMA_50'] = ta.ema(df_bt['close'], length=50)
    df_bt['ATR'] = ta.atr(df_bt['high'], df_bt['low'], df_bt['close'], length=14)
    df_bt['RSI'] = ta.rsi(df_bt['close'], length=14)
    adx_df = ta.adx(df_bt['high'], df_bt['low'], df_bt['close'], length=14)
    df_bt = pd.concat([df_bt, adx_df], axis=1)
    df_bt['ADX_14'] = df_bt['ADX_14'].fillna(0)

    if atr_threshold is not None:
        df_bt = df_bt[df_bt['ATR'] >= atr_threshold].copy()
        if df_bt.empty: return None, 0.0, 0.0, 0, 0

    signals = []
    for i in range(50, len(df_bt)):
        row = df_bt.iloc[i]
        score, max_score = 0, 0

        if filters['use_mtf']:
            max_score += 2
            trend_4h = "BULLISH" if df_bt['close'].iloc[i] > df_bt['close'].iloc[max(0,i-200):i].mean() else "BEARISH"
            if trend_4h == "BULLISH": score += 2

        if filters['use_ema']:
            max_score += 1
            if row['EMA_20'] > row['EMA_50']: score += 1

        if filters['use_vwap'] and 'VWAP' in df_bt.columns:
            max_score += 1
            if row['close'] > row['VWAP']: score += 1

        if filters['use_obi']:
            max_score += 1
            # Simulamos obi con ruido
            obi = np.random.uniform(-0.1, 0.1)
            if obi > 0.05: score += 1

        if filters.get('use_tsi', False) and 'TSI' in df_bt.columns:
            max_score += 1
            if row['TSI'] > 0: score += 1

        if filters['use_regime'] and row['ADX_14'] < 20:
            signals.append(0)
        else:
            threshold = max_score * 0.4 if max_score > 0 else 0
            if score > threshold: signals.append(1)
            elif score < -threshold: signals.append(-1)
            else: signals.append(0)

    df_signals = df_bt.iloc[50:].copy()
    if len(signals) != len(df_signals): signals = signals[:len(df_signals)]
    df_signals['signal'] = signals

    df_signals['return'] = df_signals['close'].pct_change()
    df_signals['strategy_return'] = df_signals['signal'].shift(1) * df_signals['return']
    total_return = df_signals['strategy_return'].sum()
    win_trades = (df_signals['strategy_return'] > 0).sum()
    total_trades = (df_signals['signal'] != 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    expectancy = df_signals['strategy_return'].mean() if total_trades > 0 else 0

    return df_signals, total_return, win_rate, total_trades, expectancy

# =============================================================================
# CLASIFICADOR ML DE RÉGIMEN
# =============================================================================
def train_regime_classifier(df):
    if len(df) < 100: return None, None
    df_ml = df.copy()
    df_ml['return'] = df_ml['close'].pct_change()
    df_ml['vol'] = df_ml['return'].rolling(14).std()
    df_ml['trend'] = (ta.ema(df_ml['close'], 20) > ta.ema(df_ml['close'], 50)).astype(int)
    df_ml['label'] = np.where(df_ml['return'].shift(-10) > df_ml['vol'], 1,
                             np.where(df_ml['return'].shift(-10) < -df_ml['vol'], -1, 0))
    df_ml = df_ml.dropna()
    if len(df_ml) < 50: return None, None

    features = df_ml[['vol', 'ADX_14', 'RSI', 'trend']].fillna(0)
    labels = df_ml['label']
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, labels)
    return model, scaler

# =============================================================================
# LIQUIDEZ Y CVD
# =============================================================================
def get_liquidity_and_cvd(exchange, symbol):
    try:
        book = exchange.fetch_order_book(symbol, limit=10)
        bid = book['bids'][0][0] if book['bids'] else None
        ask = book['asks'][0][0] if book['asks'] else None
        if bid and ask:
            spread_pct = (ask - bid) / ask * 100
            is_liquid = spread_pct < 0.1
            bids_vol = sum([x[1] for x in book['bids']])
            asks_vol = sum([x[1] for x in book['asks']])
            cvd = bids_vol - asks_vol
            return is_liquid, cvd
    except:
        pass
    return False, 0

# -----------------------------------------------------------------------------
# 3. INTERFAZ SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 QUIMERA v17.0")
    st.markdown(f"<div style='font-size:12px; margin-bottom:10px;'><span class='status-dot-on'>●</span> SYSTEM ONLINE</div>", unsafe_allow_html=True)
    get_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)
    run_backtest_button = st.button("▶️ Ejecutar Backtest")
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
        st.metric("Balance Disponible", f"${current_balance:,.2f}", delta=f"{current_balance-INITIAL_CAPITAL:.2f}")
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
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            ex.load_markets(); return ex, "Binance (Priv)"
    except: pass
    try:
        ex = ccxt.binance(); ex.load_markets(); return ex, "Binance (Pub)"
    except: return ccxt.kraken(), "Kraken (Fallback)"

exchange, source_name = init_exchange()

@st.cache_data(ttl=300)
def get_crypto_news():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        return [{"title": e.title, "link": e.link, "published": e.published_parsed} for e in feed.entries[:5]]
    except: return []

@st.cache_data(ttl=15)
def get_mtf_data(symbol, tf_lower):
    if not exchange: return None, 0, None, False, 0
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=600)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None, False, 0

    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
        last_4h = df_4h.iloc[-1]
        trend_4h = "BULLISH" if last_4h['close'] > last_4h['EMA_50'] else "BEARISH"
    except: pass

    is_liquid, cvd = get_liquidity_and_cvd(exchange, ticker_fix)

    return df, cvd, trend_4h, is_liquid, cvd

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
# 5. IA ANALISTA
# -----------------------------------------------------------------------------
def generate_detailed_ai_analysis_html(row, mtf_trends, mtf_score, cvd, fr, open_interest, data_src, regime, bayes_prob, is_liquid):
    t_15m = mtf_trends.get('15m', 'NEUTRO')
    t_1h = mtf_trends.get('1h', 'NEUTRO')
    t_4h = mtf_trends.get('4h', 'NEUTRO')
    if mtf_score == 3: context = "<span style='color:#00FF00'>ALCISTA FUERTE</span> (Alineación Total)"
    elif mtf_score == -3: context = "<span style='color:#FF4444'>BAJISTA FUERTE</span> (Alineación Total)"
    elif t_4h == "BULL" and t_15m == "BEAR": context = "<span style='color:#FFFF00'>CORRECCIÓN EN CURSO</span> (Macro Alcista / Micro Bajista)"
    elif t_4h == "BEAR" and t_15m == "BULL": context = "<span style='color:#FFFF00'>REBOTE TÉCNICO</span> (Macro Bajista / Micro Alcista)"
    else: context = "MERCADO MIXTO (Conflicto de Temporalidades)"

    if fr is None or open_interest is None:
        deriv_txt = "<span style='color:#FFFF00'>Fuentes Derivadas OFFLINE</span>"
        oi_txt = ""
    else:
        deriv_txt = f"Funding Rate: <b style='color:#fff'>{fr:.4f}%</b>"
        squeeze_risk = ""
        if fr > 0.02: squeeze_risk = " (HIGH Long Squeeze)"
        elif fr < -0.02: squeeze_risk = " (HIGH Short Squeeze)"
        elif fr > 0.01: squeeze_risk = " (Long Squeeze Risk)"
        elif fr < -0.01: squeeze_risk = " (Short Squeeze Risk)"
        else: squeeze_risk = " (Saludable)"
        deriv_txt += squeeze_risk
        if open_interest > 1e9: oi_fmt = f"${open_interest/1e9:.2f}B"
        elif open_interest > 1e6: oi_fmt = f"${open_interest/1e6:.2f}M"
        else: oi_fmt = f"${open_interest:,.0f}"
        oi_txt = f"Interés Abierto: <b style='color:#44AAFF'>{oi_fmt}</b>"

    mfi = row['MFI']
    adx = row['ADX_14']
    tsi = row['TSI']
    gas_status = "LLENO" if mfi > 60 else "RESERVA" if mfi < 40 else "MEDIO"
    gas_color = "#00FF00" if mfi > 60 else "#FF4444" if mfi < 40 else "#FFF"
    tsi_status = "ALCISTA" if tsi > 0 else "BAJISTA"
    tsi_color = "#00FF00" if tsi > 0 else "#FF4444"
    mom_txt = f"Gasolina (MFI): <b style='color:{gas_color}'>{gas_status}</b>. ADX: {adx:.1f}. TSI: <b style='color:{tsi_color}'>{tsi_status}</b> ({tsi:.2f})."

    cvd_txt = "🟢 Flujo Comprador" if cvd > 0 else "🔴 Flujo Vendedor" if cvd < 0 else "⚖️ Neutral"
    cvd_color = "#00FF00" if cvd > 0 else "#FF4444" if cvd < 0 else "#aaa"
    liquidity_txt = "✅ Mercado Líquido" if is_liquid else "⚠️ Ilíquido (Evitar)"

    html = f"""
    <div class='ai-box'>
        <span class='ai-title'>🤖 QUIMERA COPILOT v17.0 (ML + Backtest):</span>
        <div>📡 <b>ESTRUCTURA:</b> {context}</div>
        <div>🧠 <b>RÉGIMEN ML:</b> <b>{regime}</b></div>
        <div>📊 <b>DERIVADOS:</b> {deriv_txt}. {oi_txt}</div>
        <div>🔥 <b>MOMENTO:</b> {mom_txt}</div>
        <div>💧 <b>CVD:</b> <b style='color:{cvd_color}'>{cvd_txt}</b> ({cvd:,.0f})</div>
        <div>💧 <b>LIQUIDEZ:</b> {liquidity_txt}</div>
        <div>🎯 <b>Prob. Bayesiana:</b> <b style='color:#44AAFF'>{bayes_prob:.1%}</b></div>
    </div>
    """
    return html

def run_strategy(df, cvd, trend_4h, filters, fr, is_liquid, atr_vol_threshold):
    row = df.iloc[-1]
    if not is_liquid: return "NEUTRO", row['ATR'], 50.0, 0, ["<span class='badge-neutral'>LIQUIDEZ</span>"]
    if row['ATR'] < atr_vol_threshold: return "NEUTRO", row['ATR'], 50.0, 0, ["<span class='badge-neutral'>VOLATILIDAD</span>"]

    score, max_score, details = 0, 0, []
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2; details.append("<span class='badge-bull'>MACRO</span>")
        elif trend_4h == "BEARISH": score -= 2; details.append("<span class='badge-bear'>MACRO</span>")
        else: details.append("<span class='badge-neutral'>MACRO</span>")
    if filters['use_ema']: 
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; details.append("<span class='badge-bull'>EMA</span>")
        else: score -= 1; details.append("<span class='badge-bear'>EMA</span>")
    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; details.append("<span class='badge-bull'>VWAP</span>")
        else: score -= 1; details.append("<span class='badge-bear'>VWAP</span>")
    if filters['use_obi']:
        max_score += 1
        obi = np.sign(cvd)
        if obi > 0: score += 1; details.append("<span class='badge-bull'>OBI</span>")
        elif obi < 0: score -= 1; details.append("<span class='badge-bear'>OBI</span>")
        else: details.append("<span class='badge-neutral'>OBI</span>")
    if filters.get('use_tsi', False): 
        max_score += 1
        if row['TSI'] > 0: score += 1; details.append("<span class='badge-bull'>TSI</span>")
        else: score -= 1; details.append("<span class='badge-bear'>TSI</span>")
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
    thermo_score = (score / max_score) * 100 if max_score > 0 else 0
    if fr is not None:
        if signal == "LONG" and fr > 0.01: signal = "NEUTRO"
        elif signal == "SHORT" and fr < -0.01: signal = "NEUTRO"
        prob += 5 if abs(fr) < 0.005 else -5
    return signal, row['ATR'], prob, thermo_score, details

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

def manage_open_positions(current_price, current_high, current_low):
    df = load_trades()
    if df.empty: return
    open_idx = df.index[df['status'] == 'OPEN'].tolist()
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
            send_telegram_msg(f"💰 CIERRE {symbol}: {close_reason}\nPnL: ${pnl:.2f}")
            updated = True
    if updated or use_time_stop: save_trades(df)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: logging.warning("Telegram failed to send.")

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
    start = pd.DataFrame([{'time': 'Inicio', 'equity': INITIAL_CAPITAL}])
    curve = pd.concat([start, closed[['time', 'equity']]])
    total_profit = closed['pnl'].sum()
    fig = px.area(curve, x='time', y='equity', title="Curva de Capital (Equity Curve)")
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0))
    color = '#00FF00' if total_profit >= 0 else '#FF4444'
    fig.update_traces(line_color=color, fillcolor=color.replace("FF", "22") if total_profit>=0 else color.replace("44", "11"))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
df, cvd, trend_4h, is_liquid, _ = get_mtf_data(symbol, tf)
if df is not None:
    df = calculate_indicators(df)
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi, 'use_tsi': use_tsi}

    # ATR dinámico
    atr_vol_threshold = df['ATR'].quantile(0.2) if len(df) > 20 else 0

    # Derivados
    fr, open_interest, data_src = get_deriv_data(symbol)

    # Régimen con ML
    model, scaler = train_regime_classifier(df)
    regime = "NEUTRO"
    if model is not None:
        last_row = df.iloc[-1]
        vol = df['close'].pct_change().tail(14).std()
        trend_flag = 1 if last_row['EMA_20'] > df['EMA_50'].iloc[-1] else 0
        X = scaler.transform([[vol, last_row['ADX_14'], last_row['RSI'], trend_flag]])
        pred = model.predict(X)[0]
        regime = "TREND ↑" if pred == 1 else "TREND ↓" if pred == -1 else "RANGO"

    # Probabilidad bayesiana
    base_prob = 0.5
    if fr is not None:
        if abs(fr) < 0.005: base_prob += 0.1
        elif abs(fr) > 0.02: base_prob -= 0.15
    if df['ADX_14'].iloc[-1] > 25: base_prob += 0.1
    if regime in ["TREND ↑", "TREND ↓"]: base_prob += 0.05
    bayes_prob = min(0.95, max(0.3, base_prob))

    # Backtest on demand
    if run_backtest_button:
        with st.spinner("Ejecutando backtest..."):
            df_bt, ret, wr, n_trades, exp = run_backtest(df, filters, atr_vol_threshold)
            if df_bt is not None:
                st.sidebar.success(f"✅ Backtest Finalizado\nReturn: {ret:.2%}\nWin Rate: {wr:.1%}\nTrades: {n_trades}\nExpectancy: {exp:.4f}")
            else:
                st.sidebar.error("Insuficientes datos para backtest")

    mtf_trends, mtf_score = get_mtf_trends_analysis(symbol)
    signal, atr, prob, thermo_score, details_list = run_strategy(df, cvd, trend_4h, filters, fr, is_liquid, atr_vol_threshold)
    current_price, cur_high, cur_low = df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    fng_val, fng_label = get_fear_and_greed()
    news = get_crypto_news()
    ai_html = generate_detailed_ai_analysis_html(df.iloc[-1], mtf_trends, mtf_score, cvd, fr, open_interest, data_src, regime, bayes_prob, is_liquid)

    # ... resto de tu lógica de señales, gestión de trades, UI, etc.
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
        leverage = max(1.0, notional_value / current_balance)
        if calc_dir == "LONG":
            sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            emoji = "⬆️ LONG"
        else:
            sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            emoji = "⬇️ SHORT"
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'status': setup_type, 'qty': qty, 'lev': leverage}
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
    tab1, tab2 = st.tabs(["📊 LIVE COMMAND", "🧪 PAPER TRADING"])
    with tab1:
        col_news, col_tech, col_fng = st.columns([1.5, 1, 1])
        with col_news:
            st.markdown("### 📰 MARKET FLASH")
            if news:
                news_html = "<div class='news-box'>"
                for n in news:
                    t_struct = n.get('published', time.gmtime())
                    t_str = f"{t_struct.tm_hour:02}:{t_struct.tm_min:02}"
                    news_html += f"<div class='news-item'><span class='news-time'>{t_str}</span> <a href='{n['link']}' target='_blank' class='news-link'>{n['title']}</a></div>"
                news_html += "</div>"
                st.markdown(news_html, unsafe_allow_html=True)
            else: st.info("Sin noticias recientes.")
        with col_tech:
            fig_thermo = go.Figure(go.Indicator(
                mode = "gauge+number", value = thermo_score, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<span style='font-size:16px'>Bot Sentiment</span>"},
                gauge = {'axis': {'range': [-100, 100]}, 'bar': {'color': "white"}, 'bgcolor': "#111",
                    'steps': [{'range': [-100, -40], 'color': "#FF4444"}, {'range': [-40, 40], 'color': "#555"}, {'range': [40, 100], 'color': "#00FF00"}]}
            ))
            fig_thermo.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_thermo, use_container_width=True)
            st.markdown(f"<div style='text-align:center'>{' '.join(details_list)}</div>", unsafe_allow_html=True)
        with col_fng:
            fig_fng = go.Figure(go.Indicator(
                mode = "gauge+number", value = fng_val, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<span style='font-size:16px'>Fear & Greed</span>"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'bgcolor': "#111",
                    'steps': [{'range': [0, 40], 'color': "#FF4444"}, {'range': [40, 60], 'color': "#FFFF00"}, {'range': [60, 100], 'color': "#00FF00"}]}
            ))
            fig_fng.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_fng, use_container_width=True)
        st.markdown(ai_html, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        if fr is None:
            c2.metric("Funding Rate", "N/A", delta="OFFLINE")
        else:
            c2.metric("Funding Rate", f"{fr:.4f}%", delta=f"{fr:.4f}" if fr > 0 else f"{fr:.4f}", delta_color="inverse")
        if open_interest is None:
            c3.metric("Open Interest", "N/A")
        else:
            oi_show = f"${open_interest/1e9:.2f}B" if open_interest > 1e9 else f"${open_interest/1e6:.2f}M"
            c3.metric("Open Interest", oi_show)
        with c4:
            cols = st.columns(3)
            colors = {"BULL": "🟢", "BEAR": "🔴", "NEUTRO": "⚪"}
            cols[0].markdown(f"<div style='text-align:center; font-size:10px'>15m<br><span style='font-size:14px'>{colors[mtf_trends['15m']]}</span></div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div style='text-align:center; font-size:10px'>1h<br><span style='font-size:14px'>{colors[mtf_trends['1h']]}</span></div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='text-align:center; font-size:10px'>4h<br><span style='font-size:14px'>{colors[mtf_trends['4h']]}</span></div>", unsafe_allow_html=True)
        st.markdown("### 📊 RASTREADOR DE RENDIMIENTO (Paper Trading)")
        df_stats = load_trades()
        total_pnl_val = 0.0
        win_rate = 0.0
        open_count = 0
        total_closed = 0
        if not df_stats.empty:
            closed_s = df_stats[df_stats['status'] == 'CLOSED']
            open_s = df_stats[df_stats['status'] == 'OPEN']
            total_closed = len(closed_s)
            open_count = len(open_s)
            if total_closed > 0:
                total_pnl_val = closed_s['pnl'].sum()
                wins = len(closed_s[closed_s['pnl'] > 0])
                win_rate = (wins / total_closed) * 100
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("PnL Total", f"${total_pnl_val:.2f}", delta_color="normal")
        sc2.metric("Win Rate", f"{win_rate:.1f}%")
        sc3.metric("Trades Cerrados", total_closed)
        sc4.metric("Trades Abiertos", open_count)
        st.divider()
        if setup:
            prob_str = f"{prob:.1f}%"
            prob_color = "#00FF00" if prob >= 80 else "#FFFF00" if prob >= 60 else "#FF4444"
            if setup['status'] == "CONFIRMED":
                header_cls = "header-confirmed-long" if calc_dir == "LONG" else "header-confirmed-short"
                header_txt = f"🔥 CONFIRMADO: {setup['dir']}"
                btn_label = f"🚀 EJECUTAR {calc_dir} ({setup['qty']:.4f})"
            else:
                header_cls = "header-potential"
                header_txt = f"⚠️ POTENCIAL: {setup['dir']}"
                btn_label = f"⚠️ FORZAR ENTRADA"
            html_card = f"""
<div class="trade-setup">
    <div class="{header_cls}">{header_txt}</div>
    <div style='margin-top: 5px; margin-bottom: 10px; text-align: left;'>
        <div style='display:flex; justify-content:space-between; color:#ccc; font-size:12px; margin-bottom:2px;'>
            <span>Probabilidad de Éxito:</span><span style='color:{prob_color}; font-weight:bold;'>{prob_str}</span>
        </div>
        <div style='width: 100%; background-color: #333; border-radius: 4px; height: 6px;'>
            <div style='width: {prob}%; background-color: {prob_color}; height: 6px; border-radius: 4px; box-shadow: 0 0 5px {prob_color};'></div>
        </div>
    </div>
    <p style="color:#888; font-size:14px;">Posición: <span style="color:white; font-weight:bold">{setup['qty']:.4f} {symbol.split('/')[0]}</span> (Riesgo ${risk_amount:.1f}) | <span style="color:#44AAFF; font-weight:bold">LEV: {setup['lev']:.1f}x</span></p>
    <div style="display: flex; justify-content: space-around; margin-top: 10px;">
        <div><span class="label-mini">ENTRADA</span><br><span class="entry-blue">${setup['entry']:.2f}</span></div>
        <div><span class="label-mini">STOP LOSS</span><br><span class="sl-red">${setup['sl']:.2f}</span></div>
        <div><span class="label-mini">TP 1</span><br><span class="tp-green">${setup['tp1']:.2f}</span></div>
        <div><span class="label-mini">TP 2</span><br><span class="tp-green">${setup['tp2']:.2f}</span></div>
        <div><span class="label-mini">TP 3</span><br><span class="tp-green">${setup['tp3']:.2f}</span></div>
    </div>
</div>
"""
            st.markdown(html_card, unsafe_allow_html=True)
            if st.button(btn_label):
                execute_trade(calc_dir, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr, setup['lev'])
                st.success(f"Orden {calc_dir} lanzada.")
        else: st.info("Esperando estructura de mercado clara...")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=f'{source_name} Data'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        last_pivot, last_s1, last_r1 = df.iloc[-1]['PIVOT'], df.iloc[-1]['S1'], df.iloc[-1]['R1']
        fig.add_hline(y=last_pivot, line_dash="dash", line_color="gray", annotation_text="Pivote", row=1, col=1)
        fig.add_hline(y=last_s1, line_dash="dot", line_color="green", annotation_text="S1", row=1, col=1)
        fig.add_hline(y=last_r1, line_dash="dot", line_color="red", annotation_text="R1", row=1, col=1)
        if setup:
            fig.add_hline(y=setup['tp1'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dot", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(title=f"Chart Source: {source_name}", template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        df_trades = load_trades()
        st.subheader("📈 Rendimiento Detallado")
        render_analytics(df_trades)
        st.divider()
        if not df_trades.empty:
            open_trades = df_trades[df_trades['status'] == "OPEN"].copy()
            closed_trades = df_trades[df_trades['status'] == "CLOSED"]
            st.subheader("🟢 Posiciones Abiertas")
            if not open_trades.empty:
                open_trades['Floating PnL'] = np.where(open_trades['type'] == 'LONG', (current_price - open_trades['entry']) * open_trades['size'], (open_trades['entry'] - current_price) * open_trades['size'])
                def color_floating(val): return f'color: {"#00FF00" if val > 0 else "#FF4444"}; font-weight: bold;'
                cols_show = ['time', 'symbol', 'type', 'leverage', 'entry', 'size', 'sl', 'tp3', 'Floating PnL']
                st.dataframe(open_trades[cols_show].style.applymap(color_floating, subset=['Floating PnL']).format({'leverage': '{:.1f}x'}), use_container_width=True)
            else: st.info("No hay operaciones abiertas.")
            st.subheader("📜 Historial Cerrado")
            def color_pnl(val): return f'color: {"#228B22" if val > 0 else "#B22222" if val < 0 else "white"}'
            if not closed_trades.empty: st.dataframe(closed_trades.style.applymap(color_pnl, subset=['pnl']).format({'leverage': '{:.1f}x'}), use_container_width=True)
        else: st.info("Historial vacío.")
else: st.warning("Cargando datos...")

if auto_refresh: time.sleep(60); st.rerun()
