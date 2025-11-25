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
st.set_page_config(page_title="Quimera Pro v16.0 Ironclad", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    /* Estilos Generales */
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    
    /* Estilos Trade Setup */
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    .label-mini { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    
    /* Estilos QUIMERA COPILOT (AI BOX) - FUENTE MEJORADA */
    .ai-box {
        background-color: #0e1117;
        border-left: 4px solid #44AAFF; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 15px; 
        /* Fuente técnica moderna (tipo VS Code / Terminal) */
        font-family: 'SF Mono', 'Consolas', 'Menlo', 'Monaco', 'Liberation Mono', 'Lucida Console', monospace;
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
        /* Fuente Sans-Serif para el título para contraste */
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Relojes de Mercado */
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
    
    .status-dot-on { color: #00FF00; font-weight: bold; text-shadow: 0 0 5px #00FF00; }
    
    /* Badges */
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
    df_empty = pd.DataFrame(columns=COLUMNS_DB)
    df_empty.to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"


# -----------------------------------------------------------------------------
# 2. MOTOR DE DATOS (CCXT POWERED)
# -----------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=COLUMNS_DB)
    try:
        df = pd.read_csv(CSV_FILE)
        if 'leverage' not in df.columns: df['leverage'] = 1.0
        return df
    except: return pd.DataFrame(columns=COLUMNS_DB)
        
def get_fear_and_greed():
    """
    Obtiene el índice Fear & Greed de alternative.me
    Retorna: (valor_entero, etiqueta_texto)
    """
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            item = data['data'][0]
            return int(item['value']), str(item['value_classification'])
    except Exception as e:
        print(f"Error Fear/Greed: {e}")
    return 50, "Neutral" # Fallback por defecto

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
def get_deriv_data(symbol):
    """
    Obtiene Funding Rate y Open Interest usando CCXT (Librería Profesional)
    para evitar bloqueos HTTP simples.
    """
    # 1. Intento con CoinGlass (Si funciona la clave)
    if COINGLASS_API_KEY:
        try:
            base_coin = symbol.split('/')[0]
            headers = {"coinglassSecret": COINGLASS_API_KEY}
            url_oi = f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base_coin}"
            r_oi = requests.get(url_oi, headers=headers, timeout=2).json()
            url_fr = f"https://open-api.coinglass.com/public/v2/funding?symbol={base_coin}"
            r_fr = requests.get(url_fr, headers=headers, timeout=2).json()
            
            total_oi = 0.0
            avg_fr = 0.0
            
            if r_oi.get('success'):
                for ex in r_oi['data']: total_oi += ex.get('openInterestAmount', 0) * ex.get('price', 0)
            
            if r_fr.get('success'):
                for ex in r_fr['data']:
                    if ex['exchangeName'] == 'Binance':
                        if 'uMarginList' in ex and len(ex['uMarginList']) > 0: avg_fr = ex['uMarginList'][0]['rate']
                        break
            
            if total_oi > 0: return avg_fr, total_oi, "CoinGlass"
        except: pass

    # 2. Intento con CCXT (Binance Futures)
    try:
        # Iniciamos instancia específica de Futuros
        exchange_f = ccxt.binanceusdm({'enableRateLimit': True})
        ticker = exchange_f.fetch_ticker(symbol)
        
        # Funding Rate
        fr = float(ticker['info']['lastFundingRate']) * 100
        
        # Open Interest (A veces requiere endpoint separado)
        try:
            oi_data = exchange_f.fetch_open_interest(symbol)
            oi_val = float(oi_data['openInterestValue']) # Valor en USDT
        except:
            # Fallback si fetch_open_interest falla: estimar con volumen 24h * ratio
            oi_val = float(ticker['quoteVolume']) * 0.2 
            
        return fr, oi_val, "BinanceAPI"
    except: pass

    # 3. Intento con CCXT (Kraken Futures - Muy permisivo)
    try:
        exchange_k = ccxt.krakenfutures()
        # Kraken usa simbolos distintos, ej: PI_XBTUSD
        ticker_k = exchange_k.fetch_ticker("PF_XBTUSD") # Hardcoded BTC para fallback
        fr = float(ticker_k['info'].get('fundingRate', 0)) * 100
        oi_val = float(ticker_k['info'].get('openInterest', 0))
        return fr, oi_val, "KrakenFut"
    except: pass

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
    st.title("🦁 QUIMERA v16.0")
    st.markdown(f"<div style='font-size:12px; margin-bottom:10px;'><span class='status-dot-on'>●</span> SYSTEM ONLINE</div>", unsafe_allow_html=True)
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
    if not exchange: return None, 0, None
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
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
# 5. IA ANALISTA (SYSTEM LOGIC, NOT AI)
# -----------------------------------------------------------------------------
def generate_detailed_ai_analysis_html(row, mtf_trends, mtf_score, obi, fr, open_interest, data_src):
    # 1. CONTEXTO MULTI-TIMEFRAME
    t_15m = mtf_trends.get('15m', 'NEUTRO')
    t_1h = mtf_trends.get('1h', 'NEUTRO')
    t_4h = mtf_trends.get('4h', 'NEUTRO')
    
    if mtf_score == 3: context = "<span style='color:#00FF00'>ALCISTA FUERTE</span> (Alineación Total)"
    elif mtf_score == -3: context = "<span style='color:#FF4444'>BAJISTA FUERTE</span> (Alineación Total)"
    elif t_4h == "BULL" and t_15m == "BEAR": context = "<span style='color:#FFFF00'>CORRECCIÓN EN CURSO</span> (Macro Alcista / Micro Bajista)"
    elif t_4h == "BEAR" and t_15m == "BULL": context = "<span style='color:#FFFF00'>REBOTE TÉCNICO</span> (Macro Bajista / Micro Alcista)"
    else: context = "MERCADO MIXTO (Conflicto de Temporalidades)"
    
    # 2. DATOS DERIVADOS
    deriv_txt = f"Funding Rate: <b style='color:#fff'>{fr:.4f}%</b>"
    if fr > 0.01: deriv_txt += " (Long Squeeze Risk)"
    elif fr < -0.01: deriv_txt += " (Short Squeeze Risk)"
    else: deriv_txt += " (Saludable)"
    
    if open_interest > 1000000000: oi_fmt = f"${open_interest/1000000000:.2f}B"
    elif open_interest > 1000000: oi_fmt = f"${open_interest/1000000:.2f}M"
    else: oi_fmt = f"${open_interest:,.0f}"
    
    oi_txt = f"Interés Abierto: <b style='color:#44AAFF'>{oi_fmt}</b>"

    # 3. MOMENTO (TSI / MFI / ADX)
    mfi = row['MFI']
    adx = row['ADX_14']
    tsi = row['TSI']
    
    gas_status = "LLENO" if mfi > 60 else "RESERVA" if mfi < 40 else "MEDIO"
    gas_color = "#00FF00" if mfi > 60 else "#FF4444" if mfi < 40 else "#FFF"
    
    tsi_status = "ALCISTA" if tsi > 0 else "BAJISTA"
    tsi_color = "#00FF00" if tsi > 0 else "#FF4444"
    
    mom_txt = f"Gasolina (MFI): <b style='color:{gas_color}'>{gas_status}</b>. ADX: {adx:.1f}. TSI: <b style='color:{tsi_color}'>{tsi_status}</b> ({tsi:.2f})."

    pressure = "COMPRADORA" if obi > 0.05 else "VENDEDORA" if obi < -0.05 else "NEUTRA"
    obi_color = "#00FF00" if obi > 0.05 else "#FF4444" if obi < -0.05 else "#aaa"
    obi_txt = f"Presión Libro: <b style='color:{obi_color}'>{pressure}</b> ({obi*100:.1f}%)"

    html = f"""
    <div class='ai-box'>
        <span class='ai-title'>🤖 QUIMERA COPILOT (Data Source: {data_src}):</span>
        <div style='margin-top:5px;'>📡 <b>ESTRUCTURA:</b> {context}</div>
        <div>📊 <b>DERIVADOS:</b> {deriv_txt}. {oi_txt}</div>
        <div>🔥 <b>MOMENTO:</b> {mom_txt}</div>
        <div>⛽ <b>VOLUMEN:</b> {obi_txt}</div>
    </div>
    """
    return html

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
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
        if obi > 0.05: score += 1; details.append("<span class='badge-bull'>OBI</span>")
        elif obi < -0.05: score -= 1; details.append("<span class='badge-bear'>OBI</span>")
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
            send_telegram_msg(f"💰 CIERRE {symbol}: {close_reason}\nPnL: ${pnl:.2f}")
            updated = True
    if updated or use_time_stop: save_trades(df)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

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
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi, 'use_tsi': use_tsi}
    signal, atr, prob, thermo_score, details_list = run_strategy(df, obi, trend_4h, filters)
    current_price, cur_high, cur_low = df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    mfi_val, adx_val = df['MFI'].iloc[-1], df['ADX_14'].iloc[-1]
    
    # DATOS FIX (Coinglass / Bybit / dYdX)
    fng_val, fng_label = get_fear_and_greed()
    news = get_crypto_news()
    fr, open_interest, data_src = get_deriv_data(symbol)
    mtf_trends, mtf_score = get_mtf_trends_analysis(symbol)
    
    # IA (HTML Correcto + TSI)
    ai_html = generate_detailed_ai_analysis_html(df.iloc[-1], mtf_trends, mtf_score, obi, fr, open_interest, data_src)
    
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
    # --- PRUEBA DE NOTIFICACIONES FORZADA ---
    test_msg = "✅ PRUEBA DE NOTIFICACIÓN EXITOSA: El Bot está enviando mensajes."
    send_telegram_msg(test_msg)
    st.info("Mensaje de prueba de Telegram enviado (revisa tu chat).")
    # -----------------------------------------
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

        # HTML IA
        st.markdown(ai_html, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        
        # WIDGETS
        c2.metric("Funding Rate", f"{fr:.4f}%", delta_color="inverse")
        
        # Formateo Open Interest
        if open_interest > 1000000000: oi_show = f"${open_interest/1000000000:.2f}B"
        elif open_interest > 1000000: oi_show = f"${open_interest/1000000:.2f}M"
        else: oi_show = f"${open_interest:,.0f}"
        c3.metric("Open Interest", oi_show)
        
        # MTF
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
        
        # Titulo dinámico
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

if auto_refresh: time.sleep(60); st.rerun()# -----------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=COLUMNS_DB)
    try:
        df = pd.read_csv(CSV_FILE)
        if 'leverage' not in df.columns: df['leverage'] = 1.0
        return df
    except: return pd.DataFrame(columns=COLUMNS_DB)
        
def get_fear_and_greed():
    """
    Obtiene el índice Fear & Greed de alternative.me
    Retorna: (valor_entero, etiqueta_texto)
    """
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            item = data['data'][0]
            return int(item['value']), str(item['value_classification'])
    except Exception as e:
        print(f"Error Fear/Greed: {e}")
    return 50, "Neutral" # Fallback por defecto

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
def get_deriv_data(symbol):
    """
    Obtiene Funding Rate y Open Interest usando CCXT (Librería Profesional)
    para evitar bloqueos HTTP simples.
    """
    # 1. Intento con CoinGlass (Si funciona la clave)
    if COINGLASS_API_KEY:
        try:
            base_coin = symbol.split('/')[0]
            headers = {"coinglassSecret": COINGLASS_API_KEY}
            url_oi = f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base_coin}"
            r_oi = requests.get(url_oi, headers=headers, timeout=2).json()
            url_fr = f"https://open-api.coinglass.com/public/v2/funding?symbol={base_coin}"
            r_fr = requests.get(url_fr, headers=headers, timeout=2).json()
            
            total_oi = 0.0
            avg_fr = 0.0
            
            if r_oi.get('success'):
                for ex in r_oi['data']: total_oi += ex.get('openInterestAmount', 0) * ex.get('price', 0)
            
            if r_fr.get('success'):
                for ex in r_fr['data']:
                    if ex['exchangeName'] == 'Binance':
                        if 'uMarginList' in ex and len(ex['uMarginList']) > 0: avg_fr = ex['uMarginList'][0]['rate']
                        break
            
            if total_oi > 0: return avg_fr, total_oi, "CoinGlass"
        except: pass

    # 2. Intento con CCXT (Binance Futures)
    try:
        # Iniciamos instancia específica de Futuros
        exchange_f = ccxt.binanceusdm({'enableRateLimit': True})
        ticker = exchange_f.fetch_ticker(symbol)
        
        # Funding Rate
        fr = float(ticker['info']['lastFundingRate']) * 100
        
        # Open Interest (A veces requiere endpoint separado)
        # Intentamos calcularlo aproximado por volumen si no hay endpoint publico directo accesible
        # Pero CCXT suele tenerlo en fetch_ticker o fetch_open_interest
        try:
            oi_data = exchange_f.fetch_open_interest(symbol)
            oi_val = float(oi_data['openInterestValue']) # Valor en USDT
        except:
            # Fallback si fetch_open_interest falla: estimar con volumen 24h * ratio
            oi_val = float(ticker['quoteVolume']) * 0.2 
            
        return fr, oi_val, "BinanceAPI"
    except: pass

    # 3. Intento con CCXT (Kraken Futures - Muy permisivo)
    try:
        exchange_k = ccxt.krakenfutures()
        # Kraken usa simbolos distintos, ej: PI_XBTUSD
        ticker_k = exchange_k.fetch_ticker("PF_XBTUSD") # Hardcoded BTC para fallback
        fr = float(ticker_k['info'].get('fundingRate', 0)) * 100
        oi_val = float(ticker_k['info'].get('openInterest', 0))
        return fr, oi_val, "KrakenFut"
    except: pass

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
    st.title("🦁 QUIMERA v16.0")
    st.markdown(f"<div style='font-size:12px; margin-bottom:10px;'><span class='status-dot-on'>●</span> SYSTEM ONLINE</div>", unsafe_allow_html=True)
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
    if not exchange: return None, 0, None
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
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
# 5. IA ANALISTA (SYSTEM LOGIC, NOT AI)
# -----------------------------------------------------------------------------
def generate_detailed_ai_analysis_html(row, mtf_trends, mtf_score, obi, fr, open_interest, data_src):
    # 1. CONTEXTO MULTI-TIMEFRAME
    t_15m = mtf_trends.get('15m', 'NEUTRO')
    t_1h = mtf_trends.get('1h', 'NEUTRO')
    t_4h = mtf_trends.get('4h', 'NEUTRO')
    
    if mtf_score == 3: context = "<span style='color:#00FF00'>ALCISTA FUERTE</span> (Alineación Total)"
    elif mtf_score == -3: context = "<span style='color:#FF4444'>BAJISTA FUERTE</span> (Alineación Total)"
    elif t_4h == "BULL" and t_15m == "BEAR": context = "<span style='color:#FFFF00'>CORRECCIÓN EN CURSO</span> (Macro Alcista / Micro Bajista)"
    elif t_4h == "BEAR" and t_15m == "BULL": context = "<span style='color:#FFFF00'>REBOTE TÉCNICO</span> (Macro Bajista / Micro Alcista)"
    else: context = "MERCADO MIXTO (Conflicto de Temporalidades)"
    
    # 2. DATOS DERIVADOS
    deriv_txt = f"Funding Rate: <b style='color:#fff'>{fr:.4f}%</b>"
    if fr > 0.01: deriv_txt += " (Long Squeeze Risk)"
    elif fr < -0.01: deriv_txt += " (Short Squeeze Risk)"
    else: deriv_txt += " (Saludable)"
    
    if open_interest > 1000000000: oi_fmt = f"${open_interest/1000000000:.2f}B"
    elif open_interest > 1000000: oi_fmt = f"${open_interest/1000000:.2f}M"
    else: oi_fmt = f"${open_interest:,.0f}"
    
    oi_txt = f"Interés Abierto: <b style='color:#44AAFF'>{oi_fmt}</b>"

    # 3. MOMENTO (TSI / MFI / ADX)
    mfi = row['MFI']
    adx = row['ADX_14']
    tsi = row['TSI']
    
    gas_status = "LLENO" if mfi > 60 else "RESERVA" if mfi < 40 else "MEDIO"
    gas_color = "#00FF00" if mfi > 60 else "#FF4444" if mfi < 40 else "#FFF"
    
    tsi_status = "ALCISTA" if tsi > 0 else "BAJISTA"
    tsi_color = "#00FF00" if tsi > 0 else "#FF4444"
    
    mom_txt = f"Gasolina (MFI): <b style='color:{gas_color}'>{gas_status}</b>. ADX: {adx:.1f}. TSI: <b style='color:{tsi_color}'>{tsi_status}</b> ({tsi:.2f})."

    pressure = "COMPRADORA" if obi > 0.05 else "VENDEDORA" if obi < -0.05 else "NEUTRA"
    obi_color = "#00FF00" if obi > 0.05 else "#FF4444" if obi < -0.05 else "#aaa"
    obi_txt = f"Presión Libro: <b style='color:{obi_color}'>{pressure}</b> ({obi*100:.1f}%)"

    html = f"""
    <div class='ai-box'>
        <span class='ai-title'>🤖 QUIMERA COPILOT (Data Source: {data_src}):</span>
        <div style='margin-top:5px;'>📡 <b>ESTRUCTURA:</b> {context}</div>
        <div>📊 <b>DERIVADOS:</b> {deriv_txt}. {oi_txt}</div>
        <div>🔥 <b>MOMENTO:</b> {mom_txt}</div>
        <div>⛽ <b>VOLUMEN:</b> {obi_txt}</div>
    </div>
    """
    return html

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
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
        if obi > 0.05: score += 1; details.append("<span class='badge-bull'>OBI</span>")
        elif obi < -0.05: score -= 1; details.append("<span class='badge-bear'>OBI</span>")
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
            send_telegram_msg(f"💰 CIERRE {symbol}: {close_reason}\nPnL: ${pnl:.2f}")
            updated = True
    if updated or use_time_stop: save_trades(df)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

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
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi, 'use_tsi': use_tsi}
    signal, atr, prob, thermo_score, details_list = run_strategy(df, obi, trend_4h, filters)
    current_price, cur_high, cur_low = df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    mfi_val, adx_val = df['MFI'].iloc[-1], df['ADX_14'].iloc[-1]
    
    # DATOS FIX (Coinglass / Bybit / dYdX)
    fng_val, fng_label = get_fear_and_greed()
    news = get_crypto_news()
    fr, open_interest, data_src = get_deriv_data(symbol)
    mtf_trends, mtf_score = get_mtf_trends_analysis(symbol)
    
    # IA (HTML Correcto + TSI)
    ai_html = generate_detailed_ai_analysis_html(df.iloc[-1], mtf_trends, mtf_score, obi, fr, open_interest, data_src)
    
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

        # HTML IA
        st.markdown(ai_html, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        
        # WIDGETS
        c2.metric("Funding Rate", f"{fr:.4f}%", delta_color="inverse")
        
        # Formateo Open Interest
        if open_interest > 1000000000: oi_show = f"${open_interest/1000000000:.2f}B"
        elif open_interest > 1000000: oi_show = f"${open_interest/1000000:.2f}M"
        else: oi_show = f"${open_interest:,.0f}"
        c3.metric("Open Interest", oi_show)
        
        # MTF
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
        
        # Titulo dinámico
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
