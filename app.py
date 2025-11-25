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
st.set_page_config(page_title="Quimera Pro v13 Chronos", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    
    /* CAJA PRINCIPAL DEL TRADE */
    .trade-setup {
        background-color: #151515; 
        padding: 20px; 
        border-radius: 15px; 
        border: 1px solid #444;
        margin-top: 10px; 
        margin-bottom: 20px; 
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    .label-mini { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    
    .header-confirmed-long { color: #00FF00; font-size: 20px; font-weight: 900; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .header-confirmed-short { color: #FF4444; font-size: 20px; font-weight: 900; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .header-potential { color: #FFFF00; font-size: 18px; font-weight: bold; border-bottom: 1px dashed #555; padding-bottom: 10px; }
    
    .ai-box {
        background-color: #223344; border-left: 5px solid #44AAFF; padding: 15px; border-radius: 5px; margin-bottom: 15px; font-family: monospace;
    }
    
    .news-box {
        background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 10px; margin-bottom: 15px; height: 250px; overflow-y: auto;
    }
    .news-item { padding: 8px 0; border-bottom: 1px solid #222; font-size: 14px; }
    .news-link { text-decoration: none; color: #ddd; }
    .news-link:hover { color: #44AAFF; }
    .news-time { color: #666; font-size: 11px; margin-right: 5px; font-weight: bold;}
    
    /* ESTILO RELOJES */
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ARCHIVOS
CSV_FILE = 'paper_trades.csv'
if not os.path.exists(CSV_FILE):
    df_empty = pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
    df_empty.to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"
if 'balance' not in st.session_state: st.session_state.balance = 10000.0

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN (SIDEBAR MODULAR)
# -----------------------------------------------------------------------------
def get_market_sessions():
    """Calcula qué mercados están abiertos ahora (aprox UTC)"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    sessions = {
        "🇬🇧 LONDRES": (8, 16),
        "🇺🇸 NEW YORK": (13, 21),
        "🇯🇵 TOKYO": (0, 9),
        "🇦🇺 SYDNEY": (22, 7)
    }
    
    st.sidebar.markdown("### 🌍 SESIONES")
    for name, (start, end) in sessions.items():
        is_open = False
        if start < end:
            is_open = start <= hour < end
        else:
            is_open = hour >= start or hour < end
            
        status_icon = "🟢 ABIERTO" if is_open else "🔴 CERRADO"
        css_class = "clock-open" if is_open else "clock-closed"
        
        st.sidebar.markdown(f"""
        <div class='market-clock {css_class}'>
            <span>{name}</span>
            <span>{status_icon}</span>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🦁 QUIMERA v13.3")
    st.caption("Chronos Edition ⏳")
    
    get_market_sessions()
    st.divider()
    
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)

    with st.expander("🛡️ FILTROS DE ENTRADA", expanded=True):
        use_ema = st.checkbox("Tendencia Base (EMAs)", True)
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP (Institucional)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)

    with st.expander("🌊 MOMENTO Y VOLUMEN"):
        use_rsi = st.checkbox("RSI & Stoch", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        
    with st.expander("💰 GESTIÓN DE RIESGO"):
        st.caption("Calculadora de Lotes Automática")
        account_size = st.number_input("Tamaño Cuenta ($)", value=1000.0)
        risk_per_trade = st.slider("Riesgo (%)", 0.5, 5.0, 1.0)
        
    with st.expander("⚙️ SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 3. CAPA DE DATOS
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
        news_items = []
        for entry in feed.entries[:5]:
            news_items.append({"title": entry.title, "link": entry.link, "published": entry.published_parsed})
        return news_items
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

# -----------------------------------------------------------------------------
# 4. CAPA LÓGICA
# -----------------------------------------------------------------------------
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
    
    high_w = df['high'].rolling(20).max()
    low_w = df['low'].rolling(20).min()
    close_w = df['close']
    df['PIVOT'] = (high_w + low_w + close_w) / 3
    df['R1'] = (2 * df['PIVOT']) - low_w
    df['S1'] = (2 * df['PIVOT']) - high_w
    
    return df.fillna(method='bfill').fillna(method='ffill')

def detect_candle_patterns(row, prev_row):
    patterns = []
    body_size = abs(row['close'] - row['open'])
    full_range = row['high'] - row['low']
    if full_range > 0 and (body_size / full_range) < 0.1: patterns.append("🕯️ Doji")
    lower_wick = min(row['close'], row['open']) - row['low']
    upper_wick = row['high'] - max(row['close'], row['open'])
    if lower_wick > (body_size * 2) and upper_wick < body_size: patterns.append("🔨 Hammer")
    if row['close'] > row['open'] and prev_row['close'] < prev_row['open']:
        if row['close'] > prev_row['open'] and row['open'] < prev_row['close']: patterns.append("🔥 Engulfing")
    return patterns

def generate_ai_analysis(row, prev_row, trend_4h, obi, signal, prob, fng_val, fng_class):
    analysis = []
    if trend_4h == "BULLISH": analysis.append("Estructura Macro (4H): ALCISTA.")
    elif trend_4h == "BEARISH": analysis.append("Estructura Macro (4H): BAJISTA.")
    
    mfi = row['MFI']
    if mfi > 60: analysis.append("⛽ Flujo dinero POSITIVO.")
    elif mfi < 40: analysis.append("🪫 Flujo dinero NEGATIVO.")
    
    patterns = detect_candle_patterns(row, prev_row)
    if patterns: analysis.append(f"Patrón: {', '.join(patterns)}")
    
    if signal != "NEUTRO":
        direction = "SUBIDA" if signal == "LONG" else "BAJADA"
        analysis.append(f"🎯 CONCLUSIÓN: Probabilidad {prob:.1f}% de {direction}.")
    else:
        analysis.append("⏳ CONCLUSIÓN: Mercado indeciso. Esperar ruptura.")
        
    return " | ".join(analysis)

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
    score = 0
    max_score = 0
    
    # Sistema de Puntuación (Termómetro)
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2
        elif trend_4h == "BEARISH": score -= 2
        else: max_score -= 2 

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
    
    # Logica de Señal
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    if filters['use_rsi'] and (row['RSI'] > 70 and signal == "LONG"): signal = "NEUTRO"
    if filters['use_rsi'] and (row['RSI'] < 30 and signal == "SHORT"): signal = "NEUTRO"
    if filters['use_regime'] and row['ADX_14'] < 20: signal = "NEUTRO"
    
    if filters['use_mtf'] and signal == "LONG" and trend_4h == "BEARISH": signal = "NEUTRO"
    if filters['use_mtf'] and signal == "SHORT" and trend_4h == "BULLISH": signal = "NEUTRO"

    # Probabilidad
    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    # Puntuación para el Termómetro (-10 a +10)
    thermometer_score = 0
    if max_score > 0:
        thermometer_score = (score / max_score) * 100 # -100 a 100
    
    return signal, row['ATR'], prob, thermometer_score

# -----------------------------------------------------------------------------
# 5. GESTIÓN PAPER TRADING
# -----------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty: return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
        return df
    except: return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])

def save_trades(df):
    df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": symbol, "type": type, "entry": entry, "size": size, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": "Entry", "candles_held": 0, "atr_entry": atr}
    df = pd.concat([pd.DataFrame([new]), df], ignore_index=True)
    save_trades(df)
    return new

def manage_open_positions(current_price, current_high, current_low):
    """
    Gestiona operaciones abiertas revisando High/Low para detectar mechas y cerrar correctamente.
    """
    df = load_trades()
    if df.empty: return
    
    # Filtramos solo las abiertas
    open_idx = df.index[df['status'] == "OPEN"].tolist()
    updated = False
    
    for idx in open_idx:
        row = df.loc[idx]
        close_reason = ""
        pnl = 0
        
        # --- LÓGICA LONG ---
        if row['type'] == "LONG":
            # 1. Trailing Stop
            if use_trailing:
                new_sl = current_price - (row['atr_entry'] * 1.5)
                if new_sl > row['sl']: df.at[idx, 'sl'] = new_sl
            
            # 2. Breakeven al tocar TP1 (Asegurar entrada)
            if use_breakeven and current_high >= row['tp1'] and row['sl'] < row['entry']:
                df.at[idx, 'sl'] = row['entry'] * 1.001 

            # 3. Revisión de Salidas (Usamos High/Low para detectar mechas)
            if current_high >= row['tp3']: 
                close_reason="TP3 (Final) 🚀"
                pnl = (row['tp3'] - row['entry']) * row['size']
            elif current_low <= row['sl']: 
                close_reason="SL 🛑"
                pnl = (row['sl'] - row['entry']) * row['size']
        
        # --- LÓGICA SHORT ---
        else:
            # 1. Trailing Stop
            if use_trailing:
                new_sl = current_price + (row['atr_entry'] * 1.5)
                if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
            
            # 2. Breakeven al tocar TP1 (Asegurar entrada)
            if use_breakeven and current_low <= row['tp1'] and row['sl'] > row['entry']:
                df.at[idx, 'sl'] = row['entry'] * 0.999 

            # 3. Revisión de Salidas (Usamos High/Low para detectar mechas)
            if current_low <= row['tp3']: 
                close_reason="TP3 (Final) 🚀"
                pnl = (row['entry'] - row['tp3']) * row['size']
            elif current_high >= row['sl']: 
                close_reason="SL 🛑"
                pnl = (row['entry'] - row['sl']) * row['size']

        # 4. Time Stop (Cierre por tiempo si no hay progreso)
        if not close_reason and use_time_stop:
            df.at[idx, 'candles_held'] += 1
            current_pnl_calc = (current_price - row['entry']) * row['size'] if row['type'] == "LONG" else (row['entry'] - current_price) * row['size']
            if df.at[idx, 'candles_held'] > 12 and current_pnl_calc < 0:
                close_reason = "Time Stop ⏳"
                pnl = current_pnl_calc

        # EJECUTAR CIERRE SI CORRESPONDE
        if close_reason:
            df.at[idx, 'status'] = "CLOSED"
            df.at[idx, 'pnl'] = pnl
            df.at[idx, 'reason'] = close_reason
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
    closed['equity'] = 10000 + closed['cumulative_pnl']
    start = pd.DataFrame([{'time': 'Inicio', 'equity': 10000}])
    curve = pd.concat([start, closed[['time', 'equity']]])
    total_profit = closed['pnl'].sum()
    
    fig = px.area(curve, x='time', y='equity', title="Curva de Capital (Equity Curve)")
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0))
    color = '#00FF00' if total_profit >= 0 else '#FF4444'
    fig.update_traces(line_color=color, fillcolor=color.replace("FF", "22") if total_profit>=0 else color.replace("44", "11"))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    
    filters = {
        'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap,
        'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi,
        'use_obi': use_obi
    }
    
    signal, atr, prob, thermo_score = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    cur_high = df['high'].iloc[-1]
    cur_low = df['low'].iloc[-1]
    mfi_val = df['MFI'].iloc[-1]
    
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

    qty = 0
    if calc_dir:
        sl_dist = atr * 1.5
        risk = sl_dist
        risk_amount = account_size * (risk_per_trade / 100)
        if risk > 0: qty = risk_amount / risk
        else: qty = 0
        
        if calc_dir == "LONG":
            sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            emoji = "⬆️ LONG"
        else:
            sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            emoji = "⬇️ SHORT"
        
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'status': setup_type, 'qty': qty}

    if signal != "NEUTRO" and signal != st.session_state.last_alert and setup:
        msg = f"""🦁 *QUIMERA SIGNAL: {signal}*
Activo: {symbol}
Prob: {prob:.1f}%

🔥 *OPERACIÓN: {setup['dir']}*
⚖️ *Lote:* {qty:.4f}

🔵 *ENTRADA:* ${current_price:.2f}
🛑 *SL:* ${setup['sl']:.2f}

🎯 *TP1:* ${setup['tp1']:.2f}
🎯 *TP2:* ${setup['tp2']:.2f}
🚀 *TP3:* ${setup['tp3']:.2f}
"""
        send_telegram_msg(msg)
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    # -----------------------------------------------------
    # LÓGICA CORREGIDA: Pasar High/Low para mechas
    # -----------------------------------------------------
    manage_open_positions(current_price, cur_high, cur_low)
    
    tab1, tab2 = st.tabs(["📊 LIVE COMMAND", "🧪 PAPER TRADING"])
    
    with tab1:
        # PANEL SUPERIOR: NOTICIAS + TERMOMETRO
        col_news, col_thermo = st.columns([2, 1])
        with col_news:
            if news:
                st.markdown("### 📰 MARKET FLASH")
                news_html = "<div class='news-box'>"
                for n in news:
                    t_struct = n.get('published', time.gmtime())
                    t_str = f"{t_struct.tm_hour:02}:{t_struct.tm_min:02}"
                    news_html += f"<div class='news-item'><span class='news-time'>{t_str}</span> <a href='{n['link']}' target='_blank' class='news-link'>{n['title']}</a></div>"
                news_html += "</div>"
                st.markdown(news_html, unsafe_allow_html=True)
        
        with col_thermo:
            st.markdown("### 🎛️ TERMÓMETRO TÉCNICO")
            fig_thermo = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = thermo_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<b>Fuerza Técnica</b>"},
                gauge = {
                    'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [-100, -40], 'color': "#FF4444"},
                        {'range': [-40, 40], 'color': "gray"},
                        {'range': [40, 100], 'color': "#00FF00"}],
                }
            ))
            fig_thermo.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#111", font={'color': "white"})
            st.plotly_chart(fig_thermo, use_container_width=True)

        st.markdown(f"<div class='ai-box'>🤖 <b>QUIMERA COPILOT:</b><br>{ai_narrative}</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Tendencia 4H", trend_4h, delta="Bullish" if trend_4h=="BULLISH" else "Bearish")
        c3.metric("OBI", f"{obi:.1%}")
        
        gas_state = "⚖️ Neutro"
        if mfi_val > 60: gas_state = "⛽ Tanque Lleno"
        elif mfi_val < 40: gas_state = "🪫 Reserva"
        c4.metric("GASOLINA (MFI)", f"{mfi_val:.0f}", gas_state)

        if setup:
            # --- CORRECCIÓN HTML DE LA TARJETA ---
            prob_str = f"{prob:.1f}%"
            if prob >= 80: prob_color = "#00FF00"
            elif prob >= 60: prob_color = "#FFFF00"
            else: prob_color = "#FF4444"

            if setup['status'] == "CONFIRMED":
                header_cls = "header-confirmed-long" if calc_dir == "LONG" else "header-confirmed-short"
                header_txt = f"🔥 CONFIRMADO: {setup['dir']}"
                btn_label = f"🚀 EJECUTAR {calc_dir} ({setup['qty']:.4f})"
            else:
                header_cls = "header-potential"
                header_txt = f"⚠️ POTENCIAL: {setup['dir']}"
                btn_label = f"⚠️ FORZAR ENTRADA"

            # Construimos el HTML sin indentación para evitar el error de "código raw"
            html_card = f"""
<div class="trade-setup">
    <div class="{header_cls}">{header_txt}</div>
    <div style='margin-top: 5px; margin-bottom: 10px; text-align: left;'>
        <div style='display:flex; justify-content:space-between; color:#ccc; font-size:12px; margin-bottom:2px;'>
            <span>Probabilidad de Éxito:</span>
            <span style='color:{prob_color}; font-weight:bold;'>{prob_str}</span>
        </div>
        <div style='width: 100%; background-color: #333; border-radius: 4px; height: 6px;'>
            <div style='width: {prob}%; background-color: {prob_color}; height: 6px; border-radius: 4px; box-shadow: 0 0 5px {prob_color};'></div>
        </div>
    </div>
    <p style="color:#888; font-size:14px;">Posición Sugerida: <span style="color:white; font-weight:bold">{setup['qty']:.4f} {symbol.split('/')[0]}</span> (Riesgo ${risk_amount:.1f})</p>
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
                execute_trade(calc_dir, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr)
                st.success(f"Orden {calc_dir} lanzada.")
        else:
            st.info("Esperando estructura de mercado clara...")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        last_pivot = df.iloc[-1]['PIVOT']
        last_s1 = df.iloc[-1]['S1']
        last_r1 = df.iloc[-1]['R1']
        fig.add_hline(y=last_pivot, line_dash="dash", line_color="gray", annotation_text="Pivote", row=1, col=1)
        fig.add_hline(y=last_s1, line_dash="dot", line_color="green", annotation_text="S1", row=1, col=1)
        fig.add_hline(y=last_r1, line_dash="dot", line_color="red", annotation_text="R1", row=1, col=1)

        if setup:
            fig.add_hline(y=setup['tp1'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_trades = load_trades()
        st.subheader("📈 Rendimiento")
        render_analytics(df_trades)
        st.divider()
        if not df_trades.empty:
            open_trades = df_trades[df_trades['status'] == "OPEN"].copy()
            closed_trades = df_trades[df_trades['status'] == "CLOSED"]
            
            st.subheader("🟢 Posiciones Abiertas")
            if not open_trades.empty:
                # Calculamos PnL Flotante para ver ganancias en tiempo real
                open_trades['Floating PnL'] = np.where(
                    open_trades['type'] == 'LONG',
                    (current_price - open_trades['entry']) * open_trades['size'],
                    (open_trades['entry'] - current_price) * open_trades['size']
                )
                
                def color_floating(val):
                    color = '#00FF00' if val > 0 else '#FF4444'
                    return f'color: {color}; font-weight: bold;'
                
                # Mostramos columnas clave + el PnL flotante
                cols_to_show = ['time', 'symbol', 'type', 'entry', 'size', 'sl', 'tp3', 'Floating PnL']
                st.dataframe(open_trades[cols_to_show].style.applymap(color_floating, subset=['Floating PnL']), use_container_width=True)
            else:
                st.info("No hay operaciones abiertas.")

            st.subheader("📜 Historial Cerrado")
            def color_pnl(val):
                color = '#228B22' if val > 0 else '#B22222' if val < 0 else 'white'
                return f'color: {color}'
            if not closed_trades.empty:
                st.dataframe(closed_trades.style.applymap(color_pnl, subset=['pnl']), use_container_width=True)
                total_pnl = closed_trades['pnl'].sum()
                st.metric("PnL Total Acumulado", f"${total_pnl:.2f}", delta_color="normal")
        else:
            st.info("Historial vacío.")

else: st.warning("Cargando datos...")

if auto_refresh: time.sleep(60); st.rerun()
