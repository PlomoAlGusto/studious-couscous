import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timezone
import time
import numpy as np
import os
import feedparser

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v14.6 Analyst", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    /* ESTILOS GENERALES */
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* COLORES DE TEXTO */
    .t-green { color: #00FF00; font-weight: bold; }
    .t-red { color: #FF4444; font-weight: bold; }
    .t-blue { color: #44AAFF; font-weight: bold; }
    .t-gold { color: #FFD700; font-weight: bold; }
    
    /* CAJA DE ANÁLISIS IA MEJORADA */
    .analyst-report {
        background-color: #111;
        border-left: 4px solid #44AAFF;
        border-radius: 6px;
        padding: 20px;
        font-family: 'Consolas', 'Courier New', monospace;
        color: #e0e0e0;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    .report-section { margin-bottom: 15px; border-bottom: 1px dashed #333; padding-bottom: 10px; }
    .report-title { font-size: 16px; color: #fff; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    
    /* SEMÁFORO MTF */
    .mtf-container {
        display: flex; justify-content: space-between; background: #222; padding: 10px; border-radius: 8px; border: 1px solid #444;
    }
    .mtf-box { text-align: center; width: 30%; }
    .mtf-label { font-size: 10px; color: #888; }
    .mtf-dot { font-size: 18px; }
    
    /* TABLA OPEN TRADES */
    .stDataFrame { border: 1px solid #333; border-radius: 5px; }
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
# 2. MOTORES DE DATOS (AVANZADO)
# -----------------------------------------------------------------------------
def init_exchange():
    # Instancia Privada (Para posibles ejecuciones reales o datos spot)
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"]})
        else:
            ex = ccxt.binance() # Pública
        ex.load_markets()
        return ex
    except:
        return ccxt.kraken() # Fallback

exchange = init_exchange()

@st.cache_data(ttl=60)
def get_futures_data(symbol):
    """
    Intenta obtener Funding Rate y OI usando una instancia específica de Binance Futures 
    para evitar errores de la API Spot.
    """
    try:
        # Instancia específica solo para datos públicos de Futuros
        futures_ex = ccxt.binanceusdm({'enableRateLimit': True}) 
        ticker = futures_ex.fetch_ticker(symbol)
        
        # Funding Rate
        fr = ticker.get('info', {}).get('lastFundingRate', 0)
        fr_val = float(fr) * 100
        
        # Open Interest (A veces requiere endpoint específico)
        try:
            oi_data = futures_ex.fetch_open_interest(symbol)
            oi_val = float(oi_data['openInterestAmount']) # En monedas
            oi_usd = float(oi_data['openInterestValue'])  # En USDT
        except:
            oi_usd = 0.0
            
        return fr_val, oi_usd
    except Exception as e:
        # Fallback silencioso si falla la conexión
        return 0.0, 0.0

@st.cache_data(ttl=15)
def get_mtf_analysis(symbol):
    """Analiza tendencias en 3 temporalidades"""
    ex = ccxt.binance()
    timeframes = ['15m', '1h', '4h']
    trends = {}
    
    for tf in timeframes:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            # Estrategia simple para determinar tendencia MTF: Precio vs EMA 50
            ema = ta.ema(df['c'], length=50).iloc[-1]
            close = df['c'].iloc[-1]
            trends[tf] = "BULL" if close > ema else "BEAR"
        except:
            trends[tf] = "NEUTRO"
    return trends

@st.cache_data(ttl=15)
def get_market_data(symbol, tf):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=300)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Order Book Imbalance
        book = exchange.fetch_order_book(symbol, limit=20)
        bids = sum([x[1] for x in book['bids']])
        asks = sum([x[1] for x in book['asks']])
        obi = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
        
        return df, obi
    except:
        return None, 0

# -----------------------------------------------------------------------------
# 3. CAPA LÓGICA
# -----------------------------------------------------------------------------
def calculate_indicators(df):
    if df is None: return None
    # Trend
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    
    # Volatility & Momentum
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1)
    df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    # VWAP
    try:
        vp = ((df['high'] + df['low'] + df['close'])/3) * df['volume']
        df['VWAP'] = vp.cumsum() / df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    
    # Ichimoku (Solo visual)
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1)
    
    return df.fillna(method='bfill').fillna(method='ffill')

def generate_smart_analysis(df, mtf_trends, fr, oi, obi):
    row = df.iloc[-1]
    
    # 1. COHERENCIA MTF
    bull_count = list(mtf_trends.values()).count("BULL")
    bear_count = list(mtf_trends.values()).count("BEAR")
    
    if bull_count == 3: trend_status = "<span class='t-green'>ALCISTA FUERTE (Full Alignment)</span>"
    elif bear_count == 3: trend_status = "<span class='t-red'>BAJISTA FUERTE (Full Alignment)</span>"
    elif mtf_trends['4h'] == "BULL" and mtf_trends['15m'] == "BEAR": trend_status = "<span class='t-gold'>RETROCESO EN TENDENCIA ALCISTA</span> (Posible entrada Long)"
    elif mtf_trends['4h'] == "BEAR" and mtf_trends['15m'] == "BULL": trend_status = "<span class='t-gold'>REBOTE EN TENDENCIA BAJISTA</span> (Posible entrada Short)"
    else: trend_status = "LATERAL / INDECISIÓN"

    # 2. ANÁLISIS DE DERIVADOS
    deriv_msg = ""
    if fr > 0.01: deriv_msg = "Funding Rate muy positivo. El mercado está sobre-apalancado en Longs. <b>Peligro de Long Squeeze.</b>"
    elif fr < -0.01: deriv_msg = "Funding Rate muy negativo. Posible <b>Short Squeeze</b> inminente."
    else: deriv_msg = "Funding Rate saludable y equilibrado."
    
    # 3. MOMENTO Y VOLUMEN
    adx = row['ADX_14']
    if adx > 25: mom_txt = f"La tendencia tiene fuerza (ADX {adx:.1f})."
    else: mom_txt = "Mercado sin dirección clara (Rango)."
    
    obi_pct = obi * 100
    obi_txt = f"Libro de órdenes inclinado un {abs(obi_pct):.1f}% hacia {'COMPRAS' if obi>0 else 'VENTAS'}."

    # HTML CONSTRUIDO
    html = f"""
    <div class='analyst-report'>
        <div class='report-section'>
            <div class='report-title'>📡 ESTRUCTURA MULTI-TIMEFRAME</div>
            <div>Estado: {trend_status}</div>
            <div style='font-size:12px; color:#888; margin-top:5px;'>
                15m: {mtf_trends['15m']} | 1h: {mtf_trends['1h']} | 4h: {mtf_trends['4h']}
            </div>
        </div>
        
        <div class='report-section'>
            <div class='report-title'>📊 SALUD DEL MERCADO</div>
            <div>{deriv_msg}</div>
            <div>Interés Abierto: <span class='t-blue'>${oi/1_000_000:.1f}M</span></div>
        </div>
        
        <div class='report-section' style='border-bottom:none;'>
            <div class='report-title'>⛽ COMBUSTIBLE (VOLUMEN)</div>
            <div>{mom_txt} {obi_txt}</div>
            <div>RSI: {row['RSI']:.1f} | MFI: {row['MFI']:.1f}</div>
        </div>
    </div>
    """
    return html

def run_strategy(row, obi, mtf_trends, filters):
    score = 0
    max_score = 0
    
    # PUNTUACIÓN
    if filters['use_mtf']:
        max_score += 2
        if mtf_trends['4h'] == "BULL": score += 2
        elif mtf_trends['4h'] == "BEAR": score -= 2
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
    
    # DECISIÓN
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    # FILTROS (VETO)
    if filters['use_regime'] and row['ADX_14'] < 20: signal = "NEUTRO"
    
    # CONFLUENCIA MTF OBLIGATORIA
    if filters['use_mtf']:
        # Si quiero ir LONG, 4H no puede ser BEAR
        if signal == "LONG" and mtf_trends['4h'] == "BEAR": signal = "NEUTRO"
        # Si quiero ir SHORT, 4H no puede ser BULL
        if signal == "SHORT" and mtf_trends['4h'] == "BULL": signal = "NEUTRO"

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    return signal, row['ATR'], prob

# -----------------------------------------------------------------------------
# 6. GESTIÓN Y EJECUCIÓN
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

def save_trades(df): df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr, leverage):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": "BTC/USDT", "type": type, "entry": entry, "size": size, "leverage": leverage, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": "Entry", "candles_held": 0, "atr_entry": atr}
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
            if current_price >= row['tp3']: close_reason, pnl = "TP3 🚀", (row['tp3'] - row['entry']) * row['size']
            elif current_price <= row['sl']: close_reason, pnl = "SL 🛑", (row['sl'] - row['entry']) * row['size']
        else:
            if current_price <= row['tp3']: close_reason, pnl = "TP3 🚀", (row['entry'] - row['tp3']) * row['size']
            elif current_price >= row['sl']: close_reason, pnl = "SL 🛑", (row['entry'] - row['sl']) * row['size']

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            updated = True
    if updated: save_trades(df)

# -----------------------------------------------------------------------------
# 7. DASHBOARD
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v14.6")
    st.caption("Deep Analyst Edition")
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("TF", ["15m", "1h"], index=0)
    
    with st.expander("FILTROS", expanded=True):
        use_ema = st.checkbox("EMA Trend", True)
        use_mtf = st.checkbox("MTF Filter", True)
        use_vwap = st.checkbox("VWAP", True)
        use_ichi = st.checkbox("Ichimoku", False)
        use_regime = st.checkbox("ADX Regime", True)
        use_rsi = st.checkbox("RSI Filter", False)
        use_obi = st.checkbox("OBI", True)
    
    current_balance = get_current_balance()
    st.metric("Balance", f"${current_balance:,.0f}")
    risk_per_trade = st.slider("Riesgo %", 0.5, 5.0, 1.0)
    auto_refresh = st.checkbox("Auto-Refresh", False)

# LOGICA PRINCIPAL
df, obi = get_market_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    mtf_trends = get_mtf_analysis(symbol)
    fr, oi = get_futures_data(symbol)
    
    filters = {'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap, 'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi, 'use_obi': use_obi}
    signal, atr, prob = run_strategy(df.iloc[-1], obi, mtf_trends, filters)
    
    current_price = df['close'].iloc[-1]
    ai_html = generate_smart_analysis(df, mtf_trends, fr, oi, obi)
    
    # CALCULOS DE ENTRADA
    setup = None
    qty, leverage = 0, 1.0
    
    if signal != "NEUTRO":
        sl_dist = atr * 1.5
        risk_amount = current_balance * (risk_per_trade / 100)
        qty = risk_amount / sl_dist
        leverage = max(1.0, (qty * current_price) / current_balance)
        
        if signal == "LONG":
            sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+sl_dist, current_price+(sl_dist*2), current_price+(sl_dist*3.5)
            emoji = "⬆️ LONG"
        else:
            sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-sl_dist, current_price-(sl_dist*2), current_price-(sl_dist*3.5)
            emoji = "⬇️ SHORT"
        
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'qty': qty, 'lev': leverage}

    manage_open_positions(current_price)

    # --- VISUALIZACIÓN ---
    
    # 1. DATA WIDGETS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${current_price:,.2f}")
    c2.metric("Funding Rate", f"{fr:.4f}%", delta_color="inverse")
    c3.metric("Open Interest", f"${oi/1000000:.1f}M")
    
    # MTF VISUAL WIDGET
    with c4:
        cols = st.columns(3)
        colors = {"BULL": "🟢", "BEAR": "🔴", "NEUTRO": "⚪"}
        cols[0].markdown(f"<div style='text-align:center'><div style='font-size:10px'>15m</div><div>{colors[mtf_trends['15m']]}</div></div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='text-align:center'><div style='font-size:10px'>1h</div><div>{colors[mtf_trends['1h']]}</div></div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='text-align:center'><div style='font-size:10px'>4h</div><div>{colors[mtf_trends['4h']]}</div></div>", unsafe_allow_html=True)

    # 2. TABS
    t1, t2 = st.tabs(["TABLERO", "EJECUCIÓN"])
    
    with t1:
        # IA REPORT
        st.markdown(ai_html, unsafe_allow_html=True)
        
        # SIGNAL CARD
        if setup:
            st.info(f"🔥 SEÑAL {setup['dir']} DETECTADA | Prob: {prob:.1f}% | Lev: {setup['lev']:.1f}x")
            cols = st.columns(4)
            cols[0].metric("Entry", f"{setup['entry']:.2f}")
            cols[1].metric("SL", f"{setup['sl']:.2f}")
            cols[2].metric("TP1", f"{setup['tp1']:.2f}")
            cols[3].button(f"EJECUTAR ({setup['qty']:.3f})", on_click=lambda: execute_trade(signal, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr, setup['lev']))

        # CHART
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        if filters['use_vwap']: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        df_trades = load_trades()
        if not df_trades.empty:
            open_ops = df_trades[df_trades['status'] == 'OPEN']
            st.markdown("### 🟢 Operaciones Abiertas")
            st.dataframe(open_ops, use_container_width=True)
            
            closed_ops = df_trades[df_trades['status'] == 'CLOSED']
            st.markdown("### 📜 Historial")
            st.dataframe(closed_ops, use_container_width=True)
        else:
            st.info("No hay datos de trading.")

else:
    st.warning("Conectando a los nodos del mercado...")

if auto_refresh: time.sleep(60); st.rerun()
