import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import requests
from datetime import datetime
import time
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v6.3 Oracle", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.1); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
    
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
    
    /* Estilo para el nuevo Copilot */
    .ai-box {
        background-color: #223344;
        border-left: 5px solid #44AAFF;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ARCHIVOS (PERSISTENCIA CSV)
CSV_FILE = 'paper_trades.csv'
if not os.path.exists(CSV_FILE):
    df_empty = pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])
    df_empty.to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"
if 'balance' not in st.session_state: st.session_state.balance = 10000.0

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN (SIDEBAR MODULAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v6.3")
    st.caption("Oracle Edition")
    
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
        
    with st.expander("💰 GESTIÓN SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
        risk_per_trade = st.slider("Riesgo %", 0.5, 5.0, 1.0)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 3. CAPA DE DATOS (DATA ENGINE MTF)
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            ex.load_markets(); return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Visual)"

exchange, source_name = init_exchange()

@st.cache_data(ttl=15)
def get_mtf_data(symbol, tf_lower):
    if not exchange: return None, 0, None
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    
    # 1. Datos Principales
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf_lower, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None

    # 2. Datos Macro (4h)
    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(ticker_fix, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
        last_4h = df_4h.iloc[-1]
        if last_4h['close'] > last_4h['EMA_50']: trend_4h = "BULLISH"
        else: trend_4h = "BEARISH"
    except: pass

    # 3. OBI Live
    obi = 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids, asks = sum([x[1] for x in book['bids']]), sum([x[1] for x in book['asks']])
        if (bids + asks) > 0: obi = (bids - asks) / (bids + asks)
    except: pass
    
    return df, obi, trend_4h

# -----------------------------------------------------------------------------
# 4. CAPA LÓGICA (INDICADORES Y SEÑAL)
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
    
    return df.fillna(method='bfill').fillna(method='ffill')

# --- NUEVO MOTOR DE ANÁLISIS DE IA ---
def generate_ai_analysis(row, trend_4h, obi, signal, prob):
    analysis = []
    
    # 1. Análisis de Estructura
    if trend_4h == "BULLISH":
        analysis.append("La estructura macro (4H) es ALCISTA, lo que favorece compras.")
    elif trend_4h == "BEARISH":
        analysis.append("La estructura macro (4H) es BAJISTA, presión vendedora dominante.")
    
    # 2. Análisis de Fuerza (ADX)
    if row['ADX_14'] > 25:
        analysis.append(f"El mercado tiene una tendencia fuerte (ADX {row['ADX_14']:.1f}), movimientos explosivos probables.")
    else:
        analysis.append(f"Mercado en rango o consolidación (ADX {row['ADX_14']:.1f}). Peligro de señales falsas.")
        
    # 3. Análisis de Libro de Órdenes
    if obi > 0.1:
        analysis.append("Detecto fuerte interés comprador en el Order Book.")
    elif obi < -0.1:
        analysis.append("Detecto muro de ventas en el Order Book.")
        
    # 4. Conclusión
    if signal != "NEUTRO":
        direction = "SUBIDA" if signal == "LONG" else "BAJADA"
        analysis.append(f"🎯 CONCLUSIÓN: Alta probabilidad ({prob:.1f}%) de {direction}. El setup técnico está alineado.")
    else:
        analysis.append("⏳ CONCLUSIÓN: El mercado está indeciso o los filtros no confirman. Mejor esperar.")
        
    return " ".join(analysis)

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
    score = 0
    max_score = 0
    reasons = []

    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2
        elif trend_4h == "BEARISH": score -= 2
        else: max_score -= 2 

    if filters['use_ema']: 
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; reasons.append("Cruce EMA")
        else: score -= 1

    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; reasons.append("VWAP Support")
        else: score -= 1
        
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1; reasons.append("OrderBook Bull")
        elif obi < -0.05: score -= 1
    
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    if filters['use_rsi'] and (row['RSI'] > 70 and signal == "LONG"): signal = "NEUTRO"
    if filters['use_rsi'] and (row['RSI'] < 30 and signal == "SHORT"): signal = "NEUTRO"
    if filters['use_regime'] and row['ADX_14'] < 20: signal = "NEUTRO"; reasons = ["Rango (ADX < 20)"]
    
    if filters['use_mtf'] and signal == "LONG" and trend_4h == "BEARISH": signal = "NEUTRO"; reasons = ["Contra 4H"]
    if filters['use_mtf'] and signal == "SHORT" and trend_4h == "BULLISH": signal = "NEUTRO"; reasons = ["Contra 4H"]

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    return signal, reasons, row['ATR'], prob

# -----------------------------------------------------------------------------
# 5. PAPER TRADING
# -----------------------------------------------------------------------------
def load_trades():
    if os.path.exists(CSV_FILE): return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["id", "time", "symbol", "type", "entry", "size", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"])

def save_trades(df):
    df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr):
    df = load_trades()
    new_trade = {
        "id": int(time.time()),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "type": type,
        "entry": entry,
        "size": size,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "status": "OPEN",
        "pnl": 0.0,
        "reason": "Entry",
        "candles_held": 0,
        "atr_entry": atr
    }
    df = pd.concat([pd.DataFrame([new_trade]), df], ignore_index=True)
    save_trades(df)
    return new_trade

def manage_open_positions(current_price):
    df = load_trades()
    if df.empty: return
    
    open_trades_idx = df.index[df['status'] == "OPEN"].tolist()
    if not open_trades_idx: return

    updated = False
    for idx in open_trades_idx:
        row = df.loc[idx]
        close_reason = ""
        pnl = 0
        
        # 1. Time Stop
        if use_time_stop:
            df.at[idx, 'candles_held'] += 1
            if df.at[idx, 'candles_held'] > 12: 
                current_pnl = (current_price - row['entry']) if row['type'] == "LONG" else (row['entry'] - current_price)
                if current_pnl <= 0: close_reason = "Time Stop ⏳"
        
        # 2. Logic
        if not close_reason:
            if row['type'] == "LONG":
                if use_trailing:
                    new_sl = current_price - (row['atr_entry'] * 1.5)
                    if new_sl > row['sl']: df.at[idx, 'sl'] = new_sl
                if use_breakeven and current_price > (row['entry'] * 1.015):
                     if row['sl'] < row['entry']: df.at[index, 'sl'] = row['entry']
                
                if current_price >= row['tp3']: close_reason = "TP3 🚀"; pnl = (row['tp3']-row['entry'])*row['size']
                elif current_price <= row['sl']: close_reason = "SL 🛑"; pnl = (row['sl']-row['entry'])*row['size']
            
            else: # SHORT
                if use_trailing:
                    new_sl = current_price + (row['atr_entry'] * 1.5)
                    if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
                if use_breakeven and current_price < (row['entry'] * 0.985):
                     if row['sl'] > row['entry']: df.at[index, 'sl'] = row['entry']

                if current_price <= row['tp3']: close_reason = "TP3 🚀"; pnl = (row['entry']-row['tp3'])*row['size']
                elif current_price >= row['sl']: close_reason = "SL 🛑"; pnl = (row['entry']-row['sl'])*row['size']

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"
            df.at[idx, 'pnl'] = pnl
            df.at[idx, 'reason'] = close_reason
            st.toast(f"Cierre: {close_reason}")
            send_telegram_msg(f"💰 CIERRE {symbol}: {close_reason}\nPnL: ${pnl:.2f}")
            updated = True

    if updated or use_time_stop: save_trades(df)

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

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
    
    signal, reasons, atr, prob = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    
    # NUEVO COPILOT NARRATIVO
    ai_narrative = generate_ai_analysis(df.iloc[-1], trend_4h, obi, signal, prob)
    
    # --- CÁLCULO DE SETUP ---
    setup = None
    calc_dir = signal 
    setup_type = "CONFIRMED" if signal != "NEUTRO" else "POTENTIAL"
    
    if signal == "NEUTRO":
        if trend_4h == "BULLISH": calc_dir = "LONG"
        elif trend_4h == "BEARISH": calc_dir = "SHORT"
        else: calc_dir = None

    if calc_dir:
        sl_dist = atr * 1.5
        risk = sl_dist
        if calc_dir == "LONG":
            sl = current_price - sl_dist
            tp1, tp2, tp3 = current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            emoji = "⬆️ COMPRA"
        else:
            sl = current_price + sl_dist
            tp1, tp2, tp3 = current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            emoji = "⬇️ VENTA"
        
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'status': setup_type}

    # ALERTA TELEGRAM (RESTAURADA COMPLETA)
    if signal != "NEUTRO" and signal != st.session_state.last_alert and setup:
        msg = f"""🦁 *QUIMERA SIGNAL: {signal}*
Activo: {symbol}
Probabilidad: {prob:.1f}%

🔥 *OPERACIÓN: {setup['dir']}*

🔵 *ENTRADA:* ${current_price:.2f}

🎯 *TP1:* ${setup['tp1']:.2f}
🎯 *TP2:* ${setup['tp2']:.2f}
🚀 *TP3:* ${setup['tp3']:.2f}

🛑 *SL:* ${setup['sl']:.2f}
"""
        send_telegram_msg(msg)
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_open_positions(current_price)
    
    # --- UI ---
    tab1, tab2 = st.tabs(["📊 LIVE COMMAND", "🧪 PAPER TRADING"])
    
    with tab1:
        # CAJA IA EXTENDIDA
        st.markdown(f"<div class='ai-box'>🤖 <b>QUIMERA COPILOT:</b><br>{ai_narrative}</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Tendencia 4H", trend_4h, delta="Bullish" if trend_4h=="BULLISH" else "Bearish", delta_color="normal")
        c3.metric("OBI", f"{obi:.1%}")
        c4.metric("Probabilidad", f"{prob:.1f}%")

        if setup:
            if setup['status'] == "CONFIRMED":
                header_cls = "header-confirmed-long" if calc_dir == "LONG" else "header-confirmed-short"
                header_txt = f"🔥 CONFIRMADO: {setup['dir']}"
                btn_label = f"🚀 EJECUTAR {calc_dir}"
            else:
                header_cls = "header-potential"
                header_txt = f"⚠️ SETUP POTENCIAL: {setup['dir']}"
                btn_label = f"⚠️ FORZAR ENTRADA"

            st.markdown(f"""
            <div class="trade-setup">
                <div class="{header_cls}">{header_txt}</div>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div><span class="label-mini">ENTRADA</span><br><span class="entry-blue">${setup['entry']:.2f}</span></div>
                    <div><span class="label-mini">STOP LOSS</span><br><span class="sl-red">${setup['sl']:.2f}</span></div>
                    <div><span class="label-mini">TP 1</span><br><span class="tp-green">${setup['tp1']:.2f}</span></div>
                    <div><span class="label-mini">TP 2</span><br><span class="tp-green">${setup['tp2']:.2f}</span></div>
                    <div><span class="label-mini">TP 3</span><br><span class="tp-green">${setup['tp3']:.2f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(btn_label):
                execute_trade(calc_dir, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], 100, atr)
                st.success(f"Orden {calc_dir} lanzada al mercado.")
        else:
            st.info("Esperando estructura de mercado clara...")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        if setup:
            fig.add_hline(y=setup['tp1'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_trades = load_trades()
        if not df_trades.empty:
            open_trades = df_trades[df_trades['status'] == "OPEN"]
            closed_trades = df_trades[df_trades['status'] == "CLOSED"]
            
            st.subheader("🟢 Posiciones Abiertas")
            st.dataframe(open_trades)
            
            st.subheader("📜 Historial Cerrado")
            st.dataframe(closed_trades)
            
            total_pnl = closed_trades['pnl'].sum()
            st.metric("PnL Total", f"${total_pnl:.2f}")
        else:
            st.info("Historial vacío.")

else: st.warning("Cargando datos...")

if auto_refresh: time.sleep(60); st.rerun()
