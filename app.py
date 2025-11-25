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

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL SISTEMA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v5.0 (Institutional)", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .big-signal {font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .bullish {background-color: rgba(0, 255, 0, 0.1); border: 2px solid #00FF00; color: #00FF00;}
    .bearish {background-color: rgba(255, 0, 0, 0.1); border: 2px solid #FF0000; color: #FF0000;}
    .neutral {background-color: rgba(255, 255, 0, 0.1); border: 1px dashed #FFFF00; color: #FFFF00;}
    .stProgress > div > div > div > div { background-color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ESTADO (MEMORIA)
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = []
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. CAPA DE CONFIGURACIÓN (SIDEBAR CEREBRO)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA PRO v5")
    st.caption("Suite Institucional")
    
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)

    with st.expander("🛡️ GRUPO A: FILTROS TENDENCIA", expanded=True):
        use_ema = st.checkbox("Tendencia (EMAs)", True)
        use_vwap = st.checkbox("Filtro VWAP (Inst.)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)

    with st.expander("🌊 GRUPO B: OSCILADORES & VOLUMEN"):
        use_rsi = st.checkbox("RSI (70/30)", True)
        use_stoch = st.checkbox("StochRSI (Gatillo)", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        use_bb = st.checkbox("Bollinger Squeeze", False)

    with st.expander("💰 GRUPO C: GESTIÓN SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        risk_per_trade = st.slider("Riesgo por Trade %", 0.5, 5.0, 1.0)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 3. CAPA DE CONEXIÓN Y DATOS
# -----------------------------------------------------------------------------
def init_exchange():
    try:
        if "BINANCE_API_KEY" in st.secrets:
            ex = ccxt.binance({'apiKey': st.secrets["BINANCE_API_KEY"], 'secret': st.secrets["BINANCE_SECRET"], 'options': {'defaultType': 'spot'}})
            ex.load_markets()
            return ex, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Visual)"

exchange, source_name = init_exchange()

@st.cache_data(ttl=15)
def get_data_engine(symbol, tf):
    if not exchange: return None, 0
    
    # 1. OHLCV
    ticker_fix = symbol if "Binance" in source_name else "BTC/USDT"
    try:
        ohlcv = exchange.fetch_ohlcv(ticker_fix, tf, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0

    # 2. OBI (Order Book Imbalance) - TIEMPO REAL
    obi = 0
    try:
        book = exchange.fetch_order_book(ticker_fix, limit=20)
        bids = sum([x[1] for x in book['bids']])
        asks = sum([x[1] for x in book['asks']])
        if (bids + asks) > 0:
            obi = (bids - asks) / (bids + asks) # -1 a +1
    except: pass
    
    return df, obi

# -----------------------------------------------------------------------------
# 4. CAPA DE CÁLCULO (INDICADORES AVANZADOS)
# -----------------------------------------------------------------------------
def calculate_indicators(df):
    if df is None: return None
    
    # EMAs
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    
    # VWAP Manual (Blindado)
    try:
        vp = ((df['high'] + df['low'] + df['close'])/3) * df['volume']
        df['VWAP'] = vp.cumsum() / df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    
    # Ichimoku
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1) # ISA_9, ISB_26
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    
    # RSI & StochRSI
    df['RSI'] = ta.rsi(df['close'], length=14)
    stoch = ta.stochrsi(df['close'], length=14) # k, d
    df = pd.concat([df, stoch], axis=1)
    
    # ADX (Régimen) & ATR (Volatilidad)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1) # ADX_14
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# -----------------------------------------------------------------------------
# 5. CAPA LÓGICA (EL EMBUDO DE DECISIÓN)
# -----------------------------------------------------------------------------
def run_strategy_funnel(df, obi):
    row = df.iloc[-1]
    reasons = []
    valid_long = True
    valid_short = True
    
    # 1. FILTRO TENDENCIA (EMAs)
    if use_ema:
        if row['EMA_20'] > row['EMA_50']: 
            valid_short = False; reasons.append("Tendencia Alcista")
        else: 
            valid_long = False; reasons.append("Tendencia Bajista")

    # 2. FILTRO VWAP (Institucional)
    if use_vwap:
        if row['close'] > row['VWAP']: valid_short = False
        else: valid_long = False
        
    # 3. FILTRO ICHIMOKU (Nube)
    if use_ichi:
        # Precio debe estar sobre la nube para Long
        cloud_top = max(row['ISA_9'], row['ISB_26'])
        cloud_bottom = min(row['ISA_9'], row['ISB_26'])
        if row['close'] < cloud_top: valid_long = False
        if row['close'] > cloud_bottom: valid_short = False

    # 4. FILTRO ANTI-RANGO (ADX)
    if use_regime:
        if row['ADX_14'] < 20:
            valid_long = False; valid_short = False; reasons.append("Mercado Lateral (ADX<20)")

    # 5. FILTRO RSI
    if use_rsi:
        if row['RSI'] > 70: valid_long = False # Sobrecompra
        if row['RSI'] < 30: valid_short = False # Sobreventa
        
    # 6. FILTRO STOCHRSI (Gatillo)
    if use_stoch:
        k = row['STOCHRSIk_14_14_3_3']
        d = row['STOCHRSId_14_14_3_3']
        if k < d: valid_long = False
        if k > d: valid_short = False

    # 7. FILTRO OBI (Order Book)
    if use_obi:
        if obi < -0.05: valid_long = False # Mucha venta
        if obi > 0.05: valid_short = False # Mucha compra

    # RESULTADO FINAL
    signal = "NEUTRO"
    if valid_long: signal = "LONG"
    elif valid_short: signal = "SHORT"
    
    return signal, reasons, row['ATR']

# -----------------------------------------------------------------------------
# 6. GESTIÓN DE POSICIONES (PAPER TRADING AVANZADO)
# -----------------------------------------------------------------------------
def manage_positions(current_price):
    for pos in st.session_state.positions[:]:
        close_reason = ""
        pnl = 0
        
        # LOGICA LONG
        if pos['type'] == "LONG":
            # 1. Trailing Stop
            if use_trailing:
                new_sl = current_price - (pos['atr_at_entry'] * 1.5)
                if new_sl > pos['sl']: pos['sl'] = new_sl # Solo subimos el SL, nunca lo bajamos
            
            # 2. Breakeven
            if use_breakeven:
                profit_pct = (current_price - pos['entry']) / pos['entry']
                if profit_pct > 0.015 and pos['sl'] < pos['entry']: 
                    pos['sl'] = pos['entry'] * 1.001 # Mover a entrada + comisiones
            
            # 3. Check Salida
            if current_price >= pos['tp']:
                close_reason = "TAKE PROFIT 🎯"; pnl = (pos['tp'] - pos['entry']) * pos['size']
            elif current_price <= pos['sl']:
                close_reason = "STOP LOSS 🛑"; pnl = (pos['sl'] - pos['entry']) * pos['size']
        
        # LOGICA SHORT (Inversa)
        elif pos['type'] == "SHORT":
            if use_trailing:
                new_sl = current_price + (pos['atr_at_entry'] * 1.5)
                if new_sl < pos['sl']: pos['sl'] = new_sl
            
            if current_price <= pos['tp']:
                close_reason = "TAKE PROFIT 🎯"; pnl = (pos['entry'] - pos['tp']) * pos['size']
            elif current_price >= pos['sl']:
                close_reason = "STOP LOSS 🛑"; pnl = (pos['entry'] - pos['sl']) * pos['size']
                
        if close_reason:
            st.session_state.balance += pnl
            pos['status'] = "CLOSED"
            pos['exit'] = current_price
            pos['pnl'] = pnl
            pos['reason'] = close_reason
            st.session_state.trade_history.insert(0, pos)
            st.session_state.positions.remove(pos)
            send_telegram_msg(f"💰 CIERRE {close_reason}\nPnL: ${pnl:.2f}")

def send_telegram_msg(msg):
    t, c = st.secrets.get("TELEGRAM_TOKEN", ""), st.secrets.get("TELEGRAM_CHAT_ID", "")
    if t and c:
        try: requests.get(f"https://api.telegram.org/bot{t}/sendMessage", params={"chat_id": c, "text": msg})
        except: pass

# -----------------------------------------------------------------------------
# 7. INTERFAZ GRÁFICA (DASHBOARD)
# -----------------------------------------------------------------------------
df, obi = get_data_engine(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    signal, reasons, atr = run_strategy_funnel(df, obi)
    current_price = df['close'].iloc[-1]
    
    # Gestión de Alertas
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        send_telegram_msg(f"🦁 SIGNAL v5: {signal} en {symbol}\nPrecio: ${current_price}\n{reasons}")
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_positions(current_price)
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 LIVE COMMAND", "🧪 PAPER LAB", "🔮 AI FORECAST"])
    
    with tab1:
        # SEÑAL VISUAL
        css = "neutral"
        if signal == "LONG": css = "bullish"
        elif signal == "SHORT": css = "bearish"
        
        st.markdown(f"""
        <div class='big-signal {css}'>
            {signal}
            <div style='font-size:12px; margin-top:5px; color:#aaa;'>Estrategia: {' + '.join(reasons)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # HUD DE MÉTRICAS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Volatilidad ATR", f"${atr:.2f}")
        c3.metric("ADX (Fuerza)", f"{df['ADX_14'].iloc[-1]:.1f}")
        
        with c4:
            st.write("OBI (Presión)")
            obi_pct = (obi + 1) / 2 # Normalizar 0 a 1
            st.progress(obi_pct)
            if obi > 0: st.caption(f"🐂 Compras (+{obi:.1%})")
            else: st.caption(f"🐻 Ventas ({obi:.1%})")

        # GRÁFICO PLOTLY
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', width=1, dash='dot'), name='VWAP'), row=1, col=1)
        if use_ichi:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ISA_9'], line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ISB_26'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,255,0,0.1)', name='Cloud'), row=1, col=1)

        # STOCH RSI
        k = df['STOCHRSIk_14_14_3_3']
        d = df['STOCHRSId_14_14_3_3']
        fig.add_trace(go.Scatter(x=df['timestamp'], y=k, line=dict(color='cyan', width=1), name='Stoch K'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=d, line=dict(color='red', width=1), name='Stoch D'), row=2, col=1)
        fig.add_hline(y=80, line_dash="dot", row=2, col=1); fig.add_hline(y=20, line_dash="dot", row=2, col=1)
        
        fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # BOTONES DE EJECUCIÓN
        if signal != "NEUTRO":
            sl = current_price - (atr * 1.5) if signal == "LONG" else current_price + (atr * 1.5)
            tp = current_price + (abs(current_price-sl)*2) if signal == "LONG" else current_price - (abs(current_price-sl)*2)
            
            if st.button(f"🚀 EJECUTAR {signal} (PAPER)"):
                trade = {"time": datetime.now(), "symbol": symbol, "type": signal, "entry": current_price, "sl": sl, "tp": tp, "size": (st.session_state.balance*0.1)/current_price, "atr_at_entry": atr, "status": "OPEN"}
                st.session_state.positions.append(trade)
                st.success("Orden enviada al Laboratorio.")

    with tab2:
        st.subheader(f"Balance: ${st.session_state.balance:,.2f}")
        if st.session_state.positions:
            st.write("Posiciones Abiertas (SL Dinámico Activo):")
            st.dataframe(pd.DataFrame(st.session_state.positions)[['symbol', 'type', 'entry', 'sl', 'tp', 'pnl']])
        
        if st.session_state.trade_history:
            st.write("Historial:")
            st.dataframe(pd.DataFrame(st.session_state.trade_history))

    with tab3:
        if st.button("🔮 Consultar Oráculo (Prophet AI)"):
            with st.spinner("Analizando micro-estructura..."):
                m = Prophet()
                d_p = df[['timestamp', 'close']].rename(columns={'timestamp':'ds', 'close':'y'})
                m.fit(d_p)
                fut = m.make_future_dataframe(periods=12, freq='H')
                fcst = m.predict(fut)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=d_p['ds'], y=d_p['y'], name='Real'))
                fig2.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='IA Trend', line=dict(color='magenta')))
                st.plotly_chart(fig2, use_container_width=True)

else: st.warning("Cargando datos del mercado...")

if auto_refresh: time.sleep(60); st.rerun()
