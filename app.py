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

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL PRO
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro Suite", layout="wide", page_icon="🦁")

# CSS para dar aspecto "Fintech" oscuro
st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;}
    .signal-card {background-color: #1E1E1E; padding: 20px; border-radius: 15px; border-left: 5px solid; margin-bottom: 20px;}
    .long-signal {border-color: #00FF00;}
    .short-signal {border-color: #FF0000;}
    .neutral-signal {border-color: #FFFF00;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MOTOR DE SIMULACIÓN (PAPER TRADING INTERNO)
# -----------------------------------------------------------------------------
if 'balance' not in st.session_state: st.session_state.balance = 10000.0  # Capital inicial $10,000
if 'positions' not in st.session_state: st.session_state.positions = []   # Posiciones abiertas
if 'trade_history' not in st.session_state: st.session_state.trade_history = [] # Historial

def execute_paper_trade(signal_type, price, symbol, sl, tp, leverage):
    """Simula la apertura de una posición"""
    margin = (st.session_state.balance * 0.1)  # Usamos el 10% del saldo por operación
    size = (margin * leverage) / price
    
    trade = {
        "id": int(time.time()),
        "symbol": symbol,
        "type": signal_type,
        "entry_price": price,
        "size": size,
        "leverage": leverage,
        "sl": sl,
        "tp": tp,
        "status": "OPEN",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.positions.append(trade)
    st.toast(f"✅ Orden {signal_type} Abierta a ${price}")

def check_open_positions(current_price):
    """Revisa si el precio ha tocado TP o SL"""
    for pos in st.session_state.positions[:]: # Copia de la lista para iterar
        pnl = 0
        close_reason = ""
        
        # Lógica LONG
        if pos['type'] == "LONG":
            if current_price >= pos['tp']:
                pnl = (pos['tp'] - pos['entry_price']) * pos['size']
                close_reason = "TAKE PROFIT 🎯"
            elif current_price <= pos['sl']:
                pnl = (pos['sl'] - pos['entry_price']) * pos['size']
                close_reason = "STOP LOSS 🛑"
        
        # Lógica SHORT
        elif pos['type'] == "SHORT":
            if current_price <= pos['tp']:
                pnl = (pos['entry_price'] - pos['tp']) * pos['size']
                close_reason = "TAKE PROFIT 🎯"
            elif current_price >= pos['sl']:
                pnl = (pos['entry_price'] - pos['sl']) * pos['size']
                close_reason = "STOP LOSS 🛑"
                
        if close_reason:
            st.session_state.balance += pnl
            pos['status'] = "CLOSED"
            pos['exit_price'] = current_price
            pos['pnl'] = pnl
            pos['reason'] = close_reason
            st.session_state.trade_history.append(pos)
            st.session_state.positions.remove(pos)
            st.toast(f"{close_reason} | PnL: ${pnl:.2f}")

# -----------------------------------------------------------------------------
# 3. CONEXIÓN INTELIGENTE (ANTI-BLOQUEO)
# -----------------------------------------------------------------------------
def init_exchange():
    """Intenta Binance, y si falla por bloqueo IP, usa Kraken"""
    try:
        if "BINANCE_API_KEY" in st.secrets:
            exchange = ccxt.binance({
                'apiKey': st.secrets["BINANCE_API_KEY"],
                'secret': st.secrets["BINANCE_SECRET"],
                'options': {'defaultType': 'spot'}
            })
            exchange.load_markets()
            return exchange, "Binance"
    except: pass
    return ccxt.kraken(), "Kraken (Modo Visualización)"

exchange, source_info = init_exchange()

# -----------------------------------------------------------------------------
# 4. INTELIGENCIA ALGORÍTMICA (GENERADOR DE SEÑALES)
# -----------------------------------------------------------------------------
def generate_smart_signal(df, risk_reward=2.0, atr_multiplier=1.5):
    """Calcula entrada, SL y TP basado en Volatilidad (ATR) y Tendencia"""
    last = df.iloc[-1]
    
    # ATR para SL dinámico
    atr = last['ATR']
    
    signal = "NEUTRO"
    setup = {}
    
    # Estrategia: Cruce EMA + RSI
    if last['EMA_20'] > last['EMA_50'] and last['RSI'] < 70:
        signal = "LONG"
        sl_price = last['close'] - (atr * atr_multiplier)
        risk = last['close'] - sl_price
        tp_price = last['close'] + (risk * risk_reward)
        
    elif last['EMA_20'] < last['EMA_50'] and last['RSI'] > 30:
        signal = "SHORT"
        sl_price = last['close'] + (atr * atr_multiplier)
        risk = sl_price - last['close']
        tp_price = last['close'] - (risk * risk_reward)
    
    if signal != "NEUTRO":
        setup = {
            "entry": last['close'],
            "sl": sl_price,
            "tp": tp_price,
            "atr": atr
        }
        
    return signal, setup

# -----------------------------------------------------------------------------
# 5. SIDEBAR & DATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA PRO")
    st.caption(f"Fuente de Datos: {source_info}")
    
    symbol = st.text_input("Símbolo", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)
    
    st.divider()
    st.subheader("⚙️ Configuración IA")
    leverage = st.slider("Apalancamiento Simulado", 1, 20, 5)
    rr_ratio = st.number_input("Ratio Riesgo/Beneficio", 1.0, 5.0, 2.0)
    
    if st.button("🔄 REINICIAR CUENTA PAPER"):
        st.session_state.balance = 10000.0
        st.session_state.positions = []
        st.session_state.trade_history = []
        st.rerun()

@st.cache_data(ttl=15)
def get_data(sym, tf):
    try:
        # Arreglo para Kraken
        ticker = sym if "Binance" in source_info else "BTC/USDT"
        ohlcv = exchange.fetch_ohlcv(ticker, tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except: return None

# -----------------------------------------------------------------------------
# 6. UI PRINCIPAL
# -----------------------------------------------------------------------------
df = get_data(symbol, tf)

if df is not None:
    # Calcular Indicadores
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Generar Señal IA
    smart_signal, setup = generate_smart_signal(df, risk_reward=rr_ratio)
    current_price = df['close'].iloc[-1]
    
    # Ejecutar Check de Posiciones Abiertas
    check_open_positions(current_price)

    # --- PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD IA", "📉 CHARTING PRO", "💼 PAPER PORTFOLIO"])

    # TAB 1: INTUICIÓN DE MERCADO
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Precio Actual", f"${current_price:,.2f}", 
                     f"{current_price - df['close'].iloc[-2]:.2f}")
            
            # TARJETA DE SEÑAL IA
            color = "#FFFF00"
            if smart_signal == "LONG": color = "#00FF00"
            elif smart_signal == "SHORT": color = "#FF0000"
            
            st.markdown(f"""
            <div class="signal-card" style="border-left-color: {color};">
                <h3 style="margin:0; color:{color}">INTUICIÓN IA: {smart_signal}</h3>
                <p style="font-size:12px; margin:0">Basado en EMA Cross + Volatilidad ATR</p>
            </div>
            """, unsafe_allow_html=True)
            
            if smart_signal != "NEUTRO":
                st.write(f"🎯 **Entrada:** ${setup['entry']:.2f}")
                st.write(f"🛑 **Stop Loss:** ${setup['sl']:.2f}")
                st.write(f"💰 **Take Profit:** ${setup['tp']:.2f}")
                
                if st.button(f"EJECUTAR {smart_signal} (SIMULADO)"):
                    execute_paper_trade(smart_signal, current_price, symbol, setup['sl'], setup['tp'], leverage)
        
        with col2:
            st.subheader("🔍 Análisis de Volatilidad")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = df['RSI'].iloc[-1],
                title = {'text': "Fuerza RSI"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': color},
                         'steps': [
                             {'range': [0, 30], 'color': "green"},
                             {'range': [30, 70], 'color': "gray"},
                             {'range': [70, 100], 'color': "red"}]}))
            fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor="#1E1E1E", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

    # TAB 2: GRÁFICO INTERACTIVO CON NIVELES
    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Velas
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
        
        # Dibujar líneas de TP/SL si hay señal
        if smart_signal != "NEUTRO":
            fig.add_hline(y=setup['tp'], line_dash="dash", line_color="green", annotation_text="Target IA", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dash", line_color="red", annotation_text="Stop IA", row=1, col=1)
            
        # RSI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", 
                          dragmode="pan", # Permite mover el gráfico
                          margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: CARTERA DE PAPER TRADING
    with tab3:
        st.metric("Balance de Cuenta (Simulado)", f"${st.session_state.balance:,.2f}")
        
        st.subheader("Posiciones Abiertas")
        if st.session_state.positions:
            st.dataframe(pd.DataFrame(st.session_state.positions))
        else:
            st.info("No hay operaciones abiertas.")
            
        st.subheader("Historial de Tradeos")
        if st.session_state.trade_history:
            hist_df = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(hist_df[['timestamp', 'symbol', 'type', 'pnl', 'reason']])
        else:
            st.text("Historial vacío.")

else:
    st.warning("Cargando datos... Por favor espera.")
