import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# Importamos nuestros módulos (El cerebro, los datos y la base de datos)
from config import config
from database import TradeManager
from data_feed import DataManager
from strategy import StrategyManager
from utils import setup_logging, init_nltk

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide", page_icon="🧠")
setup_logging()
init_nltk()

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
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
        color: #44AAFF; font-weight: bold; font-size: 14px; margin-bottom: 8px; 
        display: block; border-bottom: 1px solid #333; padding-bottom: 5px;
    }
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
    .status-dot-on { color: #00FF00; font-weight: bold; text-shadow: 0 0 5px #00FF00; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES VISUALES ---
def display_market_sessions():
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("### 🌍 SESIONES (UTC)")
    for name, (start, end) in sessions.items():
        is_open = False
        if start < end: is_open = start <= hour < end
        else: is_open = hour >= start or hour < end
        status_icon = "🟢 ABIERTO" if is_open else "🔴 CERRADO"
        css_class = "clock-open" if is_open else "clock-closed"
        st.sidebar.markdown(f"<div class='market-clock {css_class}'><span>{name}</span><span>{status_icon}</span></div>", unsafe_allow_html=True)

def generate_ai_html(regime, signal, atr, details, fr, fng):
    color_regime = "#00FF00" if regime == "TENDENCIA" else "#FFFF00"
    color_signal = "#00FF00" if signal == "LONG" else "#FF4444" if signal == "SHORT" else "#aaa"
    
    html = f"""
    <div class='ai-box'>
        <span class='ai-title'>🤖 QUIMERA COPILOT v18.0:</span>
        <div>📡 <b>SEÑAL:</b> <span style='color:{color_signal}; font-weight:bold'>{signal}</span></div>
        <div>🧠 <b>RÉGIMEN ML:</b> <span style='color:{color_regime}'>{regime}</span></div>
        <div>📊 <b>VOLATILIDAD:</b> ${atr:.2f} (ATR)</div>
        <div>💰 <b>FUNDING:</b> {fr:.4f}%</div>
        <div>😨 <b>FEAR/GREED:</b> {fng}</div>
        <hr style='border-color:#333; margin:5px 0'>
        <div>🕵️ <b>DETALLES TÉCNICOS:</b></div>
        <div style='font-size:11px; color:#aaa'>{' | '.join(details)}</div>
    </div>
    """
    return html

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 QUIMERA v18.0")
    st.markdown(f"<div style='font-size:12px; margin-bottom:10px;'><span class='status-dot-on'>●</span> SYSTEM ONLINE</div>", unsafe_allow_html=True)
    
    display_market_sessions()
    
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
    
    with st.expander("🛡️ FILTROS", expanded=True):
        filters = {
            'use_ema': st.checkbox("Tendencia EMA", True),
            'use_vwap': st.checkbox("Filtro VWAP", True),
            'use_regime': st.checkbox("Filtro ML (Anti-Rango)", True)
        }
    
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
    if st.button("🔥 RESETEAR CUENTA"):
        TradeManager().reset_account()
        st.rerun()

if auto_refresh:
    st_autorefresh(interval=60 * 1000, key="data_refresh")

# --- MAIN ---
def main():
    data_mgr = DataManager()
    strat_mgr = StrategyManager()
    db_mgr = TradeManager()

    with st.spinner("🧠 Analizando mercado..."):
        # Nota: data_feed.py usa Yahoo, así que le da igual si pones BTC/USDT, él lo arregla.
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    if df is None:
        st.error(f"❌ Error cargando datos para {symbol}. Intenta refrescar.")
        return

    # Procesar estrategia
    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df)
    signal, atr, details, regime = strat_mgr.get_signal(df, filters)
    current_price = df['close'].iloc[-1]

    # --- DASHBOARD ---
    col_chart, col_ai = st.columns([2.5, 1])
    
    with col_chart:
        m1, m2, m3 = st.columns(3)
        m1.metric("Precio", f"${current_price:,.2f}", delta=f"{df['close'].pct_change().iloc[-1]:.2%}")
        m2.metric("Señal", signal, delta=regime, delta_color="off")
        m3.metric("Riesgo F&G", f"{fng_val}", fng_class)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        
        # Indicadores
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='#FFFF00', width=1), name='EMA 20'), row=1, col=1)
        if 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_ai:
        ai_html = generate_ai_html(regime, signal, atr, details, fr, fng_val)
        st.markdown(ai_html, unsafe_allow_html=True)
        
        st.markdown("### 📰 Últimas Noticias")
        st.markdown("<div style='height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
        for n in news:
            color = "#00FF00" if n['sentiment'] > 0.1 else "#FF4444" if n['sentiment'] < -0.1 else "#bbb"
            icon = "🟢" if n['sentiment'] > 0.1 else "🔴" if n['sentiment'] < -0.1 else "⚪"
            st.markdown(f"<div style='margin-bottom:10px; font-size:13px'>{icon} <a href='{n['link']}' target='_blank' style='color:{color}; text-decoration:none'>{n['title']}</a></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    tab1, tab2 = st.tabs(["🚀 EJECUTAR ORDEN", "📜 HISTORIAL"])
    
    with tab1:
        if signal != "NEUTRO":
            st.success(f"✅ OPORTUNIDAD CONFIRMADA: {signal}")
            c1, c2, c3 = st.columns(3)
            qty = c1.number_input("Tamaño (USDT)", value=1000.0)
            lev = c2.slider("Apalancamiento", 1, 50, 10)
            
            if c3.button(f"LANZAR {signal}"):
                trade = {
                    "timestamp": str(datetime.now()), "symbol": symbol, "type": signal,
                    "entry": current_price, "size": qty, "leverage": lev,
                    "sl": current_price - atr if signal == "LONG" else current_price + atr,
                    "tp1": current_price + (atr*2) if signal == "LONG" else current_price - (atr*2),
                    "tp2": 0, "tp3": 0, "status": "OPEN", "pnl": 0.0, "reason": "Signal", "candles_held": 0, "atr_entry": atr
                }
                db_mgr.add_trade(trade)
                st.balloons()
                st.success("Orden Enviada")
        else:
            st.info("Esperando señal clara...")

    with tab2:
        df_trades = db_mgr.load_trades()
        if not df_trades.empty:
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.caption("Historial vacío.")

if __name__ == "__main__":
    main()