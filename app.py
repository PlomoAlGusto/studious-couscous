import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# --- IMPORTACIONES ---
try:
    from config import config
    from database import TradeManager
    from data_feed import DataManager
    from strategy import StrategyManager
    from utils import setup_logging, init_nltk
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quimera Pro", layout="wide", page_icon="🦁")
setup_logging()
init_nltk()

# --- 2. ESTILOS CSS PROFESIONALES ---
st.markdown("""
<style>
    /* Caja de Análisis IA (Azul) */
    .ai-box {
        background-color: #0e1117;
        border-left: 4px solid #44AAFF; 
        padding: 15px; border-radius: 5px; margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    /* Caja de Noticias (Blanca - SOLICITADO) */
    .news-box {
        background-color: #0e1117;
        border-left: 4px solid #FFFFFF; 
        padding: 15px; border-radius: 5px; margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }

    .section-title { 
        font-weight: bold; font-size: 13px; margin-bottom: 10px; 
        display: block; border-bottom: 1px solid #333; padding-bottom: 5px;
        text-transform: uppercase; letter-spacing: 1px;
    }
    
    /* Tarjeta de Trade (Setup) */
    .trade-setup {
        background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d;
        margin-top: 15px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    .entry-txt { color: #44AAFF; font-weight: bold; font-size: 18px; font-family: monospace; }
    .sl-txt { color: #FF4444; font-weight: bold; font-size: 18px; font-family: monospace; }
    .tp-txt { color: #00FF00; font-weight: bold; font-size: 18px; font-family: monospace; }
    .label-mini { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    
    /* Barra de Probabilidad */
    .prob-bg { width: 100%; background-color: #333; height: 6px; border-radius: 3px; margin-top: 5px; }
    .prob-fill { height: 100%; border-radius: 3px; }

    /* Relojes */
    .clock-box { font-size: 11px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between; border: 1px solid #333; }
    .open { background-color: rgba(0,255,0,0.1); border-color: green; }
    .closed { background-color: rgba(255,255,255,0.05); color: #666; }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES ---

def display_market_sessions():
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("### 🌍 SESIONES (UTC)")
    for name, (start, end) in sessions.items():
        is_open = start <= hour < end if start < end else (hour >= start or hour < end)
        status = "🟢" if is_open else "🔴"
        css = "open" if is_open else "closed"
        st.sidebar.markdown(f"<div class='clock-box {css}'><span>{name}</span><span>{status}</span></div>", unsafe_allow_html=True)

def calculate_optimal_leverage(entry, sl):
    """Calcula lev óptimo para arriesgar max 2-3% por trade"""
    if entry == 0: return 1
    dist_pct = abs(entry - sl) / entry
    if dist_pct == 0: return 1
    # Fórmula: Riesgo deseado (2%) / Distancia a SL
    safe_lev = int(0.02 / dist_pct) 
    return max(1, min(safe_lev, 50)) # Limitamos entre 1x y 50x

def generate_ai_box(regime, signal, atr, details, fr, fng):
    c_sig = "#00FF00" if signal == "LONG" else "#FF4444" if signal == "SHORT" else "#888"
    html = f"""
    <div class='ai-box'>
        <span class='section-title' style='color:#44AAFF'>🤖 QUIMERA COPILOT</span>
        <div style='display:flex; justify-content:space-between;'><span>📡 Señal:</span><span style='color:{c_sig}; font-weight:bold'>{signal}</span></div>
        <div style='display:flex; justify-content:space-between;'><span>🧠 Régimen:</span><span>{regime}</span></div>
        <div style='display:flex; justify-content:space-between;'><span>📊 ATR:</span><span>${atr:.2f}</span></div>
        <div style='display:flex; justify-content:space-between;'><span>💰 Funding:</span><span>{fr:.4f}%</span></div>
        <hr style='border-color:#333; margin:8px 0'>
        <div style='font-size:11px; color:#aaa'>{' • '.join(details)}</div>
    </div>
    """
    return html

def generate_news_box(news):
    content = ""
    for n in news[:10]: # Aseguramos 10 noticias
        color = "#00FF00" if n['sentiment'] > 0.1 else "#FF4444" if n['sentiment'] < -0.1 else "#ccc"
        content += f"<div style='margin-bottom:8px; font-size:13px; border-bottom:1px solid #222; padding-bottom:4px;'>🔗 <a href='{n['link']}' target='_blank' style='color:{color}; text-decoration:none'>{n['title']}</a></div>"
    
    html = f"""
    <div class='news-box'>
        <span class='section-title' style='color:#FFFFFF'>📰 LIVE NEWS FEED (10)</span>
        <div style='height: 350px; overflow-y: auto; padding-right:5px;'>{content}</div>
    </div>
    """
    return html

# --- 4. INTERFAZ ---
with st.sidebar:
    st.title("🦁 QUIMERA PRO")
    display_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
    
    with st.expander("⚙️ FILTROS", expanded=True):
        filters = {
            'use_ema': st.checkbox("Tendencia EMA", True),
            'use_vwap': st.checkbox("Filtro VWAP", True),
            'use_regime': st.checkbox("Filtro ML", True)
        }
    
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
    if st.button("🗑️ RESETEAR DATA"):
        TradeManager().reset_account()
        st.rerun()

if auto_refresh:
    st_autorefresh(interval=60 * 1000, key="data_refresh")

# --- 5. MAIN ---
def main():
    data_mgr = DataManager()
    strat_mgr = StrategyManager()
    db_mgr = TradeManager()

    with st.spinner(f"Analizando {symbol}..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    if df is None:
        st.error("❌ Error de conexión. Intenta refrescar.")
        return

    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df)
    signal, atr, details, regime = strat_mgr.get_signal(df, filters)
    price = df['close'].iloc[-1]

    # Probabilidad Simulada
    prob = 50
    if signal != "NEUTRO":
        prob = 70 + (10 if regime == "TENDENCIA" else -10)
    
    prob_color = "#00FF00" if prob > 60 else "#FFFF00"

    # --- DASHBOARD ---
    col1, col2 = st.columns([2.5, 1])

    with col1:
        # Métricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${price:,.2f}")
        c2.metric("Señal", signal, delta=regime)
        c3.metric("Funding", f"{fr:.4f}%")
        c4.metric("F&G", f"{fng_val}")

        # Gráfico
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
        if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=550, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- TARJETA DE TRADE (Diseño Solicitado) ---
        if signal != "NEUTRO":
            sl_dist = atr * 1.5
            sl = price - sl_dist if signal == "LONG" else price + sl_dist
            tp1 = price + sl_dist if signal == "LONG" else price - sl_dist
            tp2 = price + (sl_dist * 2) if signal == "LONG" else price - (sl_dist * 2)
            tp3 = price + (sl_dist * 3.5) if signal == "LONG" else price - (sl_dist * 3.5)
            
            # CÁLCULO DE APALANCAMIENTO ÓPTIMO
            opt_lev = calculate_optimal_leverage(price, sl)

            # HTML SETUP CARD
            setup_html = f"""
            <div class="trade-setup">
                <div style="font-size:22px; font-weight:bold; color:{'#00FF00' if signal=='LONG' else '#FF4444'}; margin-bottom:5px;">
                    {signal} DETECTADO 🚀
                </div>
                <div style="display:flex; justify-content:space-between; font-size:13px; color:#ccc;">
                    <span>Probabilidad:</span><span style="color:{prob_color}; font-weight:bold">{prob}%</span>
                </div>
                <div class="prob-bg"><div class="prob-fill" style="width:{prob}%; background-color:{prob_color};"></div></div>
                
                <div style="margin-top:20px; display:flex; justify-content:space-around;">
                    <div><span class="label-mini">ENTRADA</span><br><span class="entry-txt">${price:.2f}</span></div>
                    <div><span class="label-mini">STOP LOSS</span><br><span class="sl-txt">${sl:.2f}</span></div>
                    <div><span class="label-mini">LEV ÓPTIMO</span><br><span style="color:white; font-size:20px; font-weight:bold">{opt_lev}x</span></div>
                </div>
                <div style="margin-top:15px; border-top:1px solid #333; padding-top:10px; display:flex; justify-content:space-around;">
                    <div><span class="label-mini">TP 1</span><br><span class="tp-txt">${tp1:.2f}</span></div>
                    <div><span class="label-mini">TP 2</span><br><span class="tp-txt">${tp2:.2f}</span></div>
                    <div><span class="label-mini">TP 3</span><br><span class="tp-txt">${tp3:.2f}</span></div>
                </div>
            </div>
            """
            st.markdown(setup_html, unsafe_allow_html=True)

            # Botón
            c_btn1, c_btn2 = st.columns([1, 2])
            size = c_btn1.number_input("Capital", value=1000.0)
            if c_btn2.button(f"🚀 EJECUTAR (Lev: {opt_lev}x)", use_container_width=True, type="primary"):
                trade = {
                    "timestamp": str(datetime.now()), "symbol": symbol, "type": signal,
                    "entry": price, "size": size, "leverage": opt_lev,
                    "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                    "status": "OPEN", "pnl": 0.0, "reason": "Signal", "candles_held": 0, "atr_entry": atr
                }
                db_mgr.add_trade(trade)
                st.success("Orden Registrada en Libro")
        else:
            st.info("Esperando configuración clara...")

    with col2:
        st.markdown(generate_ai_box(regime, signal, atr, details, fr, fng_val), unsafe_allow_html=True)
        st.markdown(generate_news_box(news), unsafe_allow_html=True)

    # --- HISTORIAL ---
    st.divider()
    st.subheader("📚 Libro de Órdenes")
    df_trades = db_mgr.load_trades()
    if not df_trades.empty:
        st.dataframe(df_trades, use_container_width=True)

if __name__ == "__main__":
    main()
