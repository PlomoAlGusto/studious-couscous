import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# --- IMPORTACIONES LOCALES ---
# Nota: NO importamos pandas_ta aquí para evitar conflictos. 
# La estrategia se encarga de eso internamente.
from config import config
from database import TradeManager
from data_feed import DataManager
from strategy import StrategyManager
from utils import setup_logging, init_nltk

# --- 1. CONFIGURACIÓN DE PÁGINA (Debe ser lo primero) ---
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide", page_icon="🧠")
setup_logging()
init_nltk()

# --- 2. ESTILOS CSS (Visuales Gamer/Pro) ---
st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
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
    
    /* Ajustes para noticias */
    .news-item { margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 8px; }
    .news-link { text-decoration: none; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES VISUALES AUXILIARES ---

def display_market_sessions():
    """Muestra el reloj de sesiones de mercado en la barra lateral"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    # Definición de sesiones en UTC
    sessions = {
        "🇬🇧 LONDRES": (8, 16), 
        "🇺🇸 NEW YORK": (13, 21), 
        "🇯🇵 TOKYO": (0, 9), 
        "🇦🇺 SYDNEY": (22, 7)
    }
    
    st.sidebar.markdown("### 🌍 SESIONES (UTC)")
    for name, (start, end) in sessions.items():
        is_open = False
        # Lógica para cruce de medianoche (ej: Sydney)
        if start < end:
            is_open = start <= hour < end
        else:
            is_open = hour >= start or hour < end
            
        status_icon = "🟢 ABIERTO" if is_open else "🔴 CERRADO"
        css_class = "clock-open" if is_open else "clock-closed"
        
        st.sidebar.markdown(
            f"<div class='market-clock {css_class}'>"
            f"<span>{name}</span><span>{status_icon}</span></div>", 
            unsafe_allow_html=True
        )

def generate_ai_html(regime, signal, atr, details, fr, fng):
    """Genera el cuadro de análisis de IA con HTML/CSS"""
    # Colores dinámicos
    color_regime = "#00FF00" if regime == "TENDENCIA" else "#FFFF00"
    
    if signal == "LONG": color_signal = "#00FF00"
    elif signal == "SHORT": color_signal = "#FF4444"
    else: color_signal = "#aaa"
    
    funding_warning = "⚠️ RIESGO" if abs(fr) > 0.03 else "NORMAL"
    
    html = f"""
    <div class='ai-box'>
        <span class='ai-title'>🤖 QUIMERA COPILOT v18.0:</span>
        <div style='display:flex; justify-content:space-between;'>
            <span>📡 SEÑAL:</span> 
            <span style='color:{color_signal}; font-weight:bold'>{signal}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>🧠 RÉGIMEN ML:</span> 
            <span style='color:{color_regime}; font-weight:bold'>{regime}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>📊 VOLATILIDAD (ATR):</span> 
            <span>${atr:.2f}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>💰 FUNDING:</span> 
            <span>{fr:.4f}% ({funding_warning})</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span>😨 FEAR/GREED:</span> 
            <span>{fng}</span>
        </div>
        <hr style='border-color:#333; margin:8px 0'>
        <div style='font-size:11px; color:#888; margin-bottom:4px;'>🕵️ DETALLES TÉCNICOS:</div>
        <div style='font-size:11px; color:#ccc; line-height:1.4;'>{' • '.join(details) if details else 'Sin confluencia clara'}</div>
    </div>
    """
    return html

# --- 4. INTERFAZ DE USUARIO (SIDEBAR) ---
with st.sidebar:
    st.title("🧠 QUIMERA v18.0")
    st.markdown(f"<div style='font-size:12px; margin-bottom:15px;'><span class='status-dot-on'>●</span> SYSTEM ONLINE</div>", unsafe_allow_html=True)
    
    # 1. Reloj de Sesiones
    display_market_sessions()
    
    st.divider()
    
    # 2. Entradas
    symbol = st.text_input("Ticker", "BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
    
    # 3. Filtros
    with st.expander("🛡️ FILTROS ACTIVOS", expanded=True):
        filters = {
            'use_ema': st.checkbox("Tendencia EMA", True),
            'use_vwap': st.checkbox("Filtro VWAP", True),
            'use_regime': st.checkbox("Filtro ML (Anti-Rango)", True)
        }
    
    # 4. Control
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
    
    if st.button("🔥 RESETEAR CUENTA"):
        TradeManager().reset_account()
        st.success("Base de datos reiniciada.")
        st.rerun()

# Auto-refresco sin bloquear la UI
if auto_refresh:
    st_autorefresh(interval=60 * 1000, key="data_refresh")

# --- 5. LÓGICA PRINCIPAL (MAIN) ---
def main():
    # Inicializar gestores
    data_mgr = DataManager()
    strat_mgr = StrategyManager()
    db_mgr = TradeManager()

    # Descarga de datos
    with st.spinner(f"🛰️ Escaneando {symbol}..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    # Verificación de errores en datos
    if df is None or df.empty:
        st.error(f"❌ Error: No se pudieron cargar datos para {symbol}. Verifica el ticker (ej: BTC/USDT).")
        return

    # Cálculo de Estrategia
    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df)
    signal, atr, details, regime = strat_mgr.get_signal(df, filters)
    
    current_price = df['close'].iloc[-1]
    price_change = df['close'].pct_change().iloc[-1]

    # --- 6. DASHBOARD ---
    
    # Columnas principales: Gráfico (Izquierda) | Panel Inteligente (Derecha)
    col_chart, col_info = st.columns([2.5, 1])
    
    with col_chart:
        # Métricas Header
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"${current_price:,.2f}", delta=f"{price_change:.2%}")
        m2.metric("Señal", signal, delta=regime, delta_color="off")
        m3.metric("Funding", f"{fr:.4f}%")
        m4.metric("F&G", f"{fng_val}", fng_class)

        # Gráfico Plotly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Velas
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Precio'), row=1, col=1)
        
        # Indicadores Superpuestos
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='#FFFF00', width=1), name='EMA 20'), row=1, col=1)
        if 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot', width=1), name='VWAP'), row=1, col=1)
        
        # Oscilador RSI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='#A020F0', width=1), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Layout Oscuro
        fig.update_layout(
            template="plotly_dark", 
            height=600, 
            margin=dict(l=0,r=0,t=0,b=0), 
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        # Panel de IA
        st.markdown(generate_ai_html(regime, signal, atr, details, fr, fng_val), unsafe_allow_html=True)
        
        # Sección de Noticias (10 ítems)
        st.markdown("#### 📰 Live News Feed")
        st.markdown("<div style='height: 400px; overflow-y: auto; padding-right:5px;'>", unsafe_allow_html=True)
        
        if news:
            for n in news:
                # Color por sentimiento
                sent_score = n.get('sentiment', 0)
                color = "#00FF00" if sent_score > 0.1 else "#FF4444" if sent_score < -0.1 else "#ccc"
                icon = "🟢" if sent_score > 0.1 else "🔴" if sent_score < -0.1 else "⚪"
                
                st.markdown(
                    f"<div class='news-item'>"
                    f"{icon} <a class='news-link' href='{n['link']}' target='_blank' style='color:{color}'>{n['title']}</a>"
                    f"<div style='font-size:10px; color:#666; margin-top:2px;'>Sentimiento: {sent_score:.2f}</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
        else:
            st.info("Sin noticias recientes.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 7. EJECUCIÓN DE TRADES ---
    st.divider()
    tab1, tab2 = st.tabs(["⚡ EJECUCIÓN MANUAL", "📚 LIBRO DE ÓRDENES"])
    
    with tab1:
        if signal != "NEUTRO":
            st.success(f"✅ CONFIGURACIÓN {signal} DETECTADA")
            
            # Calculadora de Riesgo
            c1, c2, c3 = st.columns(3)
            qty = c1.number_input("Inversión (USDT)", value=1000.0, step=100.0)
            lev = c2.slider("Apalancamiento", 1, 50, 10)
            
            # Precios Automáticos
            sl_price = current_price - (atr * 1.5) if signal == "LONG" else current_price + (atr * 1.5)
            tp_price = current_price + (atr * 3.0) if signal == "LONG" else current_price - (atr * 3.0)
            
            if c3.button(f"🚀 EJECUTAR {signal} AHORA"):
                trade = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "type": signal,
                    "entry": current_price,
                    "size": qty,
                    "leverage": lev,
                    "sl": sl_price,
                    "tp1": tp_price,
                    "tp2": 0,
                    "tp3": 0,
                    "status": "OPEN",
                    "pnl": 0.0,
                    "reason": f"Strategy ({regime})",
                    "candles_held": 0,
                    "atr_entry": atr
                }
                if db_mgr.add_trade(trade):
                    st.balloons()
                    st.success(f"Orden {signal} registrada en Base de Datos Local.")
                else:
                    st.error("Error al guardar en base de datos.")
        else:
            st.info("Esperando configuración de alta probabilidad...")

    with tab2:
        df_trades = db_mgr.load_trades()
        if not df_trades.empty:
            # Colorear PnL
            st.dataframe(
                df_trades.style.applymap(
                    lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', 
                    subset=['pnl']
                ), 
                use_container_width=True
            )
        else:
            st.caption("No hay operaciones registradas aún.")

if __name__ == "__main__":
    main()
