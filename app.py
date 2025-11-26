import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Importamos nuestros módulos profesionales
from config import config
from database import TradeManager
from data_feed import DataManager
from strategy import StrategyManager
from utils import setup_logging, init_nltk

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide", page_icon="🦁")
setup_logging()
init_nltk()

# CSS Personalizado
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .trade-box {border: 2px solid #444; padding: 20px; border-radius: 10px; background: #0e1117; text-align: center;}
    .bull {color: #00FF00; font-weight: bold;}
    .bear {color: #FF4444; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Panel de Control) ---
with st.sidebar:
    st.title("🦁 QUIMERA PRO")
    st.markdown("---")
    
    # Entradas de Usuario
    symbol = st.text_input("Symbol", "BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
    
    st.markdown("### 🛡️ Filtros de Estrategia")
    filters = {
        'use_ema': st.checkbox("Tendencia EMA", True),
        'use_vwap': st.checkbox("Filtro VWAP", True),
        'use_regime': st.checkbox("Filtro ML (Anti-Rango)", True)
    }
    
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-Refresh (60s)", False)
    if st.button("🗑️ Resetear Cuenta"):
        TradeManager().reset_account()
        st.rerun()

# Refresco Automático (Sin bloquear la app)
if auto_refresh:
    st_autorefresh(interval=60 * 1000, key="data_refresh")

# --- LÓGICA PRINCIPAL ---
def main():
    # 1. Inicializar Gestores
    data_mgr = DataManager()
    strat_mgr = StrategyManager()
    db_mgr = TradeManager()

    # 2. Obtener Datos
    with st.spinner("Analizando mercado..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    if df is None:
        st.error(f"No se pudieron cargar datos para {symbol}")
        return

    # 3. Procesar Estrategia
    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df) # Entrenar ML on-the-fly
    
    signal, atr, details, regime = strat_mgr.get_signal(df, filters)
    current_price = df['close'].iloc[-1]

    # --- DASHBOARD VISUAL ---
    
    # Fila 1: Métricas Principales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Actual", f"${current_price:,.2f}", 
              delta=f"{df['close'].pct_change().iloc[-1]:.2%}")
    c2.metric("Señal Técnica", signal, delta=regime, 
              delta_color="normal" if signal=="NEUTRO" else "inverse")
    c3.metric("Funding Rate", f"{fr:.4f}%", "Riesgo Alto" if abs(fr)>0.03 else "Normal")
    c4.metric("Fear & Greed", f"{fng_val}", fng_class)

    # Fila 2: Gráfico y Análisis AI
    col_chart, col_ai = st.columns([2, 1])
    
    with col_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        # Precio y EMAs
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        # Oscilador (RSI)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_ai:
        st.markdown("### 🤖 Quimera AI Analysis")
        st.markdown(f"""
        <div class='metric-card'>
            <b>Régimen Detectado (ML):</b> {regime}<br>
            <b>Volatilidad (ATR):</b> ${atr:.2f}<br>
            <b>Detalles:</b><br>
            {'<br>'.join([f"• {d}" for d in details])}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📰 Noticias Recientes")
        for n in news[:3]:
            icon = "🟢" if n['sentiment'] > 0 else "🔴" if n['sentiment'] < 0 else "⚪"
            st.markdown(f"{icon} [{n['title']}]({n['link']})")

    # --- SECCIÓN DE TRADING ---
    st.markdown("---")
    tab1, tab2 = st.tabs(["⚡ EJECUCIÓN", "📚 HISTORIAL (Paper Trading)"])
    
    with tab1:
        if signal != "NEUTRO":
            direction = "LONG ⬆️" if signal == "LONG" else "SHORT ⬇️"
            st.info(f"Oportunidad Detectada: {direction}")
            
            with st.form("trade_form"):
                qty = st.number_input("Tamaño (USDT)", value=100.0)
                lev = st.slider("Apalancamiento", 1, 20, 5)
                sl_price = current_price - (atr * 1.5) if signal == "LONG" else current_price + (atr * 1.5)
                tp_price = current_price + (atr * 3.0) if signal == "LONG" else current_price - (atr * 3.0)
                
                c_sub1, c_sub2 = st.columns(2)
                c_sub1.write(f"🛑 SL Sugerido: ${sl_price:.2f}")
                c_sub2.write(f"🎯 TP Sugerido: ${tp_price:.2f}")
                
                if st.form_submit_button(f"🚀 EJECUTAR {signal}"):
                    trade_data = {
                        "timestamp": str(pd.Timestamp.now()), "symbol": symbol, "type": signal,
                        "entry": current_price, "size": qty, "leverage": lev,
                        "sl": sl_price, "tp1": tp_price, "tp2": 0, "tp3": 0,
                        "status": "OPEN", "pnl": 0.0, "reason": "Signal", "candles_held": 0, "atr_entry": atr
                    }
                    if db_mgr.add_trade(trade_data):
                        st.success("Orden enviada a Paper Trading DB")
                    else:
                        st.error("Error al guardar orden")
        else:
            st.write("Esperando configuración de alta probabilidad...")

    with tab2:
        trades_df = db_mgr.load_trades()
        if not trades_df.empty:
            st.dataframe(trades_df)
            pnl_total = trades_df[trades_df['status']=='CLOSED']['pnl'].sum()
            st.metric("PnL Acumulado", f"${pnl_total:.2f}")
        else:
            st.info("No hay trades registrados aún.")

if __name__ == "__main__":
    main()