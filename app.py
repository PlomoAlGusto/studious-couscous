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
    from utils import setup_logging, init_nltk, send_telegram_alert
except ImportError as e:
    st.error(f"Error crítico: {e}")
    st.stop()

st.set_page_config(page_title="Quimera Pro", layout="wide", page_icon="🦁")
setup_logging()
init_nltk()

# --- CSS MEJORADO (VISUALES PRO) ---
st.markdown("""
<style>
    /* TAGS DE CABECERA */
    .source-tag { background-color: #21262d; color: #8b949e; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid #30363d; }
    .symbol-tag { background-color: #1f6feb; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }

    /* TARJETA DE TRADE (SETUP CARD) */
    .trade-setup {
        background-color: #0d1117; 
        border: 1px solid #30363d;
        padding: 20px; 
        border-radius: 8px; 
        margin-top: 15px; 
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    
    /* Textos de Precios */
    .price-label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .price-val { font-family: 'SF Mono', 'Consolas', monospace; font-size: 15px; font-weight: bold; }
    
    .entry-col { color: #58a6ff; } /* Azul */
    .sl-col { color: #f85149; }    /* Rojo */
    .tp-col { color: #3fb950; }    /* Verde */
    
    /* Barra de Probabilidad Neon */
    .prob-container { 
        width: 100%; background-color: #21262d; height: 10px; 
        border-radius: 5px; margin: 10px 0; position: relative;
    }
    .prob-fill { 
        height: 100%; border-radius: 5px; 
        box-shadow: 0 0 10px currentColor; transition: width 1s ease-in-out;
    }
    .prob-text { position: absolute; right: 0; top: -18px; font-size: 11px; color: #8b949e; }

    /* QUIMERA AI ANALYSIS BOX */
    .ai-analysis-box {
        background-color: #161b22;
        border-top: 3px solid #a371f7; /* Morado AI */
        padding: 15px;
        border-radius: 0 0 6px 6px;
        margin-bottom: 20px;
        font-size: 13px;
    }
    .ai-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }
    .ai-row:last-child { border-bottom: none; }
    .ai-label { color: #8b949e; display: flex; align-items: center; gap: 6px; }
    .ai-val { font-weight: bold; color: #e6edf3; }

    /* NOTICIAS */
    .news-box { background-color: #161b22; padding: 15px; border-radius: 6px; border: 1px solid #30363d; }
    .news-item { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 12px; }
    .news-link { color: #c9d1d9; text-decoration: none; }
    .news-link:hover { color: #58a6ff; }

</style>
""", unsafe_allow_html=True)

# --- FUNCIONES ---
def display_market_sessions():
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("### 🌍 SESIONES (UTC)")
    for name, (start, end) in sessions.items():
        is_open = start <= hour < end if start < end else (hour >= start or hour < end)
        status = "🟢" if is_open else "🔴"
        bg = "rgba(50,255,50,0.1)" if is_open else "rgba(255,255,255,0.05)"
        st.sidebar.markdown(f"<div style='font-size:11px; padding:4px; margin-bottom:4px; background:{bg}; border-radius:3px; display:flex; justify-content:space-between;'><span>{name}</span><span>{status}</span></div>", unsafe_allow_html=True)

def calculate_optimal_leverage(entry, sl):
    if entry == 0: return 1
    dist_pct = abs(entry - sl) / entry
    if dist_pct == 0: return 1
    safe_lev = int(0.02 / dist_pct) # Arriesgar max 2% por trade
    return max(1, min(safe_lev, 50))

# --- GENERADORES DE HTML ---

def render_trade_card(type, signal_strength, price, sl, tp1, tp2, tp3, lev, prob):
    """Genera la tarjeta de trade con todos los detalles visuales"""
    
    # Configuración de colores y textos según estado
    if signal_strength == "DIAMOND":
        header_text = f"💎 SEÑAL DIAMANTE: {type}"
        header_color = "#3fb950" if type == "LONG" else "#f85149"
        bar_color = "#3fb950" if type == "LONG" else "#f85149"
    else:
        header_text = f"⚠️ OPORTUNIDAD POTENCIAL: {type}"
        header_color = "#d29922" # Naranja/Amarillo
        bar_color = "#d29922"

    html = f"""
    <div class="trade-setup">
        <div style="text-align:center; font-size:18px; font-weight:bold; color:{header_color}; margin-bottom:15px; letter-spacing:1px;">
            {header_text}
        </div>

        <!-- BARRA DE PROBABILIDAD -->
        <div style="margin-bottom:20px;">
            <div class="prob-text">Probabilidad Estimada: <b>{prob}%</b></div>
            <div class="prob-container">
                <div class="prob-fill" style="width:{prob}%; background-color:{bar_color};"></div>
            </div>
        </div>

        <!-- FILA 1: DATOS CLAVE -->
        <div style="display:flex; justify-content:space-around; margin-bottom:15px;">
            <div style="text-align:center;">
                <div class="price-label">ENTRADA</div>
                <div class="price-val entry-col">${price:,.2f}</div>
            </div>
            <div style="text-align:center;">
                <div class="price-label">STOP LOSS</div>
                <div class="price-val sl-col">${sl:,.2f}</div>
            </div>
            <div style="text-align:center;">
                <div class="price-label">LEV ÓPTIMO</div>
                <div class="price-val" style="color:white">{lev}x</div>
            </div>
        </div>

        <!-- FILA 2: TAKE PROFITS -->
        <div style="display:flex; justify-content:space-between; background-color:#161b22; padding:10px; border-radius:6px; border:1px solid #30363d;">
            <div style="text-align:center;">
                <div class="price-label">TP 1 (Scalp)</div>
                <div class="price-val tp-col">${tp1:,.2f}</div>
            </div>
            <div style="text-align:center;">
                <div class="price-label">TP 2 (Intra)</div>
                <div class="price-val tp-col">${tp2:,.2f}</div>
            </div>
            <div style="text-align:center;">
                <div class="price-label">TP 3 (Swing)</div>
                <div class="price-val tp-col">${tp3:,.2f}</div>
            </div>
        </div>
    </div>
    """
    return html

def render_quimera_ai(regime, atr, fr, fng, rsi, trend_strength):
    """Genera el análisis detallado de IA"""
    
    # Interpretar datos
    market_mood = "Miedo Extremo" if fng < 25 else "Codicia" if fng > 75 else "Neutral"
    volatility = "Alta ⚠️" if atr > (atr*1.2) else "Normal" # Simplificado
    funding_st = "Alcista" if fr > 0.01 else "Bajista" if fr < -0.01 else "Equilibrado"
    
    html = f"""
    <div style="margin-bottom:5px; font-weight:bold; color:#a371f7; display:flex; align-items:center; gap:5px;">
        <span>🧠 QUIMERA AI ANALYSIS</span>
    </div>
    <div class="ai-analysis-box">
        <div class="ai-row">
            <span class="ai-label">🌊 Estructura de Mercado</span>
            <span class="ai-val" style="color:{'#3fb950' if 'TENDENCIA' in regime else '#d29922'}">{regime}</span>
        </div>
        <div class="ai-row">
            <span class="ai-label">📊 Fuerza de Tendencia</span>
            <span class="ai-val">{trend_strength}</span>
        </div>
        <div class="ai-row">
            <span class="ai-label">💢 Volatilidad (ATR)</span>
            <span class="ai-val">${atr:.2f}</span>
        </div>
        <div class="ai-row">
            <span class="ai-label">🐋 Flujo (Funding)</span>
            <span class="ai-val">{funding_st} ({fr:.4f}%)</span>
        </div>
        <div class="ai-row">
            <span class="ai-label">🌡️ Sentimiento (F&G)</span>
            <span class="ai-val">{fng} ({market_mood})</span>
        </div>
        <div class="ai-row">
            <span class="ai-label">🔮 Momento (RSI)</span>
            <span class="ai-val">{rsi:.1f}</span>
        </div>
    </div>
    """
    return html

# --- SIDEBAR ---
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
    if st.button("🗑️ RESETEAR DATA"): TradeManager().reset_account(); st.rerun()

if auto_refresh: st_autorefresh(interval=60000)

# --- MAIN ---
def main():
    data_mgr = DataManager()
    strat_mgr = StrategyManager()
    db_mgr = TradeManager()

    with st.spinner("Analizando mercado..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    if df is None: st.error("❌ Error de datos."); return

    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df)
    signal_raw, atr, details, regime = strat_mgr.get_signal(df, filters)
    price = df['close'].iloc[-1]
    
    # --- LOGICA DE SEÑAL ---
    # Si es NEUTRO, calculamos potencial basado en EMA pura
    display_signal = signal_raw
    signal_strength = "WEAK"
    
    if signal_raw == "NEUTRO":
        last = df.iloc[-1]
        if last['EMA_20'] > last['EMA_50']:
            display_signal = "LONG"
            signal_strength = "POTENTIAL"
        else:
            display_signal = "SHORT"
            signal_strength = "POTENTIAL"
    else:
        signal_strength = "DIAMOND"

    # Probabilidad
    prob = 55
    if signal_strength == "DIAMOND": prob = 85
    elif signal_strength == "POTENTIAL": prob = 60
    if regime == "TENDENCIA": prob += 5

    # --- DASHBOARD LAYOUT ---
    col1, col2 = st.columns([2.5, 1])

    with col1:
        # Header
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:15px;'>
            <span class='symbol-tag'>{symbol}</span>
            <span class='source-tag'>📡 YAHOO FINANCE</span>
            <span class='source-tag'>⏱️ {timeframe}</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${price:,.2f}")
        c2.metric("Tendencia", display_signal, delta=regime)
        c3.metric("Funding", f"{fr:.4f}%")
        c4.metric("F&G", f"{fng_val}")

        # Gráfico
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='#e3b341', width=1), name='EMA 20'), row=1, col=1)
        if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='#a371f7', dash='dot'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='#238636'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # --- TARJETA DE TRADE (Siempre visible) ---
        sl_dist = atr * 1.5
        sl = price - sl_dist if display_signal == "LONG" else price + sl_dist
        tp1 = price + sl_dist if display_signal == "LONG" else price - sl_dist
        tp2 = price + (sl_dist * 2) if display_signal == "LONG" else price - (sl_dist * 2)
        tp3 = price + (sl_dist * 3.5) if display_signal == "LONG" else price - (sl_dist * 3.5)
        opt_lev = calculate_optimal_leverage(price, sl)

        # Renderizar HTML de Tarjeta
        st.markdown(render_trade_card(display_signal, signal_strength, price, sl, tp1, tp2, tp3, opt_lev, prob), unsafe_allow_html=True)

        # Ejecución
        c_btn1, c_btn2 = st.columns([1, 2])
        size = c_btn1.number_input("Inversión USDT", value=1000.0)
        
        btn_label = f"🚀 EJECUTAR {display_signal}" if signal_strength == "DIAMOND" else f"⚠️ FORZAR {display_signal} (RIESGO)"
        btn_type = "primary" if signal_strength == "DIAMOND" else "secondary"

        if c_btn2.button(btn_label, use_container_width=True, type=btn_type):
            trade = {
                "timestamp": str(datetime.now()), "symbol": symbol, "type": display_signal,
                "entry": price, "size": size, "leverage": opt_lev,
                "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0, "status": "OPEN", "pnl": 0.0,
                "reason": f"{signal_strength} Setup", "candles_held": 0, "atr_entry": atr
            }
            db_mgr.add_trade(trade)
            
            # Telegram
            with st.spinner("Notificando..."):
                send_telegram_alert(symbol, display_signal, price, sl, tp1, opt_lev)
            st.success("✅ Orden Enviada")

    with col2:
        # 1. QUIMERA IA (Detallada)
        last_rsi = df['RSI'].iloc[-1]
        adx_val = df['ADX_14'].iloc[-1] if 'ADX_14' in df.columns else 0
        trend_str = "Fuerte" if adx_val > 25 else "Débil"
        
        st.markdown(render_quimera_ai(regime, atr, fr, fng_val, last_rsi, trend_str), unsafe_allow_html=True)

        # 2. NOTICIAS (Caja Blanca)
        st.markdown(f"<div style='font-weight:bold; margin-bottom:10px;'>📰 Live News Feed</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='news-box' style='height:400px; overflow-y:auto;'>{''.join([f'<div class=news-item><a class=news-link href={n['link']} target=_blank>{n['title']}</a></div>' for n in news[:10]])}</div>", unsafe_allow_html=True)

    # --- HISTORIAL ---
    st.divider()
    df_trades = db_mgr.load_trades()
    if not df_trades.empty:
        st.dataframe(df_trades, use_container_width=True)

if __name__ == "__main__":
    main()
