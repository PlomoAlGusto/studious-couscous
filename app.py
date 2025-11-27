import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import textwrap
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

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    .source-tag { background-color: #21262d; color: #8b949e; padding: 4px 8px; border-radius: 4px; font-size: 11px; border: 1px solid #30363d; font-family: monospace; }
    .symbol-tag { background-color: #1f6feb; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; font-family: monospace; }

    div.trade-card-box {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 25px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
    }

    .prob-track { width: 100%; height: 10px; background-color: #21262d; border-radius: 5px; margin: 10px 0 20px 0; overflow: hidden; }
    .price-grid-row { display: flex; justify-content: space-between; margin-bottom: 15px; gap: 10px; }
    .price-col { flex: 1; text-align: center; }
    
    .price-box-dark { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; text-align: center; flex: 1; }

    .t-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .t-val { font-family: 'Consolas', monospace; font-size: 17px; font-weight: bold; }
    .c-blue { color: #58a6ff !important; } .c-red { color: #f85149 !important; } .c-green { color: #3fb950 !important; } .c-white { color: #f0f6fc !important; }

    .news-container { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; }
    .news-row { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 12px; }
    .news-link { color: #c9d1d9; text-decoration: none; } .news-link:hover { color: #58a6ff; }
    
    .clock-open { background-color: rgba(50,255,50,0.1); } .clock-closed { background-color: rgba(255,255,255,0.05); }
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
        st.sidebar.markdown(f"<div style='font-size:11px; padding:5px; margin-bottom:5px; background:{bg}; border-radius:4px; display:flex; justify-content:space-between;'><span>{name}</span><span>{status}</span></div>", unsafe_allow_html=True)

def calculate_optimal_leverage(entry, sl):
    if entry == 0: return 1
    dist_pct = abs(entry - sl) / entry
    if dist_pct == 0: return 1
    safe_lev = int(0.02 / dist_pct)
    return max(1, min(safe_lev, 50))

def render_trade_card(type, signal_strength, price, sl, tp1, tp2, tp3, lev, prob):
    if signal_strength == "DIAMOND":
        header = f"💎 SEÑAL DIAMANTE: {type}"
        h_color = "#3fb950" if type == "LONG" else "#f85149"
        bar_color = h_color
    else:
        header = f"⚠️ OPORTUNIDAD POTENCIAL: {type}"
        h_color = "#d29922"
        bar_color = "#d29922"

    html = f"""
    <div class="trade-card-box">
        <div style="text-align:center; font-size:20px; font-weight:bold; color:{h_color}; margin-bottom:5px; letter-spacing:1px; text-transform:uppercase;">{header}</div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#8b949e; margin-bottom:5px;"><span>CONFIANZA IA</span><span style="color:{bar_color}; font-weight:bold;">{prob}%</span></div>
        <div class="prob-track"><div style="width:{prob}%; height:100%; background-color:{bar_color}; box-shadow: 0 0 15px {bar_color};"></div></div>
        <div class="price-grid-row">
            <div class="price-col"><div class="t-label">ENTRADA</div><div class="t-val c-blue">${price:,.2f}</div></div>
            <div class="price-col"><div class="t-label">STOP LOSS</div><div class="t-val c-red">${sl:,.2f}</div></div>
            <div class="price-col"><div class="t-label">LEVERAGE</div><div class="t-val c-white">{lev}x</div></div>
        </div>
        <div class="price-grid-row" style="margin-bottom:0;">
            <div class="price-box-dark"><div class="t-label">TP 1</div><div class="t-val c-green">${tp1:,.2f}</div></div>
            <div class="price-box-dark"><div class="t-label">TP 2</div><div class="t-val c-green">${tp2:,.2f}</div></div>
            <div class="price-box-dark"><div class="t-label">TP 3</div><div class="t-val c-green">${tp3:,.2f}</div></div>
        </div>
    </div>
    """
    return textwrap.dedent(html)

def render_quimera_ai(regime, atr, fr, fng, rsi, trend_strength, adr_val, tsi_val, mfi_val, trend_status):
    c_reg = "#3fb950" if "TENDENCIA" in regime else "#d29922"
    
    # Lógica de color para cambio de tendencia
    c_trend = "#e6edf3"
    if "GIRO" in trend_status: c_trend = "#a371f7" # Morado para cambios
    elif "DEBILIDAD" in trend_status: c_trend = "#d29922"
    elif "FUERTE" in trend_status: c_trend = "#3fb950"
    
    # Icono de cambio
    icon_trend = "🔄" if "GIRO" in trend_status else "📈" if "FUERTE" in trend_status else "⚠️"

    html = f"""
    <div style="margin-bottom:10px; font-weight:bold; color:#a371f7; display:flex; align-items:center; gap:5px; font-size:14px;">
        <span>🧠 QUIMERA AI ANALYSIS</span>
    </div>
    <div style="background-color:#161b22; border-top:3px solid #a371f7; padding:15px; border-radius:0 0 6px 6px; margin-bottom:20px;">
        
        <!-- ESTADO DE TENDENCIA (EL NUEVO) -->
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #30363d; background:rgba(255,255,255,0.03);">
            <span style="color:#a371f7; font-weight:bold;">{icon_trend} Estado Tendencia</span>
            <span style="color:{c_trend}; font-weight:bold">{trend_status}</span>
        </div>

        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #30363d;">
            <span style="color:#8b949e">🌊 Estructura</span><span style="color:{c_reg}; font-weight:bold">{regime}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #30363d;">
            <span style="color:#8b949e">📊 Fuerza ADX</span><span style="color:#e6edf3; font-weight:bold">{trend_strength}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #30363d;">
            <span style="color:#8b949e">📏 ADR (Rango Diario)</span><span style="color:#e6edf3; font-weight:bold">{adr_val:.2f}%</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #30363d;">
            <span style="color:#8b949e">🚀 TSI (Momento)</span><span style="color:#e6edf3; font-weight:bold">{tsi_val:.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #30363d;">
            <span style="color:#8b949e">🌡️ Sentimiento</span><span style="color:#e6edf3; font-weight:bold">{fng}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:8px 0;">
            <span style="color:#8b949e">💢 Volatilidad $</span><span style="color:#e6edf3; font-weight:bold">${atr:.2f}</span>
        </div>
    </div>
    """
    return textwrap.dedent(html)

def render_news_box(news):
    news_items_html = ""
    for n in news[:10]:
        news_items_html += f"<div class='news-row'><a class='news-link' href='{n['link']}' target='_blank'>🔗 {n['title']}</a></div>"
    
    html = f"""
    <div style="font-weight:bold; margin-bottom:10px; color:white;">📰 LIVE NEWS (10)</div>
    <div class="news-container" style="height:400px; overflow-y:auto; border-left: 4px solid white;">{news_items_html}</div>
    """
    return textwrap.dedent(html)

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
    if st.button("🗑️ RESET"): TradeManager().reset_account(); st.rerun()

if auto_refresh: st_autorefresh(interval=60000)

# --- MAIN ---
def main():
    data_mgr = DataManager()
    strat_mgr = StrategyManager()
    db_mgr = TradeManager()

    with st.spinner(f"Analizando {symbol}..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    if df is None: st.error("❌ Error de datos."); return

    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df)
    
    # Obtenemos la señal y también el nuevo "trend_status"
    signal_raw, atr, details, regime, trend_status = strat_mgr.get_signal(df, filters)
    price = df['close'].iloc[-1]
    
    # Logic Display
    display_signal = signal_raw
    signal_strength = "WEAK"
    if signal_raw == "NEUTRO":
        last = df.iloc[-1]
        if last.get('EMA_20', 0) > last.get('EMA_50', 0):
            display_signal = "LONG"; signal_strength = "POTENTIAL"
        else:
            display_signal = "SHORT"; signal_strength = "POTENTIAL"
    else:
        signal_strength = "DIAMOND"

    prob = 85 if signal_strength == "DIAMOND" else 60
    if regime == "TENDENCIA": prob += 5

    col1, col2 = st.columns([2.5, 1])
    with col1:
        st.markdown(f"<div style='display:flex; gap:10px; margin-bottom:15px;'><span class='symbol-tag'>{symbol}</span><span class='source-tag'>📡 YAHOO</span><span class='source-tag'>⏱️ {timeframe}</span></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${price:,.2f}")
        c2.metric("Tendencia", display_signal, delta=regime)
        c3.metric("Funding", f"{fr:.4f}%")
        c4.metric("F&G", f"{fng_val}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
        if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        sl_dist = atr * 1.5
        sl = price - sl_dist if display_signal == "LONG" else price + sl_dist
        tp1 = price + sl_dist if display_signal == "LONG" else price - sl_dist
        tp2 = price + (sl_dist * 2) if display_signal == "LONG" else price - (sl_dist * 2)
        tp3 = price + (sl_dist * 3.5) if display_signal == "LONG" else price - (sl_dist * 3.5)
        opt_lev = calculate_optimal_leverage(price, sl)

        st.markdown(render_trade_card(display_signal, signal_strength, price, sl, tp1, tp2, tp3, opt_lev, prob), unsafe_allow_html=True)

        c_btn1, c_btn2 = st.columns([1, 2])
        size = c_btn1.number_input("Inversión USDT", value=1000.0)
        if c_btn2.button(f"🚀 EJECUTAR {display_signal}", use_container_width=True, type="primary"):
            trade = {"timestamp": str(datetime.now()), "symbol": symbol, "type": display_signal, "entry": price, "size": size, "leverage": opt_lev, "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0, "status": "OPEN", "pnl": 0.0, "reason": f"{signal_strength}", "candles_held": 0, "atr_entry": atr}
            db_mgr.add_trade(trade)
            with st.spinner("Notificando..."):
                send_telegram_alert(symbol, display_signal, price, sl, tp1, opt_lev)
            st.success("Orden Enviada")

    with col2:
        # Extracción de datos
        rsi_val = df['RSI'].iloc[-1]
        adx_val = df['ADX_14'].iloc[-1] if 'ADX_14' in df.columns else 0
        trend_str = "Fuerte" if adx_val > 25 else "Débil"
        adr_val = df['ADR'].iloc[-1] if 'ADR' in df.columns else 0
        tsi_val = df['TSI'].iloc[-1] if 'TSI' in df.columns else 0
        mfi_val = df['MFI'].iloc[-1] if 'MFI' in df.columns else 50
        
        # Render con Trend Status
        st.markdown(render_quimera_ai(regime, atr, fr, fng_val, rsi_val, trend_str, adr_val, tsi_val, mfi_val, trend_status), unsafe_allow_html=True)
        st.markdown(render_news_box(news), unsafe_allow_html=True)

    st.divider()
    df_trades = db_mgr.load_trades()
    if not df_trades.empty: st.dataframe(df_trades, use_container_width=True)

if __name__ == "__main__":
    main()
