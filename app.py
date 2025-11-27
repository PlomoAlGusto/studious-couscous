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

# --- CSS (Estilos) ---
st.markdown("""
<style>
    .source-tag {
        background-color: #333; color: #ddd; padding: 2px 8px; 
        border-radius: 4px; font-size: 11px; border: 1px solid #555;
    }
    .symbol-tag {
        background-color: #44AAFF; color: white; padding: 2px 8px; 
        border-radius: 4px; font-size: 12px; font-weight: bold;
    }
    .trade-setup {
        background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d;
        margin-top: 15px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .diamond-badge {
        border: 1px solid #00FF00; background-color: rgba(0,255,0,0.1); 
        color: #00FF00; padding: 5px; border-radius: 5px; font-weight: bold;
        margin-bottom: 10px; display: inline-block;
    }
    .potential-badge {
        border: 1px solid #FFA500; background-color: rgba(255,165,0,0.1); 
        color: #FFA500; padding: 5px; border-radius: 5px; font-weight: bold;
        margin-bottom: 10px; display: inline-block;
    }
    /* ... (Resto de estilos previos se mantienen implícitos) ... */
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
        st.sidebar.markdown(f"<div style='font-size:12px; display:flex; justify-content:space-between; margin-bottom:4px; padding:4px; background:rgba(255,255,255,0.05); border-radius:4px;'><span>{name}</span><span>{status}</span></div>", unsafe_allow_html=True)

def calculate_optimal_leverage(entry, sl):
    if entry == 0: return 1
    dist_pct = abs(entry - sl) / entry
    if dist_pct == 0: return 1
    safe_lev = int(0.02 / dist_pct) # Arriesgar 2% de cuenta
    return max(1, min(safe_lev, 50))

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

    with st.spinner("Analizando..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        fr, oi = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()

    if df is None: st.error("❌ Error de datos."); return

    df = strat_mgr.prepare_data(df)
    strat_mgr.train_regime_model(df)
    signal_raw, atr, details, regime = strat_mgr.get_signal(df, filters)
    price = df['close'].iloc[-1]

    # --- LÓGICA DE DIAMANTE VS POTENCIAL ---
    # Si la señal oficial es NEUTRO, calculamos una "Sugerencia" basada solo en tendencia
    display_signal = signal_raw
    signal_strength = "WEAK"
    
    if signal_raw == "NEUTRO":
        # Calculamos tendencia básica para sugerir algo
        last = df.iloc[-1]
        if last['EMA_20'] > last['EMA_50']:
            display_signal = "LONG"
            signal_strength = "POTENTIAL" # Señal débil / Sugerencia
        else:
            display_signal = "SHORT"
            signal_strength = "POTENTIAL"
    else:
        # Si la estrategia dio señal, es FUERTE (Diamante)
        signal_strength = "DIAMOND"

    # --- DASHBOARD ---
    col1, col2 = st.columns([2.5, 1])

    with col1:
        # HEADER CON INFORMACIÓN DE FUENTE
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:10px;'>
            <span class='symbol-tag'>{symbol}</span>
            <span class='source-tag'>📡 DATA: YAHOO FINANCE</span>
            <span class='source-tag'>⏱️ TF: {timeframe}</span>
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
        if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- TARJETA DE TRADE INTELIGENTE ---
        # Siempre mostramos tarjeta, diferenciando Diamante vs Potencial
        
        sl_dist = atr * 1.5
        sl = price - sl_dist if display_signal == "LONG" else price + sl_dist
        tp1 = price + sl_dist if display_signal == "LONG" else price - sl_dist
        opt_lev = calculate_optimal_leverage(price, sl)
        
        # Diseño de Tarjeta
        if signal_strength == "DIAMOND":
            badge = "<div class='diamond-badge'>💎 SEÑAL DIAMANTE CONFIRMADA</div>"
            prob = 85
            color_s = "#00FF00" if display_signal == "LONG" else "#FF4444"
        else:
            badge = "<div class='potential-badge'>⚠️ OPORTUNIDAD POTENCIAL (RIESGO)</div>"
            prob = 45
            color_s = "#FFA500" # Naranja para potencial

        st.markdown(f"""
        <div class="trade-setup">
            {badge}
            <div style="font-size:20px; font-weight:bold; color:{color_s}; margin-top:5px;">
                {display_signal} DETECTADO
            </div>
            <div style="margin-top:15px; display:flex; justify-content:space-around;">
                <div><span style='color:#888; font-size:10px'>ENTRADA</span><br><span style='color:#44AAFF; font-weight:bold'>${price:.2f}</span></div>
                <div><span style='color:#888; font-size:10px'>STOP LOSS</span><br><span style='color:#FF4444; font-weight:bold'>${sl:.2f}</span></div>
                <div><span style='color:#888; font-size:10px'>LEV</span><br><span style='color:white; font-weight:bold'>{opt_lev}x</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Botón Ejecutar
        c_btn1, c_btn2 = st.columns([1, 2])
        size = c_btn1.number_input("Inversión USDT", value=1000.0)
        
        btn_label = f"🚀 EJECUTAR {display_signal} AHORA" if signal_strength == "DIAMOND" else f"⚠️ FORZAR {display_signal} (POTENCIAL)"
        
        if c_btn2.button(btn_label, use_container_width=True, type="primary"):
            trade = {
                "timestamp": str(datetime.now()), "symbol": symbol, "type": display_signal,
                "entry": price, "size": size, "leverage": opt_lev,
                "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0, "status": "OPEN", "pnl": 0.0,
                "reason": f"{signal_strength} Signal", "candles_held": 0, "atr_entry": atr
            }
            db_mgr.add_trade(trade)
            
            # --- ENVIAR TELEGRAM ---
            with st.spinner("Enviando alerta a Telegram..."):
                sent = send_telegram_alert(symbol, f"{display_signal} ({signal_strength})", price, sl, tp1, opt_lev)
            
            if sent: st.success("✅ Alerta enviada al móvil")
            else: st.warning("Trade guardado, pero falló Telegram (Revisa secrets.toml)")
            
            st.balloons()

    with col2:
        # Noticias
        st.markdown(f"""
        <div style='background-color:#161b22; padding:15px; border-radius:5px; border-left:4px solid white;'>
            <span style='font-weight:bold; color:white;'>📰 LIVE NEWS (10)</span>
            <div style='height:400px; overflow-y:auto; margin-top:10px;'>
                {''.join([f"<div style='margin-bottom:8px; border-bottom:1px solid #333; padding-bottom:4px;'><a href='{n['link']}' target='_blank' style='color:#ccc; text-decoration:none; font-size:12px;'>{n['title']}</a></div>" for n in news[:10]])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Historial
    st.divider()
    df_trades = db_mgr.load_trades()
    if not df_trades.empty:
        st.dataframe(df_trades, use_container_width=True)

if __name__ == "__main__":
    main()
