import streamlit as st
import subprocess
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# --- AUTO-FIX DEPENDENCIAS ---
try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "https://github.com/xgboosted/pandas-ta-classic/archive/main.zip"])
    import pandas_ta as ta

# --- IMPORTAMOS SOLO LO NECESARIO ---
from config import config
from database import TradeManager
from data_feed import DataManager # Usamos el nuevo data_feed
from utils import setup_logging, init_nltk

# CONFIGURACIÓN
st.set_page_config(page_title="Quimera Pro v18.0 Analyst", layout="wide", page_icon="🦁")
setup_logging()
init_nltk()

# INSTANCIAS
db_mgr = TradeManager()
data_mgr = DataManager()

# --- CSS AGRESIVO (DISEÑO SOLICITADO) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* CABECERAS DE SEÑAL */
    .header-confirmed-long { color: #00FF00; font-size: 22px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px solid #00FF00; padding-bottom: 10px; margin-bottom: 15px;}
    .header-confirmed-short { color: #FF4444; font-size: 22px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px solid #FF4444; padding-bottom: 10px; margin-bottom: 15px;}
    .header-potential { color: #FFD700; font-size: 22px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px dashed #FFD700; padding-bottom: 10px; margin-bottom: 15px;}

    /* TARJETA PRINCIPAL */
    .trade-setup {
        background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
        padding: 25px; 
        border-radius: 12px; 
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    
    /* MÉTRICAS TARJETA */
    .metric-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value-entry { font-size: 18px; font-weight: bold; color: #44AAFF; font-family: 'Consolas'; }
    .metric-value-tp { font-size: 18px; font-weight: bold; color: #00FF00; font-family: 'Consolas'; }
    .metric-value-sl { font-size: 18px; font-weight: bold; color: #FF4444; font-family: 'Consolas'; }

    /* CAJA IA */
    .ai-box {
        background-color: #080808; border-left: 3px solid #44AAFF; 
        padding: 15px; border-radius: 0 5px 5px 0; margin-bottom: 15px; 
        font-family: 'Courier New', monospace; font-size: 12px; color: #ccc;
        border: 1px solid #222;
    }
    
    .market-clock { font-size: 11px; padding: 4px; margin-bottom: 4px; border-radius: 3px; display: flex; justify-content: space-between; border: 1px solid #333; background: #111;}
</style>
""", unsafe_allow_html=True)

# --- LOGICA DE ESTRATEGIA (INTEGRADA PARA EVITAR ERRORES EXTERNOS) ---
def calculate_strategy_internal(df):
    if df is None: return None
    # Indicadores
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    try:
        if 'volume' in df.columns:
            df.ta.vwap(append=True)
            if 'VWAP_D' in df.columns: df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True)
    except: pass
    if 'VWAP' not in df.columns: df['VWAP'] = df['EMA_50'] # Fallback
    return df.fillna(method='ffill').fillna(method='bfill')

def get_signal(df):
    row = df.iloc[-1]
    score = 0
    
    # Lógica simple pero efectiva
    if row['EMA_20'] > row['EMA_50']: score += 1
    else: score -= 1
    
    if row['close'] > row['VWAP']: score += 1
    else: score -= 1
    
    signal = "NEUTRO"
    if score >= 2: signal = "LONG"
    elif score <= -2: signal = "SHORT"
    
    # Lógica de "Potencial" (Clave para el diseño amarillo)
    calc_dir = signal
    if signal == "NEUTRO":
        if row['close'] > row['EMA_50']: calc_dir = "LONG"
        else: calc_dir = "SHORT"
        
    # Probabilidad estimada
    prob = 50.0 + (abs(score) * 20)
    if prob > 95: prob = 95
    
    return signal, calc_dir, prob, row['ATR']

# --- UI ---
def main():
    with st.sidebar:
        st.title("🦁 QUIMERA v18")
        st.markdown("---")
        symbol = st.text_input("Ticker", "BTC/USDT").upper()
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h"])
        
        if st.button("🔥 RESETEAR DATOS"):
            db_mgr.reset_account()
            st.rerun()

    # Carga de datos
    with st.spinner("Analizando..."):
        df = data_mgr.fetch_market_data(symbol, timeframe)
        if df is None: st.error("Error API"); return
        
        df = calculate_strategy_internal(df)
        fr, _ = data_mgr.get_funding_rate(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()
        news = data_mgr.fetch_news(symbol)
        
        signal, calc_dir, prob, atr = get_signal(df)
        current_price = df['close'].iloc[-1]

    # --- DASHBOARD ---
    tab1, tab2 = st.tabs(["📊 TERMINAL", "📜 HISTORIAL"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Métricas Header
            m1, m2, m3 = st.columns(3)
            m1.metric("Precio", f"${current_price:,.2f}", f"{df['close'].pct_change().iloc[-1]:.2%}")
            m2.metric("Funding Rate", f"{fr:.4f}%")
            m3.metric("F&G Index", f"{fng_val}", fng_class)
            
            # Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], line=dict(color='yellow', width=1), name='EMA 50'), row=1, col=1)
            if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='cyan', dash='dot'), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="gray")
            fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="gray")
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Caja IA
            st.markdown(f"""
            <div class='ai-box'>
                <span style='color:#44AAFF; font-weight:bold'>🤖 QUIMERA AI ANALYSIS</span><br>
                <br>
                📡 <b>SEÑAL DETECTADA:</b> {signal}<br>
                🌊 <b>TENDENCIA MACRO:</b> {calc_dir}<br>
                📊 <b>VOLATILIDAD (ATR):</b> ${atr:.2f}<br>
                🎯 <b>CONFIDENCE:</b> {prob:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
            # Noticias
            st.markdown("###### 📰 Noticias")
            for n in news[:5]:
                st.markdown(f"<div style='font-size:11px; border-bottom:1px solid #333; padding:5px'>▪ <a href='{n['link']}' style='color:#ccc; text-decoration:none'>{n['title'][:50]}...</a></div>", unsafe_allow_html=True)

        st.divider()

        # --- SECCIÓN DE EJECUCIÓN (DISEÑO TXT) ---
        # Calculamos niveles
        sl_dist = atr * 1.5 if atr > 0 else current_price * 0.01
        mult = 1 if calc_dir == "LONG" else -1
        
        sl = current_price - (sl_dist * mult)
        tp1 = current_price + (sl_dist * 1.0 * mult)
        tp2 = current_price + (sl_dist * 2.0 * mult)
        tp3 = current_price + (sl_dist * 3.5 * mult)
        
        # Estilos dinámicos
        if signal == "NEUTRO":
            header_class = "header-potential"
            title_text = f"⚠️ SETUP POTENCIAL: {calc_dir}"
            btn_text = f"⚠️ FORZAR {calc_dir}"
            bar_color = "#FFD700"
        else:
            header_class = f"header-confirmed-{calc_dir.lower()}"
            title_text = f"🚀 SETUP CONFIRMADO: {calc_dir}"
            btn_text = f"🚀 LANZAR {calc_dir}"
            bar_color = "#00FF00" if calc_dir == "LONG" else "#FF4444"

        # HTML DE LA TARJETA
        st.markdown(f"""
        <div class="trade-setup">
            <div class="{header_class}">{title_text}</div>
            
            <div style="margin: 15px 0;">
                <div style="display:flex; justify-content:space-between; font-size:12px; color:#888; margin-bottom:5px;">
                    <span>Probabilidad IA</span>
                    <span style="color:{bar_color}">{prob:.1f}%</span>
                </div>
                <div style="width:100%; background:#333; height:6px; border-radius:3px;">
                    <div style="width:{prob}%; background:{bar_color}; height:6px; border-radius:3px;"></div>
                </div>
            </div>

            <div style="display:flex; justify-content:space-around; margin-top:20px;">
                <div><div class="metric-label">ENTRADA</div><div class="metric-value-entry">${current_price:,.2f}</div></div>
                <div><div class="metric-label">STOP LOSS</div><div class="metric-value-sl">${sl:,.2f}</div></div>
                <div><div class="metric-label">TP 1</div><div class="metric-value-tp">${tp1:,.2f}</div></div>
                <div><div class="metric-label">TP 3</div><div class="metric-value-tp">${tp3:,.2f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Controles
        col_exec, _ = st.columns([1, 2])
        with col_exec:
            qty = st.number_input("Capital (USD)", 100.0, 100000.0, 1000.0)
            lev = st.slider("Leverage", 1, 50, 10)
            
            if st.button(btn_text, use_container_width=True):
                trade = {
                    "timestamp": str(datetime.now()), "symbol": symbol, "type": calc_dir,
                    "entry": current_price, "size": qty, "leverage": lev,
                    "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                    "status": "OPEN", "pnl": 0.0, "reason": f"Signal: {signal}", "atr_entry": atr
                }
                db_mgr.add_trade(trade)
                st.success("Orden Enviada")

    with tab2:
        df_trades = db_mgr.load_trades()
        if not df_trades.empty:
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("Historial vacío.")

if __name__ == "__main__":
    main()
