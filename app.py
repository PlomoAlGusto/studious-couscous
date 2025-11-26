import streamlit as st
import subprocess
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timezone
import requests

# --- MÓDULOS PROPIOS ---
from config import config
from database import TradeManager
from data_feed import DataManager
from utils import setup_logging, init_nltk

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Quimera Pro v18.0 Analyst", layout="wide", page_icon="🦁")
setup_logging()
init_nltk()

# INSTANCIAS
db_mgr = TradeManager()
data_mgr = DataManager()

# --- CONTROL DE ESTADO PARA ALERTAS ---
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = "NEUTRO"

# --- MOTOR DE CÁLCULO HÍBRIDO (SOLUCIÓN AL ERROR) ---
try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

def calculate_indicators_manual(df):
    """Cálculo manual de indicadores si falla la librería externa"""
    if df is None: return None
    df = df.copy()
    
    # EMA
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # VWAP
    if 'volume' in df.columns:
        v = df['volume'].values
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    else:
        df['VWAP'] = df['EMA_50']

    df['ADX_14'] = 25 
    df['MFI'] = 50    
    
    return df.fillna(method='bfill').fillna(method='ffill')

def process_data(df):
    if HAS_TA:
        try:
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            try: 
                if 'volume' in df.columns: 
                    df.ta.vwap(append=True)
                    if 'VWAP_D' in df.columns: df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True)
            except: pass
            if 'VWAP' not in df.columns: df['VWAP'] = df['EMA_50']

            try:
                adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                if adx is not None: df = pd.concat([df, adx], axis=1)
                if 'ADX_14' not in df.columns and 'ADX' in df.columns: df['ADX_14'] = df['ADX']
            except: df['ADX_14'] = 0
            
            df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            return df.fillna(method='ffill').fillna(method='bfill')
        except:
            return calculate_indicators_manual(df)
    else:
        return calculate_indicators_manual(df)

# --- FUNCIÓN TELEGRAM ---
def send_telegram_msg(msg):
    """Envía alertas a Telegram usando secrets"""
    try:
        token = st.secrets.get("TELEGRAM_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            params = {
                "chat_id": chat_id, 
                "text": msg, 
                "parse_mode": "Markdown"
            }
            requests.get(url, params=params, timeout=5)
        else:
            print("⚠️ Telegram no configurado en .streamlit/secrets.toml")
    except Exception as e:
        print(f"Error enviando Telegram: {e}")

# --- CSS EXACTO (DISEÑO VISUAL DEL TXT) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* Headers de Señal */
    .header-confirmed-long { color: #00FF00; font-size: 24px; font-weight: bold; text-shadow: 0 0 10px rgba(0,255,0,0.5); margin-bottom: 5px; }
    .header-confirmed-short { color: #FF4444; font-size: 24px; font-weight: bold; text-shadow: 0 0 10px rgba(255,68,68,0.5); margin-bottom: 5px; }
    .header-potential { color: #FFFF00; font-size: 22px; font-weight: bold; text-shadow: 0 0 10px rgba(255,255,0,0.3); margin-bottom: 5px; }

    /* Tarjeta de Trading */
    .trade-setup {
        background-color: #151515; 
        padding: 20px; 
        border-radius: 15px; 
        border: 1px solid #444;
        margin-top: 10px; 
        margin-bottom: 20px; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Valores */
    .tp-green { color: #00FF00; font-weight: bold; font-size: 16px; font-family: 'Consolas', monospace; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 16px; font-family: 'Consolas', monospace; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 16px; font-family: 'Consolas', monospace; }
    .label-mini { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; display: block;}

    /* Caja IA */
    .ai-box {
        background-color: #0e1117; border-left: 4px solid #44AAFF; 
        padding: 15px; border-radius: 5px; margin-bottom: 15px; 
        font-family: 'SF Mono', 'Consolas', monospace; font-size: 12px; color: #e0e0e0;
    }
    .ai-title { color: #44AAFF; font-weight: bold; font-size: 14px; border-bottom: 1px solid #333; padding-bottom: 5px; display: block; margin-bottom: 8px;}

    /* Badges */
    .badge-bull { background-color: #004400; color: #00FF00; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #00FF00; }
    .badge-bear { background-color: #440000; color: #FF4444; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #FF4444; }
    .badge-neutral { background-color: #333; color: #aaa; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #555; }
    
    /* Relojes */
    .market-clock { font-size: 11px; padding: 4px; margin-bottom: 4px; border-radius: 3px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.15); border: 1px solid #005500; color: #cfc; }
    .clock-closed { background-color: rgba(50, 50, 50, 0.3); border: 1px solid #333; color: #666; }
</style>
""", unsafe_allow_html=True)

# --- LOGICA DE SEÑAL ---
def get_signal_logic(df, filters, fr):
    row = df.iloc[-1]
    score = 0
    max_score = 0
    details = []
    
    # Lógica de Puntos
    if filters['use_ema']:
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1; details.append("<span class='badge-bull'>EMA</span>")
        else: score -= 1; details.append("<span class='badge-bear'>EMA</span>")
            
    if filters['use_vwap'] and 'VWAP' in row:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1; details.append("<span class='badge-bull'>VWAP</span>")
        else: score -= 1; details.append("<span class='badge-bear'>VWAP</span>")

    # Señal Principal
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"

    # Vetos (Rango / RSI)
    veto = False
    if filters['use_regime'] and row.get('ADX_14', 25) < 20: 
        signal = "NEUTRO"; details.append("<span class='badge-neutral'>RANGO</span>"); veto = True
        
    if filters['use_rsi']:
        if row['RSI'] > 70 and signal == "LONG": signal = "NEUTRO"; details.append("<span class='badge-neutral'>SOBRECOMPRA</span>"); veto = True
        if row['RSI'] < 30 and signal == "SHORT": signal = "NEUTRO"; details.append("<span class='badge-neutral'>SOBREVENTA</span>"); veto = True

    # Probabilidad
    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score) * 45)
    if fr and abs(fr) < 0.01: prob += 2
    
    # --- DIRECCIÓN POTENCIAL ---
    calc_dir = signal
    if signal == "NEUTRO":
        if row['close'] > row['EMA_50']: calc_dir = "LONG"
        else: calc_dir = "SHORT"
    
    return signal, calc_dir, prob, details, veto

# --- MAIN ---
def main():
    with st.sidebar:
        st.title("🦁 QUIMERA v18.0")
        
        now = datetime.now(timezone.utc)
        hour = now.hour
        sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9)}
        st.sidebar.markdown("### 🌍 SESIONES")
        for name, (start, end) in sessions.items():
            is_open = start <= hour < end if start < end else (hour >= start or hour < end)
            css = "clock-open" if is_open else "clock-closed"
            icon = "🟢" if is_open else "🔴"
            st.sidebar.markdown(f"<div class='market-clock {css}'><span>{name}</span><span>{icon}</span></div>", unsafe_allow_html=True)
        
        st.divider()
        symbol = st.text_input("Ticker", "BTC/USDT").upper()
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h"])
        
        with st.expander("🛡️ FILTROS", expanded=True):
            filters = {
                'use_ema': st.checkbox("Tendencia EMAs", True),
                'use_vwap': st.checkbox("Filtro VWAP", True),
                'use_regime': st.checkbox("Filtro ADX (Rango)", True),
                'use_rsi': st.checkbox("RSI Limit", True)
            }
        
        auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
        if st.button("🔥 RESETEAR CUENTA"):
            db_mgr.reset_account(); st.rerun()

    if auto_refresh:
        st.markdown("""<meta http-equiv="refresh" content="60">""", unsafe_allow_html=True)

    # --- PROCESAMIENTO ---
    with st.spinner("Analizando..."):
        df = data_mgr.fetch_market_data(symbol, timeframe, limit=300)
        if df is None: st.error("Error API"); return
        
        # Cálculo Robusto
        df = process_data(df)
        
        fr, _ = data_mgr.get_funding_rate(symbol)
        news = data_mgr.fetch_news(symbol)
        fng_val, fng_class = data_mgr.fetch_fear_greed()
        
        signal, calc_dir, prob, details, veto = get_signal_logic(df, filters, fr)
        current_price = df['close'].iloc[-1]
        atr = df['ATR'].iloc[-1]

    # --- SISTEMA DE ALERTAS TELEGRAM ---
    if signal != "NEUTRO" and signal != st.session_state.last_alert:
        msg = f"""🚀 *SEÑAL CONFIRMADA QUIMERA*
        
📉 *Activo:* {symbol}
⚡ *Dirección:* {signal}
🎯 *Precio:* ${current_price:,.2f}
📊 *Probabilidad:* {prob:.1f}%
        """
        send_telegram_msg(msg)
        st.session_state.last_alert = signal
    elif signal == "NEUTRO":
        st.session_state.last_alert = "NEUTRO"

    # --- UI DASHBOARD ---
    tab1, tab2 = st.tabs(["📊 LIVE COMMAND", "📜 HISTORIAL"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            m1, m2, m3 = st.columns(3)
            m1.metric("Precio", f"${current_price:,.2f}", f"{df['close'].pct_change().iloc[-1]:.2%}")
            m2.metric("Funding", f"{fr:.4f}%", delta_color="inverse")
            m3.metric("F&G", f"{fng_val}", fng_class)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], line=dict(color='#FF0000', width=1), name='EMA 50'), row=1, col=1)
            if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown(f"<div style='text-align:center; margin-bottom:5px'>{' '.join(details)}</div>", unsafe_allow_html=True)
            mfi_val = df.get('MFI', pd.Series([50])).iloc[-1]
            gas_color = "#00FF00" if mfi_val > 50 else "#FF4444"
            
            ai_html = f"""
            <div class='ai-box'>
                <span class='ai-title'>🤖 QUIMERA ANALYTICS:</span>
                <div>📡 <b>SEÑAL:</b> {signal}</div>
                <div>🌊 <b>POTENCIAL:</b> {calc_dir}</div>
                <div>📊 <b>ATR:</b> ${atr:.2f}</div>
                <div>🔥 <b>MFI:</b> <b style='color:{gas_color}'>{mfi_val:.0f}</b></div>
                <div>🎯 <b>PROBABILIDAD:</b> <b style='color:#44AAFF'>{prob:.1f}%</b></div>
            </div>
            """
            st.markdown(ai_html, unsafe_allow_html=True)
            
            st.markdown("###### 📰 Noticias")
            st.markdown("<div style='height: 200px; overflow-y: auto;'>", unsafe_allow_html=True)
            for n in news[:6]:
                col = "#00FF00" if n.get('sentiment', 0) > 0 else "#FF4444"
                st.markdown(f"<div style='font-size:11px; margin-bottom:5px;'>▪ <a href='{n['link']}' style='color:{col}; text-decoration:none'>{n['title'][:60]}...</a></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # --- SETUP VISUAL (TARJETA) ---
        sl_dist = atr * 1.5 if atr > 0 else current_price * 0.01
        
        if calc_dir == "LONG":
            sl = current_price - sl_dist
            tp1, tp2, tp3 = current_price + sl_dist, current_price + (sl_dist*2), current_price + (sl_dist*3.5)
            emoji = "⬆️ LONG"
            color_prob = "#00FF00"
            if signal == "LONG":
                header_cls = "header-confirmed-long"
                setup_title = f"{emoji} (CONFIRMADO)"
            else:
                header_cls = "header-potential"
                setup_title = f"{emoji} (POTENCIAL)"
        else:
            sl = current_price + sl_dist
            tp1, tp2, tp3 = current_price - sl_dist, current_price - (sl_dist*2), current_price - (sl_dist*3.5)
            emoji = "⬇️ SHORT"
            color_prob = "#FF4444"
            if signal == "SHORT":
                header_cls = "header-confirmed-short"
                setup_title = f"{emoji} (CONFIRMADO)"
            else:
                header_cls = "header-potential"
                setup_title = f"{emoji} (POTENCIAL)"

        html_card = f"""
        <div class="trade-setup">
            <div class="{header_cls}">{setup_title}</div>
            <div style='margin-top: 5px; margin-bottom: 10px; text-align: left;'>
                <div style='display:flex; justify-content:space-between; color:#ccc; font-size:12px; margin-bottom:2px;'>
                    <span>Probabilidad Estimada:</span><span style='color:{color_prob}; font-weight:bold;'>{prob:.1f}%</span>
                </div>
                <div style='width: 100%; background-color: #333; border-radius: 4px; height: 6px;'>
                    <div style='width: {prob}%; background-color: {color_prob}; height: 6px; border-radius: 4px; box-shadow: 0 0 8px {color_prob};'></div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                <div><span class="label-mini">ENTRADA</span><br><span class="entry-blue">${current_price:,.2f}</span></div>
                <div><span class="label-mini">STOP LOSS</span><br><span class="sl-red">${sl:,.2f}</span></div>
                <div><span class="label-mini">TP 1</span><br><span class="tp-green">${tp1:,.2f}</span></div>
                <div><span class="label-mini">TP 3</span><br><span class="tp-green">${tp3:,.2f}</span></div>
            </div>
        </div>
        """
        st.markdown(html_card, unsafe_allow_html=True)

        col_exec, _ = st.columns([1, 2])
        with col_exec:
            qty = st.number_input("Tamaño (USD)", value=1000.0, step=100.0)
            lev = st.slider("Apalancamiento", 1, 50, 10)
            btn_txt = f"🚀 EJECUTAR {calc_dir}" if signal != "NEUTRO" else f"⚠️ FORZAR {calc_dir}"
            
            if st.button(btn_txt, use_container_width=True):
                trade = {
                    "timestamp": str(datetime.now()), "symbol": symbol, "type": calc_dir,
                    "entry": current_price, "size": qty, "leverage": lev,
                    "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                    "status": "OPEN", "pnl": 0.0, "reason": f"Signal: {signal}", 
                    "candles_held": 0, "atr_entry": atr
                }
                db_mgr.add_trade(trade)
                
                # Enviar Telegram al ejecutar manualmente
                send_telegram_msg(f"⚠️ *ORDEN MANUAL ENVIADA*: {calc_dir} en {symbol} @ ${current_price:,.2f}")
                
                st.success(f"Orden {calc_dir} enviada.")

    with tab2:
        df_trades = db_mgr.load_trades()
        if not df_trades.empty:
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("Sin operaciones.")

if __name__ == "__main__":
    main()
