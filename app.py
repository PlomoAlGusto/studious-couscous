import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import os
import feedparser

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN ESTRUCTURAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quimera Pro v16.1 Fixed Deriv", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    /* Estilos Generales */
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    
    /* Estilos Trade Setup */
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    .label-mini { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    
    /* Estilos QUIMERA COPILOT (AI BOX) - FUENTE MEJORADA */
    .ai-box {
        background-color: #0e1117;
        border-left: 4px solid #44AAFF; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 15px; 
        /* Fuente técnica moderna (tipo VS Code / Terminal) */
        font-family: 'SF Mono', 'Consolas', 'Menlo', 'Monaco', 'Liberation Mono', 'Lucida Console', monospace;
        font-size: 13px;
        color: #e0e0e0;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .ai-title { 
        color: #44AAFF; 
        font-weight: bold; 
        font-size: 14px; 
        margin-bottom: 8px; 
        display: block; 
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
        /* Fuente Sans-Serif para el título para contraste */
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Relojes de Mercado */
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
    
    .status-dot-on { color: #00FF00; font-weight: bold; text-shadow: 0 0 5px #00FF00; }
    
    /* Badges */
    .badge-bull { background-color: #004400; color: #00FF00; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #00FF00; margin-right: 4px; }
    .badge-bear { background-color: #440000; color: #FF4444; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #FF4444; margin-right: 4px; }
    .badge-neutral { background-color: #333; color: #aaa; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid #555; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# GESTIÓN DE ARCHIVOS
CSV_FILE = 'paper_trades.csv'
COLUMNS_DB = ["id", "time", "symbol", "type", "entry", "size", "leverage", "sl", "tp1", "tp2", "tp3", "status", "pnl", "reason", "candles_held", "atr_entry"]
INITIAL_CAPITAL = 10000.0

if not os.path.exists(CSV_FILE):
    df_empty = pd.DataFrame(columns=COLUMNS_DB)
    df_empty.to_csv(CSV_FILE, index=False)

if 'last_alert' not in st.session_state: st.session_state.last_alert = "NEUTRO"

# -----------------------------------------------------------------------------
# 2. MOTOR DE DATOS (CCXT POWERED)
# -----------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(CSV_FILE): return pd.DataFrame(columns=COLUMNS_DB)
    try:
        df = pd.read_csv(CSV_FILE)
        if 'leverage' not in df.columns: df['leverage'] = 1.0
        return df
    except: return pd.DataFrame(columns=COLUMNS_DB)
        
def get_fear_and_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            item = data['data'][0]
            return int(item['value']), str(item['value_classification'])
    except Exception as e:
        print(f"Error Fear/Greed: {e}")
    return 50, "Neutral"

def get_current_balance():
    df = load_trades()
    if df.empty: return INITIAL_CAPITAL
    realized_pnl = df[df['status'] == 'CLOSED']['pnl'].sum()
    return INITIAL_CAPITAL + realized_pnl

def reset_account():
    df_empty = pd.DataFrame(columns=COLUMNS_DB)
    df_empty.to_csv(CSV_FILE, index=False)
    st.rerun()

def get_market_sessions():
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("### 🌍 SESIONES")
    for name, (start, end) in sessions.items():
        is_open = False
        if start < end: is_open = start <= hour < end
        else: is_open = hour >= start or hour < end
        status_icon = "🟢 ABIERTO" if is_open else "🔴 CERRADO"
        css_class = "clock-open" if is_open else "clock-closed"
        st.sidebar.markdown(f"<div class='market-clock {css_class}'><span>{name}</span><span>{status_icon}</span></div>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_deriv_data(symbol):
    # Normaliza para APIs: 'BTC/USDT' -> 'BTCUSDT'
    api_symbol = symbol.replace('/', '')  # 'BTCUSDT'
    base = symbol.split('/')[0]

    COINGLASS_API_KEY = st.secrets.get("COINGLASS_API_KEY")

    # 1. Prioridad: Binance Public API (Requests directos, sin CCXT)
    try:
        # Funding Rate
        url_fr = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={api_symbol}&limit=1"
        r_fr = requests.get(url_fr, timeout=5).json()
        fr = float(r_fr[0]['fundingRate']) * 100 if r_fr else None
        
        # Open Interest
        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={api_symbol}"
        r_oi = requests.get(url_oi, timeout=5).json()
        oi_val = float(r_oi.get('openInterest', 0)) if r_oi else None
        
        if fr is not None and oi_val is not None and oi_val > 0:
            print(f"Binance Success: FR={fr}, OI={oi_val}")  # Debug console
            return fr, oi_val, "Binance Public API"
    except Exception as e:
        print(f"Error Binance Requests: {e}")

    # 2. Fallback: Bybit Public API (Requests directos)
    try:
        # Funding Rate (último)
        url_fr_bybit = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={api_symbol}&limit=1"
        r_fr_bybit = requests.get(url_fr_bybit, timeout=5).json()
        if r_fr_bybit['retCode'] == 0 and r_fr_bybit['result']['list']:
            fr = float(r_fr_bybit['result']['list'][0]['fundingRate']) * 100
        
        # Open Interest
        url_oi_bybit = f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={api_symbol}"
        r_oi_bybit = requests.get(url_oi_bybit, timeout=5).json()
        if r_oi_bybit['retCode'] == 0:
            oi_val = float(r_oi_bybit['result']['list'][0]['openInterest'])
        
        if fr is not None and oi_val is not None and oi_val > 0:
            print(f"Bybit Success: FR={fr}, OI={oi_val}")
            return fr, oi_val, "Bybit Public API"
    except Exception as e:
        print(f"Error Bybit Requests: {e}")

    # 3. Fallback: CoinGlass (Si key válida)
    if COINGLASS_API_KEY:
        try:
            headers = {"coinglassSecret": COINGLASS_API_KEY}
            url_fr = f"https://open-api.coinglass.com/public/v2/funding?symbol={base}"
            r_fr = requests.get(url_fr, headers=headers, timeout=5).json()
            url_oi = f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base}"
            r_oi = requests.get(url_oi, headers=headers, timeout=5).json()
            
            avg_fr = 0.0
            total_oi = 0.0
            if r_fr.get('success') and r_fr.get('data'):
                fr_list = [ex['uMarginList'][0]['rate'] * 100 for ex in r_fr['data'] if ex['uMarginList']]
                avg_fr = np.mean(fr_list) if fr_list else None
            if r_oi.get('success') and r_oi.get('data'):
                total_oi = sum(ex.get('openInterestAmount', 0) * ex.get('price', 1) for ex in r_oi['data'])
            
            if avg_fr is not None and total_oi > 0:
                print(f"CoinGlass Success: FR={avg_fr}, OI={total_oi}")
                return avg_fr, total_oi, "CoinGlass"
        except Exception as e:
            print(f"Error CoinGlass: {e}")

    print("All Deriv Fallbacks Failed - OFFLINE")  # Debug
    return None, None, "OFFLINE"

@st.cache_data(ttl=30)
def get_mtf_trends_analysis(symbol):
    ex = ccxt.binance()
    ticker_fix = symbol.replace("/", "USDT") if "/" not in symbol else symbol
    trends = {}
    score = 0
    for tf in ['15m', '1h', '4h']:
        try:
            ohlcv = ex.fetch_ohlcv(ticker_fix, tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            ema_fast = ta.ema(df['c'], length=20).iloc[-1]
            ema_slow = ta.ema(df['c'], length=50).iloc[-1]
            if ema_fast > ema_slow: trends[tf] = "BULL"; score += 1
            else: trends[tf] = "BEAR"; score -= 1
        except: trends[tf] = "NEUTRO"
    return trends, score

# Calculamos derivados ANTES de sidebar para que debug funcione
symbol = "BTC/USDT"  # Default, pero se sobreescribe en sidebar
fr, open_interest, data_src = get_deriv_data(symbol)
fr_debug, oi_debug, src_debug = fr, open_interest, data_src

# -----------------------------------------------------------------------------
# 3. INTERFAZ SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🦁 QUIMERA v16.1")
    st.markdown(f"<div style='font-size:12px; margin-bottom:10px;'><span class='status-dot-on'>●</span> SYSTEM ONLINE</div>", unsafe_allow_html=True)
    get_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)
    
    with st.expander("🛡️ FILTROS & CORE", expanded=True):
        use_ema = st.checkbox("Tendencia Base (EMAs)", True)
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP (Institucional)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)
    
    with st.expander("🌊 MOMENTO Y VOLUMEN"):
        use_rsi = st.checkbox("RSI & Stoch", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        use_tsi = st.checkbox("TSI (True Strength)", True)
        
    with st.expander("💰 GESTIÓN DE RIESGO"):
        current_balance = get_current_balance()
        st.metric("Balance Disponible", f"${current_balance:,.2f}", delta=f"{current_balance-INITIAL_CAPITAL:.2f}")
        risk_per_trade = st.slider("Riesgo por Trade (%)", 0.5, 5.0, 1.0)
        
    with st.expander("⚙️ SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
    
    # NUEVO: Debug para Deriv Data
    with st.expander("🔍 DEBUG DERIVADOS", expanded=False):
        if fr_debug is not None:
            st.write(f"**FR Raw:** {fr_debug:.4f}%")
            if oi_debug is not None:
                st.write(f"**OI Raw:** ${oi_debug:,.0f}")
            st.write(f"**Source:** {src_debug}")
        else:
            st.info("Datos OFFLINE - Agrega COINGLASS_API_KEY en Secrets para fallback.")
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)
    if st.button("🔥 RESETEAR CUENTA"): reset_account()

# Recalculamos derivados si symbol cambia
fr, open_interest, data_src = get_deriv_data(symbol)

# Resto del código igual...
# (Copia el resto desde tu versión original, desde # 4. CAPA DE DATOS hasta el final)

# Nota: Para no repetir todo, solo pega el resto aquí en tu editor.
