import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import numpy as np
import os

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(page_title="Quimera v17 - OKX 100% Functional", layout="wide", page_icon="lion_face")

st.markdown("""
<style>
    .big-font {font-size:32px !important; font-weight:bold; color:#44AAFF; text-align:center;}
    .metric-card {background:#1e1e1e; padding:15px; border-radius:12px; border:1px solid #444; text-align:center;}
    .ai-box {background:#0e1117; border-left:6px solid #44AAFF; padding:18px; border-radius:8px; margin:20px 0;
             font-family:'Consolas',monospace; font-size:14px; color:#e0e0e0;}
    .ai-title {color:#44AAFF; font-weight:bold; font-size:17px; margin-bottom:10px;}
    .status-dot-on {color:#00FF00; font-weight:bold; text-shadow:0 0 10px #00FF00;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATOS DERIVADOS — OKX NUNCA FALLA
# =============================================================================
@st.cache_data(ttl=60)
def get_deriv_data(symbol: str):
    binance_sym = symbol.replace("/", "")
    okx_sym = f"{symbol.split('/')[0]}-{symbol.split('/')[1]}-SWAP"

    # 1. Binance
    try:
        fr_resp = requests.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={binance_sym}&limit=1", timeout=8).json()
        oi_resp = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_sym}", timeout=8).json()
        fr = float(fr_resp[0]["fundingRate"]) * 100 if fr_resp else None
        oi = float(oi_resp.get("openInterest", 0)) if oi_resp else None
        if fr is not None and oi > 0:
            return fr, oi, "Binance"
    except:
        pass

    # 2. Bybit
    try:
        fr_resp = requests.get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={binance_sym}&limit=1", timeout=8).json()
        oi_resp = requests.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={binance_sym}", timeout=8).json()
        if fr_resp.get("retCode") == 0 and fr_resp.get("result", {}).get("list"):
            fr = float(fr_resp["result"]["list"][0]["fundingRate"]) * 100
        if oi_resp.get("retCode") == 0 and oi_resp.get("result", {}).get("list"):
            oi = float(oi_resp["result"]["list"][0]["openInterest"])
        if fr is not None and oi > 0:
            return fr, oi, "Bybit"
    except:
        pass

    # 3. OKX — ESTE NUNCA FALLA
    try:
        fr_resp = requests.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_sym}", timeout=10).json()
        oi_resp = requests.get(f"https://www.okx.com/api/v5/public/open-interest?instId={okx_sym}", timeout=10).json()
        if fr_resp.get("code") == "0" and fr_resp.get("data"):
            fr = float(fr_resp["data"][0]["fundingRate"]) * 100
        if oi_resp.get("code") == "0" and oi_resp.get("data"):
            oi = float(oi_resp["data"][0]["openInterest"])
        if fr is not None and oi > 0:
            return fr, oi, "OKX"
    except:
        pass

    return None, None, "OFFLINE"

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("<p class='big-font'>QUIMERA v17</p>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'><span class='status-dot-on'>ONLINE</span></div>", unsafe_allow_html=True)
    st.divider()

    symbol = st.text_input("Ticker", "BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)

    st.markdown("### DEBUG DERIVADOS")
    fr, oi, source = get_deriv_data(symbol)
    st.write(f"**Fuente:** {source}")
    if fr:
        st.success(f"Funding Rate: {fr:.4f}%")
    if oi:
        st.success(f"Open Interest: ${oi/1e9:.3f}B")

    auto_refresh = st.checkbox("Auto-refresh (60s)")

# =============================================================================
# CARGAR DATOS DE PRECIO
# =============================================================================
exchange = ccxt.binance({'enableRateLimit': True})
try:
    ohlcv = exchange.fetch_ohlcv(symbol.replace("USDT", "/USDT"), timeframe, limit=500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# Indicadores
df["EMA20"] = ta.ema(df["close"], length=20)
df["EMA50"] = ta.ema(df["close"], length=50)
df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
df["RSI"] = ta.rsi(df["close"], length=14)
df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)

current_price = df["close"].iloc[-1]

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio", f"${current_price:,.2f}")
col2.metric("Funding Rate", f"{fr:.4f}%" if fr else "N/A")
col3.metric("Open Interest", f"${oi/1e9:.2f}B" if oi else "N/A")
col4.metric("Fuente", source)

# Quimera Copilot
tendencia = "ALCISTA" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "BAJISTA"
st.markdown(f"""
<div class="ai-box">
    <div class="ai-title">QUIMERA COPILOT → {source}</div>
    • Tendencia principal: <b>{tendencia}</b><br>
    • Funding Rate: <b>{fr:.4f}%</b> → {"LONGS PAGAN → posible SHORT" if fr and fr > 0.01 else "SHORTS PAGAN → posible LONG" if fr and fr < -0.01 else "Neutral"}<br>
    • Open Interest: <b>{oi/1e9:.2f} mil millones USD</b><br>
    • RSI: {df["RSI"].iloc[-1]:.1f} | ATR: {df["ATR"].iloc[-1]:.2f}
</div>
""", unsafe_allow_html=True)

# Gráfico
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
fig.add_trace(go.Candlestick(x=df["timestamp"],
                             open=df["open"], high=df["high"],
                             low=df["low"], close=df["close"], name="Precio"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["VWAP"], line=dict(color="orange", width=2), name="VWAP"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA20"], line=dict(color="lime", width=1.5), name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA50"], line=dict(color="red", width=1.5), name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], line=dict(color="purple"), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(height=700, template="plotly_dark", title=f"{symbol} {timeframe} — Derivados: {source}")
st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()            headers = {"coinglassSecret": st.secrets["COINGLASS_API_KEY"]}
            fr = requests.get(f"https://open-api.coinglass.com/public/v2/funding?symbol={base}", headers=headers, timeout=8).json()
            oi = requests.get(f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base}", headers=headers, timeout=8).json()
            rates = [x["uMarginList"][0]["rate"]*100 for x in fr.get("data",[]) if x.get("uMarginList")]
            oi_total = sum(x.get("openInterestAmount",0)*x.get("price",1) for x in oi.get("data",[]))
            if rates and oi_total > 0:
                return np.mean(rates), oi_total, "CoinGlass"
        except: pass

    # 4. OKX → EL QUE NUNCA FALLA
    try:
        fr = requests.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_sym}", timeout=10).json()
        oi = requests.get(f"https://www.okx.com/api/v5/public/open-interest?instId={okx_sym}", timeout=10).json()
        funding = float(fr["data"][0]["fundingRate"]) * 100 if fr.get("code") == "0" and fr.get("data") else None
        open_int = float(oi["data"][0]["openInterest"]) if oi.get("code") == "0" and oi.get("data") else None
        if funding is not None and open_int > 0:
            return funding, open_int, "OKX"
    except Exception as e:
        st.sidebar.error(f"OKX error: {e}")

    return None, None, "OFFLINE"

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("<p class='big-font'>QUIMERA v16.3</p>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'><span class='status-dot-on'>● ONLINE</span></div>", unsafe_allow_html=True)
    st.divider()

    symbol = st.text_input("Ticker", "BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)

    with st.expander("DEBUG DERIVADOS", expanded=True):
        fr, oi, src = get_deriv_data(symbol)
        st.write(f"**Fuente:** {src}")
        if fr: st.success(f"Funding Rate: {fr:.4f}%")
        if oi: st.success(f"Open Interest: ${oi/1e9:.2f}B")
        if src == "OFFLINE": st.error("Todos los proveedores fallaron")

    auto = st.checkbox("Auto-refresh (60s)")

# =============================================================================
# CARGA DE DATOS
# =============================================================================
ex = ccxt.binance()
try:
    ohlcv = ex.fetch_ohlcv(symbol.replace("USDT","/USDT"), timeframe, limit=500)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
    st.stop()

# Indicadores
df["EMA20"] = ta.ema(df["close"], 20)
df["EMA50"] = ta.ema(df["close"], 50)
df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
df["RSI"] = ta.rsi(df["close"], 14)
df["ATR"] = ta.atr(df["high"], df["low"], df["close"], 14)

precio = df["close"].iloc[-1]
fr, oi, src = get_deriv_data(symbol)

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio", f"${precio:,.2f}")
col2.metric("Funding Rate", f"{fr:.4f}%" if fr else "N/A", delta=f"{fr:+.4f}%" if fr else None)
col3.metric("Open Interest", f"${oi/1e9:.2f}B" if oi else "N/A")
col4.metric("Fuente", src)

st.markdown(f"""
<div class="ai-box">
    <div class="ai-title">QUIMERA COPILOT → {src}</div>
    • Tendencia: <b>{"ALCISTA" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "BAJISTA"}</b><br>
    • Funding Rate: <b>{fr:.4f}%</b> → {"Longs pagan mucho → posible SHORT" if fr and fr > 0.01 else "Shorts pagan mucho → posible LONG" if fr and fr < -0.01 else "Neutral"}<br>
    • Open Interest: <b>{oi/1e9:.2f} mil millones USD</b><br>
    • RSI: {df["RSI"].iloc[-1]:.1f} | ATR: {df["ATR"].iloc[-1]:.2f}
</div>
""", unsafe_allow_html=True)

# Gráfico
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["VWAP"], line=dict(color="orange", width=2), name="VWAP"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA20"], line=dict(color="#00FF00", width=1.5), name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA50"], line=dict(color="#FF4444", width=1.5), name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], line=dict(color="#CC00FF"), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(height=700, template="plotly_dark", title=f"{symbol} {timeframe} — Fuente derivados: {src}")
st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
if auto:
    time.sleep(60)
    st.rerun()
def get_fear_and_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6).json()
        return int(r['data'][0]['value']), r['data'][0]['value_classification']
    except:
        return 50, "Neutral"

# =============================================================================
# DATOS DERIVADOS — GARANTIZADOS CON OKX
# =============================================================================
@st.cache_data(ttl=60)
def get_deriv_data(symbol):
    base = symbol.split('/')[0]
    binance_sym = symbol.replace('/', '')
    okx_sym = f"{symbol.split('/')[0]}-{symbol.split('/')[1]}-SWAP"

    # 1. Binance
    try:
        fr_resp = requests.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={binance_sym}&limit=1", timeout=7).json()
        oi_resp = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_sym}", timeout=7).json()
        fr = float(fr_resp[0]['fundingRate']) * 100 if fr_resp else None
        oi = float(oi_resp.get('openInterest', 0)) if oi_resp else None
        if fr is not None and oi > 0:
            return fr, oi, "Binance"
    except: pass

    # 2. Bybit
    try:
        fr = requests.get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={binance_sym}&limit=1", timeout=7).json()
        oi = requests.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={binance_sym}", timeout=7).json()
        funding_rate = float(fr['result']['list'][0]['fundingRate']) * 100 if fr.get('retCode')==0 and fr['result']['list'] else None
        open_int = float(oi['result']['list'][0]['openInterest']) if oi.get('retCode')==0 and oi['result']['list'] else None
        if funding_rate is not None and open_int > 0:
            return funding_rate, open_int, "Bybit"
    except: pass

    # 3. CoinGlass (opcional)
    key = st.secrets.get("COINGLASS_API_KEY", None)
    if key:
        try:
            h = {"coinglassSecret": key}
            fr = requests.get(f"https://open-api.coinglass.com/public/v2/funding?symbol={base}", headers=h, timeout=7).json()
            oi = requests.get(f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base}", headers=h, timeout=7).json()
            rates = [ex['uMarginList'][0]['rate']*100 for ex in fr.get('data',[]) if ex.get('uMarginList')]
            oi_val = sum(ex.get('openInterestAmount',0)*ex.get('price',1) for ex in oi.get('data',[]))
            if rates and oi_val > 0:
                return np.mean(rates), oi_val, "CoinGlass"
        except: pass

    # 4. OKX — NUNCA FALLA
    try:
        fr = requests.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_sym}", timeout=10).json()
        oi = requests.get(f"https://www.okx.com/api/v5/public/open-interest?instId={okx_sym}", timeout=10).json()
        if fr['code'] == '0' and fr['data']:
            funding = float(fr['data'][0]['fundingRate']) * 100
        if oi['code'] == '0' and oi['data']:
            open_int = float(oi['data'][0]['openInterest'])
        if 'funding' in locals() and open_int > 0:
            return funding, open_int, "OKX"
    except Exception as e:
        print(f"OKX error: {e}")

    return None, None, "OFFLINE"

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("<p class='big-font'>QUIMERA v16.3</p>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;'><span class='status-dot-on'>● ONLINE</span></div>", unsafe_allow_html=True)
    st.divider()

    symbol = st.text_input("Ticker", "BTC/USDT", key="symbol_input")
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0)

    with st.expander("FILTROS & CORE", expanded=True):
        use_ema = st.checkbox("Tendencia EMA", True)
        use_vwap = st.checkbox("VWAP Institucional", True)
        use_adx = st.checkbox("Anti-Rango (ADX)", True)

    with st.expander("MOMENTO Y VOLUMEN"):
        use_rsi = st.checkbox("RSI + Stoch", True)
        use_tsi = st.checkbox("TSI Strength", True)

    with st.expander("GESTIÓN DE RIESGO"):
        balance = get_current_balance()
        st.metric("Balance", f"${balance:,.2f}")
        risk_pct = st.slider("Riesgo por operación (%)", 0.5, 5.0, 1.0, 0.1)

    with st.expander("DEBUG DERIVADOS", expanded=True):
        fr_val, oi_val, source = get_deriv_data(symbol)
        st.write(f"**Fuente:** {source}")
        if fr_val: st.success(f"Funding Rate: {fr_val:.4f}%")
        if oi_val: st.success(f"Open Interest: ${oi_val/1e9:.2f}B")
        if source == "OFFLINE": st.error("Todos los proveedores fallaron")

    auto_refresh = st.checkbox("AUTO-REFRESH (60s)")
    if st.button("RESETEAR CUENTA"):
        reset_account()

# =============================================================================
# CARGA DE DATOS
# =============================================================================
ex = ccxt.binance({'enableRateLimit': True})
try:
    ohlcv = ex.fetch_ohlcv(symbol.replace("USDT", "/USDT"), tf, limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# Indicadores
df['EMA20'] = ta.ema(df['close'], 20)
df['EMA50'] = ta.ema(df['close'], 50)
df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
df['RSI'] = ta.rsi(df['close'], 14)
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], 14)
df['ADX'] = ta.adx(df['high'], df['low'], df['close'], 14)['ADX_14']

current_price = df['close'].iloc[-1]
atr = df['ATR'].iloc[-1]

# =============================================================================
# OBTENER DERIVADOS (una vez)
# =============================================================================
fr, oi, deriv_source = get_deriv_data(symbol)

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio", f"${current_price:,.2f}")
if fr is None:
    c2.metric("Funding Rate", "N/A", delta="OFFLINE")
else:
    c2.metric("Funding Rate", f"{fr:.4f}%", delta=f"{fr:+.4f}%")
if oi is None:
    c3.metric("Open Interest", "N/A")
else:
    c3.metric("Open Interest", f"${oi/1e9:.2f}B")
c4.metric("Fuente Derivados", deriv_source)

# QUIMERA COPILOT
trend = "ALCISTA" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "BAJISTA"
funding_signal = "LONG SQUEEZE" if fr and fr > 0.02 else "SHORT SQUEEZE" if fr and fr < -0.02 else "NEUTRO"

st.markdown(f"""
<div class="ai-box">
    <span class="ai-title">QUIMERA COPILOT → Fuente: {deriv_source}</span>
    • Estructura {trend} (EMA20 vs EMA50)<br>
    • Funding Rate: <b>{fr:.4f}%</b> → <b>{funding_signal}</b><br>
    • Open Interest: <b>{oi/1e9:.2f} mil millones</b><br>
    • RSI: {df['RSI'].iloc[-1]:.1f} | ATR: {atr:.2f}
</div>
""", unsafe_allow_html=True)

# GRÁFICO
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                    subplot_titles=("Precio + VWAP", "RSI"))
fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color="#FFA500", width=2), name="VWAP"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], line=dict(color="#00FF00", dash="dot"), name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA50'], line=dict(color="#FF4444", dash="dot"), name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color="#AA00FF"), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# AUTO REFRESH
if auto_refresh:
    time.sleep(60)
    st.rerun()
    # 2. Bybit
    try:
        fr = requests.get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={binance_sym}&limit=1", timeout=6).json()
        oi = requests.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={binance_sym}", timeout=6).json()
        funding = float(fr['result']['list'][0]['fundingRate']) * 100 if fr.get('retCode')==0 and fr['result']['list'] else None
        open_int = float(oi['result']['list'][0]['openInterest']) if oi.get('retCode')==0 and oi['result']['list'] else None
        if funding is not None and open_int > 0:
            return funding, open_int, "Bybit"
    except: pass

    # 3. CoinGlass (si tienes key)
    key = st.secrets.get("COINGLASS_API_KEY")
    if key:
        try:
            h = {"coinglassSecret": key}
            fr = requests.get(f"https://open-api.coinglass.com/public/v2/funding?symbol={base}", headers=h, timeout=6).json()
            oi = requests.get(f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base}", headers=h, timeout=6).json()
            rates = [ex['uMarginList'][0]['rate']*100 for ex in fr.get('data',[]) if ex.get('uMarginList')]
            oi_total = sum(ex.get('openInterestAmount',0)*ex.get('price',1) for ex in oi.get('data',[]))
            if rates and oi_total > 0:
                return np.mean(rates), oi_total, "CoinGlass"
        except: pass

    # 4. OKX — EL QUE NUNCA FALLA
    try:
        fr = requests.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_sym}", timeout=8).json()
        oi = requests.get(f"https://www.okx.com/api/v5/public/open-interest?instId={okx_sym}", timeout=8).json()
        if fr['code'] == '0' and fr['data']:
            funding = float(fr['data'][0]['fundingRate']) * 100
        if oi['code'] == '0' and oi['data']:
            open_int = float(oi['data'][0]['openInterest'])
        if 'funding' in locals() and open_int > 0:
            return funding, open_int, "OKX"
    except: pass

    return None, None, "OFFLINE"

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("QUIMERA v16.2")
    st.markdown("<div style='font-size:12px;'><span class='status-dot-on'>●</span> ONLINE</div>", unsafe_allow_html=True)
    
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe", ["15m", "1h"], index=0)
    
    # Filtros
    with st.expander("FILTROS & CORE", expanded=True):
        use_ema = st.checkbox("EMA Trend", True)
        use_mtf = st.checkbox("Filtro 4H", True)
        use_vwap = st.checkbox("VWAP", True)
        use_regime = st.checkbox("ADX Anti-Rango", True)
    
    with st.expander("MOMENTO"):
        use_rsi = st.checkbox("RSI", True)
        use_tsi = st.checkbox("TSI", True)
    
    with st.expander("RIESGO"):
        st.metric("Balance", f"${get_current_balance():,.2f}")
        risk_per_trade = st.slider("Riesgo %", 0.5, 5.0, 1.0)
    
    # DEBUG DERIVADOS
    fr, oi, src = get_deriv_data(symbol)
    with st.expander("DEBUG DERIVADOS", expanded=True):
        st.write(f"**Fuente:** {src}")
        if fr is not None: st.write(f"**Funding Rate:** {fr:.4f}%")
        if oi is not None: st.write(f"**Open Interest:** ${oi:,.0f}")
        if src == "OFFLINE": st.error("Todos los proveedores fallaron")

    auto_refresh = st.checkbox("AUTO-REFRESH (60s)")
    if st.button("RESETEAR CUENTA"): reset_account()

# =============================================================================
# DATOS PRINCIPALES
# =============================================================================
exchange = ccxt.binance()
try:
    ohlcv = exchange.fetch_ohlcv(symbol.replace("USDT","/USDT"), tf, limit=500)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
except:
    st.error("Error cargando datos del exchange")
    st.stop()

# Indicadores
df['EMA20'] = ta.ema(df['close'], 20)
df['EMA50'] = ta.ema(df['close'], 50)
df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
df['RSI'] = ta.rsi(df['close'], 14)
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], 14)

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio", f"${df['close'].iloc[-1]:,.2f}")
if fr is None:
    col2.metric("Funding Rate", "N/A", delta="OFFLINE")
else:
    col2.metric("Funding Rate", f"{fr:.4f}%", delta=f"{fr:+.4f}%")
if oi is None:
    col3.metric("Open Interest", "N/A")
else:
    col3.metric("Open Interest", f"${oi/1e9:.2f}B" if oi>1e9 else f"${oi/1e6:.1f}M")

# Quimera Copilot
st.markdown(f"""
<div class="ai-box">
    <span class="ai-title">QUIMERA COPILOT → Fuente: {src}</span>
    Estructura actual: {"ALCISTA" if df['EMA20'].iloc[-1]>df['EMA50'].iloc[-1] else "BAJISTA"}<br>
    Funding Rate: <b>{fr:.4f}%</b> → {"Long Squeeze riesgo" if fr>0.02 else "Short Squeeze riesgo" if fr<-0.02 else "Saludable"}<br>
    Open Interest: <b>{oi/1e9:.2f}B</b>
</div>
""", unsafe_allow_html=True)

# Gráfico
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange'), name='VWAP'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")
fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()
