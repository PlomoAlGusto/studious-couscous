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
st.set_page_config(page_title="Quimera Pro v17.0 Classic", layout="wide", page_icon="🦁")

st.markdown("""
<style>
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #444;}
    
    .trade-setup {
        background-color: #151515; padding: 20px; border-radius: 15px; border: 1px solid #444;
        margin-top: 10px; margin-bottom: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tp-green { color: #00FF00; font-weight: bold; font-size: 18px; }
    .sl-red { color: #FF4444; font-weight: bold; font-size: 18px; }
    .entry-blue { color: #44AAFF; font-weight: bold; font-size: 18px; }
    
    .header-confirmed-long { color: #00FF00; font-size: 20px; font-weight: 900; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .header-confirmed-short { color: #FF4444; font-size: 20px; font-weight: 900; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .header-potential { color: #FFFF00; font-size: 18px; font-weight: bold; border-bottom: 1px dashed #555; padding-bottom: 10px; }
    
    .ai-box {
        background-color: #223344; border-left: 5px solid #44AAFF; padding: 15px; border-radius: 5px; margin-bottom: 15px; font-family: monospace;
    }
    
    .news-box {
        background-color: #111; border: 1px solid #333; padding: 10px; border-radius: 10px; margin-bottom: 15px; height: 200px; overflow-y: auto;
    }
    .news-item { padding: 5px 0; border-bottom: 1px solid #222; font-size: 12px; }
    .news-link { text-decoration: none; color: #ddd; }
    .news-link:hover { color: #44AAFF; }
    .news-time { color: #666; font-size: 10px; margin-right: 5px; font-weight: bold;}
    
    .market-clock { font-size: 12px; padding: 5px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between;}
    .clock-open { background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00FF00; }
    .clock-closed { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #555; color: #888; }
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
# 2. CONFIGURACIÓN (SIDEBAR MODULAR)
# -----------------------------------------------------------------------------
def get_market_sessions():
    now = datetime.now(timezone.utc)
    hour = now.hour
    sessions = {"🇬🇧 LONDRES": (8, 16), "🇺🇸 NEW YORK": (13, 21), "🇯🇵 TOKYO": (0, 9), "🇦🇺 SYDNEY": (22, 7)}
    st.sidebar.markdown("### 🌍 SESIONES")
    for name, (start, end) in sessions.items():
        is_open = start <= hour < end
        status_icon = "🟢" if is_open else "🔴"
        css_class = "clock-open" if is_open else "clock-closed"
        st.sidebar.markdown(f"<div class='market-clock {css_class}'><span>{name}</span><span>{status_icon}</span></div>", unsafe_allow_html=True)

with st.sidebar:
    st.title("🦁 QUIMERA v17.0")
    st.caption("Classic Edition ✨")
    get_market_sessions()
    st.divider()
    symbol = st.text_input("Ticker", "BTC/USDT")
    tf = st.selectbox("Timeframe Principal", ["15m", "1h"], index=0)

    with st.expander("🛡️ FILTROS DE ENTRADA", expanded=True):
        use_ema = st.checkbox("Tendencia Base (EMAs)", True)
        use_mtf = st.checkbox("Filtro Macro (4H Trend)", True)
        use_vwap = st.checkbox("Filtro VWAP (Institucional)", True)
        use_ichi = st.checkbox("Filtro Nube Ichimoku", False)
        use_regime = st.checkbox("Filtro Anti-Rango (ADX)", True)

    with st.expander("🌊 MOMENTO Y VOLUMEN"):
        use_rsi = st.checkbox("RSI & Stoch", True)
        use_obi = st.checkbox("Order Book Imbalance", True)
        
    with st.expander("💰 GESTIÓN DE RIESGO"):
        current_balance = INITIAL_CAPITAL
        try:
            df_t = pd.read_csv(CSV_FILE)
            if not df_t.empty:
                current_balance += df_t[df_t['status']=='CLOSED']['pnl'].sum()
        except: pass
        st.metric("Balance", f"${current_balance:,.2f}")
        risk_per_trade = st.slider("Riesgo (%)", 0.5, 5.0, 1.0)
        
    with st.expander("⚙️ SALIDAS"):
        use_trailing = st.checkbox("Trailing Stop", True)
        use_breakeven = st.checkbox("Breakeven (+1.5%)", True)
        use_time_stop = st.checkbox("Time Stop (12 Velas)", True)
        
    auto_refresh = st.checkbox("🔄 AUTO-SCAN (60s)", False)

# -----------------------------------------------------------------------------
# 3. CAPA DE DATOS
# -----------------------------------------------------------------------------
def init_exchange():
    try: return ccxt.binance(), "Binance"
    except: return ccxt.kraken(), "Kraken"

exchange, source_name = init_exchange()

@st.cache_data(ttl=3600) 
def get_fear_and_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/")
        data = r.json()['data'][0]
        return int(data['value']), data['value_classification']
    except: return 50, "Neutral"

@st.cache_data(ttl=300)
def get_crypto_news():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        return [{"title": e.title, "link": e.link, "published": e.published_parsed} for e in feed.entries[:5]]
    except: return []

@st.cache_data(ttl=120)
def get_deriv_data(symbol):
    # Intentamos dYdX por ser API pública sin bloqueo regional
    base = symbol.split('/')[0]
    try:
        url = f"https://api.dydx.exchange/v3/markets/{base}-USD"
        r = requests.get(url, timeout=2).json()
        market = r['market']
        fr = float(market['nextFundingRate']) * 100
        oi = float(market['openInterest'])
        return fr, oi
    except:
        return 0.0, 0.0

@st.cache_data(ttl=15)
def get_mtf_data(symbol, tf_lower):
    if not exchange: return None, 0, None
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf_lower, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except: return None, 0, None

    trend_4h = "NEUTRO"
    try:
        ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=50)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['t','o','h','l','c','v'])
        ema_50 = ta.ema(df_4h['c'], length=50).iloc[-1]
        trend_4h = "BULLISH" if df_4h['c'].iloc[-1] > ema_50 else "BEARISH"
    except: pass

    obi = 0
    try:
        book = exchange.fetch_order_book(symbol, limit=20)
        bids = sum([x[1] for x in book['bids']])
        asks = sum([x[1] for x in book['asks']])
        obi = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
    except: pass
    
    return df, obi, trend_4h

# -----------------------------------------------------------------------------
# 4. LÓGICA
# -----------------------------------------------------------------------------
def calculate_indicators(df):
    if df is None: return None
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    try:
        vp = ((df['high'] + df['low'] + df['close'])/3) * df['volume']
        df['VWAP'] = vp.cumsum() / df['volume'].cumsum()
    except: df['VWAP'] = df['EMA_50']
    
    ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
    df = pd.concat([df, ichi], axis=1)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df = pd.concat([df, adx], axis=1)
    df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    high_w = df['high'].rolling(20).max()
    low_w = df['low'].rolling(20).min()
    close_w = df['close']
    df['PIVOT'] = (high_w + low_w + close_w) / 3
    df['R1'] = (2 * df['PIVOT']) - low_w
    df['S1'] = (2 * df['PIVOT']) - high_w
    
    return df.fillna(method='bfill').fillna(method='ffill')

def generate_ai_analysis(row, trend_4h, obi, signal, prob, fng_val, fr, oi):
    analysis = []
    if trend_4h == "BULLISH": analysis.append("Estructura Macro (4H): ALCISTA.")
    elif trend_4h == "BEARISH": analysis.append("Estructura Macro (4H): BAJISTA.")
    
    mfi = row['MFI']
    if mfi > 60: analysis.append("⛽ Flujo dinero POSITIVO (MFI Alto).")
    elif mfi < 40: analysis.append("🪫 Flujo dinero NEGATIVO (MFI Bajo).")
    
    if abs(fr) > 0.01: analysis.append(f"⚠️ Funding Rate Alto ({fr:.4f}%).")
    if oi > 0: analysis.append(f"📊 Interés Abierto: ${oi/1000000:.1f}M.")

    if signal != "NEUTRO":
        direction = "SUBIDA" if signal == "LONG" else "BAJADA"
        analysis.append(f"🎯 CONCLUSIÓN: Probabilidad {prob:.1f}% de {direction}.")
    else:
        analysis.append("⏳ CONCLUSIÓN: Mercado indeciso. Esperar ruptura.")
        
    return " | ".join(analysis)

def run_strategy(df, obi, trend_4h, filters):
    row = df.iloc[-1]
    score = 0
    max_score = 0
    
    if filters['use_mtf']:
        max_score += 2
        if trend_4h == "BULLISH": score += 2
        elif trend_4h == "BEARISH": score -= 2
        else: max_score -= 2 

    if filters['use_ema']: 
        max_score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1
        else: score -= 1

    if filters['use_vwap']:
        max_score += 1
        if row['close'] > row['VWAP']: score += 1
        else: score -= 1
        
    if filters['use_obi']:
        max_score += 1
        if obi > 0.05: score += 1
        elif obi < -0.05: score -= 1
    
    threshold = max_score * 0.4
    signal = "NEUTRO"
    if score > threshold: signal = "LONG"
    elif score < -threshold: signal = "SHORT"
    
    if filters['use_rsi'] and (row['RSI'] > 70 and signal == "LONG"): signal = "NEUTRO"
    if filters['use_rsi'] and (row['RSI'] < 30 and signal == "SHORT"): signal = "NEUTRO"
    if filters['use_regime'] and row['ADX_14'] < 20: signal = "NEUTRO"
    
    if filters['use_mtf'] and signal == "LONG" and trend_4h == "BEARISH": signal = "NEUTRO"
    if filters['use_mtf'] and signal == "SHORT" and trend_4h == "BULLISH": signal = "NEUTRO"

    prob = 50.0
    if max_score > 0: prob = 50 + ((abs(score)/max_score)*45)
    
    thermometer_score = 0
    if max_score > 0:
        thermometer_score = (score / max_score) * 100 
    
    return signal, row['ATR'], prob, thermo_score

# -----------------------------------------------------------------------------
# 5. GESTIÓN TRADES
# -----------------------------------------------------------------------------
def save_trades(df): df.to_csv(CSV_FILE, index=False)

def execute_trade(type, entry, sl, tp1, tp2, tp3, size, atr, leverage):
    df = load_trades()
    new = {"id": int(time.time()), "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": symbol, "type": type, "entry": entry, "size": size, "leverage": leverage, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "status": "OPEN", "pnl": 0.0, "reason": "Entry", "candles_held": 0, "atr_entry": atr}
    df = pd.concat([pd.DataFrame([new]), df], ignore_index=True)
    save_trades(df)
    return new

def manage_open_positions(current_price):
    df = load_trades()
    if df.empty: return
    open_idx = df.index[df['status'] == "OPEN"].tolist()
    updated = False
    for idx in open_idx:
        row = df.loc[idx]
        close_reason, pnl = "", 0
        if row['type'] == "LONG":
            if use_trailing:
                new_sl = current_price - (row['atr_entry'] * 1.5)
                if new_sl > row['sl']: df.at[idx, 'sl'] = new_sl
            if use_breakeven and current_price > (row['entry'] * 1.015) and row['sl'] < row['entry']: df.at[idx, 'sl'] = row['entry']
            if current_price >= row['tp3']: close_reason="TP3 🚀"; pnl=(row['tp3']-row['entry'])*row['size']
            elif current_price <= row['sl']: close_reason="SL 🛑"; pnl=(row['sl']-row['entry'])*row['size']
        else:
            if use_trailing:
                new_sl = current_price + (row['atr_entry'] * 1.5)
                if new_sl < row['sl']: df.at[idx, 'sl'] = new_sl
            if use_breakeven and current_price < (row['entry'] * 0.985) and row['sl'] > row['entry']: df.at[idx, 'sl'] = row['entry']
            if current_price <= row['tp3']: close_reason="TP3 🚀"; pnl=(row['entry']-row['tp3'])*row['size']
            elif current_price >= row['sl']: close_reason="SL 🛑"; pnl=(row['entry']-row['sl'])*row['size']

        if close_reason:
            df.at[idx, 'status'] = "CLOSED"; df.at[idx, 'pnl'] = pnl; df.at[idx, 'reason'] = close_reason
            updated = True
    if updated: save_trades(df)

def render_analytics(df_trades):
    if df_trades.empty: return
    closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
    if closed.empty: return
    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    closed['equity'] = INITIAL_CAPITAL + closed['cumulative_pnl']
    fig = px.area(closed, x='time', y='equity', title="Curva de Capital")
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------
df, obi, trend_4h = get_mtf_data(symbol, tf)

if df is not None:
    df = calculate_indicators(df)
    
    filters = {
        'use_mtf': use_mtf, 'use_ema': use_ema, 'use_vwap': use_vwap,
        'use_ichi': use_ichi, 'use_regime': use_regime, 'use_rsi': use_rsi,
        'use_obi': use_obi
    }
    
    signal, atr, prob, thermo_score = run_strategy(df, obi, trend_4h, filters)
    current_price = df['close'].iloc[-1]
    
    # Datos Extra
    fng_val, fng_label = get_fear_and_greed()
    news = get_crypto_news()
    fr, oi = get_deriv_data(symbol)
    
    ai_narrative = generate_ai_analysis(df.iloc[-1], trend_4h, obi, signal, prob, fng_val, fr, oi)
    
    setup = None
    calc_dir = signal 
    setup_type = "CONFIRMED" if signal != "NEUTRO" else "POTENTIAL"
    
    if signal == "NEUTRO":
        if trend_4h == "BULLISH": calc_dir = "LONG"
        elif trend_4h == "BEARISH": calc_dir = "SHORT"
        else: calc_dir = None

    qty = 0
    leverage = 1.0
    current_balance = get_current_balance()

    if calc_dir:
        sl_dist = atr * 1.5
        risk = sl_dist
        risk_amount = current_balance * (risk_per_trade / 100)
        qty = risk_amount / risk if risk > 0 else 0
        leverage = max(1.0, (qty * current_price) / current_balance)
        
        if calc_dir == "LONG":
            sl, tp1, tp2, tp3 = current_price-sl_dist, current_price+risk, current_price+(risk*2), current_price+(risk*3.5)
            emoji = "⬆️ LONG"
        else:
            sl, tp1, tp2, tp3 = current_price+sl_dist, current_price-risk, current_price-(risk*2), current_price-(risk*3.5)
            emoji = "⬇️ SHORT"
        
        setup = {'entry': current_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'dir': emoji, 'status': setup_type, 'qty': qty, 'lev': leverage}

    if signal != "NEUTRO" and signal != st.session_state.last_alert and setup:
        st.session_state.last_alert = signal
    elif signal == "NEUTRO": st.session_state.last_alert = "NEUTRO"
    
    manage_open_positions(current_price)
    
    tab1, tab2 = st.tabs(["📊 LIVE COMMAND", "🧪 PAPER TRADING"])
    
    with tab1:
        col_news, col_thermo = st.columns([2, 1])
        with col_news:
            if news:
                st.markdown("### 📰 MARKET FLASH")
                news_html = "<div class='news-box'>"
                for n in news:
                    t_struct = n.get('published', time.gmtime())
                    t_str = f"{t_struct.tm_hour:02}:{t_struct.tm_min:02}"
                    news_html += f"<div class='news-item'><span class='news-time'>{t_str}</span> <a href='{n['link']}' target='_blank' class='news-link'>{n['title']}</a></div>"
                news_html += "</div>"
                st.markdown(news_html, unsafe_allow_html=True)
        
        with col_thermo:
            # Gauge Termometro
            fig_thermo = go.Figure(go.Indicator(
                mode = "gauge+number", value = thermo_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<b>Bot Sentiment</b>"},
                gauge = {'axis': {'range': [-100, 100]}, 'bar': {'color': "white"},
                    'steps': [{'range': [-100, -40], 'color': "#FF4444"}, {'range': [-40, 40], 'color': "gray"}, {'range': [40, 100], 'color': "#00FF00"}],
                }
            ))
            fig_thermo.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#111", font={'color': "white"})
            st.plotly_chart(fig_thermo, use_container_width=True)
            
            # Gauge Fear Greed
            fig_fng = go.Figure(go.Indicator(
                mode = "gauge+number", value = fng_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "<b>Fear & Greed</b>"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                    'steps': [{'range': [0, 40], 'color': "#FF4444"}, {'range': [40, 60], 'color': "yellow"}, {'range': [60, 100], 'color': "#00FF00"}],
                }
            ))
            fig_fng.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="#111", font={'color': "white"})
            st.plotly_chart(fig_fng, use_container_width=True)

        st.markdown(f"<div class='ai-box'>🤖 <b>QUIMERA COPILOT:</b><br>{ai_narrative}</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${current_price:,.2f}")
        c2.metric("Tendencia 4H", trend_4h, delta="Bullish" if trend_4h=="BULLISH" else "Bearish")
        c3.metric("Funding Rate", f"{fr:.4f}%")
        c4.metric("GASOLINA (MFI)", f"{df['MFI'].iloc[-1]:.0f}")

        if setup:
            if setup['status'] == "CONFIRMED":
                header_cls = "header-confirmed-long" if calc_dir == "LONG" else "header-confirmed-short"
                header_txt = f"🔥 CONFIRMADO: {setup['dir']}"
                btn_label = f"🚀 EJECUTAR {calc_dir} ({setup['qty']:.4f})"
            else:
                header_cls = "header-potential"
                header_txt = f"⚠️ POTENCIAL: {setup['dir']}"
                btn_label = f"⚠️ FORZAR ENTRADA"

            st.markdown(f"""
            <div class="trade-setup">
                <div class="{header_cls}">{header_txt}</div>
                <p style="color:#888; font-size:14px;">Posición Sugerida: <span style="color:white; font-weight:bold">{setup['qty']:.4f} {symbol.split('/')[0]}</span> (Lev: {setup['lev']:.1f}x)</p>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <div><span class="label-mini">ENTRADA</span><br><span class="entry-blue">${setup['entry']:.2f}</span></div>
                    <div><span class="label-mini">STOP LOSS</span><br><span class="sl-red">${setup['sl']:.2f}</span></div>
                    <div><span class="label-mini">TP 1</span><br><span class="tp-green">${setup['tp1']:.2f}</span></div>
                    <div><span class="label-mini">TP 2</span><br><span class="tp-green">${setup['tp2']:.2f}</span></div>
                    <div><span class="label-mini">TP 3</span><br><span class="tp-green">${setup['tp3']:.2f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(btn_label):
                execute_trade(calc_dir, current_price, setup['sl'], setup['tp1'], setup['tp2'], setup['tp3'], setup['qty'], atr, setup['lev'])
                st.success(f"Orden {calc_dir} lanzada.")
        else:
            st.info("Esperando estructura de mercado clara...")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        if use_vwap: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VWAP'], line=dict(color='orange', dash='dot'), name='VWAP'), row=1, col=1)
        
        last_pivot = df.iloc[-1]['PIVOT']
        last_s1 = df.iloc[-1]['S1']
        last_r1 = df.iloc[-1]['R1']
        fig.add_hline(y=last_pivot, line_dash="dash", line_color="gray", annotation_text="Pivote", row=1, col=1)
        fig.add_hline(y=last_s1, line_dash="dot", line_color="green", annotation_text="S1", row=1, col=1)
        fig.add_hline(y=last_r1, line_dash="dot", line_color="red", annotation_text="R1", row=1, col=1)

        if setup:
            fig.add_hline(y=setup['tp1'], line_dash="dot", line_color="green", row=1, col=1)
            fig.add_hline(y=setup['sl'], line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1); fig.add_hline(y=30, row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_trades = load_trades()
        st.subheader("📈 Rendimiento")
        render_analytics(df_trades)
        st.divider()
        if not df_trades.empty:
            open_trades = df_trades[df_trades['status'] == "OPEN"].copy()
            closed_trades = df_trades[df_trades['status'] == "CLOSED"]
            
            st.subheader("🟢 Posiciones Abiertas")
            if not open_trades.empty:
                open_trades['Floating PnL'] = np.where(open_trades['type'] == 'LONG', (current_price - open_trades['entry']) * open_trades['size'], (open_trades['entry'] - current_price) * open_trades['size'])
                st.dataframe(open_trades, use_container_width=True)
            else: st.info("No hay operaciones abiertas.")
            
            st.subheader("📜 Historial Cerrado")
            def color_pnl(val): return f'color: {"#228B22" if val > 0 else "#B22222" if val < 0 else "white"}'
            if not closed_trades.empty: st.dataframe(closed_trades.style.applymap(color_pnl, subset=['pnl']), use_container_width=True)
        else:
            st.info("Historial vacío.")

else: st.warning("Cargando datos...")

if auto_refresh: time.sleep(60); st.rerun()
