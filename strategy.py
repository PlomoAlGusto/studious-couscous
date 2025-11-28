import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
# Importaciones ML
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- IMPORTACIÓN SEGURA ---
try:
    import pandas_ta_classic as ta
except ImportError:
    try:
        import pandas_ta as ta
    except ImportError:
        class DummyTA:
            def ema(self, *args, **kwargs): return pd.Series([0]*len(args[0]))
            def rsi(self, *args, **kwargs): return pd.Series([50]*len(args[0]))
            def atr(self, *args, **kwargs): return pd.Series([1]*len(args[0]))
            def adx(self, *args, **kwargs): return pd.DataFrame({'ADX_14': [0]*len(args[0])})
            def tsi(self, *args, **kwargs): return pd.Series([0]*len(args[0]))
            def mfi(self, *args, **kwargs): return pd.Series([50]*len(args[0]))
            def cdl_pattern(self, *args, **kwargs): return pd.Series([0]*len(args[0]))
        ta = DummyTA()

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(random_state=42)
        self.xgb_model = XGBClassifier(random_state=42)
        self.is_model_trained = False

    def calculate_manual_tsi(self, df, fast=13, slow=25):
        try:
            diff = df['close'].diff(1)
            slow_ema = diff.ewm(span=slow, min_periods=slow, adjust=False).mean()
            fast_slow_ema = slow_ema.ewm(span=fast, min_periods=fast, adjust=False).mean()
            abs_diff = diff.abs()
            slow_ema_abs = abs_diff.ewm(span=slow, min_periods=slow, adjust=False).mean()
            fast_slow_ema_abs = slow_ema_abs.ewm(span=fast, min_periods=fast, adjust=False).mean()
            tsi = 100 * (fast_slow_ema / fast_slow_ema_abs)
            return tsi.fillna(0)
        except: return pd.Series([0] * len(df))

    def prepare_data(self, df):
        if df is None or df.empty: return df
        d = df.copy()

        # 1. LIBRERÍA
        try:
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
            d['MFI'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None and not adx.empty: d = pd.concat([d, adx], axis=1)
        except: pass

        # 2. MANUALES (TSI, ADR, VWAP)
        d['TSI'] = self.calculate_manual_tsi(d)
        try:
            d['range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['range_pct'].rolling(window=14).mean()
        except: d['ADR'] = 0
        try:
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()
        except: d['VWAP'] = d['close']

        # 3. NUEVOS FILTROS QUANT (CVD Proxy & SFP)
        try:
            # CVD PROXY (Money Flow Multiplier)
            # (Close - Low) - (High - Close) / (High - Low) * Volume
            mrv = ((d['close'] - d['low']) - (d['high'] - d['close'])) / (d['high'] - d['low'])
            mrv = mrv.fillna(0)
            d['FLOW_VOL'] = mrv * d['volume']
            d['CVD_PROXY'] = d['FLOW_VOL'].cumsum() # Cumulative Volume Delta simulado
            
            # SFP (Swing Failure Pattern) - Detectar Liquidez
            # Buscamos fractales (máximos locales de 5 velas)
            d['swing_high'] = d['high'].rolling(5).max()
            d['swing_low'] = d['low'].rolling(5).min()
        except: pass

        d.fillna(method='bfill', inplace=True)
        d.fillna(0, inplace=True)
        return d

    def train_regime_model(self, df):
        try:
            df['ret'] = df['close'].pct_change()
            df['vol'] = df['ret'].rolling(10).std()
            future_ret = df['ret'].shift(-1)
            df['target'] = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
            data = df.dropna()
            cols = ['RSI', 'ADX_14', 'vol', 'TSI', 'MFI', 'ATR']
            valid_cols = [c for c in cols if c in data.columns]
            if len(data) > 100 and valid_cols:
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(data[valid_cols], data['target'])
                self.model.fit(X_res, y_res)
                self.xgb_model.fit(X_res, y_res)
                self.is_model_trained = True
        except: pass

    def get_signal(self, df, context_filters):
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO", "Sin Datos", "Ninguno"

        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        score = 0
        details = []

        # 1. ESTRATEGIA BASE
        ema20 = row.get('EMA_20', 0); ema50 = row.get('EMA_50', 0)
        if context_filters.get('use_ema'):
            if ema20 > ema50: score += 1; details.append("EMA Alcista")
            else: score -= 1; details.append("EMA Bajista")

        vwap = row.get('VWAP', row['close'])
        if context_filters.get('use_vwap'):
            if row['close'] > vwap: score += 1
            else: score -= 1

        # 2. ORDER FLOW & LIQUIDEZ (NUEVO)
        # Si el precio sube pero el CVD baja = Divergencia Bajista (Trampa)
        try:
            price_trend = row['close'] > df.iloc[-5]['close']
            cvd_trend = row['CVD_PROXY'] > df.iloc[-5]['CVD_PROXY']
            
            if price_trend and not cvd_trend:
                score -= 1
                details.append("⚠️ Div. CVD (Trampa Alcista)")
            elif not price_trend and cvd_trend:
                score += 1
                details.append("⚠️ Div. CVD (Trampa Bajista)")
        except: pass

        # 3. SFP (SWING FAILURE PATTERN) - FAKEOUTS
        # Si el precio rompió el máximo anterior pero cerró abajo = SFP Bearish
        sfp_signal = "Ninguno"
        try:
            last_swing_high = df.iloc[-10:-2]['high'].max() # Maximo reciente
            last_swing_low = df.iloc[-10:-2]['low'].min()   # Minimo reciente
            
            if row['high'] > last_swing_high and row['close'] < last_swing_high:
                score -= 2 # Fuerte señal de venta
                sfp_signal = "🐻 SFP Bearish (Fakeout)"
                details.append("Fakeout Arriba")
            
            if row['low'] < last_swing_low and row['close'] > last_swing_low:
                score += 2 # Fuerte señal de compra
                sfp_signal = "🐂 SFP Bullish (Fakeout)"
                details.append("Fakeout Abajo")
        except: pass

        # 4. ML REGIME
        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            try:
                cols = ['RSI', 'ADX_14', 'vol', 'TSI', 'MFI', 'ATR']
                valid_cols = [c for c in cols if c in df.columns]
                if valid_cols:
                    last_data = df[valid_cols].iloc[[-1]].fillna(0)
                    rf = self.model.predict(last_data)[0]
                    xgb = self.xgb_model.predict(last_data)[0]
                    if rf == 1 and xgb == 1: score += 2; regime = "TENDENCIA ALCISTA"
                    elif rf == -1 and xgb == -1: score -= 2; regime = "TENDENCIA BAJISTA"
                    else: score = 0; regime = "RANGO"
            except: pass

        # SEÑAL FINAL
        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        rsi = row.get('RSI', 50)
        if signal == "LONG" and rsi > 75: signal = "NEUTRO"
        if signal == "SHORT" and rsi < 25: signal = "NEUTRO"

        # Estado Visual
        trend_status = "Estable"
        if sfp_signal != "Ninguno": trend_status = f"🔥 {sfp_signal}"
        elif "TENDENCIA" in regime: trend_status = "✅ FUERTE"
        elif regime == "RANGO": trend_status = "💤 LATERAL"
        
        atr_val = row.get('ATR', 0)
        return signal, atr_val, details, regime, trend_status, sfp_signal

    # --- GESTIÓN DINÁMICA DE POSICIÓN (KELLY / VOLATILIDAD) ---
    def calculate_dynamic_position(self, account_balance, risk_per_trade_pct, entry, sl):
        """
        Calcula el tamaño de la posición basado en riesgo fijo (ej: perder max 100$).
        Wealth Generacional = No perder nunca más de lo planeado.
        """
        if entry == 0 or sl == 0: return 0, 1
        
        risk_amount = account_balance * (risk_per_trade_pct / 100)
        distance_per_token = abs(entry - sl)
        
        if distance_per_token == 0: return 0, 1
        
        # Tamaño en Monedas (ej: 0.5 BTC)
        position_size_coins = risk_amount / distance_per_token
        
        # Tamaño en USDT (Notional)
        position_size_usdt = position_size_coins * entry
        
        # Apalancamiento necesario para cubrir el notional con el balance
        leverage = position_size_usdt / account_balance
        
        # Ajustes de seguridad
        leverage = max(1, min(leverage, 50)) # Cap 50x
        return position_size_usdt, int(leverage)

    def check_and_execute_auto(self, db_mgr, symbol, signal, strength, price, atr, account_balance=10000, risk_pct=1.0):
        if strength != "DIAMOND" or signal == "NEUTRO": return False, None

        df_trades = db_mgr.load_trades()
        if not df_trades.empty:
            last = df_trades.iloc[0]
            try:
                lt = datetime.strptime(last['timestamp'], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - lt).total_seconds() < 300: return False, None
            except: pass

        sl_dist = atr * 1.5
        sl = price - sl_dist if signal == "LONG" else price + sl_dist
        tp1 = price + sl_dist if signal == "LONG" else price - sl_dist
        
        # --- CÁLCULO DINÁMICO DE TAMAÑO ---
        pos_size, lev = self.calculate_dynamic_position(account_balance, risk_pct, price, sl)
        
        trade = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "type": signal, "entry": price, "size": pos_size,
            "leverage": lev, "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0,
            "status": "OPEN", "pnl": 0.0, "reason": "AUTO-DIAMOND (Risk Managed)", "candles_held": 0, "atr_entry": atr
        }
        db_mgr.add_trade(trade)
        return True, trade

    def run_backtest_vectorized(self, df):
        if df is None or df.empty: return 0, 0, 0, 0, 0
        d = df.copy()
        d['signal'] = 0
        # Backtest con CVD y SFP simplificado
        # Solo entramos si hay tendencia Y volumen acompañando
        try:
            d['trend_ok'] = ((d['EMA_20'] > d['EMA_50']) & (d['close'] > d['VWAP'])).astype(int)
            # Proxy de volumen: si MFI > 50 es confirmación
            d.loc[(d['trend_ok'] == 1) & (d['MFI'] > 50), 'signal'] = 1
            d.loc[(d['EMA_20'] < d['EMA_50']) & (d['close'] < d['VWAP']) & (d['MFI'] < 50), 'signal'] = -1
        except: pass
        
        d['market_return'] = d['close'].pct_change()
        d['strategy_return'] = d['signal'].shift(1) * d['market_return']
        total_return = d['strategy_return'].sum()
        trades_count = d['signal'].diff().abs().sum() / 2
        wins = len(d[d['strategy_return'] > 0])
        total = len(d[d['strategy_return'] != 0])
        win_rate = wins / total if total > 0 else 0
        sharpe = d['strategy_return'].mean() / d['strategy_return'].std() * np.sqrt(252) if d['strategy_return'].std() != 0 else 0
        cum_ret = d['strategy_return'].cumsum()
        max_dd = (cum_ret.cummax() - cum_ret).max()
        return total_return, trades_count, win_rate, sharpe, max_dd
