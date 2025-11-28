import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

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
            def ichimoku(self, *args, **kwargs): return [pd.DataFrame(), pd.DataFrame()]
        ta = DummyTA()

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
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
        except:
            return pd.Series([0] * len(df))

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
            if adx is not None and not adx.empty:
                d = pd.concat([d, adx], axis=1)
        except: pass

        # 2. MANUALES
        d['TSI'] = self.calculate_manual_tsi(d)
        try:
            d['range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['range_pct'].rolling(window=14).mean()
        except: d['ADR'] = 0
        try:
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()
        except: d['VWAP'] = d['close']

        # 3. PIVOTS
        high = d['high'].rolling(1).max()
        low = d['low'].rolling(1).min()
        close = d['close']
        d['PIVOT'] = (high + low + close) / 3
        d['R1'] = (2 * d['PIVOT']) - low
        d['S1'] = (2 * d['PIVOT']) - high

        d.fillna(method='bfill', inplace=True)
        d.fillna(0, inplace=True)
        return d

    def train_regime_model(self, df):
        try:
            df['ret'] = df['close'].pct_change()
            df['vol'] = df['ret'].rolling(10).std()
            future_vol = df['vol'].shift(-5)
            df['target'] = np.where(future_vol > df['vol'].quantile(0.6), 1, 0)
            data = df.dropna()
            cols = [c for c in ['RSI', 'ADX_14', 'vol'] if c in data.columns]
            if len(data) > 50 and cols:
                self.model.fit(data[cols], data['target'])
                self.is_model_trained = True
        except: pass

    def get_signal(self, df, context_filters):
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO", "Sin Datos", "Ninguno"

        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        score = 0
        details = []

        ema20 = row.get('EMA_20', 0)
        ema50 = row.get('EMA_50', 0)
        
        if context_filters.get('use_ema'):
            if ema20 > ema50: score += 1; details.append("EMA Alcista")
            else: score -= 1; details.append("EMA Bajista")

        vwap = row.get('VWAP', row['close'])
        if context_filters.get('use_vwap'):
            if row['close'] > vwap: score += 1
            else: score -= 1

        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            try:
                cols = [c for c in ['RSI', 'ADX_14', 'vol'] if c in df.columns]
                if cols:
                    pred = self.model.predict(df[cols].iloc[[-1]])[0]
                    if pred == 0: score = 0; details.append("⛔ ML: Rango"); regime = "RANGO"
                    else: regime = "TENDENCIA"
            except: pass

        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        rsi = row.get('RSI', 50)
        if signal == "LONG" and rsi > 75: signal = "NEUTRO"
        if signal == "SHORT" and rsi < 25: signal = "NEUTRO"

        prev_ema20 = prev_row.get('EMA_20', 0)
        prev_ema50 = prev_row.get('EMA_50', 0)
        bull_cross = prev_ema20 <= prev_ema50 and ema20 > ema50
        bear_cross = prev_ema20 >= prev_ema50 and ema20 < ema50
        
        trend_status = "Estable"
        if bull_cross: trend_status = "🔄 GIRO ALCISTA"
        elif bear_cross: trend_status = "🔄 GIRO BAJISTA"
        elif signal == "LONG" and row['close'] < ema20: trend_status = "⚠️ DEBILIDAD"
        elif signal == "SHORT" and row['close'] > ema20: trend_status = "⚠️ REBOTE"
        elif regime == "TENDENCIA": trend_status = "✅ FUERTE"
        elif regime == "RANGO": trend_status = "💤 LATERAL"

        candle_pat = "Sin Patrón"
        if (prev_row['close'] < prev_row['open']) and (row['close'] > row['open']) and \
           (row['close'] > prev_row['open']) and (row['open'] < prev_row['close']):
            candle_pat = "🕯️ Bullish Engulfing"
        elif (prev_row['close'] > prev_row['open']) and (row['close'] < row['open']) and \
             (row['close'] < prev_row['open']) and (row['open'] > prev_row['close']):
            candle_pat = "🕯️ Bearish Engulfing"
        
        atr_val = row.get('ATR', 0)
        return signal, atr_val, details, regime, trend_status, candle_pat

    def check_and_execute_auto(self, db_mgr, symbol, signal, strength, price, atr):
        """Ejecuta operaciones automáticas si la señal es DIAMANTE"""
        if strength != "DIAMOND" or signal == "NEUTRO":
            return False, None

        # Cargar últimos trades para evitar spam
        df_trades = db_mgr.load_trades()
        if not df_trades.empty:
            last_trade = df_trades.iloc[0]
            try:
                last_trade_time = datetime.strptime(last_trade['timestamp'], "%Y-%m-%d %H:%M:%S")
                # Cooldown de 5 minutos
                if (datetime.now() - last_trade_time).total_seconds() < 300: 
                    return False, None
            except: pass

        # Gestión de Riesgo Automática
        sl_dist = atr * 1.5
        sl = price - sl_dist if signal == "LONG" else price + sl_dist
        tp1 = price + sl_dist if signal == "LONG" else price - sl_dist
        
        dist_pct = abs(price - sl) / price if price > 0 else 0
        lev = int(0.02 / dist_pct) if dist_pct > 0 else 1
        lev = max(1, min(lev, 50))

        trade = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "type": signal, "entry": price, "size": 1000.0,
            "leverage": lev, "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0,
            "status": "OPEN", "pnl": 0.0, "reason": "AUTO-DIAMOND", "candles_held": 0, "atr_entry": atr
        }
        
        db_mgr.add_trade(trade)
        return True, trade

    def run_backtest_vectorized(self, df):
        """Backtest rápido vectorizado para la pestaña de Backtest"""
        if df is None or df.empty: return 0, 0, 0
        
        d = df.copy()
        d['signal'] = 0
        
        # Reglas Simples para Backtest
        d.loc[(d['EMA_20'] > d['EMA_50']) & (d['close'] > d['VWAP']), 'signal'] = 1
        d.loc[(d['EMA_20'] < d['EMA_50']) & (d['close'] < d['VWAP']), 'signal'] = -1
        
        # Calcular retornos
        d['market_return'] = d['close'].pct_change()
        d['strategy_return'] = d['signal'].shift(1) * d['market_return']
        
        total_return = d['strategy_return'].sum()
        trades_count = d['signal'].diff().abs().sum() / 2
        
        winning_trades = len(d[d['strategy_return'] > 0])
        total_trades = len(d[d['strategy_return'] != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return total_return, trades_count, win_rate
