import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
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
            def ichimoku(self, *args, **kwargs): return [pd.DataFrame(), pd.DataFrame()]
            def macd(self, *args, **kwargs): return pd.DataFrame({'MACD_12_26_9': [0]*len(args[0]), 'MACDs_12_26_9': [0]*len(args[0])})
            def bbands(self, *args, **kwargs): return pd.DataFrame({'BBU_20_2.0': [0]*len(args[0]), 'BBL_20_2.0': [0]*len(args[0])})
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
        except Exception as e:
            print(f"Error in TSI calculation: {e}")
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
            macd = ta.macd(d['close'], fast=12, slow=26, signal=9)
            d['MACD'] = macd['MACD_12_26_9']
            d['MACD_signal'] = macd['MACDs_12_26_9']
            bb = ta.bbands(d['close'], length=20)
            d['BB_upper'] = bb['BBU_20_2.0']
            d['BB_lower'] = bb['BBL_20_2.0']
        except Exception as e:
            print(f"Error in TA indicators: {e}")

        # 2. MANUALES
        d['TSI'] = self.calculate_manual_tsi(d)
        try:
            d['range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['range_pct'].rolling(window=14).mean()
        except Exception as e:
            print(f"Error in ADR: {e}")
            d['ADR'] = 0
        try:
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()
            d['VWAP_std'] = ((d['close'] - d['VWAP']) / d['ATR']).abs()  # Nueva: desviación normalizada
        except Exception as e:
            print(f"Error in VWAP: {e}")
            d['VWAP'] = d['close']
            d['VWAP_std'] = 0

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

    def train_signal_model(self, df):
        try:
            df['ret'] = df['close'].pct_change()
            df['vol'] = df['ret'].rolling(10).std()
            df['ema_diff'] = (df['EMA_20'] - df['EMA_50']) / df['EMA_50']  # Nueva feature
            df['vwap_dev'] = (df['close'] - df['VWAP']) / df['VWAP']  # Nueva feature
            future_ret = df['ret'].shift(-1)
            df['target'] = np.where(future_ret > 0.001, 1, np.where(future_ret < -0.001, -1, 0))
            data = df.dropna()
            cols = ['RSI', 'ADX_14', 'vol', 'ema_diff', 'vwap_dev', 'TSI', 'MFI', 'ATR']
            cols = [c for c in cols if c in data.columns]
            if len(data) > 100 and cols:
                # Balanceo con SMOTE
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(data[cols], data['target'])
                
                # Tuning con GridSearchCV y TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=tscv)
                grid.fit(X_res, y_res)
                self.model = grid.best_estimator_
                
                # Entrenar XGBoost similarmente
                xgb_grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=tscv)
                xgb_grid.fit(X_res, y_res)
                self.xgb_model = xgb_grid.best_estimator_
                
                self.is_model_trained = True
        except Exception as e:
            print(f"Error in model training: {e}")

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
            # Uso de VWAP_std
            if row['VWAP_std'] > 1: score += 1 if row['close'] > vwap else -1; details.append("VWAP Desviación Alta")

        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            try:
                cols = ['RSI', 'ADX_14', 'vol', 'ema_diff', 'vwap_dev', 'TSI', 'MFI', 'ATR']
                cols = [c for c in cols if c in df.columns]
                if cols:
                    rf_pred = self.model.predict(df[cols].iloc[[-1]])[0]
                    xgb_pred = self.xgb_model.predict(df[cols].iloc[[-1]])[0]
                    pred = rf_pred if rf_pred == xgb_pred else 0  # Consenso ensemble
                    if pred == 1: score += 2; details.append("ML: LONG"); regime = "TENDENCIA ALCISTA"
                    elif pred == -1: score -= 2; details.append("ML: SHORT"); regime = "TENDENCIA BAJISTA"
                    else: score = 0; details.append("ML: HOLD"); regime = "RANGO"
            except Exception as e:
                print(f"Error in ML prediction: {e}")

        # Agregar MACD y BB
        if 'MACD' in row and 'MACD_signal' in row:
            if row['MACD'] > row['MACD_signal']: score += 1; details.append("MACD Alcista")
            else: score -= 1; details.append("MACD Bajista")
        if 'BB_lower' in row and row['close'] < row['BB_lower']: score += 1; details.append("BB Oversold")

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
        elif regime.startswith("TENDENCIA"): trend_status = "✅ FUERTE"
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
        if strength != "DIAMOND" or signal == "NEUTRO" or (atr / price > 0.05 if price > 0 else False):
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
            "status": "OPEN", "pnl": 0.0, "reason": "AUTO-DIAMOND", "candles_held": 0, "atr_entry": atr,
            "trailing_sl": True  # Nueva: trailing stop
        }
        
        db_mgr.add_trade(trade)
        return True, trade

    def run_backtest_vectorized(self, df):
        """Backtest rápido vectorizado para la pestaña de Backtest"""
        if df is None or df.empty: return 0, 0, 0, 0, 0
        
        d = df.copy()
        d['signal'] = 0
        
        # Reglas Simples para Backtest
        d.loc[(d['EMA_20'] > d['EMA_50']) & (d['close'] > d['VWAP']), 'signal'] = 1
        d.loc[(d['EMA_20'] < d['EMA_50']) & (d['close'] < d['VWAP']), 'signal'] = -1
        
        # Calcular retornos
        d['market_return'] = d['close'].pct_change()
        d['strategy_return'] = d['signal'].shift(1) * d['market_return']
        d['strategy_return'] -= 0.001 * d['signal'].diff().abs() / 2  # Fees por trade (aprox 0.1%)
        
        total_return = d['strategy_return'].sum()
        trades_count = d['signal'].diff().abs().sum() / 2
        
        winning_trades = len(d[d['strategy_return'] > 0])
        total_trades = len(d[d['strategy_return'] != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Sharpe y Max Drawdown
        sharpe = d['strategy_return'].mean() / d['strategy_return'].std() * np.sqrt(252) if d['strategy_return'].std() != 0 else 0
        cum_ret = d['strategy_return'].cumsum()
        max_dd = (cum_ret.cummax() - cum_ret).max()
        
        return total_return, trades_count, win_rate, sharpe, max_dd
