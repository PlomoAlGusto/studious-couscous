import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
# Importaciones ML
from sklearn.model_selection import TimeSeriesSplit
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

        # 1. LIBRERÍA TÉCNICA
        try:
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
            d['MFI'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
            
            # ADX (Vital para tu filtro)
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None and not adx.empty:
                d = pd.concat([d, adx], axis=1)
                # Normalizar nombre columna
                if 'ADX_14' not in d.columns: d['ADX_14'] = d.iloc[:, -1]
        except: pass

        # 2. CÁLCULOS MANUALES
        d['TSI'] = self.calculate_manual_tsi(d)
        
        # ADR Manual
        try:
            d['range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['range_pct'].rolling(window=14).mean()
        except: d['ADR'] = 0

        # --- MEJORA: VWAP DIARIO (ANCHORED VWAP) ---
        # Reiniciamos el VWAP cada día. Esto es como lo ven los algoritmos institucionales.
        try:
            # Aseguramos que timestamp es datetime
            if not pd.api.types.is_datetime64_any_dtype(d['timestamp']):
                d['timestamp'] = pd.to_datetime(d['timestamp'])
            
            d['date_group'] = d['timestamp'].dt.date
            
            # Cálculo eficiente con GroupBy
            def vwap_func(x):
                vol = x['volume'].replace(0, 1) # Evitar div/0
                return (x['close'] * x['volume']).cumsum() / vol.cumsum()

            # Aplicamos y reseteamos el índice para mantener alineación
            d['VWAP'] = d.groupby('date_group', group_keys=False).apply(vwap_func)
            # Si groupby devuelve multiindex, lo aplanamos (parche seguridad)
            if isinstance(d['VWAP'].index, pd.MultiIndex):
                d['VWAP'] = d['VWAP'].reset_index(level=0, drop=True)
                
        except Exception as e:
            # Fallback al acumulado total si falla el diario
            # print(f"VWAP Error: {e}")
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()

        # Limpieza
        if 'date_group' in d.columns: d.drop(columns=['date_group'], inplace=True)
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

        ema20 = row.get('EMA_20', 0); ema50 = row.get('EMA_50', 0)
        adx = row.get('ADX_14', 0)

        # 1. ESTRATEGIA (Con Filtro ADX)
        if context_filters.get('use_ema'):
            if ema20 > ema50: 
                score += 1; details.append("EMA Alcista")
            else: 
                score -= 1; details.append("EMA Bajista")

        # 2. VWAP
        vwap = row.get('VWAP', row['close'])
        if context_filters.get('use_vwap'):
            if row['close'] > vwap: score += 1
            else: score -= 1

        # 3. ML
        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            try:
                cols = ['RSI', 'ADX_14', 'vol', 'TSI', 'MFI', 'ATR']
                valid = [c for c in cols if c in df.columns]
                if valid:
                    last = df[valid].iloc[[-1]].fillna(0)
                    rf = self.model.predict(last)[0]
                    xgb = self.xgb_model.predict(last)[0]
                    if rf == 1 and xgb == 1: score += 2; regime = "TENDENCIA ALCISTA"
                    elif rf == -1 and xgb == -1: score -= 2; regime = "TENDENCIA BAJISTA"
                    else: score = 0; regime = "RANGO"
            except: pass

        # --- FILTRO ADX (TU MEJORA) ---
        # Si el ADX es bajo (<25), el mercado está muerto. Anulamos señales débiles.
        if context_filters.get('use_regime') and adx < 25:
            # Si no es una señal muy fuerte (score alto), la matamos
            if abs(score) < 3:
                score = 0
                details.append("⛔ ADX < 25 (Rango)")

        # SEÑAL FINAL
        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        # Filtro RSI
        rsi = row.get('RSI', 50)
        if signal == "LONG" and rsi > 75: signal = "NEUTRO"
        if signal == "SHORT" and rsi < 25: signal = "NEUTRO"

        # Tendencia y Patrones
        prev_ema20 = prev_row.get('EMA_20', 0); prev_ema50 = prev_row.get('EMA_50', 0)
        bull_cross = prev_ema20 <= prev_ema50 and ema20 > ema50
        bear_cross = prev_ema20 >= prev_ema50 and ema20 < ema50
        
        trend_status = "Estable"
        if bull_cross: trend_status = "🔄 GIRO ALCISTA"
        elif bear_cross: trend_status = "🔄 GIRO BAJISTA"
        elif "TENDENCIA" in regime: trend_status = "✅ FUERTE"
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

    def check_and_execute_auto(self, db_mgr, symbol, signal, strength, price, atr, account_size=10000, risk_pct=1.0):
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
        
        # Cálculo de posición
        if entry == 0 or sl == 0: return False, None
        dist_pct = abs(price - sl) / price
        if dist_pct == 0: return False, None
        
        risk_amount = account_size * (risk_pct / 100)
        pos_size_usdt = risk_amount / dist_pct # Tamaño real posición
        lev = int(pos_size_usdt / account_size)
        lev = max(1, min(lev, 50))

        trade = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "type": signal, "entry": price, "size": pos_size_usdt,
            "leverage": lev, "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0,
            "status": "OPEN", "pnl": 0.0, "reason": "AUTO-DIAMOND", "candles_held": 0, "atr_entry": atr
        }
        db_mgr.add_trade(trade)
        return True, trade

    def run_backtest_vectorized(self, df):
        """
        BACKTEST MEJORADO (INSTITUCIONAL):
        - Filtro ADX > 25 (Fuerza)
        - Filtro MFI (Volumen)
        - Fees simulados
        """
        if df is None or df.empty: return 0, 0, 0, 0, 0
        
        d = df.copy()
        d['signal'] = 0
        
        # Aseguramos columnas
        if 'ADX_14' not in d.columns: d['ADX_14'] = 0
        if 'MFI' not in d.columns: d['MFI'] = 50
        
        # --- LÓGICA VECTORIZADA ---
        # LONG: Tendencia + Valor + Fuerza + Volumen
        d.loc[
            (d['EMA_20'] > d['EMA_50']) & 
            (d['close'] > d['VWAP']) & 
            (d['ADX_14'] > 25), # TU MEJORA
            'signal'
        ] = 1
        
        # SHORT
        d.loc[
            (d['EMA_20'] < d['EMA_50']) & 
            (d['close'] < d['VWAP']) & 
            (d['ADX_14'] > 25), # TU MEJORA
            'signal'
        ] = -1
        
        d['market_return'] = d['close'].pct_change()
        d['strategy_return'] = d['signal'].shift(1) * d['market_return']
        
        # Fees (0.1%)
        d.loc[d['signal'].diff().abs() > 0, 'strategy_return'] -= 0.001
        
        total_return = d['strategy_return'].sum()
        trades_count = d['signal'].diff().abs().sum() / 2
        wins = len(d[d['strategy_return'] > 0])
        total = len(d[d['strategy_return'] != 0])
        win_rate = wins / total if total > 0 else 0
        
        sharpe = d['strategy_return'].mean() / d['strategy_return'].std() * np.sqrt(252) if d['strategy_return'].std() != 0 else 0
        cum_ret = d['strategy_return'].cumsum()
        max_dd = (cum_ret.cummax() - cum_ret).max()
        
        return total_return, trades_count, win_rate, sharpe, max_dd
