import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- IMPORTACIÓN DIRECTA (Solo la versión del ZIP) ---
try:
    import pandas_ta_classic as ta
except ImportError:
    # Si por algún motivo falla el ZIP, usamos un Dummy de emergencia
    # (Ya no intentamos importar la vieja 'pandas_ta' para evitar el aviso amarillo)
    class DummyTA:
        def ema(self, *args, **kwargs): return pd.Series([0]*len(args[0]))
        def rsi(self, *args, **kwargs): return pd.Series([50]*len(args[0]))
        def atr(self, *args, **kwargs): return pd.Series([1]*len(args[0]))
        def adx(self, *args, **kwargs): return pd.DataFrame({'ADX_14': [0]*len(args[0])})
        def tsi(self, *args, **kwargs): return pd.Series([0]*len(args[0]))
        def mfi(self, *args, **kwargs): return pd.Series([50]*len(args[0]))
        def ichimoku(self, *args, **kwargs): return [pd.DataFrame(), pd.DataFrame()]
    ta = DummyTA()
# -----------------------------------------------------

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_model_trained = False

    def prepare_data(self, df):
        if df is None or df.empty: return df
        
        d = df.copy()

        # 1. CÁLCULOS MANUALES (Respaldo robusto)
        try:
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()
        except:
            d['VWAP'] = d['close']

        try:
            d['range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['range_pct'].rolling(window=14).mean()
        except:
            d['ADR'] = 0

        # 2. INDICADORES DE LA LIBRERÍA ZIP (pandas_ta_classic)
        try:
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
        except: pass

        try:
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
        except: pass

        try:
            d['TSI'] = ta.tsi(d['close'], fast=13, slow=25)
        except: d['TSI'] = 0

        try:
            d['MFI'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
        except: d['MFI'] = 50

        try:
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None and not adx.empty:
                d = pd.concat([d, adx], axis=1)
        except: pass

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
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO", "Sin Datos"

        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        score = 0
        details = []

        # Uso de .get() para evitar errores si falta alguna columna
        ema20 = row.get('EMA_20', 0)
        ema50 = row.get('EMA_50', 0)
        
        if context_filters.get('use_ema'):
            if ema20 > ema50:
                score += 1
                details.append("EMA Alcista")
            else:
                score -= 1
                details.append("EMA Bajista")

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
                    if pred == 0:
                        score = 0
                        details.append("⛔ ML: Rango")
                        regime = "RANGO"
                    else:
                        regime = "TENDENCIA"
            except: pass

        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        rsi = row.get('RSI', 50)
        if signal == "LONG" and rsi > 75: signal = "NEUTRO"
        if signal == "SHORT" and rsi < 25: signal = "NEUTRO"

        # Tendencia
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

        atr_val = row.get('ATR', 0)
        return signal, atr_val, details, regime, trend_status

    def run_backtest_vectorized(self, df):
        return 0, 0, 0
