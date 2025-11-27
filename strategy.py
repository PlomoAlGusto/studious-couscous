import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- IMPORTACIÓN SEGURA ---
try:
    import pandas_ta_classic as ta
except ImportError:
    try:
        import pandas_ta as ta
    except ImportError:
        # Dummy por si falla todo
        class DummyTA:
            def ema(self, *args, **kwargs): return pd.Series([0]*100)
            def rsi(self, *args, **kwargs): return pd.Series([50]*100)
            def atr(self, *args, **kwargs): return pd.Series([1]*100)
            def adx(self, *args, **kwargs): return pd.DataFrame({'ADX_14': [0]*100})
            def tsi(self, *args, **kwargs): return pd.Series([0]*100)
            def mfi(self, *args, **kwargs): return pd.Series([50]*100)
            def ichimoku(self, *args, **kwargs): return [pd.DataFrame(), pd.DataFrame()]
        ta = DummyTA()

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_model_trained = False

    def prepare_data(self, df):
        if df is None or df.empty: return df
        
        d = df.copy()

        try:
            # 1. BÁSICOS
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
            
            # 2. AVANZADOS (Tus peticiones)
            # TSI (True Strength Index) - Momento puro
            d['TSI'] = ta.tsi(d['close'], fast=13, slow=25)
            
            # MFI (Money Flow Index) - RSI con Volumen (Sugerencia Pro)
            d['MFI'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
            
            # ADR (Average Daily Range) aproximado en %
            # Calculamos el rango (High - Low) promedio de 14 periodos / precio
            d['RANGE_PCT'] = ((d['high'] - d['low']) / d['close']) * 100
            d['ADR'] = d['RANGE_PCT'].rolling(14).mean()

            # 3. EXTRAS
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None and not adx.empty:
                d = pd.concat([d, adx], axis=1)

            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()

        except Exception as e:
            print(f"⚠️ Error indicadores: {e}")

        return d

    def train_regime_model(self, df):
        try:
            df['ret'] = df['close'].pct_change()
            df['vol'] = df['ret'].rolling(10).std()
            future_vol = df['vol'].shift(-5)
            df['target'] = np.where(future_vol > df['vol'].quantile(0.6), 1, 0)
            
            data = df.dropna()
            if len(data) > 50:
                cols = [c for c in ['RSI', 'ADX_14', 'vol'] if c in data.columns]
                if cols:
                    self.model.fit(data[cols], data['target'])
                    self.is_model_trained = True
        except: pass

    def get_signal(self, df, context_filters):
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO"

        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        score = 0
        details = []

        # --- FILTROS ---
        if context_filters.get('use_ema'):
            if row.get('EMA_20', 0) > row.get('EMA_50', 0):
                score += 1
                details.append("EMA Alcista")
            else:
                score -= 1
                details.append("EMA Bajista")

        if context_filters.get('use_vwap'):
            if row['close'] > row.get('VWAP', 0): score += 1
            else: score -= 1

        # --- MACHINE LEARNING ---
        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            try:
                cols = [c for c in ['RSI', 'ADX_14', 'vol'] if c in df.columns]
                if cols:
                    pred = self.model.predict(df[cols].iloc[[-1]].fillna(0))[0]
                    if pred == 0:
                        score = 0
                        details.append("⛔ ML: Rango")
                        regime = "RANGO"
                    else:
                        regime = "TENDENCIA"
            except: pass

        # --- GENERACIÓN SEÑAL ---
        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        # Filtros RSI
        rsi = row.get('RSI', 50)
        if signal == "LONG" and rsi > 75: signal = "NEUTRO"
        if signal == "SHORT" and rsi < 25: signal = "NEUTRO"

        atr_val = row.get('ATR', 0)
        return signal, atr_val, details, regime

    def run_backtest_vectorized(self, df):
        return 0, 0, 0
