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
        # Dummy mínimo por si falla la librería, pero calcularemos cosas a mano
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

    def prepare_data(self, df):
        if df is None or df.empty: return df
        
        d = df.copy()

        try:
            # --- INDICADORES DE LIBRERÍA ---
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
            d['TSI'] = ta.tsi(d['close'], fast=13, slow=25)
            d['MFI'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
            
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None and not adx.empty:
                d = pd.concat([d, adx], axis=1)

            # --- CÁLCULOS MANUALES (INFALIBLES) ---
            # ADR: (High - Low) / Low * 100 -> Promedio 14 velas
            d['candle_range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['candle_range_pct'].rolling(window=14).mean()

            # VWAP Manual
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()

        except Exception as e:
            print(f"⚠️ Error indicadores: {e}")

        # Limpiar NaNs iniciales para que no den 0.00
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
        prev_row = df.iloc[-2] # Vela anterior para detectar cruces
        score = 0
        details = []

        # --- ESTRATEGIA ---
        if context_filters.get('use_ema'):
            if row['EMA_20'] > row['EMA_50']:
                score += 1
                details.append("EMA Alcista")
            else:
                score -= 1
                details.append("EMA Bajista")

        if context_filters.get('use_vwap'):
            if row['close'] > row['VWAP']: score += 1
            else: score -= 1

        # --- RÉGIMEN ---
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

        # --- SEÑAL FINAL ---
        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        # Filtro RSI
        if signal == "LONG" and row['RSI'] > 75: signal = "NEUTRO"
        if signal == "SHORT" and row['RSI'] < 25: signal = "NEUTRO"

        # --- ANÁLISIS DE CAMBIO DE TENDENCIA (NUEVO) ---
        # Detectamos si las medias se acaban de cruzar
        bull_cross = prev_row['EMA_20'] <= prev_row['EMA_50'] and row['EMA_20'] > row['EMA_50']
        bear_cross = prev_row['EMA_20'] >= prev_row['EMA_50'] and row['EMA_20'] < row['EMA_50']
        
        trend_status = "Sin Cambios"
        if bull_cross: trend_status = "🔄 GIRO ALCISTA CONFIRMADO"
        elif bear_cross: trend_status = "🔄 GIRO BAJISTA CONFIRMADO"
        elif signal == "LONG" and row['close'] < row['EMA_20']: trend_status = "⚠️ POSIBLE DEBILIDAD"
        elif signal == "SHORT" and row['close'] > row['EMA_20']: trend_status = "⚠️ POSIBLE REBOTE"
        elif regime == "TENDENCIA": trend_status = "✅ TENDENCIA FUERTE"
        elif regime == "RANGO": trend_status = "💤 LATERAL / RANGO"

        atr_val = row['ATR']
        return signal, atr_val, details, regime, trend_status
