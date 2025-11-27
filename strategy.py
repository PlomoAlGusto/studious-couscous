import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- IMPORTACIÓN SEGURA DE LIBRERÍA TÉCNICA (Auto-detect) ---
try:
    # Intento 1: Nombre original
    import pandas_ta as ta
except ImportError:
    try:
        # Intento 2: Nombre de la versión Classic
        import pandas_ta_classic as ta
    except ImportError:
        # Intento 3: Fallo total (No detener app, usar dummy)
        st.error("⚠️ Error: Librería de Análisis Técnico no encontrada. Revisa requirements.txt")
        class DummyTA:
            def ema(self, *args, **kwargs): return pd.Series([0]*100)
            def rsi(self, *args, **kwargs): return pd.Series([50]*100)
            def atr(self, *args, **kwargs): return pd.Series([1]*100)
            def adx(self, *args, **kwargs): return pd.DataFrame({'ADX_14': [0]*100})
            def ichimoku(self, *args, **kwargs): return [pd.DataFrame(), pd.DataFrame()]
        ta = DummyTA()
# ------------------------------------------------------------

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_model_trained = False

    def prepare_data(self, df):
        if df is None or df.empty: return df
        
        d = df.copy()

        # Indicadores usando el alias 'ta' (que ya sabemos que funciona)
        try:
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
            
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None and not adx.empty:
                d = pd.concat([d, adx], axis=1)

            # VWAP Manual (Por si la librería falla)
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()

            ichi = ta.ichimoku(d['high'], d['low'], d['close'])
            if ichi and len(ichi) > 0:
                d = pd.concat([d, ichi[0]], axis=1)
                
        except Exception as e:
            print(f"Error calculando indicadores: {e}")

        return d

    def train_regime_model(self, df):
        try:
            df['ret'] = df['close'].pct_change()
            df['vol'] = df['ret'].rolling(10).std()
            
            future_vol = df['vol'].shift(-5)
            df['target'] = np.where(future_vol > df['vol'].quantile(0.6), 1, 0)
            
            data = df.dropna()
            if len(data) > 100:
                features = data[['RSI', 'ADX_14', 'vol']]
                self.model.fit(features, data['target'])
                self.is_model_trained = True
        except: pass

    def get_signal(self, df, context_filters):
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO"

        row = df.iloc[-1]
        score = 0
        details = []

        if context_filters.get('use_ema'):
            if row.get('EMA_20', 0) > row.get('EMA_50', 0):
                score += 1
                details.append("EMA Alcista")
            else:
                score -= 1
                details.append("EMA Bajista")

        if context_filters.get('use_vwap'):
            if row['close'] > row.get('VWAP', 0):
                score += 1
            else:
                score -= 1

        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            try:
                last_feat = df[['RSI', 'ADX_14', 'vol']].iloc[[-1]].fillna(0)
                pred = self.model.predict(last_feat)[0]
                if pred == 0: 
                    score = 0 
                    details.append("⛔ ML: Rango Detectado")
                    regime = "RANGO"
                else:
                    regime = "TENDENCIA"
            except: pass

        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        # Filtro RSI
        rsi_val = row.get('RSI', 50)
        if signal == "LONG" and rsi_val > 75: signal = "NEUTRO"
        if signal == "SHORT" and rsi_val < 25: signal = "NEUTRO"

        atr_val = row.get('ATR', 0)
        return signal, atr_val, details, regime

    def run_backtest_vectorized(self, df):
        return 0, 0, 0 # Desactivado temporalmente