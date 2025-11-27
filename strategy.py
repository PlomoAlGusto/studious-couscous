import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class StrategyManager:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def calculate_indicators(self, df):
        """Calcula indicadores técnicos usando pandas_ta-classic"""
        if df is None or df.empty: return df
        
        # Copia para no afectar el original inmediatamente
        df = df.copy()

        # 1. EMAs
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)

        # 2. RSI & ATR
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # 3. VWAP (Intento robusto)
        try:
            if 'volume' in df.columns:
                df.ta.vwap(append=True)
                if 'VWAP_D' in df.columns: df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True)
        except:
            pass
        
        # Fallback VWAP manual
        if 'VWAP' not in df.columns:
            vp = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
            df['VWAP'] = vp.cumsum() / df['volume'].cumsum()

        # 4. ADX
        try:
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None:
                df = pd.concat([df, adx], axis=1)
                # Normalizar nombre de columna
                if 'ADX_14' not in df.columns and 'ADX' in df.columns: 
                    df['ADX_14'] = df['ADX']
        except:
            df['ADX_14'] = 0

        # 5. MFI & TSI
        df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        try:
            tsi = ta.tsi(df['close'], fast=13, slow=25)
            if tsi is not None:
                df = pd.concat([df, tsi], axis=1)
                tsi_col = [c for c in df.columns if 'TSI' in c][0]
                df['TSI'] = df[tsi_col]
        except:
            df['TSI'] = 0

        return df.fillna(method='bfill').fillna(method='ffill')

    def train_regime_model(self, df):
        """Entrena el modelo de Machine Learning para detectar régimen"""
        if len(df) < 100: return
        
        df_ml = df.copy()
        df_ml['return'] = df_ml['close'].pct_change()
        df_ml['vol'] = df_ml['return'].rolling(14).std()
        
        # Etiquetado: 1 (Tendencia Alcista), -1 (Bajista), 0 (Rango/Ruido)
        df_ml['label'] = np.where(df_ml['return'].shift(-10) > df_ml['vol'], 1,
                                 np.where(df_ml['return'].shift(-10) < -df_ml['vol'], -1, 0))
        df_ml = df_ml.dropna()

        features = ['vol', 'ADX_14', 'RSI']
        valid_features = [f for f in features if f in df_ml.columns]
        
        if not valid_features: return

        X = df_ml[valid_features]
        y = df_ml['label']
        
        try:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            self.model.fit(self.scaler.fit_transform(X), y.astype('int'))
        except Exception as e:
            print(f"Error entrenando modelo: {e}")
            self.model = None

    def get_signal(self, df, filters):
        """Genera la señal final basada en la estrategia"""
        row = df.iloc[-1]
        
        # Inicializar
        signal = "NEUTRO"
        score = 0
        max_score = 0
        details = []
        regime = 0

        # Predicción ML
        if self.model:
            try:
                vol = df['close'].pct_change().tail(14).std()
                feats = [[vol, row.get('ADX_14', 0), row.get('RSI', 50)]]
                regime = self.model.predict(self.scaler.transform(feats))[0]
            except:
                pass

        # Lógica de Puntuación
        if filters.get('use_ema'):
            max_score += 1
            if row['EMA_20'] > row['EMA_50']:
                score += 1
                details.append("EMA: Bull")
            else:
                score -= 1
                details.append("EMA: Bear")

        if filters.get('use_vwap') and 'VWAP' in row:
            max_score += 1
            if row['close'] > row['VWAP']:
                score += 1
                details.append("VWAP: Bull")
            else:
                score -= 1
                details.append("VWAP: Bear")

        if filters.get('use_tsi') and 'TSI' in row:
            max_score += 1
            if row['TSI'] > 0:
                score += 1
                details.append("TSI: Bull")
            else:
                score -= 1
                details.append("TSI: Bear")

        # Decisión Final
        threshold = max_score * 0.4
        if score > threshold: signal = "LONG"
        elif score < -threshold: signal = "SHORT"

        # Vetos
        if filters.get('use_regime') and row.get('ADX_14', 0) < 20:
            signal = "NEUTRO"
            details.append("VETO: ADX Bajo")

        if filters.get('use_rsi'):
            if row['RSI'] > 70 and signal == "LONG": signal = "NEUTRO"
            if row['RSI'] < 30 and signal == "SHORT": signal = "NEUTRO"

        return signal, row['ATR'], details, regime