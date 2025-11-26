import pandas as pd
import pandas_ta as ta
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_model_trained = False

    def prepare_data(self, df):
        """Calcula todos los indicadores técnicos de forma vectorizada"""
        if df is None or df.empty: return df
        
        # Copia para no alterar el original
        d = df.copy()

        # 1. Tendencia
        d['EMA_20'] = ta.ema(d['close'], length=20)
        d['EMA_50'] = ta.ema(d['close'], length=50)
        
        # 2. Volatilidad y Momentum
        d['RSI'] = ta.rsi(d['close'], length=14)
        d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
        
        # 3. ADX (Fuerza de tendencia)
        adx = ta.adx(d['high'], d['low'], d['close'], length=14)
        d = pd.concat([d, adx], axis=1) # Añade ADX_14, DMP_14, DMN_14

        # 4. VWAP (Institucional)
        # Aproximación simple si no hay datos intradía precisos
        d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()

        # 5. Ichimoku (Nube base)
        ichi = ta.ichimoku(d['high'], d['low'], d['close'])[0]
        d = pd.concat([d, ichi], axis=1)

        return d

    def train_regime_model(self, df):
        """Entrena ML para detectar Régimen de Mercado (Tendencia vs Rango)"""
        # Feature Engineering para ML
        df['ret'] = df['close'].pct_change()
        df['vol'] = df['ret'].rolling(10).std()
        
        # Labeling: 1 si la volatilidad futura es alta (Tendencia), 0 si es baja (Rango)
        # Usamos shift(-5) para mirar 5 velas al futuro SOLO para entrenar
        future_vol = df['vol'].shift(-5)
        df['target'] = np.where(future_vol > df['vol'].quantile(0.6), 1, 0)
        
        data = df.dropna()
        if len(data) > 100:
            features = data[['RSI', 'ADX_14', 'vol']]
            self.model.fit(features, data['target'])
            self.is_model_trained = True

    def get_signal(self, df, context_filters):
        """
        Analiza la ÚLTIMA vela y decide: LONG, SHORT o NEUTRO.
        """
        if df is None or len(df) < 50: return "NEUTRO", 0, []

        row = df.iloc[-1]
        score = 0
        details = []

        # --- FILTRO 1: Estructura de Mercado (EMAs) ---
        if context_filters.get('use_ema'):
            if row['EMA_20'] > row['EMA_50']:
                score += 1
                details.append("EMA Alcista")
            else:
                score -= 1
                details.append("EMA Bajista")

        # --- FILTRO 2: VWAP (Precio justo) ---
        if context_filters.get('use_vwap'):
            if row['close'] > row['VWAP']:
                score += 1
            else:
                score -= 1

        # --- FILTRO 3: Machine Learning (Régimen) ---
        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            # Predecir sobre la vela actual
            last_feat = df[['RSI', 'ADX_14', 'vol']].iloc[[-1]].fillna(0)
            pred = self.model.predict(last_feat)[0]
            if pred == 0: # Rango detectado
                # Si estamos en rango, penalizamos señales de tendencia
                score = 0 
                details.append("⛔ ML: Rango Detectado")
                regime = "RANGO"
            else:
                regime = "TENDENCIA"

        # --- DECISIÓN FINAL ---
        # Umbral: Necesitamos al menos 2 puntos de confluencia
        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        # Filtro de RSI (Sobrecompra/Sobreventa)
        if signal == "LONG" and row['RSI'] > 75: signal = "NEUTRO"
        if signal == "SHORT" and row['RSI'] < 25: signal = "NEUTRO"

        return signal, row['ATR'], details, regime

    def run_backtest_vectorized(self, df):
        """Backtest ultra-rápido usando vectorización de Pandas"""
        if df is None: return 0, 0, 0
        
        df['signal'] = 0
        
        # Lógica Vectorizada (Sin bucles for)
        # Longs: EMA20 > EMA50 Y Cierre > VWAP
        df.loc[(df['EMA_20'] > df['EMA_50']) & (df['close'] > df['VWAP']), 'signal'] = 1
        
        # Shorts: EMA20 < EMA50 Y Cierre < VWAP
        df.loc[(df['EMA_20'] < df['EMA_50']) & (df['close'] < df['VWAP']), 'signal'] = -1
        
        # Calcular retornos
        df['market_return'] = df['close'].pct_change()
        df['strategy_return'] = df['signal'].shift(1) * df['market_return']
        
        total_return = df['strategy_return'].sum()
        trades_count = df['signal'].diff().abs().sum() / 2
        win_rate = len(df[df['strategy_return'] > 0]) / len(df[df['strategy_return'] != 0]) if trades_count > 0 else 0
        
        return total_return, trades_count, win_rate