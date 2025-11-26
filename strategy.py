import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- SOLUCIÓN DEL CONFLICTO DE NOMBRES ---
try:
    import pandas_ta_classic as ta
except ImportError:
    try:
        import pandas_ta as ta
    except ImportError:
        st.error("❌ Error crítico: No se encuentra la librería 'pandas_ta' ni 'pandas_ta_classic'.")
# -----------------------------------------

class StrategyManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_model_trained = False

    def prepare_data(self, df):
        if df is None or df.empty: return df
        
        d = df.copy()

        # Usamos 'ta' que ahora apunta a la librería correcta
        d['EMA_20'] = ta.ema(d['close'], length=20)
        d['EMA_50'] = ta.ema(d['close'], length=50)
        d['RSI'] = ta.rsi(d['close'], length=14)
        d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
        
        adx = ta.adx(d['high'], d['low'], d['close'], length=14)
        d = pd.concat([d, adx], axis=1)

        d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()

        ichi = ta.ichimoku(d['high'], d['low'], d['close'])[0]
        d = pd.concat([d, ichi], axis=1)

        return d

    def train_regime_model(self, df):
        df['ret'] = df['close'].pct_change()
        df['vol'] = df['ret'].rolling(10).std()
        
        future_vol = df['vol'].shift(-5)
        df['target'] = np.where(future_vol > df['vol'].quantile(0.6), 1, 0)
        
        data = df.dropna()
        if len(data) > 100:
            features = data[['RSI', 'ADX_14', 'vol']]
            self.model.fit(features, data['target'])
            self.is_model_trained = True

    def get_signal(self, df, context_filters):
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO"

        row = df.iloc[-1]
        score = 0
        details = []

        if context_filters.get('use_ema'):
            if row['EMA_20'] > row['EMA_50']:
                score += 1
                details.append("EMA Alcista")
            else:
                score -= 1
                details.append("EMA Bajista")

        if context_filters.get('use_vwap'):
            if row['close'] > row['VWAP']:
                score += 1
            else:
                score -= 1

        regime = "NEUTRO"
        if self.is_model_trained and context_filters.get('use_regime'):
            last_feat = df[['RSI', 'ADX_14', 'vol']].iloc[[-1]].fillna(0)
            pred = self.model.predict(last_feat)[0]
            if pred == 0: 
                score = 0 
                details.append("⛔ ML: Rango Detectado")
                regime = "RANGO"
            else:
                regime = "TENDENCIA"

        signal = "NEUTRO"
        if score >= 2: signal = "LONG"
        elif score <= -2: signal = "SHORT"

        if signal == "LONG" and row['RSI'] > 75: signal = "NEUTRO"
        if signal == "SHORT" and row['RSI'] < 25: signal = "NEUTRO"

        return signal, row['ATR'], details, regime

    def run_backtest_vectorized(self, df):
        if df is None: return 0, 0, 0
        
        df['signal'] = 0
        df.loc[(df['EMA_20'] > df['EMA_50']) & (df['close'] > df['VWAP']), 'signal'] = 1
        df.loc[(df['EMA_20'] < df['EMA_50']) & (df['close'] < df['VWAP']), 'signal'] = -1
        
        df['market_return'] = df['close'].pct_change()
        df['strategy_return'] = df['signal'].shift(1) * df['market_return']
        
        total_return = df['strategy_return'].sum()
        trades_count = df['signal'].diff().abs().sum() / 2
        
        winning_trades = len(df[df['strategy_return'] > 0])
        total_trades_nonzero = len(df[df['strategy_return'] != 0])
        win_rate = winning_trades / total_trades_nonzero if total_trades_nonzero > 0 else 0
        
        return total_return, trades_count, win_rate