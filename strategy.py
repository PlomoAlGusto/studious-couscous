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
        except: return pd.Series([0] * len(df))

    def calculate_optimal_leverage(self, entry, sl):
        if entry == 0: return 1
        dist_pct = abs(entry - sl) / entry
        if dist_pct == 0: return 1
        safe_lev = int(0.02 / dist_pct)
        return max(1, min(safe_lev, 50))

    def detect_candles(self, df):
        df['candle_pat'] = "Normal"
        body = (df['close'] - df['open']).abs()
        wick_upper = df['high'] - df[['open', 'close']].max(axis=1)
        wick_lower = df[['open', 'close']].min(axis=1) - df['low']
        avg_body = body.rolling(10).mean()
        is_doji = body <= (avg_body * 0.1)
        is_hammer = (wick_lower > (body * 2)) & (wick_upper < body)
        df.loc[is_doji, 'candle_pat'] = "Doji ➕"
        df.loc[is_hammer, 'candle_pat'] = "Martillo 🔨"
        return df

    def prepare_data(self, df):
        if df is None or df.empty: return df
        d = df.copy()
        try:
            # Indicadores básicos
            d['EMA_20'] = ta.ema(d['close'], length=20)
            d['EMA_50'] = ta.ema(d['close'], length=50)
            d['EMA_200'] = ta.ema(d['close'], length=200) # NUEVO: Tendencia Macro
            d['RSI'] = ta.rsi(d['close'], length=14)
            d['ATR'] = ta.atr(d['high'], d['low'], d['close'], length=14)
            d['MFI'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)
            adx = ta.adx(d['high'], d['low'], d['close'], length=14)
            if adx is not None: d = pd.concat([d, adx], axis=1)
        except: pass

        d['TSI'] = self.calculate_manual_tsi(d)
        try:
            d['range_pct'] = ((d['high'] - d['low']) / d['low']) * 100
            d['ADR'] = d['range_pct'].rolling(window=14).mean()
            d['VWAP'] = (d['close'] * d['volume']).cumsum() / d['volume'].cumsum()
        except: d['ADR'] = 0; d['VWAP'] = d['close']

        # Pivotes
        high = d['high'].rolling(1).max(); low = d['low'].rolling(1).min()
        d['PIVOT'] = (high + low + d['close']) / 3
        d['R1'] = (2 * d['PIVOT']) - low; d['S1'] = (2 * d['PIVOT']) - high
        
        d = self.detect_candles(d)
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
        if df is None or len(df) < 50: return "NEUTRO", 0, [], "NEUTRO", "Sin Datos", "Normal"

        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        score = 0
        details = []

        # --- NUEVA LÓGICA HÍBRIDA ---
        
        # 1. Definir Régimen (Tendencia vs Rango)
        adx = row.get('ADX_14', 0)
        ema200 = row.get('EMA_200', 0)
        is_trending = adx > 25
        
        regime = "TENDENCIA" if is_trending else "RANGO"

        # 2. Calcular Señal según Régimen
        signal = "NEUTRO"

        if is_trending:
            # ESTRATEGIA DE TENDENCIA (Trend Following)
            # Solo operamos a favor de la EMA 200
            if row['close'] > ema200: # Tendencia Macro Alcista
                if row['EMA_20'] > row['EMA_50'] and row['close'] > row.get('VWAP', 0):
                    signal = "LONG"
                    details.append("Tendencia Alcista (EMA+VWAP)")
            
            elif row['close'] < ema200: # Tendencia Macro Bajista
                if row['EMA_20'] < row['EMA_50'] and row['close'] < row.get('VWAP', 0):
                    signal = "SHORT"
                    details.append("Tendencia Bajista (EMA+VWAP)")

        else:
            # ESTRATEGIA DE RANGO (Mean Reversion)
            # Comprar barato, Vender caro
            rsi = row.get('RSI', 50)
            mfi = row.get('MFI', 50)
            
            if rsi < 30 and mfi < 30: # Sobreventa extrema
                signal = "LONG"
                details.append("Rebote en Rango (RSI Bajo)")
            elif rsi > 70 and mfi > 70: # Sobrecompra extrema
                signal = "SHORT"
                details.append("Rechazo en Rango (RSI Alto)")

        # Estado Tendencia Visual
        trend_status = "Estable"
        if row['close'] > ema200: trend_status = "↗️ MACRO ALCISTA"
        else: trend_status = "↘️ MACRO BAJISTA"
        
        atr_val = row.get('ATR', 0)
        candle_pat = row.get('candle_pat', 'Normal')
        
        return signal, atr_val, details, regime, trend_status, candle_pat

    def check_and_execute_auto(self, db_mgr, symbol, signal, strength, price, atr, size=1000.0):
        # Lógica de ejecución automática (igual que antes)
        if strength != "DIAMOND": return False, "No Diamond"
        last_trade_time_str = db_mgr.get_last_trade_time(symbol)
        if last_trade_time_str:
            last_trade_time = datetime.strptime(last_trade_time_str, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_trade_time).total_seconds() < 3600: return False, "Cooldown"

        sl_dist = atr * 1.5
        sl = price - sl_dist if signal == "LONG" else price + sl_dist
        tp1 = price + sl_dist if signal == "LONG" else price - sl_dist
        opt_lev = self.calculate_optimal_leverage(price, sl)

        trade = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol, "type": signal, "entry": price, "size": size, "leverage": opt_lev, "sl": sl, "tp1": tp1, "tp2": 0, "tp3": 0, "status": "OPEN", "pnl": 0.0, "reason": "AUTO-DIAMOND", "candles_held": 0, "atr_entry": atr}
        saved = db_mgr.add_trade(trade)
        if saved:
            send_telegram_alert(symbol, f"{signal} (AUTO)", price, sl, tp1, opt_lev)
            return True, "Executed"
        return False, "DB Error"

    # --- BACKTEST HÍBRIDO (V2) ---
    def run_backtest_pro(self, df, initial_capital=10000, fee_pct=0.001):
        if df is None or df.empty: return None, 0, 0

        bt = df.copy()
        
        # REGLAS HÍBRIDAS PARA EL BACKTEST
        
        # 1. Condiciones Generales
        above_200 = bt['close'] > bt['EMA_200']
        strong_trend = bt['ADX_14'] > 25
        
        # 2. Señales de Tendencia (Trend Following)
        trend_long = above_200 & strong_trend & (bt['EMA_20'] > bt['EMA_50']) & (bt['close'] > bt['VWAP'])
        trend_short = (~above_200) & strong_trend & (bt['EMA_20'] < bt['EMA_50']) & (bt['close'] < bt['VWAP'])
        
        # 3. Señales de Rango (Mean Reversion)
        range_long = (~strong_trend) & (bt['RSI'] < 30) # Comprar suelo
        range_short = (~strong_trend) & (bt['RSI'] > 70) # Vender techo
        
        # Combinar Señales
        bt['signal'] = 0
        bt.loc[trend_long | range_long, 'signal'] = 1
        bt.loc[trend_short | range_short, 'signal'] = -1
        
        # Retornos y Fees
        bt['market_return'] = bt['close'].pct_change()
        bt['strategy_return'] = bt['signal'].shift(1) * bt['market_return']
        bt['trades'] = bt['signal'].diff().abs().fillna(0)
        bt['strategy_return'] = bt['strategy_return'] - (bt['trades'] * fee_pct)
        
        bt['equity'] = initial_capital * (1 + bt['strategy_return']).cumprod()
        
        final_equity = bt['equity'].iloc[-1]
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
        cummax = bt['equity'].cummax()
        drawdown = (bt['equity'] - cummax) / cummax
        max_dd = drawdown.min() * 100

        return bt, total_return_pct, max_dd