import sqlite3
import pandas as pd
import logging
from config import config
from datetime import datetime

class TradeManager:
    def __init__(self):
        self.db_path = config.DB_NAME
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_db(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # Añadimos columnas de salida si no existen (para versiones viejas)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, type TEXT, entry REAL, size REAL,
                    leverage REAL, sl REAL, tp1 REAL, tp2 REAL, tp3 REAL,
                    status TEXT, pnl REAL, reason TEXT, candles_held INTEGER, atr_entry REAL,
                    exit_price REAL, exit_time TEXT
                )
            ''')
            try:
                cursor.execute("ALTER TABLE trades ADD COLUMN exit_price REAL")
                cursor.execute("ALTER TABLE trades ADD COLUMN exit_time TEXT")
            except: pass # Ya existen
            conn.commit()
        except Exception as e:
            logging.error(f"Error DB: {e}")
        finally:
            conn.close()

    def load_trades(self):
        conn = self.get_connection()
        try:
            return pd.read_sql("SELECT * FROM trades ORDER BY id DESC", conn)
        except: return pd.DataFrame()
        finally: conn.close()

    def add_trade(self, trade_dict):
        conn = self.get_connection()
        try:
            columns = ', '.join(trade_dict.keys())
            placeholders = ', '.join(['?'] * len(trade_dict))
            sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
            conn.cursor().execute(sql, list(trade_dict.values()))
            conn.commit()
            return True
        except: return False
        finally: conn.close()

    def get_last_trade_time(self, symbol):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT 1", (symbol,))
            res = cursor.fetchone()
            return res[0] if res else None
        finally: conn.close()

    def reset_account(self):
        conn = self.get_connection()
        conn.cursor().execute("DELETE FROM trades")
        conn.commit()
        conn.close()

    # --- NUEVO: SISTEMA DE CIERRE AUTOMÁTICO (TP/SL) ---
    def check_sl_tp(self, current_price, symbol):
        """Revisa trades abiertos y los cierra si tocan SL o TP"""
        conn = self.get_connection()
        closed_trades = []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?", (symbol,))
            # Convertir a lista de dicts
            cols = [description[0] for description in cursor.description]
            open_trades = [dict(zip(cols, row)) for row in cursor.fetchall()]

            for t in open_trades:
                reason = None
                pnl = 0.0
                
                # Lógica LONG
                if t['type'] == 'LONG':
                    if current_price <= t['sl']: reason = "STOP LOSS 🛑"
                    elif current_price >= t['tp3']: reason = "TP 3 (MOON) 🚀"
                    elif current_price >= t['tp2'] and t['reason'] != "TP2 Hit": reason = "TP 2 💰" # Simplificado
                    elif current_price >= t['tp1'] and t['reason'] != "TP1 Hit": reason = "TP 1 ✅" # Simplificado
                    
                    if reason:
                        pnl = ((current_price - t['entry']) / t['entry']) * t['size'] * t['leverage']

                # Lógica SHORT
                elif t['type'] == 'SHORT':
                    if current_price >= t['sl']: reason = "STOP LOSS 🛑"
                    elif current_price <= t['tp3']: reason = "TP 3 (MOON) 🚀"
                    elif current_price <= t['tp2']: reason = "TP 2 💰"
                    elif current_price <= t['tp1']: reason = "TP 1 ✅"
                    
                    if reason:
                        pnl = ((t['entry'] - current_price) / t['entry']) * t['size'] * t['leverage']

                if reason:
                    # Actualizar DB
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute("""
                        UPDATE trades SET status='CLOSED', pnl=?, reason=?, exit_price=?, exit_time=? 
                        WHERE id=?
                    """, (pnl, reason, current_price, now, t['id']))
                    conn.commit()
                    
                    t['pnl'] = pnl
                    t['reason'] = reason
                    closed_trades.append(t)

        except Exception as e:
            print(f"Error checking SL/TP: {e}")
        finally:
            conn.close()
        
        return closed_trades
