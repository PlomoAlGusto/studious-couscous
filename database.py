import sqlite3
import pandas as pd
import logging
from config import config
from datetime import datetime, timedelta

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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    type TEXT,
                    entry REAL,
                    size REAL,
                    leverage REAL,
                    sl REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    status TEXT,
                    pnl REAL,
                    reason TEXT,
                    candles_held INTEGER,
                    atr_entry REAL
                )
            ''')
            conn.commit()
        except Exception as e:
            logging.error(f"Error DB: {e}")
        finally:
            conn.close()

    def load_trades(self):
        conn = self.get_connection()
        try:
            return pd.read_sql("SELECT * FROM trades ORDER BY id DESC", conn)
        except:
            return pd.DataFrame()
        finally:
            conn.close()

    def add_trade(self, trade_dict):
        conn = self.get_connection()
        try:
            # Construcción dinámica
            columns = ', '.join(trade_dict.keys())
            placeholders = ', '.join(['?'] * len(trade_dict))
            sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
            conn.cursor().execute(sql, list(trade_dict.values()))
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error saving trade: {e}")
            return False
        finally:
            conn.close()

    def get_last_trade_time(self, symbol):
        """Devuelve la fecha del último trade para evitar duplicados en la misma vela"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT 1", (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def reset_account(self):
        conn = self.get_connection()
        conn.cursor().execute("DELETE FROM trades")
        conn.commit()
        conn.close()
