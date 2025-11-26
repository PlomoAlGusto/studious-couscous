import sqlite3
import pandas as pd
import logging
from config import config

class TradeManager:
    def __init__(self):
        # check_same_thread=False es vital para Streamlit
        self.conn = sqlite3.connect(config.DB_NAME, check_same_thread=False)
        self.init_db()

    def init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, symbol TEXT, type TEXT, entry REAL, size REAL,
                leverage REAL, sl REAL, tp1 REAL, tp2 REAL, tp3 REAL,
                status TEXT, pnl REAL, reason TEXT, candles_held INTEGER, atr_entry REAL
            )
        ''')
        self.conn.commit()

    def load_trades(self):
        try:
            return pd.read_sql("SELECT * FROM trades ORDER BY id DESC", self.conn)
        except:
            return pd.DataFrame()

    def add_trade(self, trade_dict):
        try:
            columns = ', '.join(trade_dict.keys())
            placeholders = ', '.join(['?'] * len(trade_dict))
            sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
            self.conn.cursor().execute(sql, list(trade_dict.values()))
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error DB: {e}")
            return False
            
    def reset_account(self):
        self.conn.cursor().execute("DELETE FROM trades")
        self.conn.commit()
