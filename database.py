import sqlite3
import pandas as pd
import logging
from config import config

class TradeManager:
    def __init__(self):
        self.db_path = config.DB_NAME
        self.init_db()

    def get_connection(self):
        # check_same_thread=False es necesario para Streamlit
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_db(self):
        """Crea la tabla si no existe"""
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
            logging.error(f"Error inicializando DB: {e}")
        finally:
            conn.close()

    def load_trades(self):
        """Devuelve un DataFrame con todos los trades"""
        conn = self.get_connection()
        try:
            df = pd.read_sql("SELECT * FROM trades ORDER BY id DESC", conn)
            return df
        except Exception as e:
            logging.error(f"Error cargando trades: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def add_trade(self, trade_dict):
        """Inserta un nuevo trade de forma segura"""
        conn = self.get_connection()
        try:
            columns = ', '.join(trade_dict.keys())
            placeholders = ', '.join(['?'] * len(trade_dict))
            sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
            cursor = conn.cursor()
            cursor.execute(sql, list(trade_dict.values()))
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error guardando trade: {e}")
            return False
        finally:
            conn.close()

    def reset_account(self):
        """Borra todos los trades (Reset)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades")
        conn.commit()
        conn.close()