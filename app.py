import asyncio
import logging
import uuid
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque

import numpy as np
import pandas as pd

# --- CONFIGURACIÓN DE NIVEL INSTITUCIONAL ---

# Configuración de Logging con formato preciso para debuggeo de latencia
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(threadName)s | %(module)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SENTINEL_HFT")

# Constantes de Configuración
MAX_HISTORY_TICKS = 1000  # Tamaño del buffer de memoria para indicadores
RISK_PER_TRADE = 0.01     # 1% del capital por operación
MAX_OPEN_POSITIONS = 3
SIMULATED_LATENCY_MS = 50 # Latencia de red simulada

# --- ESTRUCTURAS DE DATOS (Data Models) ---

class Side(Enum):
    BUY = 1
    SELL = -1

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

@dataclass
class Ticker:
    symbol: str
    timestamp: float
    bid: float
    ask: float
    last: float
    volume: float

@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    quantity: float
    price: Optional[float]
    order_type: OrderType
    timestamp: float
    status: str = "PENDING"

@dataclass
class Position:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

# --- 1. MÓDULO DE MATEMÁTICAS FINANCIERAS (QuantLib Core) ---

class QuantLib:
    """
    Librería estática para cálculos vectorizados de alta velocidad.
    Usamos NumPy para evitar la lentitud de Python nativo.
    """
    
    @staticmethod
    def rsi(prices: np.array, period: int = 14) -> float:
        if len(prices) < period + 1: return 50.0
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100. - 100. / (1. + rs)
        
        # Smoothing (Wilder's method) para los siguientes
        for i in range(period + 1, len(prices) - 1):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi = 100. - 100. / (1. + rs)
        return rsi

    @staticmethod
    def bollinger_bands(prices: np.array, window: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        if len(prices) < window: return 0.0, 0.0, 0.0
        # Slicing de los últimos 'window' datos
        slice_data = prices[-window:]
        sma = np.mean(slice_data)
        std = np.std(slice_data)
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    @staticmethod
    def volatility_ratio(prices: np.array, window: int = 20) -> float:
        """Calcula volatilidad relativa para ajuste de riesgo"""
        if len(prices) < window: return 1.0
        log_rets = np.log(prices[1:] / prices[:-1])
        vol = np.std(log_rets[-window:])
        return vol

# --- 2. MOTOR DE CONEXIÓN (Exchange Connector) ---

class ExchangeConnector:
    """
    Simula la conexión con el Exchange (Binance/Coinbase).
    En producción, esto wrappea 'ccxt' o websockets directos.
    """
    def __init__(self):
        self.connected = False
        self.account_balance = 100000.0 # $100k USD
        self.latency_simulator = SIMULATED_LATENCY_MS / 1000.0

    async def connect(self):
        logger.info("Iniciando Handshake con Exchange API...")
        await asyncio.sleep(1)
        self.connected = True
        logger.info("Conexión WebSocket Establecida [Latencia: 12ms]")

    async def fetch_ticker(self, symbol: str) -> Ticker:
        # Simulación de movimiento Browniano geométrico para precios
        await asyncio.sleep(random.uniform(0.001, 0.01)) # Micro-latencia
        base_price = 45000.0 if symbol == "BTCUSDT" else 3000.0
        noise = np.random.normal(0, 2)
        price = base_price + noise
        
        return Ticker(
            symbol=symbol,
            timestamp=time.time(),
            bid=price - 0.5, # Spread
            ask=price + 0.5,
            last=price,
            volume=random.uniform(0.1, 5.0)
        )

    async def send_order(self, order: Order) -> bool:
        """Simula envío y matching engine"""
        await asyncio.sleep(self.latency_simulator) 
        # Simulación de Slippage (Deslizamiento)
        slippage = random.uniform(0, 5) if order.order_type == OrderType.MARKET else 0
        exec_price = order.price + slippage if order.side == Side.BUY else order.price - slippage
        
        cost = exec_price * order.quantity
        if order.side == Side.BUY:
            if self.account_balance >= cost:
                self.account_balance -= cost
                logger.info(f"ORDEN COMPLETADA: COMPRA {order.quantity} {order.symbol} @ {exec_price:.2f} (Slippage: {slippage:.2f})")
                return True
        else:
            self.account_balance += cost
            logger.info(f"ORDEN COMPLETADA: VENTA {order.quantity} {order.symbol} @ {exec_price:.2f}")
            return True
        return False

# --- 3. SISTEMA CENTRAL (Sentinel Core) ---

class SentinelCore:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.exchange = ExchangeConnector()
        self.running = False
        
        # Buffers de datos para cálculo de indicadores (optimizados con deque)
        self.data_stream: Dict[str, deque] = {s: deque(maxlen=MAX_HISTORY_TICKS) for s in symbols}
        
        # Gestión de Estado
        self.positions: Dict[str, Optional[Position]] = {s: None for s in symbols}
        self.active_orders: List[Order] = []
        
        # Colas de Eventos para procesamiento asíncrono
        self.market_event_queue = asyncio.Queue()
        self.signal_queue = asyncio.Queue()

    # --- MÓDULO DE GESTIÓN DE RIESGO ---
    def risk_check(self, symbol: str, price: float) -> float:
        """
        Determina el tamaño de la posición basado en el balance actual y volatilidad.
        Retorna 0.0 si el riesgo es muy alto.
        """
        balance = self.exchange.account_balance
        
        # 1. Check de posiciones máximas
        active_count = sum(1 for p in self.positions.values() if p is not None)
        if active_count >= MAX_OPEN_POSITIONS:
            return 0.0

        # 2. Cálculo de tamaño por riesgo fijo
        risk_amount = balance * RISK_PER_TRADE
        # Asumimos un Stop Loss del 2% para calcular el tamaño
        stop_loss_distance = price * 0.02 
        position_size = risk_amount / stop_loss_distance
        
        return round(position_size, 4)

    # --- MÓDULO DE ESTRATEGIA (Mean Reversion) ---
    async def strategy_engine(self):
        """Consumidor: Analiza datos y genera señales"""
        logger.info("Iniciando Motor de Estrategia...")
        while self.running:
            ticker: Ticker = await self.market_event_queue.get()
            symbol = ticker.symbol
            
            # Actualizar buffer de precios
            self.data_stream[symbol].append(ticker.last)
            
            # Necesitamos suficientes datos para calcular indicadores
            if len(self.data_stream[symbol]) < 50:
                self.market_event_queue.task_done()
                continue

            # Conversión a NumPy para velocidad (Vectorización)
            np_prices = np.array(self.data_stream[symbol])
            
            # --- CÁLCULO DE INDICADORES (SOTA) ---
            rsi = QuantLib.rsi(np_prices)
            upper_bb, mid_bb, lower_bb = QuantLib.bollinger_bands(np_prices)
            volatility = QuantLib.volatility_ratio(np_prices)

            # --- LÓGICA DE ENTRADA ---
            # Condición: RSI Sobrevendido (<30) Y Precio rompe banda inferior Y Volatilidad estable
            if self.positions[symbol] is None:
                if rsi < 30 and ticker.last < lower_bb and volatility < 0.02:
                    logger.info(f"SEÑAL DETECTADA [{symbol}]: RSI={rsi:.2f}, Price < LowerBB")
                    await self.signal_queue.put((symbol, Side.BUY, ticker.ask))
            
            # --- LÓGICA DE SALIDA ---
            # Condición: RSI Sobrecomprado (>70) O Precio toca banda superior (Mean Reversion)
            elif self.positions[symbol] is not None:
                if rsi > 70 or ticker.last > upper_bb:
                    logger.info(f"SEÑAL DE SALIDA [{symbol}]: Take Profit trigger.")
                    await self.signal_queue.put((symbol, Side.SELL, ticker.bid))

            self.market_event_queue.task_done()

    # --- MÓDULO DE EJECUCIÓN ---
    async def execution_engine(self):
        """Consumidor: Procesa señales y ejecuta órdenes con gestión de riesgo"""
        logger.info("Iniciando Motor de Ejecución...")
        while self.running:
            symbol, side, price = await self.signal_queue.get()
            
            if side == Side.BUY:
                qty = self.risk_check(symbol, price)
                if qty > 0:
                    order = Order(str(uuid.uuid4()), symbol, side, qty, price, OrderType.MARKET, time.time())
                    success = await self.exchange.send_order(order)
                    if success:
                        self.positions[symbol] = Position(symbol, side, qty, price)
                else:
                    logger.warning(f"ORDEN RECHAZADA POR RIESGO: {symbol}")

            elif side == Side.SELL:
                pos = self.positions[symbol]
                if pos:
                    order = Order(str(uuid.uuid4()), symbol, side, pos.quantity, price, OrderType.MARKET, time.time())
                    success = await self.exchange.send_order(order)
                    if success:
                        pnl = (price - pos.entry_price) * pos.quantity
                        logger.info(f"POSICIÓN CERRADA. PnL: ${pnl:.2f}")
                        self.positions[symbol] = None # Liberar slot

            self.signal_queue.task_done()

    # --- INGESTA DE DATOS (Data Feed) ---
    async def data_feed_handler(self):
        """Productor: Simula WebSockets recibiendo datos en tiempo real"""
        while self.running:
            for symbol in self.symbols:
                ticker = await self.exchange.fetch_ticker(symbol)
                await self.market_event_queue.put(ticker)
                
                # Actualizar PnL visual de posiciones abiertas
                if self.positions[symbol]:
                    curr_pos = self.positions[symbol]
                    curr_pos.current_price = ticker.last
                    curr_pos.unrealized_pnl = (ticker.last - curr_pos.entry_price) * curr_pos.quantity
            
            # Control de frecuencia (simular 10 ticks por segundo aprox)
            await asyncio.sleep(0.1)

    # --- ORQUESTACIÓN PRINCIPAL ---
    async def start(self):
        await self.exchange.connect()
        self.running = True
        
        # Creamos las tareas asíncronas que correrán en paralelo
        tasks = [
            asyncio.create_task(self.data_feed_handler()),
            asyncio.create_task(self.strategy_engine()),
            asyncio.create_task(self.execution_engine())
        ]
        
        logger.info(f"SENTINEL ACTIVO. Capital: ${self.exchange.account_balance:.2f}")
        logger.info("Esperando datos de mercado para llenar buffers...")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Deteniendo servicios...")

# --- EJECUCIÓN ---

if __name__ == "__main__":
    bot = SentinelCore(symbols=["BTCUSDT", "ETHUSDT"])
    try:
        # Ejecución en el Event Loop principal
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nDetención manual solicitada por el usuario.")
