"""
execution/trader.py
-----------------------------------------------------------------
Order Executor
-----------------------------------------------------------------
Handles:
  - Opening long/short positions (market orders)
  - Stop-loss and take-profit management (checked internally)
  - Trailing stop logic
  - Sends real orders to Binance Futures Testnet via direct HTTP
  - Position tracking
-----------------------------------------------------------------
"""

import logging
import hmac
import hashlib
import time
import urllib.request
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Quantity precision per symbol (Binance lot size filters)
QUANTITY_PRECISION = {
    "BTCUSDT": 3, "ETHUSDT": 3, "BNBUSDT": 2, "XRPUSDT": 1,
    "DOGEUSDT": 0, "ADAUSDT": 0, "SOLUSDT": 1,
    "DOTUSDT": 1, "LINKUSDT": 1, "AVAXUSDT": 0, "1000SHIBUSDT": 0,
    "ATOMUSDT": 2, "NEARUSDT": 0, "APTUSDT": 1, "ARBUSDT": 1,
    "OPUSDT": 1, "SUIUSDT": 1, "INJUSDT": 1, "FETUSDT": 0,
    "JUPUSDT": 0, "DYDXUSDT": 1, "AAVEUSDT": 1, "RENDERUSDT": 1,
    "SEIUSDT": 0, "WLDUSDT": 0, "XLMUSDT": 0, "HBARUSDT": 0,
    "1000PEPEUSDT": 0, "1000BONKUSDT": 0, "1000FLOKIUSDT": 0, "TRXUSDT": 0,
    "ENAUSDT": 1, "LDOUSDT": 1, "MANAUSDT": 0, "KAVAUSDT": 1,
    "ALGOUSDT": 0, "RUNEUSDT": 1, "GALAUSDT": 0, "FTMUSDT": 0, "MKRUSDT": 3,
}

# Symbols that need 1000x prefix on Binance Futures
# e.g. "PEPE/USDT" trades as "1000PEPEUSDT" (price * 1000, qty / 1000)
FUTURES_1000X_SYMBOLS = {"PEPE", "SHIB", "FLOKI", "BONK"}

POSITIONS_FILE = "logs/positions.json"


@dataclass
class Position:
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    units: float
    stop_loss: float        # absolute stop price (kept for reference)
    take_profit: float      # absolute TP price (NOT used for exit — backtest match)
    atr: float
    trail_atr_mult: float = 2.2       # ATR-based trailing (matches backtest)
    highest_since_entry: Optional[float] = None
    lowest_since_entry: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def stop_loss_atr_dist(self) -> float:
        """Distance from entry to fixed stop in price units."""
        return abs(self.entry_price - self.stop_loss)


class OrderExecutor:
    """
    Manages order execution and position tracking.
    Sends real orders to Binance Futures Testnet.
    Manages stops internally (checks each scan cycle).
    Uses ATR-based trailing stops (matches backtest logic).
    Persists positions to disk for crash recovery.
    """

    TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(
        self,
        exchange,
        paper_trading: bool = True,
        fee_rate: float = 0.001,
    ):
        self.exchange = exchange
        self.paper_trading = paper_trading
        self.fee_rate = fee_rate
        self.positions: dict[str, Position] = {}

        # Extract API keys from exchange for direct HTTP calls
        self.api_key = getattr(exchange, 'apiKey', '')
        self.api_secret = getattr(exchange, 'secret', '')

        # Load persisted positions
        self._load_positions()

    @staticmethod
    def _to_api_symbol_qty(symbol: str, qty: float) -> tuple[str, float]:
        """Convert symbol to Binance Futures API format.
        Handles 1000x tokens: PEPE/USDT -> 1000PEPEUSDT (qty / 1000).
        """
        base = symbol.split("/")[0]
        api_symbol = symbol.replace("/", "")
        if base in FUTURES_1000X_SYMBOLS:
            api_symbol = "1000" + api_symbol
            qty = qty / 1000.0
        return api_symbol, qty

    def _testnet_order(self, params: dict) -> dict:
        """Send a signed order to Binance Futures Testnet."""
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        url = f"{self.TESTNET_URL}/fapi/v1/order?{query}&signature={signature}"

        req = urllib.request.Request(
            url, data=b"", method="POST",
            headers={"X-MBX-APIKEY": self.api_key}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                return result
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            logger.error(f"Testnet order error HTTP {e.code}: {error_body}")
            return {"error": error_body, "code": e.code}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    # ---- OPEN POSITIONS ----

    def open_long(
        self,
        symbol: str,
        units: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        trail_atr_mult: float = 2.2,
    ) -> dict:
        """Open a long position."""
        return self._open_position(
            symbol, "LONG", units, entry_price, stop_loss, take_profit, atr,
            trail_atr_mult,
        )

    def open_short(
        self,
        symbol: str,
        units: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        trail_atr_mult: float = 2.2,
    ) -> dict:
        """Open a short position."""
        return self._open_position(
            symbol, "SHORT", units, entry_price, stop_loss, take_profit, atr,
            trail_atr_mult,
        )

    def _open_position(
        self,
        symbol: str,
        direction: str,
        units: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        trail_atr_mult: float = 2.2,
    ) -> dict:
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}, skipping")
            return {"status": "skipped", "reason": "position_exists"}

        # Convert symbol format: "BTC/USDT" -> "BTCUSDT"
        api_symbol, api_qty = self._to_api_symbol_qty(symbol, units)
        side = "BUY" if direction == "LONG" else "SELL"

        # Round quantity to proper precision for this symbol
        precision = QUANTITY_PRECISION.get(api_symbol, 1)
        api_qty = round(api_qty, precision)
        if api_qty <= 0:
            logger.warning(f"[{symbol}] Quantity rounds to 0 (precision={precision})")
            return {"status": "error", "reason": "quantity_too_small"}

        # Send real order to testnet
        result = self._testnet_order({
            "symbol": api_symbol,
            "side": side,
            "type": "MARKET",
            "quantity": api_qty,
        })

        if "error" in result:
            logger.error(f"[TESTNET] Order failed for {symbol}: {result['error']}")
            return {"status": "error", "reason": result["error"]}

        order_id = result.get("orderId", "?")
        status = result.get("status", "?")
        logger.info(
            f"[TESTNET] {direction} {symbol}: {units} units @ ~{entry_price:.2f} | "
            f"orderId={order_id} status={status} | "
            f"SL={stop_loss:.4f}"
        )

        pos = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            trail_atr_mult=trail_atr_mult,
            highest_since_entry=entry_price,
            lowest_since_entry=entry_price,
        )
        self.positions[symbol] = pos
        self._save_positions()

        return {
            "status": "opened",
            "symbol": symbol,
            "direction": direction,
            "entry": entry_price,
            "units": units,
            "order_id": order_id,
        }

    # ---- STOP / TP CHECKS ----

    def check_stops(self, symbol: str, current_price: float) -> Optional[dict]:
        """
        Check if current price hits stop-loss or trailing ATR stop.
        Matches backtest logic: NO take-profit exit, only trailing stop.
        Returns close_result dict if position is closed, else None.
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        # Update trailing stop (ATR-based, matching backtest)
        self._update_trailing_stop(pos, current_price)

        closed = False
        close_reason = ""

        if pos.direction == "LONG":
            # Fixed stop
            fixed_stop = pos.entry_price - pos.stop_loss_atr_dist
            # Trailing stop
            trail_stop = (pos.highest_since_entry or pos.entry_price) - pos.trail_atr_mult * pos.atr
            # Use the TIGHTER (higher) of the two — matches backtest
            effective_stop = max(fixed_stop, trail_stop)
            if current_price <= effective_stop:
                closed = True
                close_reason = "trailing_stop" if trail_stop >= fixed_stop else "stop_loss"

        else:  # SHORT
            fixed_stop = pos.entry_price + pos.stop_loss_atr_dist
            trail_stop = (pos.lowest_since_entry or pos.entry_price) + pos.trail_atr_mult * pos.atr
            effective_stop = min(fixed_stop, trail_stop)
            if current_price >= effective_stop:
                closed = True
                close_reason = "trailing_stop" if trail_stop <= fixed_stop else "stop_loss"

        if closed:
            return self._close_position(symbol, current_price, close_reason)

        return None

    def _update_trailing_stop(self, pos: Position, current_price: float):
        """Update highest/lowest price since entry for ATR trailing stop."""
        if pos.direction == "LONG":
            if pos.highest_since_entry is None or current_price > pos.highest_since_entry:
                pos.highest_since_entry = current_price
        else:
            if pos.lowest_since_entry is None or current_price < pos.lowest_since_entry:
                pos.lowest_since_entry = current_price
        self._save_positions()

    def _close_position(
        self, symbol: str, exit_price: float, reason: str
    ) -> dict:
        """Close a position and calculate PnL."""
        pos = self.positions.pop(symbol)
        self._save_positions()

        # Send close order to testnet
        api_symbol, api_qty = self._to_api_symbol_qty(symbol, pos.units)
        close_side = "SELL" if pos.direction == "LONG" else "BUY"

        result = self._testnet_order({
            "symbol": api_symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": api_qty,
        })

        if "error" in result:
            logger.error(f"[TESTNET] Close order failed for {symbol}: {result['error']}")
        else:
            order_id = result.get("orderId", "?")
            logger.info(f"[TESTNET] Closed {pos.direction} {symbol} | "
                        f"orderId={order_id} reason={reason}")

        if pos.direction == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.units
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.units

        fees = pos.units * exit_price * self.fee_rate * 2  # entry + exit
        net_pnl = gross_pnl - fees
        pnl_pct = net_pnl / (pos.entry_price * pos.units)

        logger.info(
            f"[TESTNET] {pos.direction} {symbol} @ {exit_price:.4f} | "
            f"Reason: {reason} | PnL: ${net_pnl:.2f} ({pnl_pct:.2%})"
        )

        return {
            "symbol": symbol,
            "direction": pos.direction,
            "entry": pos.entry_price,
            "exit": exit_price,
            "units": pos.units,
            "gross_pnl": round(gross_pnl, 2),
            "fees": round(fees, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "reason": reason,
        }

    def close_all(self, current_prices: dict) -> list:
        """Emergency close all positions."""
        results = []
        for symbol in list(self.positions.keys()):
            price = current_prices.get(symbol, 0)
            if price > 0:
                result = self._close_position(symbol, price, "manual_close")
                results.append(result)
        return results

    # ---- POSITION PERSISTENCE ----

    def _save_positions(self):
        """Save positions to disk for crash recovery."""
        data = {}
        for sym, pos in self.positions.items():
            data[sym] = {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "units": pos.units,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "atr": pos.atr,
                "trail_atr_mult": pos.trail_atr_mult,
                "highest_since_entry": pos.highest_since_entry,
                "lowest_since_entry": pos.lowest_since_entry,
                "entry_time": pos.entry_time.isoformat(),
            }
        try:
            os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
            with open(POSITIONS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")

    def _load_positions(self):
        """Load positions from disk on startup."""
        if not os.path.exists(POSITIONS_FILE):
            return
        try:
            with open(POSITIONS_FILE, "r") as f:
                data = json.load(f)
            for sym, d in data.items():
                self.positions[sym] = Position(
                    symbol=d["symbol"],
                    direction=d["direction"],
                    entry_price=d["entry_price"],
                    units=d["units"],
                    stop_loss=d["stop_loss"],
                    take_profit=d["take_profit"],
                    atr=d["atr"],
                    trail_atr_mult=d.get("trail_atr_mult", 2.2),
                    highest_since_entry=d.get("highest_since_entry"),
                    lowest_since_entry=d.get("lowest_since_entry"),
                    entry_time=datetime.fromisoformat(d.get("entry_time", datetime.utcnow().isoformat())),
                )
            if self.positions:
                logger.info(f"Loaded {len(self.positions)} positions from disk: {list(self.positions.keys())}")
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
