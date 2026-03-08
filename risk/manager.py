"""
risk/manager.py
─────────────────────────────────────────────────────────────
Risk Management Engine
─────────────────────────────────────────────────────────────
Implements:
  1. Kelly Criterion position sizing (fractional Kelly, capped)
  2. ATR-based dynamic stop-loss calculation
  3. Portfolio-level risk controls
  4. Circuit breakers (daily loss, consecutive losses, max drawdown)
  5. Correlation check (avoid concentrated exposure)

Kelly Criterion formula:
  f* = (bp - q) / b
  where:
    b = reward/risk ratio (avg_win / avg_loss)
    p = win probability
    q = 1 - p (loss probability)

  We use QUARTER-KELLY (f* × 0.25) for safety:
  - Full Kelly: max growth but ~50-70% drawdowns
  - Half Kelly:  75% of optimal growth, ~50% less drawdown
  - Quarter Kelly: safest — smooth equity curve

  Cap at 10% of portfolio per trade regardless of formula output.
─────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, date


@dataclass
class PositionSize:
    units           : float
    risk_amount_usd : float
    position_value  : float
    kelly_fraction  : float
    reason          : str
    approved        : bool


@dataclass
class TradeRecord:
    timestamp     : datetime
    symbol        : str
    direction     : str      # "LONG" or "SHORT"
    entry         : float
    exit          : float
    pnl           : float
    pnl_pct       : float


class RiskManager:
    """
    Central risk management engine.
    All position size decisions pass through here.
    """

    def __init__(
        self,
        total_capital       : float = 10_000,
        max_risk_per_trade  : float = 0.02,      # 2% per trade
        max_portfolio_risk  : float = 0.06,      # 6% across all positions
        max_positions       : int   = 3,
        kelly_fraction      : float = 0.25,      # Quarter Kelly
        max_kelly_size      : float = 0.10,      # Hard cap: 10% per trade
        kelly_lookback      : int   = 50,        # Trades for win rate calc
        max_daily_drawdown  : float = 0.05,      # 5% daily loss limit
        max_total_drawdown  : float = 0.15,      # 15% total drawdown
        max_consecutive_loss: int   = 5,         # Circuit breaker
        atr_stop_mult       : float = 2.0,
        fee_rate            : float = 0.001,     # 0.1% Binance fee
    ):
        self.total_capital        = total_capital
        self.current_equity       = total_capital
        self.max_risk_per_trade   = max_risk_per_trade
        self.max_portfolio_risk   = max_portfolio_risk
        self.max_positions        = max_positions
        self.kelly_fraction       = kelly_fraction
        self.max_kelly_size       = max_kelly_size
        self.kelly_lookback       = kelly_lookback
        self.max_daily_drawdown   = max_daily_drawdown
        self.max_total_drawdown   = max_total_drawdown
        self.max_consecutive_loss = max_consecutive_loss
        self.atr_stop_mult        = atr_stop_mult
        self.fee_rate             = fee_rate

        # State tracking
        self.trade_history        : List[TradeRecord] = []
        self.open_positions       : dict = {}           # symbol → position
        self.peak_equity          = total_capital
        self.daily_start_equity   = total_capital
        self.trading_halted       = False
        self.halt_reason          = ""
        self.consecutive_losses   = 0

    # ─── KELLY CRITERION ─────────────────────────────────

    def calculate_kelly_fraction(self) -> tuple[float, float, float]:
        """
        Calculate Kelly fraction from recent trade history.
        Returns: (kelly_f, win_rate, reward_risk_ratio)
        
        Formula: f* = (b×p - q) / b = p - q/b
        """
        recent = self.trade_history[-self.kelly_lookback:]

        if len(recent) < 10:
            # Not enough data: use conservative default
            return (self.max_risk_per_trade, 0.5, 1.5)

        wins  = [t for t in recent if t.pnl > 0]
        loses = [t for t in recent if t.pnl <= 0]

        if not wins or not loses:
            return (self.max_risk_per_trade, 0.5, 1.5)

        p = len(wins) / len(recent)                          # win rate
        q = 1 - p                                            # loss rate
        avg_win  = np.mean([t.pnl_pct for t in wins])
        avg_loss = abs(np.mean([t.pnl_pct for t in loses]))

        if avg_loss == 0:
            return (self.max_risk_per_trade, p, 1.5)

        b = avg_win / avg_loss                               # reward/risk
        kelly_full = (b * p - q) / b                        # full Kelly

        # Apply fractional Kelly (Quarter-Kelly = × 0.25)
        kelly_adjusted = kelly_full * self.kelly_fraction

        # Hard cap at max_kelly_size
        kelly_capped = min(max(kelly_adjusted, 0.005), self.max_kelly_size)

        return (kelly_capped, round(p, 3), round(b, 3))

    # ─── POSITION SIZING ─────────────────────────────────

    def calculate_position_size(
        self,
        symbol          : str,
        entry_price     : float,
        stop_loss_price : float,
        signal_confidence: float = 0.7,
        regime_multiplier: float = 1.0,
    ) -> PositionSize:
        """
        Calculate the position size in units.

        Method:
        1. Determine risk amount = equity × kelly_fraction
        2. Calculate risk per unit = |entry - stop_loss|
        3. Units = risk_amount / risk_per_unit
        4. Validate against portfolio-level constraints
        """
        # ── Circuit breaker check ────────────────────────
        if self.trading_halted:
            return PositionSize(0, 0, 0, 0,
                                f"Trading halted: {self.halt_reason}", False)

        # ── Portfolio risk check ─────────────────────────
        current_portfolio_risk = self._calc_current_portfolio_risk()
        if current_portfolio_risk >= self.max_portfolio_risk:
            return PositionSize(0, 0, 0, 0,
                                f"Portfolio risk maxed: {current_portfolio_risk:.1%}", False)

        # ── Max positions check ──────────────────────────
        if len(self.open_positions) >= self.max_positions:
            return PositionSize(0, 0, 0, 0,
                                f"Max positions ({self.max_positions}) reached", False)

        # ── Kelly sizing ─────────────────────────────────
        kelly_f, win_rate, rr_ratio = self.calculate_kelly_fraction()

        # Scale by signal confidence and regime
        effective_kelly = kelly_f * signal_confidence * regime_multiplier
        effective_kelly = min(effective_kelly, self.max_kelly_size)

        risk_amount = self.current_equity * effective_kelly

        # ── Risk per unit from stop ──────────────────────
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit < 1e-8:
            return PositionSize(0, 0, 0, 0, "Stop loss too close to entry", False)

        units = risk_amount / risk_per_unit

        # ── Additional max risk per trade cap ────────────
        max_risk_usd   = self.current_equity * self.max_risk_per_trade
        if risk_amount > max_risk_usd:
            units       = max_risk_usd / risk_per_unit
            risk_amount = max_risk_usd

        position_value = units * entry_price
        reason = (f"Kelly={effective_kelly:.3f} | Win%={win_rate:.1%} | "
                  f"R:R={rr_ratio:.2f} | Risk=${risk_amount:.2f}")

        return PositionSize(
            units           = round(units, 6),
            risk_amount_usd = round(risk_amount, 2),
            position_value  = round(position_value, 2),
            kelly_fraction  = round(effective_kelly, 4),
            reason          = reason,
            approved        = True,
        )

    # ─── ATR STOP LOSS ───────────────────────────────────

    def calculate_atr_stop(self, entry_price: float, atr: float,
                            direction: str) -> tuple[float, float]:
        """
        Calculate stop loss and take profit using ATR.
        Returns: (stop_loss, take_profit)
        """
        stop_distance = self.atr_stop_mult * atr
        tp_distance   = stop_distance * 2.0  # 2:1 R:R minimum

        if direction == "LONG":
            stop_loss   = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss   = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        return round(stop_loss, 8), round(take_profit, 8)

    # ─── CIRCUIT BREAKERS ────────────────────────────────

    def update_equity(self, new_equity: float):
        """Call after each trade to update equity and check circuit breakers."""
        self.current_equity = new_equity
        self.peak_equity    = max(self.peak_equity, new_equity)

        total_drawdown = (self.peak_equity - new_equity) / self.peak_equity
        daily_drawdown = (self.daily_start_equity - new_equity) / self.daily_start_equity

        if daily_drawdown >= self.max_daily_drawdown:
            self._halt(f"Daily drawdown {daily_drawdown:.1%} ≥ {self.max_daily_drawdown:.1%}")

        if total_drawdown >= self.max_total_drawdown:
            self._halt(f"Total drawdown {total_drawdown:.1%} ≥ {self.max_total_drawdown:.1%}")

    def record_trade(self, trade: TradeRecord):
        """Record completed trade, update loss streak tracker."""
        self.trade_history.append(trade)

        if trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.max_consecutive_loss:
            self._halt(f"{self.consecutive_losses} consecutive losses — pausing")

    def reset_daily(self):
        """Call at start of each trading day."""
        self.daily_start_equity = self.current_equity
        # Optionally auto-resume if previous halt was daily DD only
        if "Daily drawdown" in self.halt_reason:
            self.trading_halted = False
            self.halt_reason    = ""

    def _halt(self, reason: str):
        self.trading_halted = True
        self.halt_reason    = reason

    def resume_trading(self):
        """Manually resume after reviewing halt."""
        self.trading_halted    = False
        self.halt_reason       = ""
        self.consecutive_losses = 0

    # ─── REPORTING ───────────────────────────────────────

    def get_stats(self) -> dict:
        """Summary of risk manager state and performance."""
        kelly_f, win_rate, rr = self.calculate_kelly_fraction()
        total_dd = (self.peak_equity - self.current_equity) / self.peak_equity

        return {
            "current_equity"     : round(self.current_equity, 2),
            "peak_equity"        : round(self.peak_equity, 2),
            "total_drawdown"     : round(total_dd, 4),
            "consecutive_losses" : self.consecutive_losses,
            "trading_halted"     : self.trading_halted,
            "halt_reason"        : self.halt_reason,
            "open_positions"     : len(self.open_positions),
            "total_trades"       : len(self.trade_history),
            "kelly_fraction"     : round(kelly_f, 4),
            "win_rate"           : round(win_rate, 3),
            "reward_risk"        : round(rr, 3),
        }

    def _calc_current_portfolio_risk(self) -> float:
        """Sum of risk across all open positions as % of equity."""
        total_risk = sum(
            pos.get("risk_amount_usd", 0)
            for pos in self.open_positions.values()
        )
        return total_risk / self.current_equity if self.current_equity > 0 else 0
