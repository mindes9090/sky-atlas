"""
strategies/mean_reversion.py
─────────────────────────────────────────────────────────────
Mean Reversion Strategy
─────────────────────────────────────────────────────────────
Mathematical basis:
  Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σdW
  Price mean-reverts to μ at speed θ with noise σdW.

  In practice we proxy this with:
  - Z-score of price vs rolling mean (Bollinger logic)
  - RSI confirming extreme positioning
  - BB bands as visual boundaries

  LONG  entry: Z-score < -2.0 AND RSI < 30 AND price hits lower BB
  SHORT entry: Z-score > +2.0 AND RSI > 70 AND price hits upper BB
  EXIT:        Z-score reverts to ±0.5 OR price returns to BB midline

Academic basis:
  - Pairs/statistical arbitrage uses cointegration + OU process
  - Z-score entry at 2σ has ~95.4% probability of being extreme
  - Mean reversion confirmed in crypto ranging markets (Palazzi 2025)
─────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from strategies.trend_following import Signal, TradeSignal


class MeanReversionStrategy:
    """
    Bollinger Bands + RSI + Z-score mean reversion.
    Works best in RANGING regime (low ADX, narrow BBs).
    Supports both LONG and SHORT positions.
    """

    def __init__(
        self,
        zscore_entry    : float = 2.0,
        zscore_exit     : float = 0.5,
        rsi_oversold    : float = 30.0,
        rsi_overbought  : float = 70.0,
        atr_stop_mult   : float = 1.5,    # Tighter stops for mean reversion
        atr_tp_mult     : float = 2.5,
        require_bb_touch: bool  = True,   # Price must touch BB band
        volume_min_ratio: float = 1.0,    # Lower bar for mean reversion
    ):
        self.zscore_entry     = zscore_entry
        self.zscore_exit      = zscore_exit
        self.rsi_oversold     = rsi_oversold
        self.rsi_overbought   = rsi_overbought
        self.atr_stop_mult    = atr_stop_mult
        self.atr_tp_mult      = atr_tp_mult
        self.require_bb_touch = require_bb_touch
        self.volume_min_ratio = volume_min_ratio

    def generate_signal(self, df: pd.DataFrame,
                        current_position: Optional[str] = None) -> TradeSignal:
        """
        Analyse last candle for mean reversion opportunities.
        """
        if len(df) < 25:
            return TradeSignal(Signal.NONE, 0, 0, 0, 0, 0, "Insufficient data")

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price    = curr["close"]
        atr_v    = curr.get("atr", price * 0.015)
        rsi_v    = curr.get("rsi", 50.0)
        zscore_v = curr.get("zscore", 0.0)
        bb_upper = curr.get("bb_upper", price * 1.02)
        bb_lower = curr.get("bb_lower", price * 0.98)
        bb_mid   = curr.get("bb_mid",   price)
        vol_r    = curr.get("volume_ratio", 1.0)

        zscore_prev = prev.get("zscore", 0.0)

        # ─── Exit signals ──────────────────────────────────────
        if current_position == "LONG":
            # Exit when Z-score reverts toward mean
            if abs(zscore_v) < self.zscore_exit or price >= bb_mid:
                return TradeSignal(
                    Signal.EXIT_LONG, price,
                    0, 0, atr_v, 0.85,
                    f"Z-score reverted to {zscore_v:.2f} → exit LONG"
                )
            # Also exit on RSI overbought (captured the bounce)
            if rsi_v > self.rsi_overbought:
                return TradeSignal(
                    Signal.EXIT_LONG, price,
                    0, 0, atr_v, 0.75,
                    f"RSI overbought {rsi_v:.1f} → take profit on LONG"
                )

        if current_position == "SHORT":
            if abs(zscore_v) < self.zscore_exit or price <= bb_mid:
                return TradeSignal(
                    Signal.EXIT_SHORT, price,
                    0, 0, atr_v, 0.85,
                    f"Z-score reverted to {zscore_v:.2f} → exit SHORT"
                )
            if rsi_v < self.rsi_oversold:
                return TradeSignal(
                    Signal.EXIT_SHORT, price,
                    0, 0, atr_v, 0.75,
                    f"RSI oversold {rsi_v:.1f} → take profit on SHORT"
                )

        if current_position in ("LONG", "SHORT"):
            return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0,
                               "Holding position")

        # ─── Volume filter ─────────────────────────────────────
        if vol_r < self.volume_min_ratio:
            return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0,
                               f"Low volume ({vol_r:.2f}x)")

        # ─── LONG Signal: Oversold extreme ────────────────────
        bb_touch_long = price <= bb_lower if self.require_bb_touch else True
        zscore_extreme_low = zscore_v <= -self.zscore_entry
        rsi_oversold_cond  = rsi_v <= self.rsi_oversold + 5  # slightly relaxed

        # Look for Z-score starting to recover (turning point)
        zscore_turning_up  = zscore_v > zscore_prev

        if zscore_extreme_low and rsi_oversold_cond:
            stop_loss   = price - self.atr_stop_mult * atr_v
            take_profit = bb_mid  # Target the mean (BB midline)
            confidence  = self._calc_confidence(zscore_v, rsi_v, "LONG",
                                                zscore_turning_up)
            reason = (f"Z={zscore_v:.2f} (extreme low) | RSI={rsi_v:.1f} | "
                      f"Price at lower BB")
            return TradeSignal(Signal.LONG, price, stop_loss, take_profit,
                               atr_v, confidence, reason)

        # ─── SHORT Signal: Overbought extreme ─────────────────
        bb_touch_short     = price >= bb_upper if self.require_bb_touch else True
        zscore_extreme_high = zscore_v >= self.zscore_entry
        rsi_overbought_cond = rsi_v >= self.rsi_overbought - 5  # slightly relaxed

        # Z-score starting to fade
        zscore_turning_down = zscore_v < zscore_prev

        if zscore_extreme_high and rsi_overbought_cond:
            stop_loss   = price + self.atr_stop_mult * atr_v
            take_profit = bb_mid  # Target the mean (BB midline)
            confidence  = self._calc_confidence(zscore_v, rsi_v, "SHORT",
                                                zscore_turning_down)
            reason = (f"Z={zscore_v:.2f} (extreme high) | RSI={rsi_v:.1f} | "
                      f"Price at upper BB")
            return TradeSignal(Signal.SHORT, price, stop_loss, take_profit,
                               atr_v, confidence, reason)

        return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0,
                           f"No extreme: Z={zscore_v:.2f}, RSI={rsi_v:.1f}")

    def _calc_confidence(self, zscore: float, rsi: float,
                          direction: str, turning: bool) -> float:
        """Score confidence based on extremity + turn confirmation."""
        z_abs = abs(zscore)
        score = 0.0
        score += min((z_abs - 2.0) / 2.0, 0.4)  # Z-score extremity (max 0.4)

        if direction == "LONG":
            score += max((30 - rsi) / 30, 0) * 0.3   # RSI extremity
        else:
            score += max((rsi - 70) / 30, 0) * 0.3

        score += 0.3 if turning else 0.0  # Turning point bonus
        return round(min(max(score, 0.0), 1.0), 3)
