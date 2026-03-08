"""
strategies/trend_following.py
─────────────────────────────────────────────────────────────
Trend Following Strategy
─────────────────────────────────────────────────────────────
Logic:
  LONG  entry: EMA_fast crosses ABOVE EMA_slow
               + MACD histogram turns positive
               + ADX > 25 (confirms trend strength)
               + Volume > 1.2× MA (confirms participation)
               + Price > EMA_200 (macro bull bias) [optional]

  SHORT entry: EMA_fast crosses BELOW EMA_slow
               + MACD histogram turns negative
               + ADX > 25
               + Volume > 1.2× MA
               + Price < EMA_200 (macro bear bias) [optional]

  EXIT:         Opposite EMA crossover OR ATR trailing stop hit

Academic basis:
  - EMA crossovers: widely validated momentum signal
  - ADX filter: removes whipsaw signals in ranging markets
  - Volume confirmation: reduces false breakouts
─────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Signal(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    EXIT_LONG  = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NONE  = "NONE"


@dataclass
class TradeSignal:
    signal       : Signal
    entry_price  : float
    stop_loss    : float
    take_profit  : float
    atr          : float
    confidence   : float      # 0.0 – 1.0
    reason       : str


class TrendFollowingStrategy:
    """
    EMA Crossover + MACD + ADX + Volume Confirmation
    Works best in TRENDING regime (ADX > 25).
    """

    def __init__(
        self,
        ema_fast        : int   = 12,
        ema_slow        : int   = 26,
        adx_min         : float = 25.0,
        volume_min_ratio: float = 1.2,
        atr_stop_mult   : float = 2.0,
        atr_tp_mult     : float = 4.0,
        use_ema200_filter: bool = True,
    ):
        self.ema_fast         = ema_fast
        self.ema_slow         = ema_slow
        self.adx_min          = adx_min
        self.volume_min_ratio = volume_min_ratio
        self.atr_stop_mult    = atr_stop_mult
        self.atr_tp_mult      = atr_tp_mult
        self.use_ema200_filter = use_ema200_filter

    def generate_signal(self, df: pd.DataFrame,
                        current_position: Optional[str] = None) -> TradeSignal:
        """
        Analyse the last few candles and return a TradeSignal.
        df must have indicators applied (add_all_indicators).
        """
        if len(df) < 30:
            return TradeSignal(Signal.NONE, 0, 0, 0, 0, 0, "Insufficient data")

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price  = curr["close"]
        atr_v  = curr.get("atr", price * 0.02)
        adx_v  = curr.get("adx", 0)
        vol_r  = curr.get("volume_ratio", 1.0)
        ema200 = curr.get("ema_200", price)
        macd_h = curr.get("macd_hist", 0)
        macd_h_prev = prev.get("macd_hist", 0)

        ema_fast_curr = curr.get("ema_fast", price)
        ema_slow_curr = curr.get("ema_slow", price)
        ema_fast_prev = prev.get("ema_fast", price)
        ema_slow_prev = prev.get("ema_slow", price)

        # ─── Exit signals first (for existing positions) ──────
        if current_position == "LONG":
            if ema_fast_curr < ema_slow_curr and ema_fast_prev >= ema_slow_prev:
                return TradeSignal(
                    Signal.EXIT_LONG, price,
                    price - atr_v, price + atr_v, atr_v,
                    0.9, "EMA bearish crossover → exit LONG"
                )

        if current_position == "SHORT":
            if ema_fast_curr > ema_slow_curr and ema_fast_prev <= ema_slow_prev:
                return TradeSignal(
                    Signal.EXIT_SHORT, price,
                    price + atr_v, price - atr_v, atr_v,
                    0.9, "EMA bullish crossover → exit SHORT"
                )

        # ─── Guard: No new entry if already in position ────────
        if current_position in ("LONG", "SHORT"):
            return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0,
                               "Already in position")

        # ─── Filter: ADX must confirm trend ────────────────────
        if adx_v < self.adx_min:
            return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0,
                               f"ADX {adx_v:.1f} < {self.adx_min} — no trend")

        # ─── Filter: Volume confirmation ───────────────────────
        if vol_r < self.volume_min_ratio:
            return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0,
                               f"Volume ratio {vol_r:.2f} < {self.volume_min_ratio}")

        # ─── EMA Bullish Crossover (LONG signal) ───────────────
        ema_bull_cross = (ema_fast_curr > ema_slow_curr and
                          ema_fast_prev <= ema_slow_prev)
        macd_bull      = macd_h > 0
        macro_bull     = (price > ema200) if self.use_ema200_filter else True

        if ema_bull_cross and macd_bull:
            stop_loss   = price - self.atr_stop_mult * atr_v
            take_profit = price + self.atr_tp_mult  * atr_v
            confidence  = self._calc_confidence(adx_v, vol_r, macro_bull)
            reason      = (f"EMA bullish cross | ADX={adx_v:.1f} | "
                           f"Vol×{vol_r:.2f} | MACD↑")
            return TradeSignal(Signal.LONG, price, stop_loss, take_profit,
                               atr_v, confidence, reason)

        # ─── EMA Bearish Crossover (SHORT signal) ──────────────
        ema_bear_cross = (ema_fast_curr < ema_slow_curr and
                          ema_fast_prev >= ema_slow_prev)
        macd_bear      = macd_h < 0
        macro_bear     = (price < ema200) if self.use_ema200_filter else True

        if ema_bear_cross and macd_bear:
            stop_loss   = price + self.atr_stop_mult * atr_v
            take_profit = price - self.atr_tp_mult  * atr_v
            confidence  = self._calc_confidence(adx_v, vol_r, macro_bear)
            reason      = (f"EMA bearish cross | ADX={adx_v:.1f} | "
                           f"Vol×{vol_r:.2f} | MACD↓")
            return TradeSignal(Signal.SHORT, price, stop_loss, take_profit,
                               atr_v, confidence, reason)

        return TradeSignal(Signal.NONE, price, 0, 0, atr_v, 0, "No signal")

    def _calc_confidence(self, adx: float, vol_ratio: float,
                          macro_aligned: bool) -> float:
        """Composite confidence 0.0–1.0."""
        score = 0.0
        score += min((adx - 25) / 25, 0.4)   # ADX contribution (max 0.4)
        score += min((vol_ratio - 1) / 2, 0.3)  # Volume contribution (max 0.3)
        score += 0.3 if macro_aligned else 0.0   # Macro alignment (max 0.3)
        return round(min(max(score, 0.0), 1.0), 3)
