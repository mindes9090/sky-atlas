"""
risk/black_swan.py
-----------------------------------------------------------------
Black Swan Protection System
-----------------------------------------------------------------
Detects and responds to extreme market events:

  1. Flash Crash Detection
     - Price drops >3% in 1 bar (1h)
     - Price drops >5% in 4 bars (4h)
     - Triggers immediate position closure

  2. Volatility Explosion
     - ATR jumps to >3x its 20-bar average
     - Reduces position sizes to 25%

  3. Volume Anomaly
     - Volume spikes to >5x normal
     - Often precedes major moves

  4. Correlation Breakdown
     - All pairs moving in same direction with high magnitude
     - Indicates systemic risk / contagion

  5. Cascade Stop
     - Rapid sequence of stop-losses hit
     - Halts all trading for cooldown period

Protection actions:
  - REDUCE: Cut position sizes by 50-75%
  - CLOSE_ALL: Emergency close all positions
  - HALT: Stop trading for N hours
  - HEDGE: (future) Open hedge positions
-----------------------------------------------------------------
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BlackSwanAction(Enum):
    NONE = "NONE"
    REDUCE = "REDUCE"         # Cut position sizes
    CLOSE_ALL = "CLOSE_ALL"   # Emergency liquidation
    HALT = "HALT"             # Stop trading


@dataclass
class BlackSwanAlert:
    action: BlackSwanAction
    reason: str
    severity: float           # 0.0-1.0
    size_multiplier: float    # Applied to position sizing
    cooldown_hours: int       # Hours to wait before resuming


class BlackSwanDetector:
    """
    Detects extreme market events and triggers protective actions.
    """

    def __init__(
        self,
        flash_crash_1h_pct: float = 0.03,    # 3% drop in 1 bar
        flash_crash_4h_pct: float = 0.05,    # 5% drop in 4 bars
        atr_spike_mult: float = 3.0,          # ATR > 3x average
        volume_spike_mult: float = 5.0,       # Volume > 5x average
        max_stops_per_hour: int = 3,           # Cascade detection
        correlation_threshold: float = 0.95,   # All-pair correlation
    ):
        self.flash_crash_1h_pct = flash_crash_1h_pct
        self.flash_crash_4h_pct = flash_crash_4h_pct
        self.atr_spike_mult = atr_spike_mult
        self.volume_spike_mult = volume_spike_mult
        self.max_stops_per_hour = max_stops_per_hour
        self.correlation_threshold = correlation_threshold

        # State
        self.recent_stops: list[datetime] = []
        self.halt_until: Optional[datetime] = None
        self.active_alert: Optional[BlackSwanAlert] = None

    def check(self, df: pd.DataFrame, symbol: str = "") -> BlackSwanAlert:
        """
        Run all black swan checks on current data.
        Returns the most severe alert found.
        """
        alerts = []

        # Check if in cooldown
        if self.halt_until and datetime.utcnow() < self.halt_until:
            remaining = (self.halt_until - datetime.utcnow()).seconds // 60
            return BlackSwanAlert(
                BlackSwanAction.HALT,
                f"Cooldown active ({remaining}m remaining)",
                0.9, 0.0, 0
            )

        alerts.append(self._check_flash_crash(df, symbol))
        alerts.append(self._check_volatility_explosion(df, symbol))
        alerts.append(self._check_volume_anomaly(df, symbol))
        alerts.append(self._check_cascade_stops())

        # Return most severe alert
        alerts = [a for a in alerts if a.action != BlackSwanAction.NONE]
        if not alerts:
            self.active_alert = None
            return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

        worst = max(alerts, key=lambda a: a.severity)
        self.active_alert = worst

        if worst.action == BlackSwanAction.HALT:
            self.halt_until = datetime.utcnow() + timedelta(hours=worst.cooldown_hours)
            logger.warning(f"BLACK SWAN [{symbol}]: {worst.reason} -> HALT {worst.cooldown_hours}h")
        elif worst.action == BlackSwanAction.CLOSE_ALL:
            logger.warning(f"BLACK SWAN [{symbol}]: {worst.reason} -> CLOSE ALL")
        elif worst.action == BlackSwanAction.REDUCE:
            logger.warning(f"BLACK SWAN [{symbol}]: {worst.reason} -> REDUCE x{worst.size_multiplier}")

        return worst

    def check_multi_asset(
        self, all_returns: dict[str, float]
    ) -> BlackSwanAlert:
        """
        Check for correlated crash across all assets.
        all_returns: {symbol: last_1h_return}
        """
        if len(all_returns) < 2:
            return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

        returns = list(all_returns.values())
        avg_ret = np.mean(returns)
        all_negative = all(r < -0.01 for r in returns)
        all_big_drop = all(r < -0.02 for r in returns)

        if all_big_drop:
            return BlackSwanAlert(
                BlackSwanAction.CLOSE_ALL,
                f"Correlated crash: all {len(returns)} assets down >2% "
                f"(avg {avg_ret:.2%})",
                0.95, 0.0, 4
            )
        elif all_negative and avg_ret < -0.015:
            return BlackSwanAlert(
                BlackSwanAction.REDUCE,
                f"Correlated sell-off: avg return {avg_ret:.2%}",
                0.7, 0.3, 0
            )

        return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

    def record_stop_hit(self):
        """Record a stop-loss event for cascade detection."""
        self.recent_stops.append(datetime.utcnow())
        # Clean old entries
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.recent_stops = [s for s in self.recent_stops if s > cutoff]

    def _check_flash_crash(self, df: pd.DataFrame, symbol: str) -> BlackSwanAlert:
        """Detect rapid price drops."""
        if len(df) < 5:
            return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

        # 1-bar crash
        last_ret = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
        if last_ret < -self.flash_crash_1h_pct:
            return BlackSwanAlert(
                BlackSwanAction.CLOSE_ALL,
                f"Flash crash {symbol}: {last_ret:.2%} in 1 bar",
                0.95, 0.0, 2
            )

        # 4-bar crash
        if len(df) >= 5:
            ret_4h = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]
            if ret_4h < -self.flash_crash_4h_pct:
                return BlackSwanAlert(
                    BlackSwanAction.CLOSE_ALL,
                    f"Sustained crash {symbol}: {ret_4h:.2%} in 4 bars",
                    0.9, 0.0, 3
                )

        # Rapid spike UP (potential manipulation / squeeze)
        if last_ret > self.flash_crash_1h_pct:
            return BlackSwanAlert(
                BlackSwanAction.REDUCE,
                f"Price spike {symbol}: +{last_ret:.2%} (potential squeeze)",
                0.6, 0.5, 0
            )

        return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

    def _check_volatility_explosion(self, df: pd.DataFrame, symbol: str) -> BlackSwanAlert:
        """Detect ATR spikes indicating extreme volatility."""
        if "atr" not in df.columns or len(df) < 25:
            return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

        current_atr = df["atr"].iloc[-1]
        avg_atr = df["atr"].iloc[-21:-1].mean()

        if avg_atr == 0:
            return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

        atr_ratio = current_atr / avg_atr

        if atr_ratio > self.atr_spike_mult * 1.5:
            return BlackSwanAlert(
                BlackSwanAction.REDUCE,
                f"Extreme volatility {symbol}: ATR {atr_ratio:.1f}x normal",
                0.8, 0.25, 0
            )
        elif atr_ratio > self.atr_spike_mult:
            return BlackSwanAlert(
                BlackSwanAction.REDUCE,
                f"High volatility {symbol}: ATR {atr_ratio:.1f}x normal",
                0.5, 0.5, 0
            )

        return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

    def _check_volume_anomaly(self, df: pd.DataFrame, symbol: str) -> BlackSwanAlert:
        """Detect extreme volume spikes."""
        if "volume_ratio" not in df.columns:
            return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

        vol_ratio = df["volume_ratio"].iloc[-1]

        if vol_ratio > self.volume_spike_mult * 2:
            return BlackSwanAlert(
                BlackSwanAction.REDUCE,
                f"Extreme volume {symbol}: {vol_ratio:.1f}x normal",
                0.7, 0.4, 0
            )
        elif vol_ratio > self.volume_spike_mult:
            return BlackSwanAlert(
                BlackSwanAction.REDUCE,
                f"Volume spike {symbol}: {vol_ratio:.1f}x normal",
                0.4, 0.6, 0
            )

        return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

    def _check_cascade_stops(self) -> BlackSwanAlert:
        """Detect rapid sequence of stop-losses."""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent = [s for s in self.recent_stops if s > cutoff]

        if len(recent) >= self.max_stops_per_hour:
            return BlackSwanAlert(
                BlackSwanAction.HALT,
                f"Cascade: {len(recent)} stops hit in 1 hour",
                0.85, 0.0, 2
            )

        return BlackSwanAlert(BlackSwanAction.NONE, "OK", 0, 1.0, 0)

    def get_backtest_size_multiplier(self, df: pd.DataFrame, i: int) -> float:
        """
        For backtesting: compute black swan size multiplier at bar i.
        Returns 0.0-1.0 multiplier (0 = don't trade, 1 = full size).
        """
        if i < 5:
            return 1.0

        price = df["close"].iloc[i]
        prev_price = df["close"].iloc[i - 1]
        prev_4 = df["close"].iloc[i - 4] if i >= 4 else prev_price

        # Flash crash check
        ret_1 = (price - prev_price) / prev_price
        ret_4 = (price - prev_4) / prev_4

        if abs(ret_1) > self.flash_crash_1h_pct:
            return 0.0  # Don't enter during flash crash
        if abs(ret_4) > self.flash_crash_4h_pct:
            return 0.0

        # Volatility check
        if "atr" in df.columns and i >= 25:
            current_atr = df["atr"].iloc[i]
            avg_atr = df["atr"].iloc[max(0, i-20):i].mean()
            if avg_atr > 0:
                atr_ratio = current_atr / avg_atr
                if atr_ratio > self.atr_spike_mult * 1.5:
                    return 0.25
                elif atr_ratio > self.atr_spike_mult:
                    return 0.5

        # Volume check
        if "volume_ratio" in df.columns:
            vol_r = df["volume_ratio"].iloc[i]
            if vol_r > self.volume_spike_mult * 2:
                return 0.4
            elif vol_r > self.volume_spike_mult:
                return 0.6

        return 1.0
