"""
regime/detector.py
-----------------------------------------------------------------
Market Regime Detector
-----------------------------------------------------------------
Classifies the market into one of three regimes:
  TRENDING  - Strong directional movement (ADX > 25, wide BBs)
  RANGING   - Mean-reverting, low volatility (ADX < 15, narrow BBs)
  VOLATILE  - High volatility, no clear direction (reduce size or skip)

Uses a 5-factor scoring system:
  1. ADX level
  2. Bollinger Band width
  3. Volatility rank (ATR percentile)
  4. Short/long volatility ratio
  5. Price distance from EMA-200
-----------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Regime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


@dataclass
class RegimeResult:
    regime: Regime
    confidence: float
    adx: float
    bb_width: float
    vol_rank: float
    trend_score: float
    range_score: float


class RegimeDetector:
    """
    Multi-factor market regime classifier.
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        adx_low: float = 15.0,
        bb_wide: float = 0.10,
        bb_narrow: float = 0.04,
        vol_rank_high: float = 0.80,
        vol_rank_low: float = 0.30,
        vol_ratio_trend: float = 1.5,
        vol_ratio_range: float = 0.7,
        ema_dist_pct: float = 0.03,
    ):
        self.adx_threshold = adx_threshold
        self.adx_low = adx_low
        self.bb_wide = bb_wide
        self.bb_narrow = bb_narrow
        self.vol_rank_high = vol_rank_high
        self.vol_rank_low = vol_rank_low
        self.vol_ratio_trend = vol_ratio_trend
        self.vol_ratio_range = vol_ratio_range
        self.ema_dist_pct = ema_dist_pct

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        Classify current market regime from indicator DataFrame.
        Returns RegimeResult with scores and classification.
        """
        curr = df.iloc[-1]

        adx = float(curr.get("adx", 20))
        bb_width = float(curr.get("bb_width", 0.05))
        vol_rank = float(curr.get("vol_rank", 0.5))
        vol_ratio_sl = float(curr.get("vol_ratio_sl", 1.0))
        price = float(curr["close"])
        ema200 = float(curr.get("ema_200", price))
        ema_dist = abs(price - ema200) / (ema200 + 1e-10)

        # Score for TRENDING (0-5)
        trend_score = 0.0
        trend_score += 1.0 if adx > self.adx_threshold else 0.0
        trend_score += 1.0 if bb_width > self.bb_wide else 0.0
        trend_score += 1.0 if vol_rank > self.vol_rank_high else 0.0
        trend_score += 1.0 if vol_ratio_sl > self.vol_ratio_trend else 0.0
        trend_score += 1.0 if ema_dist > self.ema_dist_pct else 0.0

        # Score for RANGING (0-5)
        range_score = 0.0
        range_score += 1.0 if adx < self.adx_low else 0.0
        range_score += 1.0 if bb_width < self.bb_narrow else 0.0
        range_score += 1.0 if vol_rank < self.vol_rank_low else 0.0
        range_score += 1.0 if vol_ratio_sl < self.vol_ratio_range else 0.0
        range_score += 1.0 if ema_dist < self.ema_dist_pct * 0.5 else 0.0

        # Classification
        if trend_score >= 3:
            regime = Regime.TRENDING
            confidence = trend_score / 5.0
        elif range_score >= 3:
            regime = Regime.RANGING
            confidence = range_score / 5.0
        elif vol_rank > 0.9 and vol_ratio_sl > 2.0:
            regime = Regime.VOLATILE
            confidence = 0.8
        else:
            # Ambiguous: default to the higher score
            if trend_score > range_score:
                regime = Regime.TRENDING
                confidence = trend_score / 5.0
            elif range_score > trend_score:
                regime = Regime.RANGING
                confidence = range_score / 5.0
            else:
                regime = Regime.VOLATILE
                confidence = 0.5

        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            adx=round(adx, 1),
            bb_width=round(bb_width, 4),
            vol_rank=round(vol_rank, 3),
            trend_score=trend_score,
            range_score=range_score,
        )

    def get_position_size_multiplier(self, result: RegimeResult) -> float:
        """
        Regime-based position size multiplier.
        TRENDING  = 1.0 (full size)
        RANGING   = 0.8 (slightly reduced)
        VOLATILE  = 0.4 (significantly reduced)
        """
        if result.regime == Regime.TRENDING:
            return 1.0
        elif result.regime == Regime.RANGING:
            return 0.8
        else:
            return 0.4
