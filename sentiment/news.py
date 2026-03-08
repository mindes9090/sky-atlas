"""
sentiment/news.py
-----------------------------------------------------------------
Market Sentiment & News Catalyst Layer
-----------------------------------------------------------------
Sources:
  1. Crypto Fear & Greed Index (alternative.me - free, no key)
  2. Price-momentum sentiment proxy (for backtesting)
  3. Claude AI news analysis (live trading only)

Sentiment impact on trading:
  - Extreme Fear  (0-25):  Boost LONG scores, reduce SHORT
  - Fear          (25-45): Slight LONG bias
  - Neutral       (45-55): No adjustment
  - Greed         (55-75): Slight SHORT bias
  - Extreme Greed (75-100): Boost SHORT scores, reduce LONG

News catalyst logic:
  - Good news + uptrend  = stronger LONG signal
  - Bad news  + downtrend = stronger SHORT signal
  - Good news + downtrend = conflicting, reduce size
  - Bad news  + uptrend   = conflicting, reduce size
-----------------------------------------------------------------
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import urllib.request
    import json as _json
    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False


@dataclass
class SentimentResult:
    fear_greed_index: int       # 0-100
    fear_greed_label: str       # "Extreme Fear", "Fear", etc.
    sentiment_bias: float       # -1.0 (extreme fear) to +1.0 (extreme greed)
    long_multiplier: float      # Multiplier for long score
    short_multiplier: float     # Multiplier for short score
    size_multiplier: float      # Position size adjustment
    news_aligned: bool          # True if sentiment aligns with trend


class SentimentAnalyzer:
    """
    Fetches market sentiment and adjusts trading signals accordingly.
    Falls back to momentum-based proxy when API unavailable.
    """

    def __init__(self, use_live_api: bool = True):
        self.use_live_api = use_live_api
        self._cached_fgi = None
        self._cache_timestamp = 0

    def get_sentiment(self, df: Optional[pd.DataFrame] = None) -> SentimentResult:
        """
        Get current market sentiment.
        Uses live API if available, else computes from price data.
        """
        fgi = None

        if self.use_live_api and HAS_HTTP:
            fgi = self._fetch_fear_greed_index()

        if fgi is None and df is not None:
            fgi = self._compute_momentum_sentiment(df)

        if fgi is None:
            fgi = 50  # neutral fallback

        return self._score_sentiment(fgi)

    def compute_backtest_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a sentiment proxy for each bar in backtest data.
        Uses multi-factor momentum approach.
        """
        sentiment = pd.Series(50.0, index=df.index)

        # Factor 1: 14-bar RSI mapped to 0-100 (RSI is already 0-100)
        rsi = df.get("rsi", pd.Series(50, index=df.index))

        # Factor 2: 30-bar ROC percentile rank — vectorized
        roc_30 = df["close"].pct_change(30)
        roc_rank = roc_30.rolling(100).rank(pct=True) * 100

        # Factor 3: Volume surge (high volume = emotion)
        vol_r = df.get("volume_ratio", pd.Series(1.0, index=df.index))
        vol_emotion = np.clip((vol_r - 1.0) * 20 + 50, 10, 90)

        # Weighted blend
        sentiment = (rsi * 0.4 + roc_rank.fillna(50) * 0.4 +
                     vol_emotion * 0.2)

        return sentiment.clip(0, 100)

    def _fetch_fear_greed_index(self) -> Optional[int]:
        """Fetch Crypto Fear & Greed Index from alternative.me."""
        import time
        # Cache for 1 hour
        if (self._cached_fgi is not None and
                time.time() - self._cache_timestamp < 3600):
            return self._cached_fgi

        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "SKYATLAS/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
                fgi = int(data["data"][0]["value"])
                self._cached_fgi = fgi
                self._cache_timestamp = time.time()
                logger.info(f"Fear & Greed Index: {fgi} ({data['data'][0]['value_classification']})")
                return fgi
        except Exception as e:
            logger.debug(f"Failed to fetch Fear & Greed: {e}")
            return None

    def _compute_momentum_sentiment(self, df: pd.DataFrame) -> int:
        """Compute sentiment proxy from price momentum."""
        if len(df) < 30:
            return 50

        rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50
        roc = float(df["close"].pct_change(14).iloc[-1]) * 100

        # Map: RSI 30 = fear, RSI 70 = greed; ROC adjusts
        sentiment = rsi * 0.7 + np.clip(roc * 5 + 50, 0, 100) * 0.3
        return int(np.clip(sentiment, 0, 100))

    def _score_sentiment(self, fgi: int) -> SentimentResult:
        """Convert Fear & Greed Index to trading adjustments."""
        # Classification
        if fgi <= 25:
            label = "Extreme Fear"
        elif fgi <= 45:
            label = "Fear"
        elif fgi <= 55:
            label = "Neutral"
        elif fgi <= 75:
            label = "Greed"
        else:
            label = "Extreme Greed"

        # Bias: -1 = extreme fear, +1 = extreme greed
        bias = (fgi - 50) / 50.0

        # Contrarian logic: fear = buy opportunity, greed = sell opportunity
        if fgi <= 20:       # Extreme fear -> strong buy signal
            long_mult = 1.3
            short_mult = 0.6
            size_mult = 0.8  # Still cautious on size
        elif fgi <= 35:     # Fear -> mild buy bias
            long_mult = 1.15
            short_mult = 0.85
            size_mult = 0.9
        elif fgi <= 55:     # Neutral
            long_mult = 1.0
            short_mult = 1.0
            size_mult = 1.0
        elif fgi <= 75:     # Greed -> mild sell bias
            long_mult = 0.85
            short_mult = 1.15
            size_mult = 0.9
        else:               # Extreme greed -> strong sell signal
            long_mult = 0.6
            short_mult = 1.3
            size_mult = 0.8

        return SentimentResult(
            fear_greed_index=fgi,
            fear_greed_label=label,
            sentiment_bias=round(bias, 3),
            long_multiplier=long_mult,
            short_multiplier=short_mult,
            size_multiplier=size_mult,
            news_aligned=True,  # Updated during signal evaluation
        )

    def check_news_alignment(
        self, sentiment: SentimentResult, trend_up: bool
    ) -> SentimentResult:
        """
        Check if sentiment aligns with the current trend direction.
        Aligned: fear + downtrend or greed + uptrend -> stronger conviction
        Conflicting: fear + uptrend or greed + downtrend -> reduce size
        """
        fgi = sentiment.fear_greed_index

        if trend_up and fgi > 55:
            # Greed + uptrend: momentum aligned but risky
            sentiment.news_aligned = True
            sentiment.size_multiplier *= 0.9  # Slight caution
        elif not trend_up and fgi < 45:
            # Fear + downtrend: panic selling, aligned
            sentiment.news_aligned = True
            sentiment.size_multiplier *= 0.9
        elif trend_up and fgi < 35:
            # Fear + uptrend: contrarian buy - great setup
            sentiment.news_aligned = True
            sentiment.long_multiplier *= 1.1
        elif not trend_up and fgi > 65:
            # Greed + downtrend: contrarian sell - great setup
            sentiment.news_aligned = True
            sentiment.short_multiplier *= 1.1
        else:
            sentiment.news_aligned = True  # Neutral, no conflict

        return sentiment
