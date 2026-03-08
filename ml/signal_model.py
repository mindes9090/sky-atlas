"""
ml/signal_model.py
-----------------------------------------------------------------
Machine Learning Signal Layer
-----------------------------------------------------------------
Gradient Boosted Tree model that learns optimal signal weights
from historical data.

Features used:
  - RSI, ADX, Z-score, MACD histogram
  - Volume ratio, EMA slope, ROC
  - BB width, ATR percentage
  - Sentiment proxy, funding rate proxy
  - Multi-timeframe trend alignment

Training approach:
  - Walk-forward: train on past N bars, predict next window
  - Target: future N-bar return sign (+1 profitable, -1 loss)
  - Output: probability used as confidence multiplier

Falls back gracefully if sklearn not installed.
-----------------------------------------------------------------
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.info("scikit-learn not installed. ML features disabled.")


FEATURE_COLUMNS = [
    "rsi", "adx", "zscore", "macd_hist", "volume_ratio",
    "ema_slope", "roc_12", "roc_24", "bb_width", "atr_pct",
    "sentiment", "htf_trend", "vol_rank",
]


class SignalModel:
    """
    ML-based signal confidence estimator.
    Trains on historical data to predict profitable trade direction.
    """

    def __init__(
        self,
        lookforward: int = 12,        # Bars to look ahead for target
        min_return_pct: float = 0.005, # 0.5% min to count as profitable
        train_window: int = 500,       # Bars for training
        retrain_every: int = 100,      # Retrain every N bars
    ):
        self.lookforward = lookforward
        self.min_return_pct = min_return_pct
        self.train_window = train_window
        self.retrain_every = retrain_every

        self.model_long = None
        self.model_short = None
        self.scaler = None
        self.is_trained = False
        self.last_train_idx = 0

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ML features from indicator DataFrame."""
        features = pd.DataFrame(index=df.index)

        features["rsi"] = df.get("rsi", 50)
        features["adx"] = df.get("adx", 20)
        features["zscore"] = df.get("zscore", 0)
        features["macd_hist"] = df.get("macd_hist", 0)
        features["volume_ratio"] = df.get("volume_ratio", 1)
        features["bb_width"] = df.get("bb_width", 0.05)
        features["atr_pct"] = df.get("atr_pct", 0.02)
        features["vol_rank"] = df.get("vol_rank", 0.5)

        # Computed features
        features["ema_slope"] = df.get("ema_slope",
            df["close"].ewm(span=12, adjust=False).mean().pct_change(3)
            if "close" in df.columns else 0)
        features["roc_12"] = df.get("roc_12",
            df["close"].pct_change(12) if "close" in df.columns else 0)
        features["roc_24"] = df.get("roc_24",
            df["close"].pct_change(24) if "close" in df.columns else 0)

        # Sentiment and HTF (filled externally, default neutral)
        features["sentiment"] = df.get("sentiment", 50)
        features["htf_trend"] = df.get("htf_trend", 0)

        features = features.fillna(0)
        return features

    def prepare_targets(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable: future return direction.
        +1 = long profitable, -1 = short profitable, 0 = flat
        """
        future_ret = df["close"].shift(-self.lookforward) / df["close"] - 1

        target = pd.Series(0, index=df.index)
        target[future_ret > self.min_return_pct] = 1    # Long profitable
        target[future_ret < -self.min_return_pct] = -1   # Short profitable

        return target

    def train(self, df: pd.DataFrame, features: pd.DataFrame,
              targets: pd.Series, end_idx: int) -> bool:
        """
        Train the model on data up to end_idx.
        Returns True if training succeeded.
        """
        if not HAS_SKLEARN:
            return False

        start_idx = max(0, end_idx - self.train_window)
        train_feat = features.iloc[start_idx:end_idx]
        train_tgt = targets.iloc[start_idx:end_idx]

        # Need targets that aren't NaN (lookforward window)
        valid = train_tgt.notna() & (train_tgt != 0)
        if valid.sum() < 50:
            return False

        X = train_feat[valid].values
        y = train_tgt[valid].values

        # Need both classes
        if len(np.unique(y)) < 2:
            return False

        try:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Binary classifier: is LONG profitable?
            y_long = (y == 1).astype(int)
            self.model_long = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            self.model_long.fit(X_scaled, y_long)

            # Binary classifier: is SHORT profitable?
            y_short = (y == -1).astype(int)
            self.model_short = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            self.model_short.fit(X_scaled, y_short)

            self.is_trained = True
            self.last_train_idx = end_idx
            return True

        except Exception as e:
            logger.debug(f"ML training failed: {e}")
            return False

    def predict_confidence(
        self, features: pd.DataFrame, idx: int
    ) -> tuple[float, float]:
        """
        Predict confidence for long and short at given index.
        Returns (long_confidence, short_confidence) in [0, 1].
        """
        if not self.is_trained or self.model_long is None:
            return (0.5, 0.5)

        try:
            X = features.iloc[idx:idx+1].values
            X_scaled = self.scaler.transform(X)

            long_prob = self.model_long.predict_proba(X_scaled)[0][1]
            short_prob = self.model_short.predict_proba(X_scaled)[0][1]

            return (float(long_prob), float(short_prob))
        except Exception:
            return (0.5, 0.5)

    def should_retrain(self, current_idx: int) -> bool:
        """Check if model needs retraining."""
        return (current_idx - self.last_train_idx) >= self.retrain_every

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model_long is None:
            return None

        cols = FEATURE_COLUMNS
        imp = self.model_long.feature_importances_
        return {col: round(float(imp[i]), 4)
                for i, col in enumerate(cols) if i < len(imp)}


class AssetProfile:
    """
    Per-asset optimization parameters.
    Different thresholds for BTC (low vol) vs altcoins (high vol).
    """

    # Per-asset direction restrictions based on 2-year backtest analysis.
    # "both" = longs and shorts allowed
    # "long_only" = shorts disabled (asset chops too much to short profitably)
    # "short_only" = longs disabled (rare, only for extreme cases)
    #
    # Rationale (2yr backtest, Mar 2024 - Mar 2026):
    #   ETH shorts:  avg -0.25%/trade, 52% WR → loses despite decent WR (chops)
    #   AVAX shorts: avg +0.22%/trade, 49% WR → marginal, not worth the risk
    #   APT shorts:  avg  0.00%/trade, 48% WR → break even, wastes capital
    #   XLM shorts:  avg -0.38%/trade, 30% WR → terrible
    #   BONK shorts: avg +0.04%/trade, 50% WR → break even
    #   BNB shorts:  loses in backtest, choppy price action
    #   ADA shorts:  loses in backtest over 2yr
    #   LDO shorts:  loses in backtest over 2yr
    #   MKR shorts:  loses in backtest over 2yr
    #
    #   BOME shorts: avg +1.69%/trade, 67% WR → excellent
    #   ENA shorts:  avg +1.48%/trade, 62% WR → excellent
    #   DYDX shorts: avg +2.24%/trade, 71% WR → best in universe
    #   ALGO shorts: avg +1.34%/trade, 67% WR → excellent
    #   DOGE shorts: avg +0.38%/trade, 47% WR → decent

    DIRECTION_ALLOWED = {
        # Long-only: shorts unprofitable over 2 years
        "BTC/USDT":    "long_only",
        "ETH/USDT":    "long_only",
        "BONK/USDT":   "long_only",
        "LDO/USDT":    "long_only",
        "JUP/USDT":    "long_only",
        "FET/USDT":    "long_only",
        "SUI/USDT":    "long_only",
        "RUNE/USDT":   "long_only",
        "SOL/USDT":    "long_only",
    }

    # BTC-regime gated symbols: only trade when BTC > EMA200 (bullish).
    # These symbols bleed during BTC sideways — only trade in BTC uptrends.
    BTC_REGIME_GATED = {
        "ETH/USDT", "GALA/USDT", "BONK/USDT",
        "JUP/USDT", "RUNE/USDT",
    }

    PROFILES = {
        # BTC: widest stops — BTC trends slowly but far
        "BTC/USDT": {
            "stop_mult": 3.5,
            "tp_mult": 10.0,
            "trail_atr_mult": 3.0,
            "size_mult": 1.0,
            "vol_class": "low",
        },
        # ETH: wide stops
        "ETH/USDT": {
            "stop_mult": 3.0,
            "tp_mult": 8.0,
            "trail_atr_mult": 2.5,
            "size_mult": 0.9,
            "vol_class": "medium",
        },
    }

    # Default for altcoins — still wide to let trends develop
    DEFAULT_ALTCOIN = {
        "stop_mult": 3.0,
        "tp_mult": 7.0,
        "trail_atr_mult": 2.2,
        "size_mult": 0.8,
        "vol_class": "high",
    }

    @classmethod
    def get(cls, symbol: str) -> dict:
        return cls.PROFILES.get(symbol, cls.DEFAULT_ALTCOIN)

    @classmethod
    def allowed_direction(cls, symbol: str) -> str:
        """Return 'both', 'long_only', or 'short_only' for this symbol."""
        return cls.DIRECTION_ALLOWED.get(symbol, "both")

    @classmethod
    def is_btc_regime_gated(cls, symbol: str) -> bool:
        """Return True if this symbol should only trade when BTC is bullish."""
        return symbol in cls.BTC_REGIME_GATED
