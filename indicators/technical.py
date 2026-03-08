"""
indicators/technical.py
-----------------------------------------------------------------
Technical Indicator Library
Computes all indicators needed by both strategies.
-----------------------------------------------------------------
Indicators:
  - EMA (fast, slow, 200)
  - MACD + histogram
  - RSI (Wilder's smoothing)
  - Bollinger Bands (upper, mid, lower, width)
  - ATR (Average True Range) + ATR%
  - ADX (Average Directional Index)
  - Z-score (price deviation from rolling mean)
  - Volume ratio (volume / volume MA)
-----------------------------------------------------------------
"""

import numpy as np
import pandas as pd


def add_all_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    ema_signal: int = 9,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    adx_period: int = 14,
    vol_ma_period: int = 20,
) -> pd.DataFrame:
    """Add all technical indicators to OHLCV DataFrame."""
    df = df.copy()

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # MACD
    df["macd_line"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd_line"].ewm(span=ema_signal, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # RSI (Wilder's smoothing)
    df["rsi"] = _rsi(df["close"], rsi_period)

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(bb_period).mean()
    bb_rolling_std = df["close"].rolling(bb_period).std()
    df["bb_upper"] = df["bb_mid"] + bb_std * bb_rolling_std
    df["bb_lower"] = df["bb_mid"] - bb_std * bb_rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR
    df["atr"] = _atr(df, atr_period)
    df["atr_pct"] = df["atr"] / df["close"]

    # ADX
    df["adx"] = _adx(df, adx_period)

    # Z-score (deviation from rolling mean)
    rolling_mean = df["close"].rolling(bb_period).mean()
    rolling_std = df["close"].rolling(bb_period).std()
    df["zscore"] = (df["close"] - rolling_mean) / (rolling_std + 1e-10)

    # Volume ratio
    df["volume_ma"] = df["volume"].rolling(vol_ma_period).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + 1e-10)

    # Volatility rank (percentile of recent ATR) — vectorized
    df["vol_rank"] = df["atr_pct"].rolling(100).rank(pct=True)

    # Short/long volatility ratio (for regime detection)
    vol_short = df["close"].pct_change().rolling(24).std()
    vol_long = df["close"].pct_change().rolling(72).std()
    df["vol_ratio_sl"] = vol_short / (vol_long + 1e-10)

    # Fill NaN from warmup period
    df.bfill(inplace=True)

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (exponential)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    # ATR for ADX
    atr = _atr(df, period)

    # Smoothed +DI and -DI
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx
