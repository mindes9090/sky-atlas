"""
data/fetcher.py
-----------------------------------------------------------------
Binance Data Fetcher (Testnet + Live)
-----------------------------------------------------------------
Features:
  - OHLCV candles (any timeframe)
  - Multi-timeframe data (1h + 4h simultaneously)
  - Funding rate for futures
  - Account balance and positions
-----------------------------------------------------------------
"""

import ccxt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Fetches OHLCV candles, funding rates, and account data from Binance.
    Supports both testnet (paper) and live modes.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        paper_trading: bool = True,
    ):
        self.paper_trading = paper_trading

        # Public exchange for market data (no auth needed)
        self.public_exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if paper_trading:
            # Connect to Binance futures testnet for balance
            # CCXT dropped sandbox support, so set URLs manually
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            # Override to testnet URLs directly
            self.exchange.urls["api"] = {
                "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                "public": "https://testnet.binancefuture.com/api/v3",
                "private": "https://testnet.binancefuture.com/api/v3",
            }
            logger.info("Using Binance FUTURES TESTNET for balance")
        else:
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            logger.info("Using Binance LIVE")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles."""
        logger.debug(f"Fetching {limit} {timeframe} candles for {symbol}")
        raw = self.public_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        logger.debug(f"Fetched {len(df)} candles for {symbol}")
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str,
        tf_primary: str = "1h",
        tf_higher: str = "4h",
        limit_primary: int = 500,
        limit_higher: int = 200,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch both primary and higher timeframe data.
        Returns (primary_df, higher_tf_df).
        """
        df_primary = self.fetch_ohlcv(symbol, tf_primary, limit_primary)
        df_higher = self.fetch_ohlcv(symbol, tf_higher, limit_higher)
        return df_primary, df_higher

    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch current funding rate for a futures symbol.
        Positive = longs pay shorts (bearish pressure)
        Negative = shorts pay longs (bullish pressure)
        Returns 0.0 on error.
        """
        try:
            # Convert symbol format for API
            ticker = self.public_exchange.fetch_ticker(symbol)
            info = ticker.get("info", {})
            funding = float(info.get("lastFundingRate", 0))
            logger.debug(f"Funding rate {symbol}: {funding:.6f}")
            return funding
        except Exception as e:
            logger.debug(f"Failed to fetch funding rate for {symbol}: {e}")
            return 0.0

    def fetch_funding_rate_history(
        self, symbol: str, limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        Returns DataFrame with timestamp and funding_rate columns.
        """
        try:
            # Use CCXT's funding rate history
            rates = self.public_exchange.fetch_funding_rate_history(
                symbol, limit=limit
            )
            if not rates:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])

            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["funding_rate"] = df["fundingRate"].astype(float)
            return df[["timestamp", "funding_rate"]]
        except Exception as e:
            logger.debug(f"Funding history unavailable for {symbol}: {e}")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

    def fetch_balance(self) -> dict:
        """Fetch current account balance for all stablecoins."""
        if self.paper_trading:
            return self._fetch_testnet_balance()
        else:
            return self._fetch_live_balance()

    def _fetch_live_balance(self) -> dict:
        """Fetch balance from live Binance."""
        balance = self.exchange.fetch_balance()
        return self._parse_balance(balance)

    def _fetch_testnet_balance(self) -> dict:
        """Fetch balance from Binance futures testnet directly."""
        import hmac
        import hashlib
        import time
        import urllib.request
        import json

        base_url = "https://testnet.binancefuture.com"
        api_key = self.exchange.apiKey
        api_secret = self.exchange.secret

        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}"
        signature = hmac.new(
            api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

        url = f"{base_url}/fapi/v2/account?{query}&signature={signature}"
        req = urllib.request.Request(url, headers={
            "X-MBX-APIKEY": api_key,
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        # Parse futures account assets
        assets = {a["asset"]: a for a in data.get("assets", [])}
        usdt = assets.get("USDT", {})
        usdc = assets.get("USDC", {})

        usdt_balance = float(usdt.get("walletBalance", 0))
        usdt_available = float(usdt.get("availableBalance", 0))
        usdt_used = usdt_balance - usdt_available

        usdc_balance = float(usdc.get("walletBalance", 0))
        usdc_available = float(usdc.get("availableBalance", 0))
        usdc_used = usdc_balance - usdc_available

        return {
            "USDT": {"total": usdt_balance, "free": usdt_available, "used": usdt_used},
            "USDC": {"total": usdc_balance, "free": usdc_available, "used": usdc_used},
            "total": usdt_balance + usdc_balance,
            "free": usdt_available + usdc_available,
            "used": usdt_used + usdc_used,
        }

    def _parse_balance(self, balance: dict) -> dict:
        """Parse CCXT balance response."""
        usdt = balance.get("USDT", {})
        usdc = balance.get("USDC", {})

        usdt_total = float(usdt.get("total", 0) or 0)
        usdt_free = float(usdt.get("free", 0) or 0)
        usdt_used = float(usdt.get("used", 0) or 0)

        usdc_total = float(usdc.get("total", 0) or 0)
        usdc_free = float(usdc.get("free", 0) or 0)
        usdc_used = float(usdc.get("used", 0) or 0)

        return {
            "USDT": {"total": usdt_total, "free": usdt_free, "used": usdt_used},
            "USDC": {"total": usdc_total, "free": usdc_free, "used": usdc_used},
            "total": usdt_total + usdc_total,
            "free": usdt_free + usdc_free,
            "used": usdt_used + usdc_used,
        }

    def fetch_positions(self) -> list:
        """Fetch open positions (futures)."""
        positions = self.exchange.fetch_positions()
        return [
            p for p in positions
            if float(p.get("contracts", 0)) > 0
        ]


def compute_htf_trend(
    df_higher: pd.DataFrame, df_primary: pd.DataFrame
) -> pd.Series:
    """
    Compute higher-timeframe trend direction and merge into primary timeframe.

    Returns Series aligned to primary index:
      +1 = HTF bullish (price > EMA20 on 4h)
      -1 = HTF bearish (price < EMA20 on 4h)
       0 = neutral
    """
    htf = df_higher.copy()
    htf["ema_20"] = htf["close"].ewm(span=20, adjust=False).mean()
    htf["ema_50"] = htf["close"].ewm(span=50, adjust=False).mean()

    # HTF trend: above both EMAs = bullish, below both = bearish
    htf["htf_signal"] = 0
    htf.loc[(htf["close"] > htf["ema_20"]) & (htf["close"] > htf["ema_50"]),
            "htf_signal"] = 1
    htf.loc[(htf["close"] < htf["ema_20"]) & (htf["close"] < htf["ema_50"]),
            "htf_signal"] = -1

    # Forward-fill HTF signal to primary timeframe
    htf_resampled = htf["htf_signal"].reindex(
        df_primary.index, method="ffill"
    )

    return htf_resampled.fillna(0).astype(int)


def compute_htf_trend_from_primary(df: pd.DataFrame) -> pd.Series:
    """
    Approximate higher-timeframe trend from primary data only.
    Uses longer EMAs as proxy for multi-TF confirmation.
    """
    ema_48 = df["close"].ewm(span=48, adjust=False).mean()
    ema_96 = df["close"].ewm(span=96, adjust=False).mean()

    htf = pd.Series(0, index=df.index)
    htf[(df["close"] > ema_48) & (df["close"] > ema_96)] = 1
    htf[(df["close"] < ema_48) & (df["close"] < ema_96)] = -1

    return htf
