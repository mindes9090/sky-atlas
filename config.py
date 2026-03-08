"""
=============================================================
  CRYPTO ALGO TRADING BOT - CONFIGURATION
  Based on: Kelly Criterion, GARCH Regime Detection,
  Trend Following + Mean Reversion adaptive strategy
  + Sentiment, Black Swan, ML, Multi-Timeframe
=============================================================
"""
import os

# Load .env file if it exists
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ─── EXCHANGE ────────────────────────────────────────────
EXCHANGE_ID = "binance"
PAPER_TRADING = True   # Set False for live trading

# API Keys loaded from .env (never hardcode secrets)
BINANCE_TESTNET_API_KEY    = os.environ.get("BINANCE_TESTNET_API_KEY", "")
BINANCE_TESTNET_API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
BINANCE_LIVE_API_KEY       = os.environ.get("BINANCE_LIVE_API_KEY", "")
BINANCE_LIVE_API_SECRET    = os.environ.get("BINANCE_LIVE_API_SECRET", "")
ANTHROPIC_API_KEY          = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── TRADING UNIVERSE ────────────────────────────────────
# Expanded: volatile altcoins perform best with our strategy
SYMBOLS = [
    # Core
    "BTC/USDT",
    "ETH/USDT",
    "DOGE/USDT",
    # Original v6 winners
    "NEAR/USDT",
    "JUP/USDT",
    "INJ/USDT",
    "FET/USDT",
    "SUI/USDT",
    "ATOM/USDT",
    "DYDX/USDT",
    # Scan winners (Mar 2026) — sorted by Sharpe
    "XRP/USDT",      # +20.8% Sharpe 7.21
    "ARB/USDT",      # +15.3% Sharpe 4.89
    "PEPE/USDT",     # +15.5% Sharpe 4.67
    "HBAR/USDT",     # +13.4% Sharpe 4.65
    "AAVE/USDT",     #  +9.4% Sharpe 4.59
    "RENDER/USDT",   # +11.5% Sharpe 4.40
    "BNB/USDT",      # +11.9% Sharpe 4.20
    "FLOKI/USDT",    # +13.6% Sharpe 4.14
    "APT/USDT",      # +14.3% Sharpe 4.13
    "SHIB/USDT",     # +11.3% Sharpe 4.08
    "ADA/USDT",      # +10.0% Sharpe 4.04
    "XLM/USDT",      # +11.2% Sharpe 4.04
    "BONK/USDT",     #  +7.6% Sharpe 2.70
    "OP/USDT",       #  +6.4% Sharpe 2.18
    "WLD/USDT",      #  +4.2% Sharpe 1.77
    "AVAX/USDT",     #  +3.5% Sharpe 1.68
    "SEI/USDT",      #  +3.8% Sharpe 1.33
    # Scan winners (Mar 2026 round 2)
    "ENA/USDT",      # +25.1% Sharpe 6.93
    "LDO/USDT",      # +12.1% Sharpe 3.90
    "MANA/USDT",     # +11.6% Sharpe 3.73
    "KAVA/USDT",     # +31.3% Sharpe 3.56
    "ALGO/USDT",     #  +8.7% Sharpe 3.01
    "RUNE/USDT",     #  +7.2% Sharpe 2.71
    "GALA/USDT",     #  +6.1% Sharpe 2.42
    "FTM/USDT",      #  +7.0% Sharpe 2.03
    "MKR/USDT",      #  +6.2% Sharpe 1.97
]

TIMEFRAME       = "1h"    # Primary timeframe
TIMEFRAME_HIGH  = "4h"    # Higher timeframe for multi-TF confirmation
CANDLES_LIMIT   = 500     # Number of candles to fetch

# ─── POSITION & RISK ────────────────────────────────────
TOTAL_CAPITAL        = 10_000     # USDT starting capital
MAX_RISK_PER_TRADE   = 0.02       # 2% max risk per trade
MAX_PORTFOLIO_RISK   = 0.06       # 6% max simultaneous portfolio risk
MAX_POSITIONS        = 15         # Max open positions (28 pairs in universe)
LEVERAGE             = 1          # 1x = no leverage

# Kelly Criterion settings
KELLY_FRACTION       = 0.25       # Quarter-Kelly for conservative sizing
KELLY_LOOKBACK       = 50         # Trades lookback for win rate estimation
MAX_KELLY_SIZE       = 0.10       # Cap: never more than 10% on one trade

# Stop Loss / Take Profit
ATR_STOP_MULTIPLIER  = 2.0        # Stop = entry +/- 2xATR
ATR_TP_MULTIPLIER    = 4.0        # TP   = entry +/- 4xATR
TRAILING_STOP_FACTOR = 0.015      # 1.5% trailing stop activation

# ─── RISK CIRCUIT BREAKERS ───────────────────────────────
MAX_DAILY_DRAWDOWN   = 0.05       # Halt trading if daily loss > 5%
MAX_TOTAL_DRAWDOWN   = 0.15       # Halt trading if total drawdown > 15%
MAX_CONSECUTIVE_LOSS = 5          # Pause after 5 losses in a row

# ─── BLACK SWAN PROTECTION ──────────────────────────────
FLASH_CRASH_1H_PCT   = 0.03       # 3% drop in 1h = flash crash
FLASH_CRASH_4H_PCT   = 0.05       # 5% drop in 4h = sustained crash
ATR_SPIKE_MULT       = 3.0        # ATR > 3x normal = vol explosion
VOLUME_SPIKE_MULT    = 5.0        # Volume > 5x normal
MAX_STOPS_PER_HOUR   = 3          # Cascade stop detection

# ─── SENTIMENT ──────────────────────────────────────────
USE_SENTIMENT        = True        # Enable sentiment layer
SENTIMENT_WEIGHT     = 0.3         # How much sentiment affects signals (0-1)

# ─── ML SIGNAL LAYER ───────────────────────────────────
USE_ML_SIGNALS       = True        # Enable ML confidence layer
ML_LOOKFORWARD       = 12          # Bars ahead for target
ML_TRAIN_WINDOW      = 500         # Training window size
ML_RETRAIN_EVERY     = 100         # Retrain interval

# ─── REGIME DETECTION ────────────────────────────────────
REGIME_VOL_WINDOW    = 24
REGIME_ADX_THRESHOLD = 25
REGIME_RSI_NEUTRAL   = (40, 60)

# ─── TECHNICAL INDICATORS ────────────────────────────────
EMA_FAST     = 12
EMA_SLOW     = 26
EMA_SIGNAL   = 9
ADX_PERIOD   = 14
ADX_MIN      = 25

BB_PERIOD    = 20
BB_STD       = 2.0
RSI_PERIOD   = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT  = 0.5

ATR_PERIOD   = 14

VOLUME_MA_PERIOD = 20
VOLUME_MIN_RATIO = 1.2

# ─── EXECUTION ───────────────────────────────────────────
ORDER_TYPE       = "limit"
SLIPPAGE_BPS     = 5
FEE_BPS          = 10

# ─── LOGGING & MONITORING ───────────────────────────────
LOG_LEVEL        = "INFO"
LOG_FILE         = "logs/bot.log"
TRADE_LOG_FILE   = "logs/trades.csv"
PERFORMANCE_FILE = "logs/performance.json"

LOOP_INTERVAL_SECONDS = 60

# ─── BACKTEST ────────────────────────────────────────────
BACKTEST_START   = "2023-01-01"
BACKTEST_END     = "2024-12-31"
INITIAL_CAPITAL  = 10_000
COMMISSION       = 0.001

# ─── PER-ASSET PROFILES (override defaults) ─────────────
# Defined in ml/signal_model.py AssetProfile class
# BTC: conservative (higher thresholds, wider stops)
# ETH: moderate
# Altcoins: aggressive (lower thresholds, tighter management)
