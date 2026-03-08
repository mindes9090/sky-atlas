"""
Scan candidate trading pairs through v2 backtest to find profitable ones.
Usage: python scan_pairs.py
"""
import sys
import logging
import ccxt
import pandas as pd
import numpy as np

logging.disable(logging.CRITICAL)  # Silence all logging

sys.path.insert(0, "/Users/mindaugasseredis/SKYATLAS")
from indicators.technical import add_all_indicators
from data.fetcher import compute_htf_trend_from_primary
from sentiment.news import SentimentAnalyzer
from risk.black_swan import BlackSwanDetector
from ml.signal_model import AssetProfile
from backtest.engine import Backtester
import config


# ── Big scan: ~50 candidates not in current universe ──
# Includes user requests: ETHBTC proxy, SUSHI, EGLD, JASMY, ANKR, TIA
CANDIDATES = [
    # User-requested pairs
    "SOL/USDT",       # Adding to universe
    "SUSHI/USDT",     # DeFi
    "EGLD/USDT",      # MultiversX L1
    "JASMY/USDT",     # IoT/Data
    "ANKR/USDT",      # Infrastructure
    "TIA/USDT",       # Celestia modular blockchain
    # Layer 1s & Layer 2s
    "LINK/USDT", "DOT/USDT", "LTC/USDT", "TRX/USDT",
    "ICP/USDT", "VET/USDT", "EOS/USDT", "THETA/USDT",
    "FIL/USDT", "STX/USDT", "STRK/USDT", "ZK/USDT",
    "MATIC/USDT", "TON/USDT", "KAS/USDT",
    # DeFi protocols
    "UNI/USDT", "CRV/USDT", "COMP/USDT", "SNX/USDT", "PENDLE/USDT",
    "GRT/USDT", "ENS/USDT", "1INCH/USDT", "YFI/USDT",
    "GMX/USDT", "JTO/USDT", "PYTH/USDT",
    # AI & Data
    "TAO/USDT", "RNDR/USDT", "AGIX/USDT", "OCEAN/USDT",
    # Gaming / Metaverse
    "SAND/USDT", "IMX/USDT", "AXS/USDT", "APE/USDT",
    "PIXEL/USDT", "RONIN/USDT",
    # Memes & trending
    "WIF/USDT", "ORDI/USDT", "LUNC/USDT",
    "TURBO/USDT", "NEIRO/USDT", "MEW/USDT",
    # Infrastructure / Storage / Misc
    "CAKE/USDT", "ZEC/USDT", "ONE/USDT",
    "CHZ/USDT", "CELO/USDT", "ZIL/USDT", "IOTA/USDT", "BAT/USDT",
    "RSR/USDT", "MASK/USDT", "FLOW/USDT", "CFX/USDT",
    "BLUR/USDT", "AEVO/USDT", "W/USDT",
    "SUPER/USDT", "NOT/USDT", "IO/USDT",
]


def generate_signals(df, profile):
    """Signal generation using main.py strategy."""
    from main import _generate_backtest_signals_v2
    return _generate_backtest_signals_v2(df, profile, "SCAN")


def main():
    exchange = ccxt.binance({"enableRateLimit": True})

    # Filter out pairs already in our universe
    existing = set(config.SYMBOLS)
    to_scan = [s for s in CANDIDATES if s not in existing]

    header = "{:<14s} | {:>8s} | {:>7s} | {:>6s} | {:>6s} | {:>6s} | {}".format(
        "Symbol", "Return", "Sharpe", "WR", "PF", "Trades", "Status"
    )
    print(f"Scanning {len(to_scan)} candidates (excluding {len(existing)} already in universe)")
    print(header)
    print("-" * 75)

    winners = []

    for symbol in to_scan:
        try:
            raw = exchange.fetch_ohlcv(symbol, "1h", limit=2000)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            df = add_all_indicators(df)

            profile = AssetProfile.get(symbol)
            df = generate_signals(df, profile)

            bt = Backtester(initial_capital=10000, commission=0.0005)
            report, _ = bt.run(df, "signal")

            is_good = (report.total_return_pct > 3.0
                       and report.sharpe_ratio > 0.5
                       and report.total_trades >= 3)
            status = "<<< ADD" if is_good else "SKIP"

            if is_good:
                winners.append((symbol, report))

            print("{:<14s} | {:>+7.2f}% | {:>7.3f} | {:>5.1f}% | {:>5.2f} | {:>6d} | {}".format(
                symbol, report.total_return_pct, report.sharpe_ratio,
                report.win_rate_pct, report.profit_factor,
                report.total_trades, status
            ))
        except Exception as e:
            err = str(e)[:45]
            print("{:<14s} | ERROR: {}".format(symbol, err))

    print("\n" + "=" * 75)
    print(f"RECOMMENDED ADDITIONS ({len(winners)} winners):")
    print("=" * 75)
    for sym, r in sorted(winners, key=lambda x: x[1].sharpe_ratio, reverse=True):
        print("  {:14s}  Return={:>+7.2f}%  Sharpe={:.3f}  PF={:.2f}  Trades={}".format(
            sym, r.total_return_pct, r.sharpe_ratio, r.profit_factor, r.total_trades
        ))


if __name__ == "__main__":
    main()
