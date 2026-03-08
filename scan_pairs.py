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


CANDIDATES = [
    # New scan — popular tokens not yet in our universe
    "SOL/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT", "UNI/USDT",
    "FIL/USDT", "TRX/USDT", "WIF/USDT", "CRV/USDT", "PENDLE/USDT",
    "TAO/USDT", "ENA/USDT", "TIA/USDT", "STX/USDT",
    "ALGO/USDT", "SAND/USDT", "MANA/USDT", "GRT/USDT", "IMX/USDT",
    "MKR/USDT", "COMP/USDT", "SNX/USDT", "LDO/USDT", "RUNE/USDT",
    "FTM/USDT", "GALA/USDT", "AXS/USDT", "APE/USDT",
    "THETA/USDT", "ICP/USDT", "VET/USDT", "EOS/USDT", "EGLD/USDT",
    "CAKE/USDT", "ZEC/USDT", "KAVA/USDT", "ONE/USDT", "ENS/USDT",
    "ASTR/USDT", "ORDI/USDT",
]


def generate_signals(df, profile):
    """v4 checklist-based signal generation — same logic as main.py."""
    from main import _generate_backtest_signals_v2
    return _generate_backtest_signals_v2(df, profile, "SCAN")


def main():
    exchange = ccxt.binance({"enableRateLimit": True})

    header = "{:<14s} | {:>8s} | {:>7s} | {:>6s} | {:>6s} | {:>6s} | {}".format(
        "Symbol", "Return", "Sharpe", "WR", "PF", "Trades", "Status"
    )
    print(header)
    print("-" * 75)

    winners = []

    for symbol in CANDIDATES:
        try:
            raw = exchange.fetch_ohlcv(symbol, "1h", limit=2000)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            df = add_all_indicators(df)

            profile = AssetProfile.get(symbol)
            df = generate_signals(df, profile)

            bt = Backtester(initial_capital=10000, commission=0.001)
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
    print("RECOMMENDED ADDITIONS:")
    print("=" * 75)
    for sym, r in sorted(winners, key=lambda x: x[1].sharpe_ratio, reverse=True):
        print("  {:14s}  Return={:>+7.2f}%  Sharpe={:.3f}  Trades={}".format(
            sym, r.total_return_pct, r.sharpe_ratio, r.total_trades
        ))


if __name__ == "__main__":
    main()
