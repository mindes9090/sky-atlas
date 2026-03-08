"""
Historical backtest — fetch extended data going back 1-2 years.
Usage: python backtest_historical.py
"""
import sys
import time
import logging
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HistBacktest")

sys.path.insert(0, "/Users/mindaugasseredis/SKYATLAS")
from indicators.technical import add_all_indicators
from data.fetcher import compute_htf_trend_from_primary
from sentiment.news import SentimentAnalyzer
from risk.black_swan import BlackSwanDetector
from ml.signal_model import SignalModel, AssetProfile
from backtest.engine import Backtester, compute_performance, print_report
import config


def fetch_extended_ohlcv(exchange, symbol, timeframe="1h", days_back=365):
    """Fetch historical candles by paginating backwards."""
    all_candles = []
    tf_ms = {"1h": 3600000, "4h": 14400000, "1d": 86400000}
    interval_ms = tf_ms.get(timeframe, 3600000)

    end_ts = int(datetime.utcnow().timestamp() * 1000)
    start_ts = int((datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000)

    since = start_ts
    batch = 0

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + interval_ms
            batch += 1
            if batch % 5 == 0:
                logger.info(f"  {symbol}: fetched {len(all_candles)} candles so far...")
            time.sleep(0.2)  # rate limit
        except Exception as e:
            logger.warning(f"  {symbol}: fetch error at batch {batch}: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df.astype(float)

    return df


def generate_signals(df, profile, symbol):
    """v3 signal generator — uses main.py's _generate_backtest_signals_v2."""
    from main import _generate_backtest_signals_v2

    # Count longs/shorts from the signal column
    result_df = _generate_backtest_signals_v2(df, profile, symbol)
    signals = result_df["signal"]
    positions = signals.diff().fillna(0)

    long_entries = ((signals == 1) & (signals.shift(1) != 1)).sum()
    short_entries = ((signals == -1) & (signals.shift(1) != -1)).sum()

    return result_df, int(long_entries), int(short_entries)


def main():
    exchange = ccxt.binance({"enableRateLimit": True})

    # Test periods
    periods = [
        ("6 MONTHS", 180),
        ("1 YEAR", 365),
        ("2 YEARS", 730),
    ]

    for period_name, days in periods:
        print(f"\n{'='*70}")
        print(f"  HISTORICAL BACKTEST — {period_name} ({days} days)")
        print(f"{'='*70}")

        all_results = {}
        all_longs = 0
        all_shorts = 0

        for symbol in config.SYMBOLS:
            logger.info(f"Fetching {days} days of {symbol}...")
            try:
                df = fetch_extended_ohlcv(exchange, symbol, "1h", days_back=days)
                if len(df) < 500:
                    logger.warning(f"  {symbol}: only {len(df)} candles, skipping")
                    continue

                logger.info(f"  {symbol}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                df = add_all_indicators(df)

                profile = AssetProfile.get(symbol)
                df, longs, shorts = generate_signals(df, profile, symbol)
                all_longs += longs
                all_shorts += shorts

                bt = Backtester(initial_capital=config.INITIAL_CAPITAL, commission=config.COMMISSION)
                report, _ = bt.run(df, "signal")
                all_results[symbol] = report

                status = "+" if report.total_return_pct > 0 else " "
                print(f"  {symbol:14s} | {status}{report.total_return_pct:>7.2f}% | "
                      f"Sharpe {report.sharpe_ratio:>6.2f} | "
                      f"WR {report.win_rate_pct:>5.1f}% | "
                      f"PF {report.profit_factor:>5.2f} | "
                      f"{report.total_trades:>3d} trades | "
                      f"MaxDD {report.max_drawdown_pct:>6.2f}%")

            except Exception as e:
                logger.error(f"  {symbol}: {e}")

        if all_results:
            total_ret = sum(r.total_return_pct for r in all_results.values())
            avg_sharpe = np.mean([r.sharpe_ratio for r in all_results.values()])
            avg_wr = np.mean([r.win_rate_pct for r in all_results.values()])
            avg_pf = np.mean([r.profit_factor for r in all_results.values()])
            total_trades = sum(r.total_trades for r in all_results.values())
            avg_dd = np.mean([r.max_drawdown_pct for r in all_results.values()])

            print(f"{'─'*70}")
            print(f"  {'TOTAL':14s} | {total_ret:>+7.2f}% | "
                  f"Sharpe {avg_sharpe:>6.2f} | "
                  f"WR {avg_wr:>5.1f}% | "
                  f"PF {avg_pf:>5.2f} | "
                  f"{total_trades:>3d} trades | "
                  f"AvgDD {avg_dd:>6.2f}%")
            print(f"  Direction split: {all_longs} longs / {all_shorts} shorts "
                  f"({all_longs/(all_longs+all_shorts+1e-10)*100:.0f}% / "
                  f"{all_shorts/(all_longs+all_shorts+1e-10)*100:.0f}%)")


if __name__ == "__main__":
    main()
