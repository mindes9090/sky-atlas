"""
main.py
-------------------------------------------------------------
CRYPTO ALGO TRADING BOT - Main Orchestrator (v2)
-------------------------------------------------------------
Architecture:
  1. Fetch OHLCV data (primary + higher timeframe)
  2. Add technical indicators
  3. Check black swan protection
  4. Detect market regime (trending / ranging / volatile)
  5. Fetch sentiment (Fear & Greed + news catalyst)
  6. Select strategy + apply per-asset profile
  7. ML confidence layer
  8. Calculate position size (Kelly + sentiment + ML)
  9. Execute order (paper or live)
  10. Monitor stops / exits

Run:
  python main.py              -> Live bot loop (paper trading)
  python main.py --backtest   -> Backtest mode
  python main.py --once       -> Single scan, no loop
-------------------------------------------------------------
"""

import sys
import time
import logging
import argparse
import json
from datetime import datetime, date

import numpy as np
import pandas as pd

# Local imports
import config
from data.fetcher import BinanceDataFetcher, compute_htf_trend_from_primary
from indicators.technical import add_all_indicators
from regime.detector import RegimeDetector, Regime
from strategies.trend_following import TrendFollowingStrategy, Signal, TradeSignal
from strategies.mean_reversion import MeanReversionStrategy
from risk.manager import RiskManager, TradeRecord
from risk.black_swan import BlackSwanDetector, BlackSwanAction
from execution.trader import OrderExecutor
from backtest.engine import Backtester, print_report
from claude_agent.analyzer import ClaudeAnalyzer
from sentiment.news import SentimentAnalyzer
from ml.signal_model import SignalModel, AssetProfile

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ]
)
logger = logging.getLogger("CryptoBot")


# =============================================================
#  BOT CORE
# =============================================================

class CryptoTradingBot:
    """
    Regime-Adaptive Crypto Trading Bot (v2).
    Integrates: multi-TF, sentiment, black swan, ML, per-asset profiles.
    """

    def __init__(self):
        logger.info("Initialising CryptoTradingBot v2...")

        self.fetcher = BinanceDataFetcher(
            api_key=config.BINANCE_TESTNET_API_KEY,
            api_secret=config.BINANCE_TESTNET_API_SECRET,
            paper_trading=config.PAPER_TRADING,
        )
        self.regime_detector = RegimeDetector(
            adx_threshold=config.REGIME_ADX_THRESHOLD,
        )
        self.trend_strategy = TrendFollowingStrategy(
            ema_fast=config.EMA_FAST,
            ema_slow=config.EMA_SLOW,
            adx_min=config.ADX_MIN,
            atr_stop_mult=config.ATR_STOP_MULTIPLIER,
            atr_tp_mult=config.ATR_TP_MULTIPLIER,
        )
        self.mean_rev_strategy = MeanReversionStrategy(
            zscore_entry=config.ZSCORE_ENTRY,
            zscore_exit=config.ZSCORE_EXIT,
            rsi_oversold=config.RSI_OVERSOLD,
            rsi_overbought=config.RSI_OVERBOUGHT,
        )
        self.risk_manager = RiskManager(
            total_capital=config.TOTAL_CAPITAL,
            max_risk_per_trade=config.MAX_RISK_PER_TRADE,
            max_portfolio_risk=config.MAX_PORTFOLIO_RISK,
            max_positions=config.MAX_POSITIONS,
            kelly_fraction=config.KELLY_FRACTION,
            max_kelly_size=config.MAX_KELLY_SIZE,
            max_daily_drawdown=config.MAX_DAILY_DRAWDOWN,
            max_total_drawdown=config.MAX_TOTAL_DRAWDOWN,
        )
        self.executor = OrderExecutor(
            exchange=self.fetcher.exchange,
            paper_trading=config.PAPER_TRADING,
            fee_rate=config.FEE_BPS / 10_000,
        )
        self.claude = ClaudeAnalyzer(api_key=config.ANTHROPIC_API_KEY)
        self.black_swan = BlackSwanDetector(
            flash_crash_1h_pct=config.FLASH_CRASH_1H_PCT,
            flash_crash_4h_pct=config.FLASH_CRASH_4H_PCT,
        )
        self.sentiment = SentimentAnalyzer(use_live_api=True)

        self.last_daily_reset = date.today()
        self.last_exit_time: dict[str, datetime] = {}  # Cooldown tracking per symbol
        self.failed_symbols: dict[str, datetime] = {}  # Cooldown for failed orders
        self.COOLDOWN_SECONDS = 48 * 3600  # 48 bars * 1h = 2 days (matches backtest)
        self.FAILED_COOLDOWN_SECONDS = 3600  # 1h cooldown after failed order

        # Scan real account balance and use it for position sizing
        self._sync_balance()
        logger.info(f"Bot initialised. Symbols: {config.SYMBOLS}")

    def _sync_balance(self):
        """Fetch real account balance and update risk manager capital."""
        try:
            bal = self.fetcher.fetch_balance()
            total_funds = bal["total"]
            free_funds = bal["free"]

            usdt_info = bal["USDT"]
            usdc_info = bal["USDC"]

            logger.info(f"Account balance: "
                        f"USDT={usdt_info['total']:.2f} (free={usdt_info['free']:.2f}) | "
                        f"USDC={usdc_info['total']:.2f} (free={usdc_info['free']:.2f}) | "
                        f"TOTAL={total_funds:.2f} (free={free_funds:.2f})")

            if total_funds > 0:
                self.risk_manager.total_capital = total_funds
                self.risk_manager.current_equity = total_funds
                self.risk_manager.peak_equity = max(self.risk_manager.peak_equity, total_funds)
                self.risk_manager.daily_start_equity = total_funds
                logger.info(f"Risk manager capital set to ${total_funds:.2f}")
            else:
                logger.warning(f"Account balance is ${total_funds:.2f} — using config default ${config.TOTAL_CAPITAL}")
        except Exception as e:
            logger.warning(f"Failed to fetch balance: {e} — using config default ${config.TOTAL_CAPITAL}")

    def run(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")
        while True:
            try:
                self._daily_reset_if_needed()
                self._sync_balance()
                self._scan_all_symbols()
                logger.info(f"Sleeping {config.LOOP_INTERVAL_SECONDS}s...")
                time.sleep(config.LOOP_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                self._shutdown_report()
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(30)

    def run_once(self):
        self._sync_balance()
        self._scan_all_symbols()

    def _scan_all_symbols(self):
        # Fetch data for all symbols once (reused for crash check + processing)
        symbol_data = {}
        for symbol in config.SYMBOLS:
            try:
                df = self.fetcher.fetch_ohlcv(symbol, config.TIMEFRAME, config.CANDLES_LIMIT)
                df = add_all_indicators(df)
                symbol_data[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")

        # Check multi-asset correlation crash from fetched data
        all_returns = {}
        for symbol, df in symbol_data.items():
            if len(df) >= 2:
                ret = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
                all_returns[symbol] = ret

        crash_alert = self.black_swan.check_multi_asset(all_returns)
        if crash_alert.action == BlackSwanAction.CLOSE_ALL:
            logger.warning(f"MULTI-ASSET CRASH: {crash_alert.reason}")
            for sym in list(self.executor.positions.keys()):
                if sym in symbol_data:
                    price = symbol_data[sym]["close"].iloc[-1]
                    result = self.executor._close_position(sym, price, "correlated_crash")
                    if result:
                        self._record_closed_trade(result)
            return

        for symbol, df in symbol_data.items():
            try:
                self._process_symbol_v2(symbol, df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _process_symbol_v2(self, symbol: str, df: pd.DataFrame):
        """Live trading with v6 Donchian breakout (matches backtest logic)."""
        profile = AssetProfile.get(symbol)
        price = df["close"].iloc[-1]
        atr_v = df["atr"].iloc[-1] if "atr" in df.columns else price * 0.02

        # 1. Black swan check
        bs_alert = self.black_swan.check(df, symbol)
        if bs_alert.action in (BlackSwanAction.CLOSE_ALL, BlackSwanAction.HALT):
            if self.executor.has_position(symbol):
                result = self.executor._close_position(symbol, price, "black_swan")
                if result:
                    self._record_closed_trade(result)
            return

        # 2. Regime detection (for logging and position sizing)
        regime = self.regime_detector.detect(df)
        logger.info(f"[{symbol}] Regime: {regime.regime.value} "
                    f"(conf={regime.confidence:.0%} | ADX={regime.adx:.1f})")

        # 3. Check stops on existing positions
        if self.executor.has_position(symbol):
            close_result = self.executor.check_stops(symbol, price)
            if close_result:
                self._record_closed_trade(close_result)
                if close_result["reason"] == "stop_loss":
                    self.black_swan.record_stop_hit()
            return

        if self.risk_manager.trading_halted:
            logger.warning(f"Trading halted: {self.risk_manager.halt_reason}")
            return

        # 3b. Cooldown check (matches backtest COOLDOWN_BARS = 48)
        if symbol in self.last_exit_time:
            elapsed = (datetime.utcnow() - self.last_exit_time[symbol]).total_seconds()
            if elapsed < self.COOLDOWN_SECONDS:
                return

        # 3c. Failed order cooldown (don't retry for 1h)
        if symbol in self.failed_symbols:
            elapsed = (datetime.utcnow() - self.failed_symbols[symbol]).total_seconds()
            if elapsed < self.FAILED_COOLDOWN_SECONDS:
                return

        # 4. Donchian breakout detection (v6 strategy)
        ENTRY_LOOKBACK = 96  # 4-day high/low
        dc_high = df["high"].rolling(ENTRY_LOOKBACK).max()
        dc_low = df["low"].rolling(ENTRY_LOOKBACK).min()

        curr = df.iloc[-1]
        prev_dc_high = dc_high.iloc[-2] if len(dc_high) > 1 else price
        prev_dc_low = dc_low.iloc[-2] if len(dc_low) > 1 else price

        rsi = float(curr.get("rsi", 50))
        adx = float(curr.get("adx", 20))
        ema200 = float(curr.get("ema_200", price))

        breakout_up = (price > prev_dc_high)
        breakout_down = (price < prev_dc_low)

        # Long: breakout above 96-bar high + macro uptrend
        long_ok = breakout_up and price > ema200 and rsi < 80 and adx > 15
        # Short: breakout below 96-bar low + confirmed bear (very strict)
        short_ok = breakout_down and price < ema200 * 0.95 and rsi > 25 and adx > 20

        stop_mult = profile["stop_mult"]
        tp_mult = profile["tp_mult"]

        # Log signal status
        logger.info(f"[{symbol}] DC: high={prev_dc_high:.2f} low={prev_dc_low:.2f} | "
                    f"Price={price:.2f} EMA200={ema200:.2f} | "
                    f"RSI={rsi:.1f} ADX={adx:.1f} | "
                    f"Breakout={'UP' if breakout_up else 'DOWN' if breakout_down else 'NONE'}")

        direction = None
        if long_ok:
            direction = "LONG"
            stop_loss = price - stop_mult * atr_v
            take_profit = price + tp_mult * atr_v
            confidence = 0.7
        elif short_ok:
            direction = "SHORT"
            stop_loss = price + stop_mult * atr_v
            take_profit = price - tp_mult * atr_v
            confidence = 0.6
        else:
            return

        logger.info(f"[{symbol}] {direction} BREAKOUT signal | "
                    f"RSI={rsi:.1f} ADX={adx:.1f}")

        # 5. Position sizing
        regime_mult = self.regime_detector.get_position_size_multiplier(regime)
        regime_mult *= bs_alert.size_multiplier

        pos_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=price,
            stop_loss_price=stop_loss,
            signal_confidence=confidence,
            regime_multiplier=regime_mult * profile["size_mult"],
        )

        if not pos_size.approved:
            return

        # 6. Execute
        trade_signal = TradeSignal(
            signal=Signal.LONG if direction == "LONG" else Signal.SHORT,
            entry_price=price, stop_loss=stop_loss, take_profit=take_profit,
            atr=atr_v, confidence=confidence,
            reason=f"v6 Donchian breakout {direction}"
        )
        self._execute_trade(symbol, direction, trade_signal, pos_size, regime, profile)

    def _execute_trade(self, symbol, direction, trade_signal, pos_size, regime, profile):
        logger.info(f"[{symbol}] {direction} SIGNAL | {trade_signal.reason}")
        logger.info(f"[{symbol}] Size: {pos_size.units:.6f} | "
                    f"Risk: ${pos_size.risk_amount_usd:.2f}")

        # Log trade details (no Claude API call — saves ~$1.5/night)
        rr = abs(trade_signal.take_profit - trade_signal.entry_price) / (
            abs(trade_signal.stop_loss - trade_signal.entry_price) + 1e-10)
        logger.info(f"[{symbol}] {direction} | {trade_signal.reason} | "
                    f"R:R={rr:.1f} | Risk=${pos_size.risk_amount_usd:.2f}")

        trail_mult = profile["trail_atr_mult"]
        if direction == "LONG":
            result = self.executor.open_long(
                symbol, pos_size.units, trade_signal.entry_price,
                trade_signal.stop_loss, trade_signal.take_profit, trade_signal.atr,
                trail_atr_mult=trail_mult)
        else:
            result = self.executor.open_short(
                symbol, pos_size.units, trade_signal.entry_price,
                trade_signal.stop_loss, trade_signal.take_profit, trade_signal.atr,
                trail_atr_mult=trail_mult)

        # Only track in risk manager if order actually succeeded
        if result.get("status") == "opened":
            self.risk_manager.open_positions[symbol] = {
                "risk_amount_usd": pos_size.risk_amount_usd,
                "direction": direction,
            }
            self.failed_symbols.pop(symbol, None)
        else:
            logger.warning(f"[{symbol}] Order not opened: {result}")
            self.failed_symbols[symbol] = datetime.utcnow()

    def _record_closed_trade(self, close_result: dict):
        symbol = close_result["symbol"]
        trade = TradeRecord(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=close_result["direction"],
            entry=close_result["entry"],
            exit=close_result["exit"],
            pnl=close_result["net_pnl"],
            pnl_pct=close_result["pnl_pct"],
        )
        self.risk_manager.record_trade(trade)
        new_equity = self.risk_manager.current_equity + close_result["net_pnl"]
        self.risk_manager.update_equity(new_equity)
        self.risk_manager.open_positions.pop(symbol, None)
        # Record exit time for cooldown
        self.last_exit_time[symbol] = datetime.utcnow()

    def _daily_reset_if_needed(self):
        today = date.today()
        if today != self.last_daily_reset:
            # Claude AI: daily performance review
            stats = self.risk_manager.get_stats()
            review = self.claude.daily_performance_review(stats)
            logger.info(f"[DAILY REVIEW]\n{review}")

            self.risk_manager.reset_daily()
            self.last_daily_reset = today

    def _shutdown_report(self):
        stats = self.risk_manager.get_stats()
        logger.info(f"Final stats: {json.dumps(stats, indent=2)}")


# =============================================================
#  BACKTEST MODE (v2 - Full Integration)
# =============================================================

def run_backtest():
    """
    Run backtest with all v2 features:
    - Multi-timeframe confirmation
    - Per-asset optimization
    - Sentiment proxy
    - Black swan protection
    - ML confidence layer
    """
    import ccxt

    logger.info("Running backtest v2 (all features)...")

    exchange = ccxt.binance({"enableRateLimit": True})

    all_results = {}

    for symbol in config.SYMBOLS:
        logger.info(f"Fetching data for {symbol}...")
        try:
            raw = exchange.fetch_ohlcv(symbol, "1h", limit=2000)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            df = add_all_indicators(df)

            # Generate signals with all features
            profile = AssetProfile.get(symbol)
            df = _generate_backtest_signals_v2(df, profile, symbol)

            # Run backtest
            bt = Backtester(
                initial_capital=config.INITIAL_CAPITAL,
                commission=config.COMMISSION,
            )
            report, bt_df = bt.run(df, "signal")
            print_report(report, f"{symbol} v2 Strategy")
            all_results[symbol] = report

            # Walk-forward validation
            wf_results = bt.walk_forward(df, "signal", n_splits=5)
            wf_sharpes = [r["report"].sharpe_ratio for r in wf_results]
            wf_returns = [r["report"].total_return_pct for r in wf_results]
            logger.info(f"{symbol} Walk-forward Sharpes: "
                        f"{[round(float(s), 3) for s in wf_sharpes]}")
            logger.info(f"{symbol} Walk-forward Returns: "
                        f"{[round(float(r), 2) for r in wf_returns]}")

        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}", exc_info=True)

    # Portfolio summary
    if all_results:
        _print_portfolio_summary(all_results)


def _generate_backtest_signals_v2(
    df: pd.DataFrame, profile: dict, symbol: str
) -> pd.DataFrame:
    """
    v6 signal generator — Donchian breakout + trend filter.

    Based on Turtle Traders methodology adapted for crypto:
    - LONG: price breaks above 72-bar high in bull regime
    - SHORT: price breaks below 72-bar low in confirmed bear regime
    - EXIT: trailing stop (ATR) or break of 24-bar low/high (tighter channel)

    Why this works:
    - Breakouts have built-in momentum confirmation
    - Very few trades (~5-15/year/symbol)
    - Rides big trends for days/weeks
    - Proven across asset classes since 1980s
    """
    df = df.copy()
    df["signal"] = 0

    # Donchian channels
    ENTRY_LOOKBACK = 96     # 4-day high/low for entry
    EXIT_LOOKBACK = 48      # 2-day high/low for exit (kept for reference)
    df["dc_high"] = df["high"].rolling(ENTRY_LOOKBACK).max()
    df["dc_low"] = df["low"].rolling(ENTRY_LOOKBACK).min()
    df["dc_exit_low"] = df["low"].rolling(EXIT_LOOKBACK).min()
    df["dc_exit_high"] = df["high"].rolling(EXIT_LOOKBACK).max()

    bs_detector = BlackSwanDetector()

    trail_mult = profile["trail_atr_mult"]
    stop_mult = profile["stop_mult"]

    COOLDOWN_BARS = 48      # 2 days between trades

    current_position = None
    entry_price = 0.0
    entry_atr = 0.0
    bars_in_trade = 0
    bars_since_last_exit = 999
    highest_since_entry = 0.0
    lowest_since_entry = float("inf")

    sig_col = df.columns.get_loc("signal")

    for i in range(200, len(df) - 1):
        curr = df.iloc[i]
        price = curr["close"]
        atr_v = curr.get("atr", price * 0.02)
        ema200 = curr.get("ema_200", price)

        # Black swan check
        bs_mult = bs_detector.get_backtest_size_multiplier(df, i)
        if bs_mult == 0.0 and current_position is not None:
            df.iloc[i, sig_col] = 0
            current_position = None
            bars_in_trade = 0
            bars_since_last_exit = 0
            continue

        # ═══ MANAGE OPEN POSITIONS ═══
        if current_position is not None:
            bars_in_trade += 1

            if current_position == "LONG":
                highest_since_entry = max(highest_since_entry, price)
                trail_stop = highest_since_entry - trail_mult * entry_atr
                fixed_stop = entry_price - stop_mult * entry_atr
                hit_stop = price <= max(fixed_stop, trail_stop)
            else:
                lowest_since_entry = min(lowest_since_entry, price)
                trail_stop = lowest_since_entry + trail_mult * entry_atr
                fixed_stop = entry_price + stop_mult * entry_atr
                hit_stop = price >= min(fixed_stop, trail_stop)

            if hit_stop:
                df.iloc[i, sig_col] = 0
                current_position = None
                bars_in_trade = 0
                bars_since_last_exit = 0
                continue

            df.iloc[i, sig_col] = 1 if current_position == "LONG" else -1
            continue

        # ═══ NOT IN POSITION ═══
        bars_since_last_exit += 1
        if bs_mult == 0.0:
            continue
        if bars_since_last_exit < COOLDOWN_BARS:
            continue

        rsi = curr.get("rsi", 50)
        adx = curr.get("adx", 20)

        # Donchian breakout signals
        prev_dc_high = df.iloc[i - 1].get("dc_high", price)
        prev_dc_low = df.iloc[i - 1].get("dc_low", price)

        breakout_up = (price > prev_dc_high)
        breakout_down = (price < prev_dc_low)

        # ═══ LONG ENTRY: breakout above 72-bar high ═══
        long_ok = breakout_up
        if long_ok:
            # Bull regime: price above EMA200
            if price < ema200:
                long_ok = False
            # Not extremely overbought (RSI < 80)
            if rsi > 80:
                long_ok = False
            # Some trend strength
            if adx < 15:
                long_ok = False

        # ═══ SHORT ENTRY: breakout below 72-bar low (very strict) ═══
        short_ok = breakout_down
        if short_ok:
            # Bear regime: price well below EMA200
            if price > ema200 * 0.95:
                short_ok = False
            # Not oversold
            if rsi < 25:
                short_ok = False
            # Strong trend
            if adx < 20:
                short_ok = False

        # ═══ ENTRY DECISION ═══
        if long_ok:
            df.iloc[i, sig_col] = 1
            current_position = "LONG"
            entry_price = price
            entry_atr = atr_v
            bars_in_trade = 0
            highest_since_entry = price
            lowest_since_entry = price

        elif short_ok:
            df.iloc[i, sig_col] = -1
            current_position = "SHORT"
            entry_price = price
            entry_atr = atr_v
            bars_in_trade = 0
            highest_since_entry = price
            lowest_since_entry = price

    return df


def _print_portfolio_summary(results: dict):
    """Print aggregated portfolio performance."""
    print(f"\n{'='*60}")
    print(f"  PORTFOLIO SUMMARY ({len(results)} assets)")
    print(f"{'='*60}")

    total_trades = 0
    weighted_return = 0
    all_sharpes = []
    all_win_rates = []
    all_pf = []

    for sym, r in results.items():
        total_trades += r.total_trades
        weighted_return += r.total_return_pct
        all_sharpes.append(r.sharpe_ratio)
        all_win_rates.append(r.win_rate_pct)
        all_pf.append(r.profit_factor)

        status = "+" if r.total_return_pct > 0 else ""
        print(f"  {sym:12s} | {status}{r.total_return_pct:>7.2f}% | "
              f"Sharpe {r.sharpe_ratio:>6.2f} | "
              f"WR {r.win_rate_pct:>5.1f}% | "
              f"PF {r.profit_factor:>5.2f} | "
              f"{r.total_trades:>3d} trades")

    avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0
    avg_wr = np.mean(all_win_rates) if all_win_rates else 0
    avg_pf = np.mean(all_pf) if all_pf else 0

    print(f"{'─'*60}")
    print(f"  {'TOTAL':12s} | {weighted_return:>+7.2f}% | "
          f"Sharpe {avg_sharpe:>6.2f} | "
          f"WR {avg_wr:>5.1f}% | "
          f"PF {avg_pf:>5.2f} | "
          f"{total_trades:>3d} trades")
    print(f"{'='*60}\n")


# =============================================================
#  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Algo Trading Bot v2")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest mode")
    parser.add_argument("--once", action="store_true",
                        help="Run single scan cycle and exit")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.once:
        bot = CryptoTradingBot()
        bot.run_once()
    else:
        bot = CryptoTradingBot()
        bot.run()
