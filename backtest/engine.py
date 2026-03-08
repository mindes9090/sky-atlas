"""
backtest/engine.py  +  metrics/performance.py
─────────────────────────────────────────────────────────────
Vectorized Backtesting Engine + Performance Analytics
─────────────────────────────────────────────────────────────
Metrics computed:
  - Total return, CAGR
  - Sharpe Ratio       (excess return / volatility)
  - Sortino Ratio      (excess return / downside deviation)
  - Calmar Ratio       (CAGR / max drawdown)
  - Max Drawdown + duration
  - Win rate, avg win/loss, profit factor
  - Expectancy (expected $ per trade)
─────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════

@dataclass
class PerformanceReport:
    # Returns
    total_return_pct : float
    cagr_pct         : float
    # Risk-adjusted
    sharpe_ratio     : float
    sortino_ratio    : float
    calmar_ratio     : float
    # Drawdown
    max_drawdown_pct : float
    max_dd_duration  : int       # bars in drawdown
    # Trade stats
    total_trades     : int
    win_rate_pct     : float
    avg_win_pct      : float
    avg_loss_pct     : float
    profit_factor    : float
    expectancy       : float     # avg $ per trade
    # Other
    sharpe_annualized: float
    best_trade_pct   : float
    worst_trade_pct  : float


def compute_performance(equity_curve: pd.Series,
                         trade_returns: pd.Series,
                         risk_free_rate: float = 0.04,  # 4% annual
                         periods_per_year: int = 8760,  # hourly
                         ) -> PerformanceReport:
    """
    Compute full performance report from equity curve + trade returns.
    
    equity_curve: pd.Series of portfolio value over time
    trade_returns: pd.Series of per-trade return % (e.g. 0.02 = 2%)
    """
    if len(equity_curve) < 2:
        return _empty_report()

    # ── Returns ─────────────────────────────────────────
    period_returns  = equity_curve.pct_change().dropna()
    total_return    = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    n_periods       = len(period_returns)
    n_years         = n_periods / periods_per_year
    cagr            = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # ── Sharpe Ratio ─────────────────────────────────────
    # (Annualised excess return) / (Annualised std)
    rf_per_period   = risk_free_rate / periods_per_year
    excess_returns  = period_returns - rf_per_period
    sharpe          = (excess_returns.mean() / (excess_returns.std() + 1e-10)
                       * np.sqrt(periods_per_year))

    # ── Sortino Ratio ────────────────────────────────────
    # Like Sharpe but only penalises downside deviation
    downside_ret    = excess_returns[excess_returns < 0]
    downside_std    = downside_ret.std() if len(downside_ret) > 0 else 1e-10
    sortino         = (excess_returns.mean() / downside_std
                       * np.sqrt(periods_per_year))

    # ── Max Drawdown ─────────────────────────────────────
    rolling_max     = equity_curve.cummax()
    drawdown        = (equity_curve - rolling_max) / rolling_max
    max_dd          = drawdown.min()

    # Drawdown duration (longest period below previous peak)
    in_dd           = (drawdown < 0).astype(int)
    dd_groups       = (in_dd != in_dd.shift()).cumsum()
    dd_dur_series   = in_dd.groupby(dd_groups).sum()
    max_dd_dur      = int(dd_dur_series.max()) if len(dd_dur_series) > 0 else 0

    # ── Calmar Ratio ─────────────────────────────────────
    calmar          = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # ── Trade Statistics ─────────────────────────────────
    if len(trade_returns) > 0:
        wins        = trade_returns[trade_returns > 0]
        losses      = trade_returns[trade_returns <= 0]
        win_rate    = len(wins) / len(trade_returns)
        avg_win     = wins.mean()  if len(wins)   > 0 else 0
        avg_loss    = losses.mean() if len(losses) > 0 else 0

        gross_profit  = wins.sum()
        gross_loss    = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 9.99

        # Expectancy = (win_rate × avg_win) + (loss_rate × avg_loss)
        expectancy  = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        best_trade  = trade_returns.max()
        worst_trade = trade_returns.min()
    else:
        win_rate = avg_win = avg_loss = profit_factor = expectancy = 0
        best_trade = worst_trade = 0

    return PerformanceReport(
        total_return_pct  = round(total_return * 100, 2),
        cagr_pct          = round(cagr * 100, 2),
        sharpe_ratio      = round(sharpe, 3),
        sortino_ratio     = round(sortino, 3),
        calmar_ratio      = round(calmar, 3),
        max_drawdown_pct  = round(max_dd * 100, 2),
        max_dd_duration   = max_dd_dur,
        total_trades      = len(trade_returns),
        win_rate_pct      = round(win_rate * 100, 2),
        avg_win_pct       = round(avg_win * 100, 4),
        avg_loss_pct      = round(avg_loss * 100, 4),
        profit_factor     = round(profit_factor, 3),
        expectancy        = round(expectancy * 100, 4),
        sharpe_annualized = round(sharpe, 3),
        best_trade_pct    = round(best_trade * 100, 4),
        worst_trade_pct   = round(worst_trade * 100, 4),
    )


def _empty_report() -> PerformanceReport:
    return PerformanceReport(*([0.0] * 16))


def print_report(report: PerformanceReport, strategy_name: str = "Strategy"):
    """Pretty-print performance report to console."""
    print(f"\n{'='*55}")
    print(f"  📊 PERFORMANCE REPORT — {strategy_name}")
    print(f"{'='*55}")
    print(f"  Returns")
    print(f"    Total Return     : {report.total_return_pct:>8.2f}%")
    print(f"    CAGR             : {report.cagr_pct:>8.2f}%")
    print(f"  Risk-Adjusted")
    print(f"    Sharpe Ratio     : {report.sharpe_ratio:>8.3f}  (>1.0 = good)")
    print(f"    Sortino Ratio    : {report.sortino_ratio:>8.3f}  (>2.0 = great)")
    print(f"    Calmar Ratio     : {report.calmar_ratio:>8.3f}")
    print(f"  Drawdown")
    print(f"    Max Drawdown     : {report.max_drawdown_pct:>8.2f}%")
    print(f"    Max DD Duration  : {report.max_dd_duration:>8d} bars")
    print(f"  Trade Stats")
    print(f"    Total Trades     : {report.total_trades:>8d}")
    print(f"    Win Rate         : {report.win_rate_pct:>8.2f}%")
    print(f"    Avg Win          : {report.avg_win_pct:>8.4f}%")
    print(f"    Avg Loss         : {report.avg_loss_pct:>8.4f}%")
    print(f"    Profit Factor    : {report.profit_factor:>8.3f}  (>1.5 = good)")
    print(f"    Expectancy       : {report.expectancy:>8.4f}%/trade")
    print(f"    Best Trade       : {report.best_trade_pct:>8.4f}%")
    print(f"    Worst Trade      : {report.worst_trade_pct:>8.4f}%")
    print(f"{'='*55}\n")


# ═══════════════════════════════════════════════════════════
#  VECTORIZED BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════

class Backtester:
    """
    Fast vectorized backtester for strategy evaluation.
    
    Usage:
        bt = Backtester(initial_capital=10000, commission=0.001)
        report = bt.run(df, signals)
    """

    def __init__(
        self,
        initial_capital : float = 10_000,
        commission       : float = 0.0005,  # 0.05% taker fee (Binance Futures)
        slippage         : float = 0.0005,  # 0.05% avg slippage
        funding_per_bar  : float = 0.0000125, # 0.01%/8h on 1h bars (0.01/100/8)
    ):
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        self.funding_per_bar = funding_per_bar

    def run(self, df: pd.DataFrame,
            signal_col: str = "signal") -> PerformanceReport:
        """
        Run backtest on a DataFrame with pre-computed signals.
        
        df must have columns: close, signal
        signal: +1 = long, -1 = short, 0 = flat
        
        Returns PerformanceReport.
        """
        df = df.copy()
        df["position"] = df[signal_col].shift(1).fillna(0)  # Execute next bar

        # ── Gross returns ─────────────────────────────────
        df["returns"]   = df["close"].pct_change().fillna(0)
        df["strat_ret"] = df["position"] * df["returns"]

        # ── Deduct costs on entry/exit ────────────────────
        trades    = df["position"].diff().fillna(0).abs()
        cost      = trades * (self.commission + self.slippage)
        df["strat_ret"] -= cost

        # ── Funding rate cost (charged while in position) ──
        in_position = (df["position"] != 0).astype(float)
        df["strat_ret"] -= in_position * self.funding_per_bar

        df["strat_ret"] = df["strat_ret"].fillna(0)

        # ── Equity curve ─────────────────────────────────
        df["equity"] = self.initial_capital * (1 + df["strat_ret"]).cumprod()

        # ── Trade-level returns ───────────────────────────
        trade_rets = self._extract_trade_returns(df)

        report = compute_performance(df["equity"], pd.Series(trade_rets))

        logger.info(f"Backtest complete — {report.total_trades} trades | "
                    f"Return: {report.total_return_pct:.2f}% | "
                    f"Sharpe: {report.sharpe_ratio:.3f}")
        return report, df

    def _extract_trade_returns(self, df: pd.DataFrame) -> list:
        """Extract individual trade P&L from vectorized signals."""
        trade_rets = []
        in_trade   = False
        entry_price = 0.0
        direction   = 0

        for _, row in df.iterrows():
            pos = row.get("position", 0)
            price = row["close"]

            if not in_trade and pos != 0:
                in_trade    = True
                entry_price = price
                direction   = pos

            elif in_trade and (pos == 0 or pos != direction):
                pct = direction * (price - entry_price) / entry_price
                cost = (self.commission + self.slippage) * 2
                trade_rets.append(pct - cost)
                in_trade = (pos != 0)
                if in_trade:
                    entry_price = price
                    direction   = pos

        return trade_rets

    def walk_forward(self, df: pd.DataFrame, signal_col: str = "signal",
                     n_splits: int = 5) -> list:
        """
        Walk-forward validation: train on past, test on future.
        Splits data into n windows and runs backtest on each test portion.
        Returns list of PerformanceReport for each window.
        """
        results = []
        split_size = len(df) // n_splits

        for i in range(1, n_splits):
            test_start = i * split_size
            test_end   = min((i + 1) * split_size, len(df))
            test_df    = df.iloc[test_start:test_end]

            if len(test_df) < 50:
                continue

            report, _ = self.run(test_df, signal_col)
            results.append({
                "window" : i,
                "start"  : test_df.index[0],
                "end"    : test_df.index[-1],
                "report" : report,
            })
            logger.info(f"Walk-forward window {i}: "
                        f"Return={report.total_return_pct:.2f}% "
                        f"Sharpe={report.sharpe_ratio:.3f}")

        return results
