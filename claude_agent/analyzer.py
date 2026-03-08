"""
claude_agent/analyzer.py
-----------------------------------------------------------------
Claude AI Analysis Layer
-----------------------------------------------------------------
Uses Claude API for:
  1. Trade signal sanity checks (anomaly detection)
  2. Trade explanation logging
  3. Daily performance review
-----------------------------------------------------------------
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("anthropic package not installed. Claude features disabled.")


class ClaudeAnalyzer:
    """
    Claude AI layer for trade analysis and anomaly detection.
    Gracefully degrades if API key is missing or invalid.
    """

    def __init__(self, api_key: str = ""):
        self.enabled = False
        self.client = None

        if not HAS_ANTHROPIC:
            logger.info("Claude analyzer disabled (no anthropic package)")
            return

        if not api_key or api_key.startswith("YOUR_"):
            logger.info("Claude analyzer disabled (no API key configured)")
            return

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.enabled = True
            logger.info("Claude analyzer enabled")
        except Exception as e:
            logger.warning(f"Failed to init Claude client: {e}")

    def flag_anomalies(
        self, symbol: str, indicators: dict
    ) -> Optional[str]:
        """
        Ask Claude to check if indicator readings look anomalous.
        Returns warning string if anomaly detected, else None.
        """
        if not self.enabled:
            return None

        prompt = (
            f"You are a crypto trading risk analyst. Check these indicators for {symbol} "
            f"and flag any anomalies that would make opening a new trade dangerous.\n\n"
            f"Indicators: {indicators}\n\n"
            f"Think hard about correlations between indicators, unusual divergences, "
            f"and market microstructure risks.\n\n"
            f"Reply with ONLY one of:\n"
            f"- 'OK' if everything looks normal\n"
            f"- A brief warning (max 50 words) if something looks anomalous\n"
            f"Do NOT explain your reasoning."
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.upper() == "OK":
                return None
            return text
        except Exception as e:
            logger.debug(f"Claude anomaly check failed: {e}")
            return None

    def explain_trade_signal(
        self,
        symbol: str,
        direction: str,
        signal_reason: str,
        regime,
        risk_reward: float,
        kelly_pct: float,
    ) -> str:
        """
        Get Claude to provide a brief trade explanation for logging.
        """
        if not self.enabled:
            return f"{direction} {symbol}: {signal_reason}"

        prompt = (
            f"You are a crypto trading log writer. Write a 1-sentence trade summary:\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Reason: {signal_reason}\n"
            f"Regime: {regime.regime.value} (conf={regime.confidence:.0%})\n"
            f"R:R ratio: {risk_reward:.1f}\n"
            f"Kelly size: {kelly_pct:.1%}\n\n"
            f"Reply with ONLY the 1-sentence summary. Be concise."
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.debug(f"Claude explain failed: {e}")
            return f"{direction} {symbol}: {signal_reason}"

    def daily_performance_review(self, stats: dict) -> str:
        """
        Get Claude to write a daily performance review.
        """
        if not self.enabled:
            return f"Stats: {stats}"

        prompt = (
            f"You are a trading performance analyst. Write a brief daily review "
            f"(3-5 bullet points) based on these stats:\n\n{stats}\n\n"
            f"Focus on: risk levels, drawdown, consecutive losses, and actionable advice. "
            f"Keep it under 100 words."
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.debug(f"Claude review failed: {e}")
            return f"Stats: {stats}"
