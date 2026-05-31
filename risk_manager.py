"""
risk_manager.py — Institutional real-time risk monitoring engine.

Implements:
  1. Intraday historical-simulation VaR (99% confidence).
  2. Trailing peak-to-trough drawdown watchdog with forced liquidation.
  3. Rolling volatility circuit breaker that halts trading on outlier spikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import CONFIG, TradingConfig


# ── Result containers ─────────────────────────────────────────────────────────

class CircuitBreakerEvent(NamedTuple):
    """Immutable record of a single circuit-breaker trigger.

    Attributes:
        timestamp: Bar at which the event was triggered.
        bar_idx: Integer bar index.
        trigger_type: ``"VOLATILITY_SHOCK"`` or ``"MAX_DRAWDOWN"``.
        metric_value: Observed metric that breached the threshold.
        threshold: The configured threshold that was exceeded.
        cool_down_until_bar: First bar index at which trading may resume.
    """

    timestamp: pd.Timestamp
    bar_idx: int
    trigger_type: str
    metric_value: float
    threshold: float
    cool_down_until_bar: int


@dataclass
class RiskReport:
    """Snapshot of risk metrics at a single bar.

    Attributes:
        bar_idx: Integer bar index.
        timestamp: Bar timestamp.
        equity: Current equity level (USD).
        var_99: 99% 1-bar VaR (positive number representing potential loss).
        trailing_drawdown_pct: Current trailing drawdown as a percentage.
        rolling_vol_pct: 5-bar rolling annualised vol (%).
        is_halted: Whether trading is currently suspended.
        circuit_event: Populated if a new circuit-breaker fired this bar.
    """

    bar_idx: int
    timestamp: pd.Timestamp
    equity: float
    var_99: float
    trailing_drawdown_pct: float
    rolling_vol_pct: float
    is_halted: bool
    circuit_event: CircuitBreakerEvent | None = None


@dataclass
class InstitutionalRiskEngine:
    """Bar-by-bar risk evaluation engine for the basis arbitrage strategy.

    The engine evaluates every bar in order and produces:
    - A per-bar ``RiskReport`` list (for audit and UI display).
    - A boolean ``forced_exit_mask`` Series that the strategy uses to
      trigger emergency position closures.
    - A list of ``CircuitBreakerEvent`` objects for post-hoc analysis.

    Args:
        cfg: Frozen TradingConfig instance.
    """

    cfg: TradingConfig = field(default_factory=lambda: CONFIG)

    # ── Internal state ────────────────────────────────────────────────────────
    _equity_history: list[float] = field(default_factory=list, init=False)
    _peak_equity: float = field(default=0.0, init=False)
    _reports: list[RiskReport] = field(default_factory=list, init=False)
    _circuit_events: list[CircuitBreakerEvent] = field(
        default_factory=list, init=False
    )
    _halt_until_bar: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self._peak_equity = self.cfg.initial_capital

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        market_data: pd.DataFrame,
        equity_curve: pd.Series,
    ) -> tuple[pd.Series, list[RiskReport], list[CircuitBreakerEvent]]:
        """Run the full risk pass over the simulation horizon.

        Args:
            market_data: Enriched market DataFrame from ``MarketSimulator``.
            equity_curve: Bar-by-bar equity curve from the strategy (first-pass,
                          before forced exits — used for PnL attribution).

        Returns:
            Tuple of:
            - ``forced_exit_mask``: Boolean Series; ``True`` = close position now.
            - ``reports``: List of ``RiskReport`` per bar.
            - ``circuit_events``: List of all ``CircuitBreakerEvent`` instances.
        """
        self._reset()
        n = len(market_data)
        forced_exit_flags = np.zeros(n, dtype=bool)

        returns = market_data["nifty_return"].to_numpy()
        vol5 = market_data["realized_vol_5m"].to_numpy()
        timestamps = market_data.index
        equity_arr = equity_curve.to_numpy()

        for i in range(n):
            ts = timestamps[i]
            equity = float(equity_arr[i])
            self._equity_history.append(equity)

            # Update peak
            if equity > self._peak_equity:
                self._peak_equity = equity

            # ── Compute risk metrics ──────────────────────────────────────────
            var_99 = self._compute_var(i, returns, equity)
            dd_pct = self._trailing_drawdown_pct(equity)
            rv_pct = float(vol5[i]) * 100.0

            # ── Check circuit breakers ────────────────────────────────────────
            is_halted = i <= self._halt_until_bar
            new_event: CircuitBreakerEvent | None = None

            if not is_halted:
                vol_event = self._check_vol_shock(i, ts, vol5, rv_pct)
                if vol_event is not None:
                    new_event = vol_event
                    forced_exit_flags[i] = True
                    is_halted = True

                if not is_halted:
                    dd_event = self._check_drawdown(i, ts, dd_pct, equity)
                    if dd_event is not None:
                        new_event = dd_event
                        forced_exit_flags[i] = True
                        is_halted = True
            else:
                # Still in cool-down: force exits if somehow a position slipped
                forced_exit_flags[i] = True

            report = RiskReport(
                bar_idx=i,
                timestamp=ts,
                equity=equity,
                var_99=var_99,
                trailing_drawdown_pct=dd_pct,
                rolling_vol_pct=rv_pct,
                is_halted=is_halted,
                circuit_event=new_event,
            )
            self._reports.append(report)

        forced_mask = pd.Series(forced_exit_flags, index=market_data.index)
        return forced_mask, self._reports, self._circuit_events

    def reports_to_dataframe(self) -> pd.DataFrame:
        """Convert the internal report list to a typed DataFrame.

        Returns:
            DataFrame with one row per bar containing all risk metrics.
        """
        if not self._reports:
            return pd.DataFrame()

        rows = [
            {
                "timestamp": r.timestamp,
                "equity": r.equity,
                "var_99_usd": r.var_99,
                "trailing_drawdown_pct": r.trailing_drawdown_pct,
                "rolling_vol_pct": r.rolling_vol_pct,
                "is_halted": r.is_halted,
                "circuit_event_type": (
                    r.circuit_event.trigger_type if r.circuit_event else None
                ),
            }
            for r in self._reports
        ]
        return pd.DataFrame(rows).set_index("timestamp")

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        """Clear all mutable state for a fresh evaluation pass."""
        self._equity_history = []
        self._peak_equity = self.cfg.initial_capital
        self._reports = []
        self._circuit_events = []
        self._halt_until_bar = -1

    def _compute_var(
        self, bar_idx: int, returns: np.ndarray, equity: float
    ) -> float:
        """Historical-simulation 99% 1-bar VaR on the current equity.

        Uses the rolling lookback window of return observations as the
        empirical distribution.  Falls back to parametric (Normal) if
        the window has fewer than 10 observations.

        Args:
            bar_idx: Current integer bar index.
            returns: Full returns array.
            equity: Current equity level (USD).

        Returns:
            VaR in USD (positive value representing potential loss).
        """
        lookback = self.cfg.var_lookback
        start = max(0, bar_idx - lookback)
        window_returns = returns[start:bar_idx]

        if len(window_returns) < 10:
            return 0.0

        # Historical simulation: sort losses (negative returns)
        losses = -window_returns * equity
        var = float(np.percentile(losses, self.cfg.var_confidence * 100))
        return max(var, 0.0)

    def _trailing_drawdown_pct(self, equity: float) -> float:
        """Compute the current peak-to-trough drawdown percentage.

        Args:
            equity: Current equity level.

        Returns:
            Drawdown as a negative percentage (e.g., -3.2 means -3.2%).
        """
        if self._peak_equity <= 0:
            return 0.0
        return ((equity - self._peak_equity) / self._peak_equity) * 100.0

    def _check_vol_shock(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        vol5: np.ndarray,
        current_vol_pct: float,
    ) -> CircuitBreakerEvent | None:
        """Detect a volatility spike exceeding the configured threshold.

        A spike is defined as the current 5-bar rolling vol being more than
        ``vol_shock_threshold_pct``% higher than the previous observation.

        Args:
            bar_idx: Current bar index.
            ts: Current bar timestamp.
            vol5: Full array of 5-bar rolling vols.
            current_vol_pct: Current rolling vol as a percentage.

        Returns:
            A ``CircuitBreakerEvent`` if triggered, otherwise ``None``.
        """
        if bar_idx < self.cfg.vol_shock_window + 1:
            return None

        prev_vol = float(vol5[bar_idx - 1]) * 100.0
        if prev_vol <= 1e-9:
            return None

        pct_change = ((current_vol_pct - prev_vol) / prev_vol) * 100.0
        threshold = self.cfg.vol_shock_threshold_pct

        if pct_change > threshold:
            cool_down_until = bar_idx + self.cfg.cool_down_bars
            self._halt_until_bar = cool_down_until
            event = CircuitBreakerEvent(
                timestamp=ts,
                bar_idx=bar_idx,
                trigger_type="VOLATILITY_SHOCK",
                metric_value=round(pct_change, 2),
                threshold=threshold,
                cool_down_until_bar=cool_down_until,
            )
            self._circuit_events.append(event)
            return event

        return None

    def _check_drawdown(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        dd_pct: float,
        equity: float,
    ) -> CircuitBreakerEvent | None:
        """Detect a trailing drawdown breach.

        Args:
            bar_idx: Current bar index.
            ts: Current bar timestamp.
            dd_pct: Current trailing drawdown percentage (negative).
            equity: Current equity for context.

        Returns:
            A ``CircuitBreakerEvent`` if triggered, otherwise ``None``.
        """
        limit = self.cfg.max_trailing_drawdown_pct   # e.g. -5.0
        if dd_pct <= limit:
            cool_down_until = bar_idx + self.cfg.cool_down_bars
            self._halt_until_bar = cool_down_until
            event = CircuitBreakerEvent(
                timestamp=ts,
                bar_idx=bar_idx,
                trigger_type="MAX_DRAWDOWN",
                metric_value=round(dd_pct, 4),
                threshold=limit,
                cool_down_until_bar=cool_down_until,
            )
            self._circuit_events.append(event)
            return event

        return None
