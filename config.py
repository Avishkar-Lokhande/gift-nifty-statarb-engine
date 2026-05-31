"""
config.py — Frozen trading configuration for the GIFT Nifty Basis Arbitrage Engine.
All monetary values are in USD. Time units are in minutes unless specified.
"""

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class TradingConfig:
    """Immutable configuration for the basis arbitrage strategy.

    Args:
        initial_capital: Starting equity in USD.
        transaction_cost_per_lot: Slippage + fees per lot per side (USD).
        lot_size: Number of units per lot.
        max_leverage: Maximum allowable gross leverage multiple.
        sampling_frequency_minutes: Bar interval in minutes.
        rolling_window: Lookback period (bars) for Z-score computation.
        entry_zscore: Absolute Z-score threshold to open a spread position.
        exit_zscore: Absolute Z-score threshold to square off a position.
        var_confidence: Confidence level for intraday VaR computation.
        var_lookback: Rolling window (bars) for historical VaR estimation.
        max_trailing_drawdown_pct: Forced liquidation drawdown floor (negative %).
        vol_shock_window: Rolling window (bars) for volatility circuit breaker.
        vol_shock_threshold_pct: % spike in rolling vol that triggers circuit breaker.
        cool_down_bars: Number of bars to halt trading after a circuit-breaker event.
        trading_hours_per_day: Liquid hours per session.
        trading_days_per_quarter: Simulated calendar days per quarter.
        usd_inr_fx: Approximate USD/INR rate for GIFT Nifty price conversion.
        nifty_base_price: Starting synthetic Nifty 50 index level.
        annual_drift: Annualized drift used in GBM data simulation.
        annual_vol: Annualized volatility used in GBM data simulation.
        basis_mean_reversion_speed: Ornstein-Uhlenbeck mean-reversion coefficient (κ).
        basis_long_run_mean: Long-run equilibrium basis spread (USD).
        basis_vol: Volatility of the basis spread process.
    """

    # ── Capital & execution ────────────────────────────────────────────────────
    initial_capital: float = 150_000.0
    transaction_cost_per_lot: float = 0.40
    lot_size: int = 50
    max_leverage: float = 4.0
    sampling_frequency_minutes: int = 1

    # ── Z-score signal thresholds ─────────────────────────────────────────────
    rolling_window: int = 60          # 1-hour lookback
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0

    # ── Risk limits ───────────────────────────────────────────────────────────
    var_confidence: float = 0.99
    var_lookback: int = 120           # 2-hour rolling VaR window
    max_trailing_drawdown_pct: float = -5.0   # -5 % → forced liquidation
    vol_shock_window: int = 5
    vol_shock_threshold_pct: float = 20.0     # 20 % intra-window spike
    cool_down_bars: int = 30          # 30-minute halt post circuit-breaker

    # ── Simulation parameters ─────────────────────────────────────────────────
    trading_hours_per_day: int = 6
    trading_days_per_quarter: int = 63
    usd_inr_fx: float = 83.5
    nifty_base_price: float = 22_000.0
    annual_drift: float = 0.08
    annual_vol: float = 0.18
    basis_mean_reversion_speed: float = 0.35  # κ — faster = tighter spread
    basis_long_run_mean: float = 0.0
    basis_vol: float = 6.0            # USD spread noise


# Module-level singleton — import this everywhere else.
CONFIG: Final[TradingConfig] = TradingConfig()
