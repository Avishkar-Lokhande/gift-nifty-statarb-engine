"""
data_pipeline.py — Synthetic intraday market data generator.

Simulates cointegrated NSE Nifty 50 and GIFT Nifty price series at 1-minute
resolution over a full quarter, including realistic anomalies (vol bursts,
liquidity droughts, FX-driven basis dislocations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from config import CONFIG, TradingConfig


@dataclass
class MarketSimulator:
    """Generates synthetic intraday tick data for Nifty 50 and GIFT Nifty.

    The underlying NSE Nifty 50 follows a Geometric Brownian Motion (GBM)
    with stochastic volatility bursts.  GIFT Nifty is modelled as Nifty 50
    plus an Ornstein-Uhlenbeck basis spread that mean-reverts throughout
    the session (mimicking the cointegration observed empirically).

    Args:
        cfg: Frozen TradingConfig instance.
        seed: NumPy random seed for reproducibility.
    """

    cfg: TradingConfig = CONFIG
    seed: int = 42

    # ── derived constants set in __post_init__ ────────────────────────────────
    _bars_per_day: int = 0
    _total_bars: int = 0

    def __post_init__(self) -> None:
        self._bars_per_day = self.cfg.trading_hours_per_day * 60
        self._total_bars = self._bars_per_day * self.cfg.trading_days_per_quarter

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate and return the full synthetic intraday dataset.

        Returns:
            DataFrame indexed by a trading DatetimeIndex with columns:
            ``timestamp``, ``nifty_price``, ``gift_nifty_price``, ``basis``,
            ``nifty_return``, ``gift_return``, ``realized_vol_5m``.
        """
        rng = np.random.default_rng(self.seed)

        timestamps = self._build_timestamps()
        nifty_prices = self._simulate_nifty(rng)
        basis = self._simulate_basis(rng, nifty_prices)
        gift_prices = self._compute_gift_prices(nifty_prices, basis)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "nifty_price": nifty_prices,
                "gift_nifty_price": gift_prices,
                "basis": basis,
            }
        )
        df = df.set_index("timestamp")
        df = self._inject_anomalies(df, rng)
        df = self._compute_derived_features(df)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Build a DatetimeIndex spanning the quarter at 1-minute frequency,
        restricted to intraday trading hours (09:15–15:15 IST proxy)."""
        start = pd.Timestamp("2025-01-01 09:15:00")
        all_times: list[pd.Timestamp] = []
        current_day = start

        for _ in range(self.cfg.trading_days_per_quarter):
            session_start = current_day.replace(hour=9, minute=15, second=0)
            session_bars = pd.date_range(
                start=session_start,
                periods=self._bars_per_day,
                freq="1min",
            )
            all_times.extend(session_bars)
            # Advance to the next weekday
            current_day += pd.offsets.BDay(1)

        return pd.DatetimeIndex(all_times)

    def _simulate_nifty(self, rng: np.random.Generator) -> np.ndarray:
        """Simulate NSE Nifty 50 via GBM with stochastic volatility clustering.

        Volatility follows a basic GARCH-like regime: occasional 3× vol spikes
        lasting 15–30 bars simulate macro shock events.

        Args:
            rng: Seeded NumPy Generator instance.

        Returns:
            Array of Nifty price levels, shape (total_bars,).
        """
        dt = 1.0 / (252 * 390)          # 1-min bar as fraction of a trading year
        mu = self.cfg.annual_drift
        base_sigma = self.cfg.annual_vol
        n = self._total_bars

        # Stochastic volatility multiplier
        vol_multiplier = np.ones(n)
        shock_starts = rng.integers(0, n, size=15)
        for s in shock_starts:
            duration = int(rng.integers(15, 45))
            end = min(s + duration, n)
            vol_multiplier[s:end] = rng.uniform(2.0, 4.5)

        sigma_t = base_sigma * vol_multiplier
        z = rng.standard_normal(n)
        log_returns = (mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * z
        prices = self.cfg.nifty_base_price * np.exp(np.cumsum(log_returns))
        return prices

    def _simulate_basis(
        self, rng: np.random.Generator, nifty: np.ndarray
    ) -> np.ndarray:
        """Simulate the basis spread as a pure Ornstein-Uhlenbeck process.

        The OU SDE is:  dX = κ(μ - X)dt + σ dW

        where κ controls how fast the spread snaps back to equilibrium μ and
        σ is the annual diffusion coefficient.  The initial value is drawn
        from the theoretical stationary distribution N(μ, σ/√(2κ)) so the
        process is already in steady-state at bar 0 — no warm-up burn-in.

        This guarantees smooth, autocorrelated Z-score waves rather than
        instantaneous noise spikes, producing clean, tradeable mean-reversion
        signals throughout the quarter.

        Args:
            rng: Seeded NumPy Generator instance.
            nifty: Unused; kept for signature compatibility.

        Returns:
            Array of basis spread values (USD), shape (total_bars,).
        """
        n = self._total_bars
        kappa = self.cfg.basis_mean_reversion_speed   # e.g. 1500 annual
        mu_b = self.cfg.basis_long_run_mean           # long-run mean (USD)
        sigma_b = self.cfg.basis_vol                  # annual diffusion vol
        dt = 1.0 / (252 * 390)                        # 1-min bar fraction

        # Stationary std of the OU process: avoids extreme warm-up transients
        sigma_stat = sigma_b / np.sqrt(2.0 * kappa)
        per_bar_vol = sigma_b * np.sqrt(dt)

        # Euler-Maruyama discretisation — vectorised draw, scalar update loop
        innovations = rng.standard_normal(n)
        basis = np.empty(n)
        basis[0] = rng.normal(mu_b, sigma_stat)

        mean_rev = kappa * dt   # fraction pulled back each bar
        for t in range(1, n):
            basis[t] = (
                basis[t - 1]
                + mean_rev * (mu_b - basis[t - 1])
                + per_bar_vol * innovations[t]
            )

        return basis

    def _compute_gift_prices(
        self, nifty: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        """Convert Nifty INR price to USD and add OU basis spread.

        Args:
            nifty: NSE Nifty price array in INR.
            basis: OU basis spread array in USD.

        Returns:
            GIFT Nifty price array in USD, shape (total_bars,).
        """
        nifty_usd = nifty / self.cfg.usd_inr_fx
        return nifty_usd + basis

    def _inject_anomalies(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Layer in fat-tail jump events on both series simultaneously.

        5 random bars receive a correlated price gap of ±0.5–1.5% to
        simulate news-driven co-jumps (e.g., RBI announcements, global macro).

        Args:
            df: Base OHLC-equivalent DataFrame.
            rng: Seeded NumPy Generator instance.

        Returns:
            DataFrame with co-jump anomalies applied.
        """
        n = len(df)
        jump_indices = rng.integers(200, n - 200, size=5)
        for idx in jump_indices:
            jump_pct = rng.choice([-1, 1]) * rng.uniform(0.005, 0.015)
            df.iloc[idx, df.columns.get_loc("nifty_price")] *= (1 + jump_pct)
            df.iloc[idx, df.columns.get_loc("gift_nifty_price")] *= (
                1 + jump_pct * rng.uniform(0.8, 1.2)
            )
            # Recalculate basis at that bar
            df.iloc[idx, df.columns.get_loc("basis")] = (
                df.iloc[idx]["gift_nifty_price"]
                - df.iloc[idx]["nifty_price"] / self.cfg.usd_inr_fx
            )
        return df

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return columns and rolling realised volatility.

        Args:
            df: Price DataFrame.

        Returns:
            Augmented DataFrame with ``nifty_return``, ``gift_return``,
            and ``realized_vol_5m`` columns.
        """
        df["nifty_return"] = df["nifty_price"].pct_change().fillna(0.0)
        df["gift_return"] = df["gift_nifty_price"].pct_change().fillna(0.0)
        df["realized_vol_5m"] = (
            df["nifty_return"]
            .rolling(self.cfg.vol_shock_window)
            .std()
            .fillna(0.0)
        )
        return df
