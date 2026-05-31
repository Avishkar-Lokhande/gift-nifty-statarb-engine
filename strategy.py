"""
strategy.py — Statistical basis arbitrage execution engine.

Ingests the synthetic market data, computes rolling Z-score spread metrics,
generates trade signals, and produces a full execution ledger with realized
PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from config import CONFIG, TradingConfig


# ── Type aliases ─────────────────────────────────────────────────────────────
Direction = Literal["long_spread", "short_spread", "flat"]


@dataclass
class Position:
    """Represents an open spread position.

    Args:
        direction: ``"long_spread"`` (buy NSE / short GIFT) or
                   ``"short_spread"`` (short NSE / long GIFT).
        entry_bar: Integer bar index at entry.
        entry_timestamp: Pandas Timestamp at entry.
        entry_nifty_price: NSE Nifty price at entry (INR).
        entry_gift_price: GIFT Nifty price at entry (USD).
        entry_zscore: Z-score that triggered the entry.
        lots: Number of lots traded.
        transaction_cost: Total transaction cost incurred at entry (USD).
    """

    direction: Direction
    entry_bar: int
    entry_timestamp: pd.Timestamp
    entry_nifty_price: float
    entry_gift_price: float
    entry_zscore: float
    lots: int
    transaction_cost: float


@dataclass
class BasisArbitrageStrategy:
    """Rolling Z-score mean-reversion strategy on the NSE/GIFT Nifty basis.

    The strategy maintains at most one open spread position at a time.
    All arithmetic is performed in USD; the Nifty INR leg is converted at
    the configured FX rate for PnL attribution.

    Args:
        cfg: Frozen TradingConfig instance.
    """

    cfg: TradingConfig = field(default_factory=lambda: CONFIG)

    # ── Internal state ───────────────────────────────────────────────────────
    _current_position: Position | None = field(default=None, init=False)
    _equity: float = field(default=0.0, init=False)
    _ledger: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._equity = self.cfg.initial_capital

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        market_data: pd.DataFrame,
        forced_exit_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Execute the full backtest over the supplied market data.

        Args:
            market_data: Output of ``MarketSimulator.generate()``.
            forced_exit_mask: Boolean Series (same index as market_data)
                indicating bars where the risk engine forces position closure.
                ``True`` at bar i means: close any open position at bar i
                before evaluating any new entry signal.

        Returns:
            Execution ledger as a DataFrame with columns:
            ``trade_id``, ``direction``, ``entry_timestamp``,
            ``exit_timestamp``, ``entry_nifty``, ``exit_nifty``,
            ``entry_gift``, ``exit_gift``, ``entry_zscore``,
            ``exit_zscore``, ``lots``, ``gross_pnl_usd``,
            ``transaction_cost_usd``, ``net_pnl_usd``, ``exit_reason``.
        """
        enriched = self._enrich(market_data)
        if forced_exit_mask is None:
            forced_exit_mask = pd.Series(False, index=enriched.index)

        for bar_idx, (ts, row) in enumerate(enriched.iterrows()):
            in_mask = ts in forced_exit_mask.index
            forced = bool(forced_exit_mask.loc[ts]) if in_mask else False

            if np.isnan(row["zscore"]):
                continue

            # Force-close and halt new entries if risk engine demands it
            if forced:
                if self._current_position is not None:
                    self._close_position(row, bar_idx, "risk_forced_exit")
                continue   # no new entries during halt / cool-down bars

            self._process_bar(bar_idx, ts, row)

        # Close any residual position at end of simulation
        if self._current_position is not None and not enriched.empty:
            last_row = enriched.iloc[-1]
            self._close_position(
                last_row, len(enriched) - 1, "end_of_simulation"
            )

        return self._build_ledger_df()

    def get_equity_curve(
        self, market_data: pd.DataFrame, ledger: pd.DataFrame
    ) -> pd.Series:
        """Reconstruct a bar-by-bar equity curve from the execution ledger.

        Args:
            market_data: Original market data DataFrame (for the index).
            ledger: Output of ``run()``.

        Returns:
            Equity curve Series indexed by timestamp.
        """
        equity = pd.Series(
            self.cfg.initial_capital, index=market_data.index, dtype=float
        )
        if ledger.empty:
            return equity

        cumulative_pnl = 0.0
        for _, trade in ledger.iterrows():
            exit_ts = trade["exit_timestamp"]
            if exit_ts in equity.index:
                cumulative_pnl += trade["net_pnl_usd"]
                equity.loc[exit_ts:] = (
                    self.cfg.initial_capital + cumulative_pnl
                )

        return equity

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach rolling Z-score columns to the market DataFrame.

        Args:
            df: Raw market data.

        Returns:
            Augmented DataFrame with ``spread``, ``roll_mean``,
            ``roll_std``, and ``zscore`` columns.
        """
        w = self.cfg.rolling_window
        df = df.copy()
        df["spread"] = df["basis"]   # already = GIFT - Nifty_USD
        df["roll_mean"] = df["spread"].rolling(w).mean()
        df["roll_std"] = df["spread"].rolling(w).std()
        std = df["roll_std"].replace(0.0, np.nan)
        df["zscore"] = (df["spread"] - df["roll_mean"]) / std
        return df

    def _process_bar(
        self, bar_idx: int, ts: pd.Timestamp, row: pd.Series
    ) -> None:
        """Evaluate signal and update position state for a single bar.

        Args:
            bar_idx: Integer bar index.
            ts: Timestamp of the current bar.
            row: Enriched market data row.
        """
        z = row["zscore"]
        entry_th = self.cfg.entry_zscore
        exit_th = self.cfg.exit_zscore

        if self._current_position is None:
            # Check for new entry
            if z <= -entry_th:
                self._open_position(bar_idx, ts, row, "long_spread")
            elif z >= entry_th:
                self._open_position(bar_idx, ts, row, "short_spread")
        else:
            # Check for mean-reversion exit
            direction = self._current_position.direction
            if direction == "long_spread" and z >= exit_th:
                self._close_position(row, bar_idx, "mean_reversion")
            elif direction == "short_spread" and z <= exit_th:
                self._close_position(row, bar_idx, "mean_reversion")

    def _open_position(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        row: pd.Series,
        direction: Direction,
    ) -> None:
        """Open a new spread position.

        Lot sizing is fixed at 1 lot per trade to keep leverage controlled
        within the configured limits.

        Args:
            bar_idx: Integer bar index.
            ts: Entry timestamp.
            row: Market data row at entry.
            direction: Trade direction — ``"long_spread"`` or
                ``"short_spread"``.
        """
        lots = self._compute_lot_size(row["nifty_price"])
        # Reserve cost covers both entry and exit sides
        cost = lots * self.cfg.transaction_cost_per_lot * 2
        self._current_position = Position(
            direction=direction,
            entry_bar=bar_idx,
            entry_timestamp=ts,
            entry_nifty_price=row["nifty_price"],
            entry_gift_price=row["gift_nifty_price"],
            entry_zscore=row["zscore"],
            lots=lots,
            transaction_cost=cost,
        )

    def _close_position(
        self,
        row: pd.Series,
        bar_idx: int,
        reason: str,
    ) -> None:
        """Square off the open position and record the trade in the ledger.

        PnL attribution (spread = GIFT − NSE_USD):
        - Long spread  (Z ≤ −2): long GIFT + short NSE.
          Profit = Δbasis × notional  (basis rises back to zero).
        - Short spread (Z ≥ +2): short GIFT + long NSE.
          Profit = −Δbasis × notional (basis falls back to zero).

        Args:
            row: Market data row at exit.
            bar_idx: Integer bar index at exit.
            reason: Human-readable exit reason tag.
        """
        pos = self._current_position
        if pos is None:
            return

        exit_nifty = row["nifty_price"]
        exit_gift = row["gift_nifty_price"]
        exit_z = row.get("zscore", float("nan"))

        entry_nifty_usd = pos.entry_nifty_price / self.cfg.usd_inr_fx
        exit_nifty_usd = exit_nifty / self.cfg.usd_inr_fx

        notional = pos.lots * self.cfg.lot_size

        if pos.direction == "long_spread":
            # Z <= -2: GIFT cheap → long GIFT, short NSE (USD)
            # Profit when spread rises back toward zero
            nse_leg_pnl = (entry_nifty_usd - exit_nifty_usd) * notional
            gift_leg_pnl = (exit_gift - pos.entry_gift_price) * notional
        else:
            # Z >= +2: GIFT expensive → short GIFT, long NSE (USD)
            # Profit when spread falls back toward zero
            nse_leg_pnl = (exit_nifty_usd - entry_nifty_usd) * notional
            gift_leg_pnl = (pos.entry_gift_price - exit_gift) * notional

        gross_pnl = nse_leg_pnl + gift_leg_pnl
        net_pnl = gross_pnl - pos.transaction_cost

        self._ledger.append(
            {
                "trade_id": len(self._ledger) + 1,
                "direction": pos.direction,
                "entry_timestamp": pos.entry_timestamp,
                "exit_timestamp": row.name,
                "entry_nifty": pos.entry_nifty_price,
                "exit_nifty": exit_nifty,
                "entry_gift": pos.entry_gift_price,
                "exit_gift": exit_gift,
                "entry_zscore": pos.entry_zscore,
                "exit_zscore": exit_z,
                "lots": pos.lots,
                "gross_pnl_usd": round(gross_pnl, 4),
                "transaction_cost_usd": round(pos.transaction_cost, 4),
                "net_pnl_usd": round(net_pnl, 4),
                "exit_reason": reason,
            }
        )
        self._current_position = None

    def _compute_lot_size(self, nifty_price: float) -> int:
        """Determine position size respecting leverage and capital limits.

        Args:
            nifty_price: Current Nifty 50 index level (INR).

        Returns:
            Number of lots (minimum 1).
        """
        nifty_usd = nifty_price / self.cfg.usd_inr_fx
        notional_per_lot = nifty_usd * self.cfg.lot_size
        max_notional = self._equity * self.cfg.max_leverage
        lots = max(1, int(max_notional / notional_per_lot / 10))
        return lots

    def _build_ledger_df(self) -> pd.DataFrame:
        """Convert the internal ledger list to a typed DataFrame.

        Returns:
            Typed execution ledger DataFrame, or an empty DataFrame with the
            correct schema if no trades were recorded.
        """
        if not self._ledger:
            return pd.DataFrame(
                columns=[
                    "trade_id", "direction", "entry_timestamp",
                    "exit_timestamp", "entry_nifty", "exit_nifty",
                    "entry_gift", "exit_gift", "entry_zscore",
                    "exit_zscore", "lots", "gross_pnl_usd",
                    "transaction_cost_usd", "net_pnl_usd", "exit_reason",
                ]
            )
        df = pd.DataFrame(self._ledger)
        df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"])
        df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])
        return df
