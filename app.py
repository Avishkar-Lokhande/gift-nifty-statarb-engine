"""
app.py — Institutional-grade Streamlit analytics dashboard for the
GIFT Nifty Basis Arbitrage Engine.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import CONFIG, TradingConfig
from data_pipeline import MarketSimulator
from risk_manager import CircuitBreakerEvent, InstitutionalRiskEngine
from strategy import BasisArbitrageStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GIFT Nifty Basis Arb — Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
        border: 1px solid #30304a;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-label { color: #8888aa; font-size: 0.78rem; letter-spacing: 0.05em; }
    .kpi-value { color: #e8e8f0; font-size: 1.55rem; font-weight: 700; }
    .kpi-value.positive { color: #3ddc84; }
    .kpi-value.negative { color: #ff4b4b; }
    .cb-banner-active {
        background: #ff1a1a;
        border-radius: 8px;
        padding: 14px 20px;
        animation: pulse 1.2s infinite;
        font-weight: 700;
        font-size: 1.05rem;
        color: white;
    }
    .cb-banner-clear {
        background: #1a6e40;
        border-radius: 8px;
        padding: 14px 20px;
        font-weight: 600;
        color: #cfffdd;
    }
    @keyframes pulse {
        0%   { opacity: 1.0; }
        50%  { opacity: 0.55; }
        100% { opacity: 1.0; }
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Simulation Controls")
    st.markdown("---")

    seed = st.slider("Random Seed", 1, 200, 42)
    rolling_window = st.slider(
        "Z-Score Rolling Window (bars)", 20, 240, CONFIG.rolling_window
    )
    entry_z = st.slider("Entry Z-Score Threshold", 1.0, 4.0, CONFIG.entry_zscore, 0.1)
    exit_z = st.slider("Exit Z-Score Threshold", -1.0, 1.5, CONFIG.exit_zscore, 0.1)
    max_dd = st.slider(
        "Max Trailing Drawdown (%)", -15.0, -1.0, CONFIG.max_trailing_drawdown_pct, 0.5
    )
    vol_threshold = st.slider(
        "Vol Shock Threshold (%)", 5.0, 100.0, CONFIG.vol_shock_threshold_pct, 5.0
    )
    cool_down = st.slider(
        "Cool-Down Period (bars)", 5, 120, CONFIG.cool_down_bars
    )

    st.markdown("---")
    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Computation (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running simulation…")
def run_full_simulation(
    seed: int,
    rolling_window: int,
    entry_z: float,
    exit_z: float,
    max_dd: float,
    vol_threshold: float,
    cool_down: int,
) -> dict[str, Any]:
    """Execute data generation, strategy, and risk engine; cache results.

    Args:
        seed: RNG seed for data generation.
        rolling_window: Z-score rolling lookback (bars).
        entry_z: Entry Z-score threshold.
        exit_z: Exit Z-score threshold.
        max_dd: Max trailing drawdown limit (%).
        vol_threshold: Volatility shock trigger threshold (%).
        cool_down: Cool-down bars after circuit-breaker event.

    Returns:
        Dictionary containing all artefacts needed by the dashboard.
    """
    # Build a custom config from sidebar overrides
    cfg = TradingConfig(
        rolling_window=rolling_window,
        entry_zscore=entry_z,
        exit_zscore=exit_z,
        max_trailing_drawdown_pct=max_dd,
        vol_shock_threshold_pct=vol_threshold,
        cool_down_bars=cool_down,
    )

    # 1 — Generate market data
    simulator = MarketSimulator(cfg=cfg, seed=seed)
    market_data = simulator.generate()

    # 2 — First-pass strategy (no forced exits yet — needed to build equity curve)
    strat_first = BasisArbitrageStrategy(cfg=cfg)
    ledger_first = strat_first.run(market_data)
    equity_first = strat_first.get_equity_curve(market_data, ledger_first)

    # 3 — Risk engine evaluation
    risk_engine = InstitutionalRiskEngine(cfg=cfg)
    forced_mask, reports, circuit_events = risk_engine.evaluate(
        market_data, equity_first
    )
    risk_df = risk_engine.reports_to_dataframe()

    # 4 — Second-pass strategy with forced exits applied
    strat_final = BasisArbitrageStrategy(cfg=cfg)
    ledger = strat_final.run(market_data, forced_exit_mask=forced_mask)
    equity_curve = strat_final.get_equity_curve(market_data, ledger)

    # 5 — Build benchmark (buy-and-hold Nifty 50 in USD terms)
    nifty_usd = market_data["nifty_price"] / cfg.usd_inr_fx
    bh_returns = nifty_usd / nifty_usd.iloc[0]
    benchmark_curve = cfg.initial_capital * bh_returns

    # 6 — Drawdown series
    rolling_peak = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_peak) / rolling_peak * 100.0

    # 7 — Enriched market data for charts (add zscore)
    w = cfg.rolling_window
    market_data["spread"] = market_data["basis"]
    market_data["roll_mean"] = market_data["spread"].rolling(w).mean()
    market_data["roll_std"] = market_data["spread"].rolling(w).std()
    market_data["zscore"] = (
        (market_data["spread"] - market_data["roll_mean"])
        / market_data["roll_std"].replace(0.0, float("nan"))
    )
    market_data["upper_band"] = market_data["roll_mean"] + entry_z * market_data["roll_std"]
    market_data["lower_band"] = market_data["roll_mean"] - entry_z * market_data["roll_std"]

    return {
        "market_data": market_data,
        "ledger": ledger,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "drawdown_series": drawdown_series,
        "risk_df": risk_df,
        "circuit_events": circuit_events,
        "cfg": cfg,
    }


def _compute_kpis(
    equity_curve: pd.Series,
    ledger: pd.DataFrame,
    cfg: TradingConfig,
) -> dict[str, float | str]:
    """Compute all strategy KPIs from the equity curve and execution ledger.

    Args:
        equity_curve: Bar-by-bar equity Series.
        ledger: Execution ledger DataFrame.
        cfg: Trading configuration.

    Returns:
        Dictionary of labelled KPI values.
    """
    bars_per_year = cfg.trading_hours_per_day * 60 * 252

    returns = equity_curve.pct_change().dropna()
    total_return_pct = (
        (equity_curve.iloc[-1] / cfg.initial_capital) - 1.0
    ) * 100.0

    # Sharpe (annualised, risk-free = 5%)
    rf_per_bar = 0.05 / bars_per_year
    excess = returns - rf_per_bar
    sharpe = (
        (excess.mean() / excess.std() * np.sqrt(bars_per_year))
        if excess.std() > 0
        else 0.0
    )

    # Sortino
    downside = returns[returns < 0]
    sortino = (
        (excess.mean() / downside.std() * np.sqrt(bars_per_year))
        if len(downside) > 0 and downside.std() > 0
        else 0.0
    )

    # Max drawdown
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak * 100.0
    max_dd = float(drawdown.min())

    # Calmar
    ann_return = (
        (1 + total_return_pct / 100.0) ** (bars_per_year / max(len(equity_curve), 1)) - 1.0
    ) * 100.0
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # Win rate & total costs
    if ledger.empty:
        win_rate = 0.0
        total_costs = 0.0
    else:
        win_rate = float((ledger["net_pnl_usd"] > 0).mean()) * 100.0
        total_costs = float(ledger["transaction_cost_usd"].sum())

    return {
        "Total Return (%)": round(total_return_pct, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Calmar Ratio": round(calmar, 3),
        "Max Drawdown (%)": round(max_dd, 2),
        "Win Rate (%)": round(win_rate, 1),
        "Total Tx Costs ($)": round(total_costs, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

_CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0e0e1a",
    plot_bgcolor="#0e0e1a",
    font_color="#ccccdd",
)


def _build_spread_chart(market_data: pd.DataFrame, entry_z: float) -> go.Figure:
    """Dual-axis chart: basis spread vs. Z-score with execution bands.

    Args:
        market_data: Enriched market DataFrame.
        entry_z: Entry threshold (for band labelling).

    Returns:
        Plotly Figure object.
    """
    # Downsample to every 5 bars for chart performance
    md = market_data.iloc[::5]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04,
        subplot_titles=["Basis Spread with Execution Bands", "Rolling Z-Score"],
    )

    # Spread
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["spread"],
            name="Basis Spread",
            line=dict(color="#4f9cf7", width=1.2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["upper_band"],
            name=f"Upper Band (+{entry_z}σ)",
            line=dict(color="#ff6b6b", width=1, dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["lower_band"],
            name=f"Lower Band (-{entry_z}σ)",
            line=dict(color="#3ddc84", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(61,220,132,0.05)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["roll_mean"],
            name="Rolling Mean",
            line=dict(color="#ffd700", width=1, dash="dot"),
        ),
        row=1, col=1,
    )

    # Z-score
    zscore_color = md["zscore"].apply(
        lambda z: "#ff6b6b" if z >= entry_z else ("#3ddc84" if z <= -entry_z else "#8888aa")
    )
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["zscore"],
            name="Z-Score",
            line=dict(color="#c8aaff", width=1.2),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=entry_z, line_color="#ff6b6b", line_dash="dash", row=2, col=1)
    fig.add_hline(y=-entry_z, line_color="#3ddc84", line_dash="dash", row=2, col=1)
    fig.add_hline(y=0, line_color="#555566", line_dash="dot", row=2, col=1)

    fig.update_layout(
        height=500,
        legend=dict(orientation="h", y=-0.12),
        **_CHART_THEME,
    )
    return fig


def _build_equity_chart(
    equity_curve: pd.Series, benchmark_curve: pd.Series
) -> go.Figure:
    """Equity curve vs. buy-and-hold benchmark.

    Args:
        equity_curve: Strategy equity Series (USD).
        benchmark_curve: Benchmark equity Series (USD).

    Returns:
        Plotly Figure object.
    """
    eq = equity_curve.iloc[::5]
    bm = benchmark_curve.iloc[::5]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=eq,
            name="Basis Arb Strategy",
            line=dict(color="#4f9cf7", width=2),
            fill="tozeroy",
            fillcolor="rgba(79,156,247,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bm.index, y=bm,
            name="Buy & Hold Benchmark",
            line=dict(color="#ffd700", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        title="Equity Curve vs. Benchmark",
        yaxis_title="Portfolio Value (USD)",
        height=380,
        legend=dict(orientation="h", y=-0.15),
        **_CHART_THEME,
    )
    return fig


def _build_drawdown_chart(drawdown_series: pd.Series) -> go.Figure:
    """Underwater drawdown profile.

    Args:
        drawdown_series: Bar-by-bar drawdown percentage Series.

    Returns:
        Plotly Figure object.
    """
    dd = drawdown_series.iloc[::5]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd,
            name="Drawdown (%)",
            line=dict(color="#ff4b4b", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.15)",
        )
    )
    fig.update_layout(
        title="Drawdown Profile (Underwater Periods)",
        yaxis_title="Drawdown (%)",
        height=280,
        showlegend=False,
        **_CHART_THEME,
    )
    return fig


def _build_vol_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Rolling 5-bar realised volatility from the risk engine report.

    Args:
        risk_df: Risk engine DataFrame.

    Returns:
        Plotly Figure object.
    """
    rv = risk_df["rolling_vol_pct"].iloc[::5]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rv.index, y=rv,
            name="Realised Vol (5-bar, %)",
            line=dict(color="#ff9f43", width=1.2),
        )
    )
    fig.update_layout(
        title="Rolling 5-Bar Realised Volatility",
        yaxis_title="Volatility (%)",
        height=240,
        showlegend=False,
        **_CHART_THEME,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI card renderer
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: float | str, is_pct: bool = False) -> str:
    """Render a single KPI card as an HTML string.

    Args:
        label: Display label.
        value: Numeric or string value.
        is_pct: Whether to suffix a ``%`` sign and apply pos/neg colouring.

    Returns:
        HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    if isinstance(value, float):
        fmt = f"{value:+.2f}%" if is_pct else f"{value:,.3g}"
    else:
        fmt = str(value)

    css_class = "kpi-value"
    if is_pct and isinstance(value, float):
        css_class += " positive" if value >= 0 else " negative"

    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{css_class}">{fmt}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker banner
# ─────────────────────────────────────────────────────────────────────────────

def _render_circuit_banner(circuit_events: list[CircuitBreakerEvent]) -> None:
    """Render the risk status banner.

    Shows a flashing red panel with event details if any circuit breaker
    fired; otherwise shows a green "all-clear" panel.

    Args:
        circuit_events: List of circuit-breaker events from the risk engine.
    """
    if circuit_events:
        # Show the most recent event prominently
        latest = circuit_events[-1]
        trigger_label = (
            "VOLATILITY SHOCK OUTLIER DETECTED"
            if latest.trigger_type == "VOLATILITY_SHOCK"
            else "MAX TRAILING DRAWDOWN BREACH"
        )
        lines = [
            f"CIRCUIT BREAKER TRIGGERED — {trigger_label}",
            f"Time: {latest.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"Metric: {latest.metric_value:.2f} | Threshold: {latest.threshold:.2f}",
            f"Total events this session: {len(circuit_events)}",
        ]
        html = (
            '<div class="cb-banner-active">'
            + "<br>".join(lines)
            + "</div>"
        )
    else:
        html = (
            '<div class="cb-banner-clear">'
            "RISK STATUS: ALL SYSTEMS NOMINAL — No circuit-breaker events"
            "</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("GIFT Nifty Intraday Basis Arbitrage — Institutional Risk Dashboard")
st.caption(
    "Simulated statistical arbitrage between NSE Nifty 50 (onshore INR) "
    "and GIFT Nifty (offshore USD). All figures are hypothetical."
)

# Auto-run on first load; re-run only if the button is pressed after that.
if "results" not in st.session_state or run_btn:
    st.session_state["results"] = run_full_simulation(
        seed, rolling_window, entry_z, exit_z, max_dd, vol_threshold, cool_down
    )

res = st.session_state["results"]
market_data: pd.DataFrame = res["market_data"]
ledger: pd.DataFrame = res["ledger"]
equity_curve: pd.Series = res["equity_curve"]
benchmark_curve: pd.Series = res["benchmark_curve"]
drawdown_series: pd.Series = res["drawdown_series"]
risk_df: pd.DataFrame = res["risk_df"]
circuit_events: list[CircuitBreakerEvent] = res["circuit_events"]
cfg: TradingConfig = res["cfg"]

# ── Circuit breaker banner ────────────────────────────────────────────────────
st.markdown("### Risk Control Panel")
_render_circuit_banner(circuit_events)
st.markdown("")

if circuit_events:
    with st.expander(f"View all {len(circuit_events)} circuit-breaker events"):
        cb_df = pd.DataFrame(
            [
                {
                    "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Bar": e.bar_idx,
                    "Type": e.trigger_type,
                    "Metric Value": e.metric_value,
                    "Threshold": e.threshold,
                    "Cool-Down Until Bar": e.cool_down_until_bar,
                }
                for e in circuit_events
            ]
        )
        st.dataframe(cb_df, use_container_width=True)

st.markdown("---")

# ── KPI Ribbon ────────────────────────────────────────────────────────────────
st.markdown("### Performance KPIs")
kpis = _compute_kpis(equity_curve, ledger, cfg)

kpi_keys = list(kpis.keys())
cols = st.columns(len(kpi_keys))
pct_metrics = {"Total Return (%)", "Win Rate (%)", "Max Drawdown (%)"}

for col, key in zip(cols, kpi_keys):
    with col:
        st.markdown(
            _kpi_card(key, kpis[key], is_pct=(key in pct_metrics)),
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("### Spread & Signal Analysis")
st.plotly_chart(
    _build_spread_chart(market_data, entry_z),
    use_container_width=True,
)

col_left, col_right = st.columns([3, 2])
with col_left:
    st.plotly_chart(_build_equity_chart(equity_curve, benchmark_curve), use_container_width=True)
with col_right:
    st.plotly_chart(_build_drawdown_chart(drawdown_series), use_container_width=True)

st.plotly_chart(_build_vol_chart(risk_df), use_container_width=True)

st.markdown("---")

# ── Execution Ledger ──────────────────────────────────────────────────────────
st.markdown("### Execution Ledger")

if ledger.empty:
    st.info("No trades executed under the current parameters.")
else:
    # Format display columns
    display_ledger = ledger.copy()
    for col in ["entry_timestamp", "exit_timestamp"]:
        display_ledger[col] = pd.to_datetime(display_ledger[col]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    for col in ["entry_nifty", "exit_nifty"]:
        display_ledger[col] = display_ledger[col].map("{:,.2f}".format)
    for col in ["entry_gift", "exit_gift"]:
        display_ledger[col] = display_ledger[col].map("{:,.4f}".format)
    for col in ["gross_pnl_usd", "transaction_cost_usd", "net_pnl_usd"]:
        display_ledger[col] = display_ledger[col].map("${:,.2f}".format)

    st.dataframe(display_ledger, use_container_width=True, height=360)

    # CSV export
    csv_bytes = ledger.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="Export Ledger as CSV",
        data=csv_bytes,
        file_name="gift_nifty_execution_ledger.csv",
        mime="text/csv",
    )

# ── Risk report export ────────────────────────────────────────────────────────
if not risk_df.empty:
    with st.expander("View Risk Engine Report"):
        st.dataframe(risk_df.iloc[::5].reset_index(), use_container_width=True)
        risk_csv = risk_df.reset_index().to_csv(
            index=False, encoding="utf-8-sig"
        ).encode("utf-8-sig")
        st.download_button(
            label="Export Risk Report as CSV",
            data=risk_csv,
            file_name="gift_nifty_risk_report.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Powered by Claude Code | All results are simulated and for educational "
    "purposes only. Not financial advice."
)
