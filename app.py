"""
app.py — Institutional-grade Streamlit analytics dashboard for the
GIFT Nifty Basis Arbitrage Engine.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

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
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
        border: 1px solid #30304a;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        color: #9090b0;
        font-size: 0.72rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #e8e8f0;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-value.positive { color: #3ddc84; }
    .kpi-value.negative { color: #ff4b4b; }

    /* Circuit breaker banners */
    .cb-banner-active {
        background: linear-gradient(90deg, #8b0000 0%, #cc0000 100%);
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 18px 24px;
        animation: pulse 1.1s ease-in-out infinite;
        font-weight: 700;
        font-size: 1.1rem;
        color: #ffffff;
        line-height: 1.7;
        letter-spacing: 0.02em;
    }
    .cb-banner-clear {
        background: linear-gradient(90deg, #0d3d22 0%, #1a6e40 100%);
        border: 2px solid #3ddc84;
        border-radius: 10px;
        padding: 18px 24px;
        font-weight: 600;
        font-size: 1.05rem;
        color: #cfffdd;
        line-height: 1.6;
    }
    @keyframes pulse {
        0%   { opacity: 1.0;  box-shadow: 0 0 0 0 rgba(255,68,68,0.6); }
        50%  { opacity: 0.75; box-shadow: 0 0 18px 6px rgba(255,68,68,0.3); }
        100% { opacity: 1.0;  box-shadow: 0 0 0 0 rgba(255,68,68,0.0); }
    }

    /* Section dividers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #c8c8e8;
        border-left: 4px solid #4f9cf7;
        padding-left: 10px;
        margin: 28px 0 10px 0;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Simulation Controls")
    st.caption("Adjust parameters and click Run to re-simulate.")
    st.markdown("---")

    seed = st.slider("Random Seed", 1, 200, 42)
    rolling_window = st.slider(
        "Z-Score Rolling Window (bars)", 20, 240, CONFIG.rolling_window
    )
    entry_z = st.slider(
        "Entry Z-Score Threshold", 1.0, 4.0, CONFIG.entry_zscore, 0.1
    )
    exit_z = st.slider(
        "Exit Z-Score Threshold", -1.0, 1.5, CONFIG.exit_zscore, 0.1
    )
    max_dd = st.slider(
        "Max Trailing Drawdown (%)",
        -15.0, -1.0, CONFIG.max_trailing_drawdown_pct, 0.5,
    )
    vol_threshold = st.slider(
        "Vol Shock Threshold (%)",
        5.0, 100.0, CONFIG.vol_shock_threshold_pct, 5.0,
    )
    cool_down = st.slider(
        "Cool-Down Period (bars)", 5, 120, CONFIG.cool_down_bars
    )

    st.markdown("---")
    run_btn = st.button(
        "Run Simulation", use_container_width=True, type="primary"
    )

    st.markdown("---")
    st.caption(
        "**How parameters work**\n\n"
        "- **Rolling Window** — how many past 1-minute bars the algorithm "
        "uses to measure 'normal' spread behaviour.\n"
        "- **Entry Z-Score** — how far the spread must stretch before a "
        "trade fires (higher = rarer, bigger moves only).\n"
        "- **Vol Shock Threshold** — % spike in volatility that triggers an "
        "emergency halt.\n"
        "- **Cool-Down** — minutes of forced inactivity after a halt."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation engine (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running simulation — this takes a few seconds…")
def run_full_simulation(
    seed: int,
    rolling_window: int,
    entry_z: float,
    exit_z: float,
    max_dd: float,
    vol_threshold: float,
    cool_down: int,
) -> dict[str, Any]:
    """Run the full pipeline and return all dashboard artefacts.

    Args:
        seed: RNG seed for reproducible data generation.
        rolling_window: Z-score lookback in bars.
        entry_z: Entry Z-score threshold.
        exit_z: Exit Z-score threshold.
        max_dd: Max trailing drawdown limit (%).
        vol_threshold: Volatility shock trigger (%).
        cool_down: Bars to halt after a circuit-breaker event.

    Returns:
        Dictionary of DataFrames, Series, and metadata for the UI.
    """
    cfg = TradingConfig(
        rolling_window=rolling_window,
        entry_zscore=entry_z,
        exit_zscore=exit_z,
        max_trailing_drawdown_pct=max_dd,
        vol_shock_threshold_pct=vol_threshold,
        cool_down_bars=cool_down,
    )

    simulator = MarketSimulator(cfg=cfg, seed=seed)
    market_data = simulator.generate()

    strat_first = BasisArbitrageStrategy(cfg=cfg)
    ledger_first = strat_first.run(market_data)
    equity_first = strat_first.get_equity_curve(market_data, ledger_first)

    risk_engine = InstitutionalRiskEngine(cfg=cfg)
    forced_mask, _, circuit_events = risk_engine.evaluate(
        market_data, equity_first
    )
    risk_df = risk_engine.reports_to_dataframe()

    strat_final = BasisArbitrageStrategy(cfg=cfg)
    ledger = strat_final.run(market_data, forced_exit_mask=forced_mask)
    equity_curve = strat_final.get_equity_curve(market_data, ledger)

    nifty_usd = market_data["nifty_price"] / cfg.usd_inr_fx
    benchmark_curve = cfg.initial_capital * (nifty_usd / nifty_usd.iloc[0])

    rolling_peak = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_peak) / rolling_peak * 100.0

    w = cfg.rolling_window
    market_data["spread"] = market_data["basis"]
    market_data["roll_mean"] = market_data["spread"].rolling(w).mean()
    market_data["roll_std"] = market_data["spread"].rolling(w).std()
    std = market_data["roll_std"].replace(0.0, float("nan"))
    market_data["zscore"] = (
        (market_data["spread"] - market_data["roll_mean"]) / std
    )
    market_data["upper_band"] = (
        market_data["roll_mean"] + entry_z * market_data["roll_std"]
    )
    market_data["lower_band"] = (
        market_data["roll_mean"] - entry_z * market_data["roll_std"]
    )
    market_data["nifty_usd"] = market_data["nifty_price"] / cfg.usd_inr_fx

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


# ─────────────────────────────────────────────────────────────────────────────
# KPI computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_kpis(
    equity_curve: pd.Series,
    ledger: pd.DataFrame,
    cfg: TradingConfig,
) -> dict[str, float]:
    """Compute all strategy performance KPIs.

    Args:
        equity_curve: Bar-by-bar equity Series.
        ledger: Execution ledger DataFrame.
        cfg: Trading configuration.

    Returns:
        Dictionary mapping KPI label to numeric value.
    """
    bars_per_year = cfg.trading_hours_per_day * 60 * 252
    returns = equity_curve.pct_change().dropna()
    total_return_pct = (
        (equity_curve.iloc[-1] / cfg.initial_capital) - 1.0
    ) * 100.0

    rf_per_bar = 0.05 / bars_per_year
    excess = returns - rf_per_bar
    sharpe = (
        excess.mean() / excess.std() * np.sqrt(bars_per_year)
        if excess.std() > 0 else 0.0
    )

    downside = returns[returns < 0]
    sortino = (
        excess.mean() / downside.std() * np.sqrt(bars_per_year)
        if len(downside) > 0 and downside.std() > 0 else 0.0
    )

    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak * 100.0
    max_dd = float(drawdown.min())

    ann_return = (
        (1 + total_return_pct / 100.0)
        ** (bars_per_year / max(len(equity_curve), 1)) - 1.0
    ) * 100.0
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

    if ledger.empty:
        win_rate, total_costs = 0.0, 0.0
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

_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0e0e1a",
    plot_bgcolor="#0e0e1a",
    font_color="#ccccdd",
    font_family="Inter, sans-serif",
)


def _build_convergence_chart(
    market_data: pd.DataFrame, entry_z: float
) -> go.Figure:
    """Two-panel chart: price convergence on top, Z-score rubber band below.

    Args:
        market_data: Enriched market DataFrame.
        entry_z: Entry Z-score threshold for trigger line labels.

    Returns:
        Plotly Figure.
    """
    md = market_data.iloc[::5]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=[
            "Asset Price Convergence: NSE Nifty vs. GIFT Nifty",
            "Basis Spread Variance  (The Market Rubber Band)",
        ],
    )

    # ── Top panel: price convergence ─────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["nifty_usd"],
            name="NSE Nifty 50 (USD equiv.)",
            line=dict(color="#4f9cf7", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["gift_nifty_price"],
            name="GIFT Nifty (USD)",
            line=dict(color="#ffd700", width=1.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # ── Bottom panel: Z-score with trigger lines ──────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=md.index, y=md["zscore"],
            name="Spread Z-Score",
            line=dict(color="#c8aaff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(200,170,255,0.06)",
        ),
        row=2, col=1,
    )

    # Sell trigger (upper boundary)
    fig.add_hline(
        y=entry_z,
        line_color="#ff6b6b",
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"  Sell Trigger — Stretched Upper Boundary (+{entry_z}σ)",
        annotation_position="top left",
        annotation_font_color="#ff6b6b",
        annotation_font_size=11,
        row=2, col=1,
    )
    # Buy trigger (lower boundary)
    fig.add_hline(
        y=-entry_z,
        line_color="#3ddc84",
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"  Buy Trigger — Stretched Lower Boundary (-{entry_z}σ)",
        annotation_position="bottom left",
        annotation_font_color="#3ddc84",
        annotation_font_size=11,
        row=2, col=1,
    )
    # Centre equilibrium
    fig.add_hline(
        y=0,
        line_color="#555577",
        line_dash="dot",
        annotation_text="  Equilibrium (Fair Value)",
        annotation_position="top left",
        annotation_font_color="#888899",
        annotation_font_size=10,
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score (σ)", row=2, col=1)
    fig.update_layout(
        height=560,
        legend=dict(orientation="h", y=-0.10, x=0),
        margin=dict(t=60, b=40),
        **_THEME,
    )
    return fig


def _build_equity_chart(
    equity_curve: pd.Series, benchmark_curve: pd.Series
) -> go.Figure:
    """Equity curve vs. buy-and-hold benchmark.

    Args:
        equity_curve: Strategy equity Series (USD).
        benchmark_curve: Passive benchmark Series (USD).

    Returns:
        Plotly Figure.
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
            fillcolor="rgba(79,156,247,0.07)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bm.index, y=bm,
            name="Buy & Hold Nifty (benchmark)",
            line=dict(color="#ffd700", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        title="Strategy Equity Curve vs. Passive Benchmark",
        yaxis_title="Portfolio Value (USD)",
        height=360,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=50, b=40),
        **_THEME,
    )
    return fig


def _build_drawdown_chart(drawdown_series: pd.Series) -> go.Figure:
    """Underwater drawdown profile.

    Args:
        drawdown_series: Bar-by-bar drawdown % Series.

    Returns:
        Plotly Figure.
    """
    dd = drawdown_series.iloc[::5]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd,
            name="Drawdown (%)",
            line=dict(color="#ff4b4b", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.14)",
        )
    )
    fig.update_layout(
        title="Drawdown Profile — Underwater Periods",
        yaxis_title="Drawdown (%)",
        height=280,
        showlegend=False,
        margin=dict(t=50, b=20),
        **_THEME,
    )
    return fig


def _build_vol_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Rolling 5-bar realised volatility.

    Args:
        risk_df: Risk engine output DataFrame.

    Returns:
        Plotly Figure.
    """
    rv = risk_df["rolling_vol_pct"].iloc[::5]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rv.index, y=rv,
            name="Realised Vol (5-bar %)",
            line=dict(color="#ff9f43", width=1.3),
            fill="tozeroy",
            fillcolor="rgba(255,159,67,0.07)",
        )
    )
    fig.update_layout(
        title="Rolling 5-Minute Realised Volatility",
        yaxis_title="Volatility (%)",
        height=240,
        showlegend=False,
        margin=dict(t=50, b=20),
        **_THEME,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI card renderer
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: float, is_pct: bool = False) -> str:
    """Return an HTML KPI card string.

    Args:
        label: Metric display label.
        value: Numeric value.
        is_pct: Apply +/- colouring and % suffix.

    Returns:
        HTML string.
    """
    fmt = f"{value:+.2f}%" if is_pct else f"{value:,.3g}"
    css_class = "kpi-value"
    if is_pct:
        css_class += " positive" if value >= 0 else " negative"

    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="{css_class}">{fmt}</div>'
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_circuit_banner(circuit_events: list[CircuitBreakerEvent]) -> None:
    """Render a large, prominent risk status indicator.

    Green block when all clear; flashing red block with root-cause detail
    when any circuit breaker has fired.

    Args:
        circuit_events: All events emitted by the risk engine this session.
    """
    if circuit_events:
        latest = circuit_events[-1]
        trigger_label = (
            "VOLATILITY SHOCK OUTLIER DETECTED"
            if latest.trigger_type == "VOLATILITY_SHOCK"
            else "MAX TRAILING DRAWDOWN BREACH"
        )
        html = (
            '<div class="cb-banner-active">'
            f"⛔  CIRCUIT BREAKER TRIGGERED — {trigger_label}<br>"
            f"<span style='font-size:0.92rem; font-weight:400;'>"
            f"First event: {circuit_events[0].timestamp.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
            f"Last event: {latest.timestamp.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
            f"Metric recorded: {latest.metric_value:.2f} &nbsp;|&nbsp; "
            f"Threshold: {latest.threshold:.2f} &nbsp;|&nbsp; "
            f"Total events this session: {len(circuit_events)}"
            f"</span>"
            "</div>"
        )
    else:
        html = (
            '<div class="cb-banner-clear">'
            "✅  RISK STATUS: ALL SYSTEMS NOMINAL<br>"
            "<span style='font-size:0.9rem; font-weight:400;'>"
            "No circuit-breaker events detected. The strategy operated within "
            "all configured risk limits throughout the simulation."
            "</span>"
            "</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("GIFT Nifty Intraday Basis Arbitrage")
st.caption(
    "Simulated statistical arbitrage between NSE Nifty 50 (Mumbai, INR) and "
    "GIFT Nifty (Gujarat, USD). All figures are hypothetical and for "
    "educational purposes only."
)

# ─────────────────────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────────────────────

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

kpis = _compute_kpis(equity_curve, ledger, cfg)

# ─────────────────────────────────────────────────────────────────────────────
# ① Top-line scorecard — st.metric() ribbon, 4 columns
# ─────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)

_ret = kpis["Total Return (%)"]
m1.metric(
    label="Total Net Return",
    value=f"{_ret:+.2f}%",
    delta=f"{'above' if _ret >= 0 else 'below'} breakeven",
    delta_color="normal" if _ret >= 0 else "inverse",
)

_sharpe = kpis["Sharpe Ratio"]
m2.metric(
    label="Annualized Sharpe Ratio",
    value=f"{_sharpe:.3f}",
    delta="risk-adjusted return",
    delta_color="off",
)

_dd = kpis["Max Drawdown (%)"]
m3.metric(
    label="Max Drawdown",
    value=f"{_dd:.2f}%",
    delta=f"limit: {cfg.max_trailing_drawdown_pct:.1f}%",
    delta_color="inverse" if _dd < cfg.max_trailing_drawdown_pct else "off",
)

_trades = len(ledger)
m4.metric(
    label="Total Trades Executed",
    value=f"{_trades:,}",
    delta=f"win rate {kpis['Win Rate (%)']:.1f}%",
    delta_color="off",
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ② Detailed KPI cards
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Performance Summary</div>',
    unsafe_allow_html=True,
)

pct_metrics = {"Total Return (%)", "Win Rate (%)", "Max Drawdown (%)"}
kpi_cols = st.columns(len(kpis))
for col, (label, value) in zip(kpi_cols, kpis.items()):
    with col:
        st.markdown(
            _kpi_card(label, value, is_pct=(label in pct_metrics)),
            unsafe_allow_html=True,
        )

st.markdown("")   # breathing room

# ─────────────────────────────────────────────────────────────────────────────
# ② Risk Control Panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Live Risk Control Panel</div>',
    unsafe_allow_html=True,
)
_render_circuit_banner(circuit_events)

if circuit_events:
    with st.expander(
        f"View full log — {len(circuit_events)} circuit-breaker event(s)"
    ):
        cb_df = pd.DataFrame(
            [
                {
                    "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Bar Index": e.bar_idx,
                    "Trigger Type": e.trigger_type,
                    "Metric Value": e.metric_value,
                    "Threshold": e.threshold,
                    "Cool-Down Until Bar": e.cool_down_until_bar,
                }
                for e in circuit_events
            ]
        )
        st.dataframe(cb_df, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ③ Main signal chart
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Spread & Signal Analysis</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    _build_convergence_chart(market_data, entry_z),
    use_container_width=True,
)
st.info(
    "**How to read this chart**  \n"
    "The top panel shows both markets trading in USD — they should track "
    "each other closely because they represent the same underlying index.  \n"
    "The bottom panel (The Market Rubber Band) measures *how far* the two "
    "prices have drifted apart relative to recent history.  \n"
    "When the line stretches past the **red dashed boundary**, the algorithm "
    "sells the expensive market and buys the cheap one.  \n"
    "When the line stretches past the **green dashed boundary**, it does the "
    "opposite.  \n"
    "In both cases the algorithm expects prices to snap back to the centre "
    "line, capturing the difference as profit."
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ④ Equity curve & drawdown
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Portfolio Performance</div>',
    unsafe_allow_html=True,
)
col_left, col_right = st.columns([3, 2])
with col_left:
    st.plotly_chart(
        _build_equity_chart(equity_curve, benchmark_curve),
        use_container_width=True,
    )
    st.caption(
        "Blue = strategy portfolio value over time. "
        "Gold dashed = a simple 'do nothing, hold Nifty' benchmark. "
        "When blue stays above gold, the arbitrage strategy is adding value."
    )
with col_right:
    st.plotly_chart(
        _build_drawdown_chart(drawdown_series),
        use_container_width=True,
    )
    st.caption(
        "Shows how far the portfolio has fallen from its peak at any point. "
        "Deeper red regions = larger temporary losses. "
        "The strategy auto-halts if this breaches the configured limit."
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Volatility monitor
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Volatility Monitor</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(_build_vol_chart(risk_df), use_container_width=True)
st.caption(
    "Tracks how rapidly prices are moving in each 5-minute window. "
    "Sudden orange spikes indicate market stress events (e.g. macro news). "
    "If a spike exceeds the configured Vol Shock Threshold, the circuit "
    "breaker triggers and all positions are immediately closed."
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ⑥ Execution ledger
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header">Execution Ledger</div>',
    unsafe_allow_html=True,
)

if ledger.empty:
    st.info(
        "No trades were executed under the current parameters. "
        "Try lowering the Entry Z-Score threshold or the Vol Shock Threshold."
    )
else:
    display_ledger = ledger.copy()
    for col in ["entry_timestamp", "exit_timestamp"]:
        display_ledger[col] = pd.to_datetime(
            display_ledger[col]
        ).dt.strftime("%Y-%m-%d %H:%M")
    for col in ["entry_nifty", "exit_nifty"]:
        display_ledger[col] = display_ledger[col].map("{:,.2f}".format)
    for col in ["entry_gift", "exit_gift"]:
        display_ledger[col] = display_ledger[col].map("{:,.4f}".format)
    for col in ["gross_pnl_usd", "transaction_cost_usd", "net_pnl_usd"]:
        display_ledger[col] = display_ledger[col].map("${:,.2f}".format)

    st.caption(
        f"{len(ledger):,} trades recorded. "
        "Green net PnL rows = profitable trades. "
        "Exit reason 'risk_forced_exit' = closed by the circuit breaker."
    )
    st.dataframe(display_ledger, use_container_width=True, height=360)

    csv_bytes = ledger.to_csv(
        index=False, encoding="utf-8-sig"
    ).encode("utf-8-sig")
    st.download_button(
        label="Export Ledger as CSV",
        data=csv_bytes,
        file_name="gift_nifty_execution_ledger.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# ⑦ Risk engine report (collapsible)
# ─────────────────────────────────────────────────────────────────────────────

if not risk_df.empty:
    with st.expander("View full Risk Engine Report (advanced)"):
        st.caption(
            "One row per 5-minute sample. "
            "var_99_usd = the maximum expected 1-bar loss at 99% confidence. "
            "trailing_drawdown_pct = current peak-to-trough drop."
        )
        st.dataframe(
            risk_df.iloc[::5].reset_index(), use_container_width=True
        )
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
    "Built with Claude Code · All results are simulated · "
    "Not financial advice · For educational purposes only."
)
