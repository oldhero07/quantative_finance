# -*- coding: utf-8 -*-
"""
dashboard/app.py - Streamlit Dashboard
Tech GC 2026 | IIT Roorkee
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Absolute path to repo root (dashboard/app.py -> parent = repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

st.set_page_config(
    page_title="Quant Trading System - Tech GC 2026",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1A1A2E;
        border: 1px solid #333355;
        border-radius: 8px;
        padding: 12px;
    }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "strategy": "#00BFFF",
    "benchmark": "#FF7F50",
    "positive":  "#2ECC71",
    "negative":  "#E74C3C",
    "neutral":   "#F39C12",
    "bg":        "#0F0F1A",
    "panel":     "#111128",
}

REGIME_COLORS = {
    "BULL":     "#2ECC71",
    "BEAR":     "#E74C3C",
    "SIDEWAYS": "#F39C12",
    "CRISIS":   "#8E44AD",
}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def abs_path(*parts):
    """Build an absolute path from repo root."""
    return os.path.join(REPO_ROOT, *parts)


def read_csv(path, **kwargs):
    """Read CSV, return None on any failure."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


@st.cache_data
def load_all_outputs():
    data = {}

    # Find output dir — prefer outputs/ then outputs/test/
    out_dir = None
    for candidate in ["outputs", "outputs/test"]:
        if os.path.exists(abs_path(candidate, "equity_curve.csv")):
            out_dir = abs_path(candidate)
            break

    if out_dir is None:
        return data

    # ---- Equity curve ----
    eq = read_csv(os.path.join(out_dir, "equity_curve.csv"), parse_dates=True)
    if eq is not None:
        if eq.index.name != "date":
            eq = eq.set_index(eq.columns[0])
        eq.index = pd.to_datetime(eq.index)
        data["equity"] = eq

    # ---- Regime labels ----
    reg = read_csv(os.path.join(out_dir, "regime_labels.csv"))
    if reg is not None:
        reg = reg.set_index(reg.columns[0])
        reg.index = pd.to_datetime(reg.index)
        reg.columns = ["regime"]
        data["regimes"] = reg

    # ---- Position scale ----
    sc = read_csv(os.path.join(out_dir, "position_scale.csv"))
    if sc is not None:
        sc = sc.set_index(sc.columns[0])
        sc.index = pd.to_datetime(sc.index)
        data["scale"] = sc

    # ---- Options signals ----
    opt = read_csv(os.path.join(out_dir, "options_signals.csv"), parse_dates=True)
    if opt is not None:
        if opt.index.name not in ("Date", "date"):
            opt = opt.set_index(opt.columns[0])
        opt.index = pd.to_datetime(opt.index)
        # Fill NaN smoothed PCR with raw PCR
        if "pcr_smoothed" in opt.columns:
            opt["pcr_smoothed"] = opt["pcr_smoothed"].fillna(opt["pcr_raw"])
        data["options"] = opt

    # ---- Trade log ----
    tr = read_csv(os.path.join(out_dir, "trade_log.csv"), parse_dates=["date"])
    if tr is not None:
        data["trades"] = tr

    # ---- Performance metrics ----
    perf = read_csv(os.path.join(out_dir, "performance.csv"), index_col=0)
    if perf is not None:
        data["metrics"] = perf

    # ---- Alpha decay ----
    alpha = read_csv(os.path.join(out_dir, "alpha_decay.csv"), index_col=0)
    if alpha is not None:
        data["alpha_decay"] = alpha

    # ---- Regime summary ----
    rs = read_csv(os.path.join(out_dir, "regime_summary.csv"), index_col=0)
    if rs is not None:
        data["regime_summary"] = rs

    # ---- Nifty benchmark ----
    for nifty_rel in ["data/test/nifty.csv", "data/nifty.csv"]:
        nifty_path = abs_path(nifty_rel)
        if os.path.exists(nifty_path):
            nf = read_csv(nifty_path)
            if nf is not None:
                nf = nf.set_index(nf.columns[0])
                nf.index = pd.to_datetime(nf.index)
                data["nifty"] = nf.squeeze()
            break

    return data


def dark_layout(title="", height=400):
    return dict(
        title=dict(text=title, font=dict(color="white", size=14)),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color="white"),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor="#222244", zerolinecolor="#333355"),
        yaxis=dict(gridcolor="#222244", zerolinecolor="#333355"),
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## \U0001f4ca Quant Trading System")
        st.markdown("**Tech GC 2026 | IIT Roorkee**")
        st.markdown("---")
        st.markdown("### Strategy")
        st.markdown("- **Universe**: Nifty 500 (499 stocks)")
        st.markdown("- **Style**: Equity Long-Short")
        st.markdown("- **Signal**: Multi-Factor (4 factors)")
        st.markdown("- **Regime**: HMM (4 states) + FII/DII")
        st.markdown("- **Options**: PCR + IV Skew")
        st.markdown("- **Rebalance**: Weekly")
        st.markdown("---")
        st.markdown("### Cost Model")
        st.markdown("- Commission: 0.05% / side")
        st.markdown("- Bid-Ask: 0.05% / side")
        st.markdown("- Slippage: Almgren-Chriss")
        st.markdown("---")
        page = st.radio("Navigate", [
            "\U0001f4ca Strategy Overview",
            "\U0001f30d Regime Analysis",
            "\U0001f4c8 Options Signals",
            "\U0001f4dd Trade Log",
            "\U0001f9ea Factor Analysis",
            "\U0001f3db Architecture",
        ])
    return page


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

def page_overview(data):
    st.title("\U0001f4ca Strategy Overview")

    equity  = data.get("equity")
    metrics = data.get("metrics")
    nifty   = data.get("nifty")

    if equity is None:
        st.error("No backtest data found.")
        st.code("python quick_test.py --stocks 50", language="bash")
        return

    nav = equity["nav"]
    ret = equity["daily_return"]

    total_ret   = nav.iloc[-1] / nav.iloc[0] - 1
    n_days      = len(ret)
    ann_ret     = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol     = ret.std() * np.sqrt(252)
    sharpe      = (ann_ret - 0.065) / ann_vol if ann_vol > 0 else 0
    rolling_max = nav.cummax()
    drawdown    = (nav - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    win_rate    = (ret > 0).mean()

    st.markdown("### Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Return",    f"{total_ret:.1%}")
    c2.metric("Ann. Return",     f"{ann_ret:.1%}")
    c3.metric("Sharpe Ratio",    f"{sharpe:.2f}")
    c4.metric("Max Drawdown",    f"{max_dd:.1%}")
    c5.metric("Ann. Volatility", f"{ann_vol:.1%}")
    c6.metric("Win Rate",        f"{win_rate:.1%}")
    st.markdown("---")

    st.markdown("### Equity Curve vs Nifty 50 Benchmark")
    fig = make_subplots(rows=3, cols=1, row_heights=[0.55, 0.25, 0.20],
                        shared_xaxes=True, vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=nav.index, y=nav / 1e7, name="Strategy",
                             line=dict(color=COLORS["strategy"], width=2)), row=1, col=1)

    if nifty is not None:
        nifty_a = nifty.reindex(nav.index).ffill().bfill()
        if nifty_a.iloc[0] > 0:
            nifty_n = nifty_a / nifty_a.iloc[0] * nav.iloc[0]
            fig.add_trace(go.Scatter(x=nifty_n.index, y=nifty_n / 1e7, name="Nifty 50",
                                     line=dict(color=COLORS["benchmark"], width=1.5, dash="dash"),
                                     opacity=0.8), row=1, col=1)

    fig.add_vline(x="2022-12-31", line_dash="dot", line_color="yellow", opacity=0.6)

    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown %",
                             fill="tozeroy", line=dict(color=COLORS["negative"], width=1),
                             fillcolor="rgba(231,76,60,0.3)"), row=2, col=1)

    bar_colors = [COLORS["positive"] if r > 0 else COLORS["negative"] for r in ret.values]
    fig.add_trace(go.Bar(x=ret.index, y=ret * 100, name="Daily Return %",
                         marker_color=bar_colors, opacity=0.7), row=3, col=1)

    fig.update_layout(**dark_layout(height=650))
    fig.update_yaxes(title_text="NAV (INR Cr)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %",   row=2, col=1)
    fig.update_yaxes(title_text="Return %",     row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    if metrics is not None:
        st.markdown("### Walk-Forward: Train vs Test")
        st.dataframe(metrics.T)


# --------------------------------------------------
# PAGE 2: REGIME ANALYSIS
# --------------------------------------------------

def page_regime(data):
    st.title("\U0001f30d Market Regime Analysis")
    st.markdown("*4-state Hidden Markov Model trained on daily returns, volatility, trend, and VIX*")

    regimes = data.get("regimes")
    scale   = data.get("scale")
    summary = data.get("regime_summary")
    nifty   = data.get("nifty")

    if regimes is None:
        st.error("Regime data not found. Run `python quick_test.py` and redeploy.")
        return

    regime_series = regimes["regime"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Market Regimes Over Time")
        fig = go.Figure()

        if nifty is not None:
            nifty_a = nifty.reindex(regime_series.index).ffill().bfill()
            for rname, colour in REGIME_COLORS.items():
                mask = (regime_series == rname)
                if not mask.any():
                    continue
                idx   = regime_series[mask].index
                yvals = nifty_a.reindex(idx).values
                r = int(colour[1:3], 16)
                g = int(colour[3:5], 16)
                b = int(colour[5:7], 16)
                fig.add_trace(go.Scatter(
                    x=list(idx) + [idx[-1], idx[0]],
                    y=list(yvals) + [float(nifty_a.min()), float(nifty_a.min())],
                    fill="toself", fillcolor=f"rgba({r},{g},{b},0.18)",
                    line=dict(width=0), name=rname, showlegend=True,
                ))
            fig.add_trace(go.Scatter(
                x=nifty_a.index, y=nifty_a.values,
                name="Nifty 50", line=dict(color="white", width=1.5)
            ))
        else:
            # No Nifty — show regime bar chart
            for rname, colour in REGIME_COLORS.items():
                mask = (regime_series == rname)
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=regime_series[mask].index,
                    y=[1] * mask.sum(),
                    mode="markers", marker=dict(color=colour, size=4),
                    name=rname
                ))

        fig.update_layout(**dark_layout(height=380))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Regime Distribution")
        counts = regime_series.value_counts()
        fig_pie = px.pie(values=counts.values, names=counts.index,
                         color=counts.index, color_discrete_map=REGIME_COLORS, hole=0.4)
        fig_pie.update_layout(**dark_layout(height=300))
        st.plotly_chart(fig_pie, use_container_width=True)

    if scale is not None:
        st.markdown("#### Position Scale Factor")
        st.caption("1.0 = full size | 0.15 = crisis (almost flat)")
        scale_s = scale.squeeze()
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=scale_s.index, y=scale_s.values,
                                   fill="tozeroy", name="Position Scale",
                                   line=dict(color=COLORS["strategy"], width=1.5),
                                   fillcolor="rgba(0,191,255,0.2)"))
        fig_s.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.4)
        fig_s.update_layout(**dark_layout(height=250))
        st.plotly_chart(fig_s, use_container_width=True)

    if summary is not None:
        st.markdown("#### Regime Statistics")
        st.dataframe(summary, use_container_width=True)


# --------------------------------------------------
# PAGE 3: OPTIONS SIGNALS
# --------------------------------------------------

def page_options(data):
    st.title("\U0001f4c8 Options Market Signals")
    st.markdown("*Put-Call Ratio and IV Skew as regime confirmation signals*")

    options = data.get("options")
    if options is None:
        st.error("Options data not found. Run `python quick_test.py` and redeploy.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg PCR",         f"{options['pcr_raw'].mean():.3f}")
    c2.metric("Avg IV Skew",     f"{options['iv_skew'].mean():.4f}")
    c3.metric("Crash Risk Days", f"{int(options['crash_risk_flag'].sum())}")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=options.index, y=options["pcr_smoothed"],
                             name="PCR (smoothed)", line=dict(color=COLORS["strategy"], width=1.5)),
                  row=1, col=1)
    fig.add_hline(y=1.2, line_dash="dash", line_color=COLORS["negative"], opacity=0.6,
                  annotation_text="Bearish zone", row=1, col=1)
    fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS["positive"], opacity=0.6,
                  annotation_text="Bullish zone", row=1, col=1)

    fig.add_trace(go.Scatter(x=options.index, y=options["iv_skew"],
                             name="IV Skew", line=dict(color=COLORS["neutral"], width=1.5)),
                  row=2, col=1)
    fig.add_hline(y=0.08, line_dash="dash", line_color=COLORS["negative"], opacity=0.6,
                  annotation_text="High crash risk", row=2, col=1)

    sig_colors = [COLORS["positive"] if v > 0 else COLORS["negative"]
                  for v in options["combined_signal"].values]
    fig.add_trace(go.Bar(x=options.index, y=options["combined_signal"],
                         name="Combined Signal", marker_color=sig_colors),
                  row=3, col=1)

    fig.update_layout(**dark_layout(title="PCR + IV Skew Signals", height=580))
    fig.update_yaxes(title_text="Put-Call Ratio", row=1, col=1)
    fig.update_yaxes(title_text="IV Skew",        row=2, col=1)
    fig.update_yaxes(title_text="Signal",         row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Reading the signals:**
- **PCR > 1.2** — heavy put buying, market is fearful — contrarian buy signal
- **PCR < 0.7** — heavy call buying, market is greedy — reduce longs
- **IV Skew > 0.08** — options market pricing crash risk — cut long exposure
    """)


# --------------------------------------------------
# PAGE 4: TRADE LOG
# --------------------------------------------------

def page_trades(data):
    st.title("\U0001f4dd Trade Log")

    trades = data.get("trades")
    if trades is None or len(trades) == 0:
        st.error("Trade log not found. Run `python quick_test.py` and redeploy.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades",   f"{len(trades):,}")
    c2.metric("Total Cost",     f"INR {trades['cost'].sum():,.0f}")
    c3.metric("Avg Cost (bps)", f"{trades['cost_bps'].mean():.1f}")

    col1, col2, col3 = st.columns(3)
    direction = col1.selectbox("Direction", ["All", "BUY", "SELL"])
    min_val   = col2.number_input("Min Trade Value (INR)", value=0, step=10000)
    search    = col3.text_input("Search Ticker")

    filtered = trades.copy()
    if direction != "All":
        filtered = filtered[filtered["direction"] == direction]
    if min_val > 0:
        filtered = filtered[filtered["trade_value"] >= min_val]
    if search:
        filtered = filtered[filtered["ticker"].str.contains(search.upper(), na=False)]

    st.markdown(f"**Showing {len(filtered):,} of {len(trades):,} trades**")
    st.dataframe(filtered.tail(500), use_container_width=True)


# --------------------------------------------------
# PAGE 5: FACTOR ANALYSIS
# --------------------------------------------------

def page_factors(data):
    st.title("\U0001f9ea Factor Analysis")

    alpha = data.get("alpha_decay")
    if alpha is not None:
        st.markdown("### Alpha Decay — Information Coefficient by Horizon")
        st.markdown("""
IC (Information Coefficient) measures how well the signal predicts future returns.
- IC > 0.05 = useful signal &nbsp;|&nbsp; IC > 0.10 = strong signal
- IR (IC / std) > 0.5 = statistically robust
        """)
        horizons = ["IC_1d", "IC_5d", "IC_10d", "IC_21d"]
        labels   = ["1 day", "5 days", "10 days", "21 days"]
        mean_ic  = [float(alpha.loc[h, "mean_IC"]) if h in alpha.index and not pd.isna(alpha.loc[h, "mean_IC"]) else 0.0 for h in horizons]
        std_ic   = [float(alpha.loc[h, "std_IC"])  if h in alpha.index and not pd.isna(alpha.loc[h, "std_IC"])  else 0.0 for h in horizons]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=mean_ic,
                             error_y=dict(type="data", array=std_ic, visible=True),
                             marker_color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in mean_ic],
                             name="Mean IC"))
        fig.add_hline(y=0.05, line_dash="dash", line_color="yellow",
                      opacity=0.6, annotation_text="IC = 0.05 (useful)")
        fig.update_layout(**dark_layout(title="Signal IC by Forward Return Horizon", height=350))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(alpha, use_container_width=True)
    else:
        st.info("Alpha decay data not found.")

    st.markdown("### Factor Weights in Composite Signal")
    factors = {"Momentum (12m-1m)": 0.40, "RSI Mean-Reversion": 0.25,
               "Volume Confirmation": 0.20, "Volatility Penalty": 0.15}
    fig2 = px.pie(values=list(factors.values()), names=list(factors.keys()),
                  color_discrete_sequence=["#00BFFF", "#2ECC71", "#F39C12", "#E74C3C"], hole=0.4)
    fig2.update_layout(**dark_layout(height=320))
    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# PAGE 6: ARCHITECTURE
# --------------------------------------------------

def page_architecture():
    st.title("\U0001f3db System Architecture")
    st.markdown("""
### Pipeline

```
data_loader.py       ->  499 Nifty 500 stocks | 5Y daily OHLCV | parquet (offline)
       |
signals.py           ->  4-factor composite + 52-week filter + confidence score (0-85)
       |
regime_detection.py  ->  HMM 4 states + FII/DII flow overlay -> position scale
       |
options_signal.py    ->  PCR + IV Skew overlay signals
       |
portfolio.py         ->  Inverse-vol risk parity x confidence x regime scale
       |
backtest.py          ->  Walk-forward + Almgren-Chriss costs + full metrics
       |
dashboard/app.py     ->  This Streamlit interface
```

### Regime-Aware Position Sizing

| Regime   | HMM Scale | Rationale |
|----------|-----------|-----------|
| Bull     | 1.00x     | Full size — trend is favourable |
| Bear     | 0.70x     | Reduce — directional risk elevated |
| Sideways | 0.40x     | Cut — signal quality drops in choppy markets |
| Crisis   | 0.15x     | Almost flat — capital preservation mode |

FII/DII overlay multiplies the above: Bullish 1.0x, Neutral 0.7x, Bearish 0.4x.

### 52-Week Proximity Filter

- **Longs** only within 15% of 52-week high (structural strength confirmed)
- **Shorts** only within 15% of 52-week low (structural weakness confirmed)
- Removes ~62% of raw signals, keeping only high-conviction setups

### Confidence Scoring

Each signal gets a score 0-85 based on: momentum rank, factor agreement,
volume confirmation, RSI zone quality, proximity to 52-week extreme.

Position size scales as: `0.5 + 0.5 x (confidence / 85)`

### Cost Model

```
Total cost per trade =
    Commission (0.05%/side)
  + Bid-Ask spread (0.05%/side)
  + Almgren-Chriss: sigma x sqrt(order_size / ADV) x 0.10
```

### Validation

Walk-forward split: Train 2021-2022, Test OOS 2023-2026.
No look-ahead bias. Alpha decay tested at 1, 5, 10, 21 day horizons.
    """)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    data = load_all_outputs()
    page = render_sidebar()

    if   page == "\U0001f4ca Strategy Overview": page_overview(data)
    elif page == "\U0001f30d Regime Analysis":   page_regime(data)
    elif page == "\U0001f4c8 Options Signals":   page_options(data)
    elif page == "\U0001f4dd Trade Log":         page_trades(data)
    elif page == "\U0001f9ea Factor Analysis":   page_factors(data)
    elif page == "\U0001f3db Architecture":      page_architecture()


if __name__ == "__main__":
    main()
