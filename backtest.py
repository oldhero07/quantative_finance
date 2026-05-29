"""
backtest.py -- Backtesting Engine
=================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

What this does:
    Simulates running your strategy over historical data day by day.
    Applies realistic transaction costs: commission + spread + slippage.
    Tracks portfolio value, trades, and P&L every single day.
    Produces the equity curve, trade log, and performance metrics.

Cost Model:
    commission  = 0.10% round-trip (0.05% each side)
    bid_ask     = 0.10% round-trip (0.05% each side, cost of crossing the spread)
    slippage    = f(order_size, volatility) using square-root model
    total       ~ 0.20-0.35% round trip per position

Walk-Forward Validation:
    Train: 2021      (signal calibration)
    Test:  2022-2026 (out-of-sample evaluation)

Output files:
    equity_curve.csv   -- portfolio NAV every trading day
    trade_log.csv      -- every trade with entry/exit/cost/PnL
    performance.csv    -- all performance metrics (Sharpe, drawdown, etc.)
    backtest_chart.png -- equity curve vs Nifty benchmark chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

INITIAL_CAPITAL  = 10_000_000   # INR1 crore starting capital
COMMISSION_RT    = 0.0010       # 0.10% round-trip commission (0.05% each side)
BID_ASK_RT       = 0.0010       # 0.10% round-trip bid-ask spread
SLIPPAGE_FACTOR  = 0.10         # slippage scaling factor (calibrated)
TRAIN_END        = "2021-12-31" # walk-forward split date
RISK_FREE_RATE   = 0.065        # Indian 10Y Gsec yield ~ 6.5%


# ---------------------------------------------
# TRANSACTION COST MODEL
# ---------------------------------------------

def compute_transaction_cost(trade_value, stock_vol, adv):
    """
    Realistic transaction cost model combining 3 components.

    Component 1 -- Commission:
        Fixed percentage per trade. Covers broker fees, exchange fees, STT.
        commission = COMMISSION_RT x |trade_value|

    Component 2 -- Bid-Ask Spread:
        Cost of crossing the bid-ask spread when entering/exiting.
        In liquid Indian markets, spread ~ 0.05% for large caps.
        bid_ask_cost = BID_ASK_RT x |trade_value|

    Component 3 -- Slippage (Market Impact):
        When you buy, your buying pushes price up slightly against you.
        When you sell, your selling pushes price down against you.
        Larger orders in less-liquid stocks = more slippage.

        We use the Square-Root Model (Almgren & Chriss):
        slippage = SLIPPAGE_FACTOR x volatility x sqrt(order_size / ADV)

        Where:
            volatility = daily vol of the stock
            order_size = value of your order in rupees
            ADV        = average daily volume x price (rupees)

    Args:
        trade_value : float -- absolute value of trade in rupees
        stock_vol   : float -- annualised daily volatility of stock
        adv         : float -- average daily value traded (rupees)

    Returns:
        total_cost : float -- total transaction cost in rupees
    """
    commission  = COMMISSION_RT * abs(trade_value)
    bid_ask     = BID_ASK_RT    * abs(trade_value)

    # Square-root market impact model
    if adv > 0 and not np.isnan(stock_vol):
        participation = abs(trade_value) / max(adv, 1)
        slippage = SLIPPAGE_FACTOR * stock_vol * np.sqrt(participation) * abs(trade_value)
    else:
        slippage = 0.0

    total = commission + bid_ask + slippage
    return total


# ---------------------------------------------
# CORE BACKTEST ENGINE
# ---------------------------------------------

def run_backtest(daily_weights, close, returns, volume,
                 initial_capital=INITIAL_CAPITAL, output_dir="outputs/"):
    """
    Simulate strategy performance day by day over the full history.

    Algorithm:
        For each trading day:
            1. Get current portfolio holdings (from yesterday's weights)
            2. Get target weights for today (from daily_weights)
            3. Compute trades needed to move from current to target
            4. Compute transaction costs for those trades
            5. Update portfolio value:
               new_NAV = old_NAV x (1 + portfolio_return) - transaction_costs
            6. Record: NAV, trades, costs, positions

    Args:
        daily_weights   : pd.DataFrame -- target weights per stock per day
        close           : pd.DataFrame -- closing prices
        returns         : pd.DataFrame -- daily returns
        volume          : pd.DataFrame -- daily trading volume
        initial_capital : float -- starting capital in rupees
        output_dir      : str

    Returns:
        equity_curve : pd.Series -- portfolio NAV each day
        trade_log    : pd.DataFrame -- all trades
        metrics      : dict -- performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BACKTESTING ENGINE")
    print(f"Capital   : INR{initial_capital:,.0f}")
    print(f"Cost model: Commission + Bid-Ask + Almgren-Chriss Slippage")
    print("=" * 60)

    # Align all data to same dates
    common_dates = daily_weights.index.intersection(returns.index)
    common_dates = common_dates.intersection(close.index)
    weights_aligned = daily_weights.reindex(common_dates).fillna(0)
    returns_aligned = returns.reindex(common_dates).fillna(0)
    close_aligned   = close.reindex(common_dates).ffill()
    volume_aligned  = volume.reindex(common_dates).fillna(0)

    # Pre-compute rolling volatility and ADV (average daily value)
    vol_20d = returns_aligned.rolling(20).std() * np.sqrt(252)
    adv_20d = (close_aligned * volume_aligned).rolling(20).mean()

    # Initialise tracking variables
    nav          = initial_capital
    nav_series   = []
    trade_log    = []
    prev_weights = pd.Series(0.0, index=weights_aligned.columns)
    total_costs  = 0.0

    print(f"[Backtest] Simulating {len(common_dates)} trading days...")

    for i, date in enumerate(common_dates):
        target_w   = weights_aligned.loc[date]
        day_return = returns_aligned.loc[date]

        # -- Step 1: Compute portfolio return from yesterday's holdings --
        portfolio_return = (prev_weights * day_return).sum()

        # -- Step 2: Compute required trades (weight changes) --
        weight_changes = target_w - prev_weights
        trades_needed  = weight_changes[weight_changes.abs() > 0.001]   # ignore tiny rebalances

        # -- Step 3: Compute transaction costs --
        day_cost = 0.0
        for ticker in trades_needed.index:
            if ticker not in close_aligned.columns:
                continue

            trade_value = abs(trades_needed[ticker]) * nav
            stock_vol   = vol_20d.loc[date, ticker] if ticker in vol_20d.columns else 0.15
            stock_adv   = adv_20d.loc[date, ticker] if ticker in adv_20d.columns else nav * 0.01

            cost = compute_transaction_cost(trade_value, stock_vol, stock_adv)
            day_cost += cost

            # Log the trade
            if trade_value > 1000:   # only log meaningful trades
                trade_log.append({
                    "date":         date,
                    "ticker":       ticker,
                    "direction":    "BUY" if trades_needed[ticker] > 0 else "SELL",
                    "weight_from":  round(prev_weights.get(ticker, 0), 4),
                    "weight_to":    round(target_w[ticker], 4),
                    "trade_value":  round(trade_value, 2),
                    "cost":         round(cost, 2),
                    "cost_bps":     round(cost / trade_value * 10000, 2),
                })

        # -- Step 4: Update NAV --
        nav = nav * (1 + portfolio_return) - day_cost
        total_costs += day_cost

        nav_series.append({"date": date, "nav": nav,
                           "daily_return": portfolio_return,
                           "cost": day_cost})

        # Update weights for next day
        prev_weights = target_w.copy()

        if (i + 1) % 250 == 0:
            print(f"[Backtest] Progress: {i+1}/{len(common_dates)} days | "
                  f"NAV: INR{nav:,.0f}")

    # -- Compile results --
    equity_curve = pd.DataFrame(nav_series).set_index("date")
    trade_log_df = pd.DataFrame(trade_log)

    print(f"\n[Backtest] Simulation complete.")
    print(f"[Backtest] Total transaction costs: INR{total_costs:,.0f} "
          f"({total_costs/initial_capital*100:.2f}% of initial capital)")

    return equity_curve, trade_log_df


# ---------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------

def compute_metrics(equity_curve, nifty_close,
                    risk_free=RISK_FREE_RATE, output_dir="outputs/"):
    """
    Compute all performance metrics that judges will evaluate.

    Metrics computed:
        Sharpe Ratio     : (annual_return - risk_free) / annual_vol
        Sortino Ratio    : (annual_return - risk_free) / downside_vol
        Calmar Ratio     : annual_return / max_drawdown
        Max Drawdown     : worst peak-to-trough loss
        Win Rate         : % of profitable trading days
        Annualised Return: CAGR over full period
        Annualised Vol   : std of daily returns x sqrt(252)
        VaR 95%          : 95th percentile daily loss
        CVaR 95%         : average loss on worst 5% days

    Walk-forward split:
        Reports metrics separately for train (2018-2021)
        and test (2022-2024) periods.

    Args:
        equity_curve : pd.DataFrame from run_backtest()
        nifty_close  : pd.Series for benchmark comparison
        risk_free    : annual risk-free rate (decimal)
        output_dir   : str

    Returns:
        metrics_df : pd.DataFrame -- all metrics for train/test/full periods
    """
    nav = equity_curve["nav"]
    ret = equity_curve["daily_return"]

    def period_metrics(nav_s, ret_s, label):
        """Compute all metrics for a given time period."""
        n_days       = len(ret_s)
        total_ret    = nav_s.iloc[-1] / nav_s.iloc[0] - 1
        ann_ret      = (1 + total_ret) ** (252 / n_days) - 1
        ann_vol      = ret_s.std() * np.sqrt(252)
        sharpe       = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0

        # Downside volatility (only negative days) for Sortino
        neg_ret      = ret_s[ret_s < 0]
        down_vol     = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else ann_vol
        sortino      = (ann_ret - risk_free) / down_vol if down_vol > 0 else 0

        # Drawdown
        rolling_max  = nav_s.cummax()
        drawdown     = (nav_s - rolling_max) / rolling_max
        max_dd       = drawdown.min()
        calmar       = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Win rate
        win_rate     = (ret_s > 0).mean()

        # VaR and CVaR
        var_95       = np.percentile(ret_s.dropna(), 5)
        cvar_95      = ret_s[ret_s <= var_95].mean()

        return {
            "Period":          label,
            "Start":           nav_s.index[0].strftime("%Y-%m-%d"),
            "End":             nav_s.index[-1].strftime("%Y-%m-%d"),
            "Trading Days":    n_days,
            "Total Return":    f"{total_ret:.2%}",
            "Ann. Return":     f"{ann_ret:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio":    f"{sharpe:.3f}",
            "Sortino Ratio":   f"{sortino:.3f}",
            "Calmar Ratio":    f"{calmar:.3f}",
            "Max Drawdown":    f"{max_dd:.2%}",
            "Win Rate":        f"{win_rate:.2%}",
            "VaR (95%)":       f"{var_95:.2%}",
            "CVaR (95%)":      f"{cvar_95:.2%}",
        }

    metrics = []

    # Full period
    metrics.append(period_metrics(nav, ret, "Full Period"))

    # Train period
    train_mask = nav.index <= TRAIN_END
    if train_mask.sum() > 10:
        metrics.append(period_metrics(nav[train_mask], ret[train_mask], "Train (2018-2021)"))

    # Test period (out-of-sample -- most important)
    test_mask = nav.index > TRAIN_END
    if test_mask.sum() > 10:
        metrics.append(period_metrics(nav[test_mask], ret[test_mask], "Test OOS (2022-2024)"))

    metrics_df = pd.DataFrame(metrics).set_index("Period")
    metrics_df.to_csv(f"{output_dir}performance.csv")

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(metrics_df.T.to_string())
    print("=" * 60)

    return metrics_df


# ---------------------------------------------
# CHART
# ---------------------------------------------

def plot_backtest(equity_curve, nifty_close, output_path="outputs/backtest_chart.png"):
    """
    Plot equity curve, drawdown, and daily returns.
    This goes directly into the Streamlit dashboard.
    """
    nav   = equity_curve["nav"]
    ret   = equity_curve["daily_return"]

    # Normalise Nifty to same starting point
    nifty_aligned = nifty_close.reindex(nav.index).ffill()
    nifty_norm    = nifty_aligned / nifty_aligned.iloc[0] * nav.iloc[0]

    # Drawdown
    rolling_max = nav.cummax()
    drawdown    = (nav - rolling_max) / rolling_max

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.patch.set_facecolor("#0F0F1A")

    for ax in axes:
        ax.set_facecolor("#0F0F1A")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 1: Equity curve
    axes[0].plot(nav.index, nav / 1e7, color="#00BFFF",
                 linewidth=2, label="Strategy")
    axes[0].plot(nifty_norm.index, nifty_norm / 1e7, color="#FF7F50",
                 linewidth=1.5, linestyle="--", alpha=0.8, label="Nifty 50 (benchmark)")
    axes[0].axvline(pd.Timestamp(TRAIN_END), color="yellow",
                    linestyle=":", alpha=0.7, linewidth=1.5, label="Train/Test split")
    axes[0].set_title("Portfolio Equity Curve vs Nifty 50",
                      color="white", fontsize=13, pad=10)
    axes[0].set_ylabel("Portfolio Value (INR Cr)", color="white")
    axes[0].legend(facecolor="#1A1A2E", edgecolor="#333355",
                   labelcolor="white", fontsize=10)

    # Panel 2: Drawdown
    axes[1].fill_between(drawdown.index, drawdown.values * 100, 0,
                         alpha=0.7, color="#E74C3C")
    axes[1].set_title("Drawdown (%)", color="white", fontsize=11, pad=8)
    axes[1].set_ylabel("Drawdown %", color="white")

    # Panel 3: Daily returns histogram
    axes[2].hist(ret.dropna() * 100, bins=80, color="#2ECC71",
                 alpha=0.7, edgecolor="none")
    axes[2].axvline(0, color="white", linestyle="--", alpha=0.5)
    axes[2].set_title("Daily Return Distribution", color="white", fontsize=11, pad=8)
    axes[2].set_xlabel("Daily Return (%)", color="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0F0F1A")
    plt.close()
    print(f"[Backtest] Chart saved -> {output_path}")


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def run_full_backtest(daily_weights, close, returns, volume,
                      nifty_close, output_dir="outputs/"):
    """Master function: run backtest + metrics + chart."""
    equity_curve, trade_log = run_backtest(
        daily_weights, close, returns, volume, output_dir=output_dir
    )
    metrics = compute_metrics(equity_curve, nifty_close, output_dir=output_dir)
    plot_backtest(equity_curve, nifty_close,
                  output_path=f"{output_dir}backtest_chart.png")

    equity_curve.to_csv(f"{output_dir}equity_curve.csv")
    if not trade_log.empty:
        trade_log.to_csv(f"{output_dir}trade_log.csv", index=False)

    print(f"\n[Backtest] All outputs saved to {output_dir}")
    return equity_curve, trade_log, metrics


if __name__ == "__main__":
    from data_loader import load_data
    from signals import generate_signals
    from portfolio import build_portfolio

    close, _, _, _, volume, returns, nifty, vix = load_data()
    positions, signals, factors = generate_signals(close, volume)
    daily_weights, _ = build_portfolio(positions, returns)
    run_full_backtest(daily_weights, close, returns, volume, nifty)
