"""
portfolio.py -- Portfolio Construction & Optimisation Module
============================================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

What this does:
    Takes signal positions (+1/0/-1) from signals.py.
    Converts them into actual portfolio weights (e.g. 5.2% in RELIANCE).
    Applies risk parity weighting, enforces constraints, integrates regime scaling.

Method: Risk Parity (Inverse Volatility Weighting)
    Allocate more capital to low-volatility stocks (stable movers).
    Allocate less capital to high-volatility stocks (risky, expensive to trade).
    Result: each stock contributes EQUAL RISK to the portfolio.

Output:
    weights.csv        -- target weight per stock per rebalance date
    weights_scaled.csv -- weights after regime scaling applied
    portfolio_log.csv  -- rebalance history with turnover stats
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

REBALANCE_FREQ   = "W-FRI"    # rebalance every Friday
VOL_WINDOW       = 20         # days for volatility estimation
MAX_POSITION     = 0.08       # max 8% in any single stock
MAX_SECTOR_CONC  = 0.30       # max 30% in any one sector
MAX_TURNOVER     = 0.25       # max 25% portfolio turnover per rebalance
LONG_LEVERAGE    = 1.0        # total long exposure = 100% of capital
SHORT_LEVERAGE   = 0.5        # total short exposure = 50% of capital


# ---------------------------------------------
# RISK PARITY WEIGHTS
# ---------------------------------------------

def compute_risk_parity_weights(positions, returns, vol_window=VOL_WINDOW):
    """
    Compute risk parity weights for each rebalance date.

    Method: Inverse Volatility Weighting
        weight_i = (1 / vol_i) / sum(1 / vol_j for all active positions j)

    Why Risk Parity?
        In equal-weight portfolios, high-vol stocks dominate risk.
        One volatile stock can drive 40% of portfolio risk even at 5% weight.
        Risk parity ensures every stock contributes equally to total risk.

    Steps per rebalance date:
        1. Identify active positions (long or short)
        2. Compute 20-day realised volatility for each active stock
        3. Assign weight = 1/vol, separately for longs and shorts
        4. Normalise: long weights sum to +LONG_LEVERAGE
                      short weights sum to -SHORT_LEVERAGE

    Args:
        positions  : pd.DataFrame -- signal positions (+1/0/-1) from signals.py
        returns    : pd.DataFrame -- daily returns for all stocks
        vol_window : int -- rolling window for volatility

    Returns:
        weights    : pd.DataFrame -- portfolio weights (rows=rebalance dates, cols=stocks)
    """
    # Get rebalance dates (every Friday)
    rebalance_dates = positions.resample(REBALANCE_FREQ).last().index
    rebalance_dates = rebalance_dates[rebalance_dates >= positions.index[0]]

    all_weights = []

    for date in rebalance_dates:
        # Get position on this date
        if date not in positions.index:
            # Find nearest previous date
            prev_dates = positions.index[positions.index <= date]
            if len(prev_dates) == 0:
                continue
            date_pos = positions.loc[prev_dates[-1]]
        else:
            date_pos = positions.loc[date]

        # Active long and short positions
        longs  = date_pos[date_pos == 1].index.tolist()
        shorts = date_pos[date_pos == -1].index.tolist()

        if len(longs) == 0 and len(shorts) == 0:
            all_weights.append(pd.Series(0.0, index=positions.columns, name=date))
            continue

        # Compute volatility for each active stock
        # Use data available UP TO this date (no look-ahead)
        hist_ret = returns.loc[:date].tail(vol_window)
        vol      = hist_ret.std().replace(0, np.nan)

        weights  = pd.Series(0.0, index=positions.columns)

        # Long weights: inverse vol, normalised to sum = LONG_LEVERAGE
        if longs:
            long_vol     = vol[longs].fillna(vol[longs].mean())
            inv_vol_long = 1.0 / long_vol
            long_weights = inv_vol_long / inv_vol_long.sum() * LONG_LEVERAGE
            long_weights = long_weights.clip(upper=MAX_POSITION)
            # Re-normalise after clipping
            long_weights = long_weights / long_weights.sum() * LONG_LEVERAGE
            weights[longs] = long_weights

        # Short weights: inverse vol, normalised to sum = -SHORT_LEVERAGE
        if shorts:
            short_vol     = vol[shorts].fillna(vol[shorts].mean())
            inv_vol_short = 1.0 / short_vol
            short_weights = inv_vol_short / inv_vol_short.sum() * SHORT_LEVERAGE
            short_weights = short_weights.clip(upper=MAX_POSITION)
            short_weights = short_weights / short_weights.sum() * SHORT_LEVERAGE
            weights[shorts] = -short_weights   # negative = short

        weights.name = date
        all_weights.append(weights)

    if not all_weights:
        return pd.DataFrame()

    weights_df = pd.DataFrame(all_weights)
    weights_df.index = pd.DatetimeIndex(weights_df.index)

    print(f"[Portfolio] Weights computed for {len(weights_df)} rebalance dates")
    return weights_df


# ---------------------------------------------
# CONSTRAINTS
# ---------------------------------------------

def apply_constraints(weights):
    """
    Enforce portfolio constraints on computed weights.

    Constraints applied:
        1. Max position size: clip any weight to MAX_POSITION
        2. Turnover limit: if rebalance changes > MAX_TURNOVER, scale changes down
        3. Re-normalise after clipping so weights still sum correctly

    Args:
        weights : pd.DataFrame of target weights

    Returns:
        constrained : pd.DataFrame of weights after constraints
        turnover    : pd.Series of turnover at each rebalance
    """
    constrained = weights.copy()
    turnovers   = []

    prev_weights = pd.Series(0.0, index=weights.columns)

    for date, row in constrained.iterrows():
        # Max position constraint
        longs  = row[row > 0]
        shorts = row[row < 0]

        if len(longs) > 0:
            longs  = longs.clip(upper=MAX_POSITION)
            longs  = longs / longs.sum() * LONG_LEVERAGE
            constrained.loc[date, longs.index] = longs

        if len(shorts) > 0:
            shorts = shorts.clip(lower=-MAX_POSITION)
            shorts = shorts / shorts.abs().sum() * SHORT_LEVERAGE
            constrained.loc[date, shorts.index] = -shorts.abs()

        # Turnover calculation
        turnover = (constrained.loc[date] - prev_weights).abs().sum() / 2
        turnovers.append({"date": date, "turnover": turnover})

        # Turnover limit: if too high, blend old and new weights
        if turnover > MAX_TURNOVER and len(prev_weights[prev_weights != 0]) > 0:
            blend = MAX_TURNOVER / turnover
            constrained.loc[date] = (
                blend * constrained.loc[date] + (1 - blend) * prev_weights
            )

        prev_weights = constrained.loc[date].copy()

    turnover_df = pd.DataFrame(turnovers).set_index("date")
    print(f"[Portfolio] Avg turnover per rebalance: {turnover_df['turnover'].mean():.1%}")

    return constrained, turnover_df


# ---------------------------------------------
# FORWARD FILL WEIGHTS
# ---------------------------------------------

def expand_weights_to_daily(weights, date_index):
    """
    Weights are computed weekly (rebalance dates).
    Expand to daily frequency by forward-filling.

    Between rebalances, the portfolio holds its positions.
    We forward fill to get a weight for every trading day.

    Args:
        weights    : pd.DataFrame of weekly weights
        date_index : pd.DatetimeIndex of all trading days

    Returns:
        daily_weights : pd.DataFrame, one row per trading day
    """
    daily = weights.reindex(date_index).ffill()
    daily = daily.fillna(0.0)
    print(f"[Portfolio] Weights expanded to {len(daily)} daily observations")
    return daily


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------

def build_portfolio(positions, returns, position_scale=None,
                    confidence=None, output_dir="outputs/"):
    """
    Master function: run full portfolio construction pipeline.

    Args:
        positions      : pd.DataFrame -- signal positions from signals.py
        returns        : pd.DataFrame -- daily returns from data_loader.py
        position_scale : pd.Series -- regime scaling factors from regime_detection.py
                         If None, no regime scaling is applied
        confidence     : pd.DataFrame -- confidence scores (0-85) from signals.py
                         Used to scale individual position weights
        output_dir     : str -- where to save outputs

    Returns:
        daily_weights  : pd.DataFrame -- daily portfolio weights
        turnover       : pd.Series -- turnover per rebalance
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PORTFOLIO CONSTRUCTION MODULE")
    print("Method: Risk Parity (Inverse Volatility) x Confidence Scaling")
    print("=" * 60)

    # Step 1: Compute risk parity weights
    weights = compute_risk_parity_weights(positions, returns)

    # Step 2: Apply constraints
    weights, turnover = apply_constraints(weights)

    # Step 3: Apply confidence-based scaling per stock
    # position_size = base_weight x (confidence / 85)
    # Confidence ranges 0-85 from price signals; we normalise to 0.5-1.0 range
    # to avoid zeroing out positions entirely on low-confidence days
    if confidence is not None:
        print("[Portfolio] Applying confidence-based position scaling...")
        rebal_dates = weights.index
        for date in rebal_dates:
            prev_dates = confidence.index[confidence.index <= date]
            if len(prev_dates) == 0:
                continue
            conf_row = confidence.loc[prev_dates[-1]]
            # Normalise: confidence 0 -> 0.5x, confidence 85 -> 1.0x
            conf_scale = 0.5 + 0.5 * (conf_row.reindex(weights.columns).fillna(42) / 85)
            weights.loc[date] = weights.loc[date] * conf_scale
        print(f"[Portfolio] Confidence scaling applied to {len(rebal_dates)} rebalance dates")

    # Step 4: Apply regime scaling (if provided)
    if position_scale is not None:
        print("[Portfolio] Applying regime-based position scaling (HMM x FII)...")
        scale_aligned = position_scale.reindex(weights.index, method="ffill").fillna(1.0)
        weights = weights.multiply(scale_aligned, axis=0)
        print(f"[Portfolio] Average combined regime scale: {scale_aligned.mean():.3f}")

    # Step 5: Expand to daily frequency
    daily_weights = expand_weights_to_daily(weights, returns.index)

    # Step 5: Save outputs
    weights.to_csv(f"{output_dir}weights.csv")
    daily_weights.to_csv(f"{output_dir}weights_daily.csv")
    turnover.to_csv(f"{output_dir}portfolio_turnover.csv")

    # Summary
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    avg_longs  = (daily_weights > 0).sum(axis=1).mean()
    avg_shorts = (daily_weights < 0).sum(axis=1).mean()
    avg_gross  = daily_weights.abs().sum(axis=1).mean()
    print(f"Avg long positions : {avg_longs:.1f} stocks")
    print(f"Avg short positions: {avg_shorts:.1f} stocks")
    print(f"Avg gross exposure : {avg_gross:.2f}x")
    print(f"Max single weight  : {daily_weights.abs().max().max():.2%}")
    print("=" * 60)

    return daily_weights, turnover


if __name__ == "__main__":
    from data_loader import load_data
    from signals import generate_signals

    print("Loading data...")
    close, _, _, _, volume, returns, nifty, vix = load_data()

    print("Generating signals...")
    positions, signals, factors = generate_signals(close, volume)

    print("Building portfolio...")
    weights, turnover = build_portfolio(positions, returns)

    print("\nSample weights (last rebalance):")
    latest = weights[weights != 0].dropna(how="all").tail(1)
    active = latest.loc[:, (latest != 0).any()].T
    print(active.sort_values(by=active.columns[0], ascending=False))
