"""
signals.py -- Alpha & Signal Generation Module
==============================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

Strategy: Multi-Factor Cross-Sectional Signal
Factors  : Momentum + RSI Mean-Reversion + Volume Confirmation + Volatility Penalty

How it works:
    1. For every stock, compute 4 factor scores on every date
    2. Z-score each factor (normalise to same scale)
    3. Combine into one composite signal with weights
    4. Rank all stocks by composite signal
    5. Top 20% -> Long (+1), Bottom 20% -> Short (-1), Rest -> No position (0)

Output:
    signals.csv  -- signal scores and directions for every stock x every date
    alpha_decay.csv -- how quickly each signal loses predictive power
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

# Factor weights (must sum to 1.0)
FACTOR_WEIGHTS = {
    "momentum":   0.40,   # 12-month minus 1-month return
    "rsi":        0.25,   # mean-reversion via RSI
    "volume":     0.20,   # volume confirmation
    "volatility": 0.15,   # low-volatility preference
}

# Portfolio construction params
LONG_PERCENTILE  = 0.80   # top 20% of ranked stocks -> long
SHORT_PERCENTILE = 0.20   # bottom 20% of ranked stocks -> short

# Lookback windows (in trading days)
MOMENTUM_LONG  = 252      # 12 months
MOMENTUM_SHORT = 21       # 1 month (subtracted to remove reversal)
RSI_WINDOW     = 14       # standard RSI period
VOLUME_WINDOW  = 20       # rolling average volume window
VOLATILITY_WINDOW = 20    # rolling std of daily returns


# ---------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------

def zscore_cross_section(df):
    """
    Z-score each row (date) across all stocks.
    This normalises factors so they are on the same scale.

    Formula: z = (x - mean) / std
    After this, each factor has mean=0 and std=1 across stocks on any given day.

    Args:
        df: DataFrame where rows=dates, columns=stocks

    Returns:
        DataFrame of same shape with z-scored values
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def compute_rsi(prices, window=14):
    """
    Compute RSI (Relative Strength Index) for a single price series.

    RSI measures speed and magnitude of recent price moves.
    RSI > 70 -> overbought (expect reversal downward)
    RSI < 30 -> oversold  (expect reversal upward)

    Formula:
        RS  = avg_gain / avg_loss over `window` periods
        RSI = 100 - (100 / (1 + RS))

    Args:
        prices: pd.Series of closing prices
        window: lookback period (default 14 days)

    Returns:
        pd.Series of RSI values (0 to 100)
    """
    delta  = prices.diff()
    gain   = delta.clip(lower=0)          # keep only positive moves
    loss   = (-delta).clip(lower=0)       # keep only negative moves (as positive number)

    # Exponential moving average of gains and losses
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)    # avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ---------------------------------------------
# FACTOR 1: MOMENTUM
# ---------------------------------------------

def compute_momentum(close_prices):
    """
    Cross-sectional momentum factor.

    Formula:
        momentum = 12-month return - 1-month return

    Why subtract 1 month?
        The last month often shows short-term reversal (stocks
        that ran too far, too fast, tend to pull back slightly).
        Subtracting removes this noise and keeps the medium-term trend.

    Academic basis:
        Jegadeesh & Titman (1993) showed 3-12 month momentum
        predicts future 3-12 month returns. One of the most
        replicated findings in finance.

    Args:
        close_prices: DataFrame (rows=dates, columns=stock tickers)

    Returns:
        DataFrame of momentum scores, same shape
    """
    # 12-month return: how much has each stock moved over last 252 days?
    return_long  = close_prices.pct_change(MOMENTUM_LONG)

    # 1-month return: recent short-term move (we will subtract this)
    return_short = close_prices.pct_change(MOMENTUM_SHORT)

    # Final momentum: 12-month trend, minus short-term reversal noise
    momentum = return_long - return_short

    return momentum


# ---------------------------------------------
# FACTOR 2: RSI MEAN-REVERSION
# ---------------------------------------------

def compute_rsi_factor(close_prices):
    """
    Mean-reversion factor based on RSI.

    We flip RSI so that:
        Low RSI  (oversold)  -> positive score  (expect bounce up)
        High RSI (overbought) -> negative score  (expect pullback)

    Formula:
        rsi_score = 50 - RSI

    Why this works:
        In the short term, extreme moves tend to revert.
        A stock down 15% in 2 weeks is likely oversold --
        some buyers will step in, pushing it back up.

    Args:
        close_prices: DataFrame (rows=dates, columns=stock tickers)

    Returns:
        DataFrame of RSI factor scores
    """
    rsi_scores = close_prices.apply(
        lambda col: compute_rsi(col, RSI_WINDOW)
    )

    # Flip: low RSI = positive signal, high RSI = negative signal
    rsi_factor = 50 - rsi_scores

    return rsi_factor


# ---------------------------------------------
# FACTOR 3: VOLUME CONFIRMATION
# ---------------------------------------------

def compute_volume_factor(close_prices, volume):
    """
    Volume confirmation factor.

    Idea: A price move is more trustworthy when backed by high volume.
          Volume is the "fuel" of a trend.

    Formula:
        volume_ratio = (today_volume - avg_20day_volume) / avg_20day_volume
        direction    = sign of 5-day return (up or down)
        factor       = volume_ratio x direction

    So:
        Stock rising  + high volume -> positive score  (strong uptrend)
        Stock falling + high volume -> negative score  (strong downtrend)
        Stock moving  + low volume  -> near zero score (weak move, ignore)

    Args:
        close_prices: DataFrame (rows=dates, columns=stock tickers)
        volume:       DataFrame (rows=dates, columns=stock tickers)

    Returns:
        DataFrame of volume factor scores
    """
    # Rolling 20-day average volume for each stock
    avg_volume = volume.rolling(VOLUME_WINDOW).mean()

    # Volume ratio: how much above/below average is today's volume?
    volume_ratio = (volume - avg_volume) / avg_volume.replace(0, np.nan)

    # Price direction over last 5 days
    price_direction = close_prices.pct_change(5).apply(np.sign)

    # Combine: strong volume in direction of trend = strong signal
    volume_factor = volume_ratio * price_direction

    return volume_factor


# ---------------------------------------------
# FACTOR 4: VOLATILITY PENALTY
# ---------------------------------------------

def compute_volatility_factor(close_prices):
    """
    Volatility penalty factor (low-volatility preference).

    Idea: All else equal, prefer lower-volatility stocks.
          They produce smoother returns, lower drawdowns,
          and lower transaction costs (less slippage).

    Formula:
        daily_returns = pct_change(close)
        volatility    = rolling_std(daily_returns, 20) x sqrt(252)  # annualised
        factor        = -1 x volatility  (lower vol = higher score)

    Academic basis:
        The Low Volatility Anomaly (Blitz & van Vliet, 2007):
        lower-volatility stocks historically outperform
        higher-volatility stocks on a risk-adjusted basis.

    Args:
        close_prices: DataFrame (rows=dates, columns=stock tickers)

    Returns:
        DataFrame of volatility factor scores (negative = more volatile = penalised)
    """
    daily_returns = close_prices.pct_change()

    # Annualised volatility: std x sqrt(252 trading days per year)
    volatility = daily_returns.rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)

    # Negate: low volatility should score HIGH (we prefer low-vol stocks)
    volatility_factor = -1 * volatility

    return volatility_factor


# ---------------------------------------------
# COMPOSITE SIGNAL
# ---------------------------------------------

def compute_composite_signal(close_prices, volume):
    """
    Combine all 4 factors into one composite signal score.

    Steps:
        1. Compute each raw factor
        2. Z-score each factor cross-sectionally (per date, across stocks)
           -> brings all factors to same scale (mean=0, std=1)
        3. Weighted average of z-scored factors
        4. Final composite signal: higher = better stock to be long

    Args:
        close_prices: DataFrame (rows=dates, columns=stock tickers)
        volume:       DataFrame (rows=dates, columns=stock tickers)

    Returns:
        composite: DataFrame of final composite signal scores
        factors:   dict of individual factor DataFrames (for analysis)
    """
    print("[Signal] Computing Factor 1: Momentum...")
    momentum_raw = compute_momentum(close_prices)

    print("[Signal] Computing Factor 2: RSI Mean-Reversion...")
    rsi_raw = compute_rsi_factor(close_prices)

    print("[Signal] Computing Factor 3: Volume Confirmation...")
    volume_raw = compute_volume_factor(close_prices, volume)

    print("[Signal] Computing Factor 4: Volatility Penalty...")
    volatility_raw = compute_volatility_factor(close_prices)

    print("[Signal] Z-scoring all factors cross-sectionally...")
    momentum_z   = zscore_cross_section(momentum_raw)
    rsi_z        = zscore_cross_section(rsi_raw)
    volume_z     = zscore_cross_section(volume_raw)
    volatility_z = zscore_cross_section(volatility_raw)

    print("[Signal] Computing weighted composite signal...")
    composite = (
        FACTOR_WEIGHTS["momentum"]   * momentum_z   +
        FACTOR_WEIGHTS["rsi"]        * rsi_z         +
        FACTOR_WEIGHTS["volume"]     * volume_z      +
        FACTOR_WEIGHTS["volatility"] * volatility_z
    )

    factors = {
        "momentum_raw":   momentum_raw,
        "rsi_raw":        rsi_raw,
        "volume_raw":     volume_raw,
        "volatility_raw": volatility_raw,
        "momentum_z":     momentum_z,
        "rsi_z":          rsi_z,
        "volume_z":       volume_z,
        "volatility_z":   volatility_z,
    }

    return composite, factors


# ---------------------------------------------
# FILTER: 52-WEEK HIGH / LOW DISTANCE
# ---------------------------------------------

def compute_52week_filter(close_prices, high_prices=None, low_prices=None):
    """
    52-week high/low distance filter.

    Strategy logic from the guide:
        For LONGS : only trade if within 15% of 52-week high
                    (breakout momentum works near highs, not bottoms)
        For SHORTS: only trade if within 15% of 52-week low
                    (falling knives continue to fall)

    Formula:
        dist_from_high = (price - 52wk_high) / 52wk_high   [negative for longs]
        dist_from_low  = (price - 52wk_low)  / 52wk_low    [positive for shorts]

    Returns:
        long_eligible  : DataFrame -- True if stock is eligible as a LONG candidate
        short_eligible : DataFrame -- True if stock is eligible as a SHORT candidate
    """
    WINDOW_52W = 252   # 252 trading days ~ 1 year

    if high_prices is not None:
        high_52w = high_prices.rolling(WINDOW_52W, min_periods=100).max()
    else:
        high_52w = close_prices.rolling(WINDOW_52W, min_periods=100).max()

    if low_prices is not None:
        low_52w = low_prices.rolling(WINDOW_52W, min_periods=100).min()
    else:
        low_52w = close_prices.rolling(WINDOW_52W, min_periods=100).min()

    dist_from_high = (close_prices - high_52w) / high_52w.replace(0, np.nan)
    dist_from_low  = (close_prices - low_52w)  / low_52w.replace(0, np.nan)

    # Long eligible: within 15% of 52-week high (dist_from_high >= -0.15)
    long_eligible  = dist_from_high >= -0.15

    # Short eligible: within 15% of 52-week low (dist_from_low <= 0.15)
    short_eligible = dist_from_low  <= 0.15

    return long_eligible, short_eligible, dist_from_high, dist_from_low


# ---------------------------------------------
# CONFIDENCE SCORE
# ---------------------------------------------

def compute_confidence_score(composite_signal, momentum_z, rsi_raw,
                              volume_z, close_prices):
    """
    Compute a confidence score (0-100) for each signal on each date.

    From the strategy document:
        +30  Momentum rank in top/bottom 10%
        +20  All 3 momentum components agree (12m, 3m, 1m all same direction)
        +15  Volume > 1.5x average
        +10  RSI in favorable zone (40-60, room to run)
        +10  Near 52-week high/low
        +15  Reserved for FII flow alignment (applied in portfolio.py)

    Higher confidence -> larger position size.

    Args:
        composite_signal : DataFrame of composite signal scores
        momentum_z       : DataFrame of z-scored momentum
        rsi_raw          : DataFrame of raw RSI values
        volume_z         : DataFrame of z-scored volume factor
        close_prices     : DataFrame of close prices

    Returns:
        confidence : DataFrame of confidence scores (0 to 85 from price data)
    """
    confidence = pd.DataFrame(0.0, index=composite_signal.index,
                              columns=composite_signal.columns)

    # +30: momentum rank in top/bottom 10%
    rank_pct = composite_signal.rank(axis=1, pct=True)
    top10    = (rank_pct >= 0.90) | (rank_pct <= 0.10)
    confidence[top10] += 30

    # +20: all 3 momentum components align
    # We use the composite z-score direction as a proxy
    mom_positive = momentum_z > 0.5
    mom_negative = momentum_z < -0.5
    long_pos     = composite_signal >= 0
    short_pos    = composite_signal < 0
    all_align    = (mom_positive & long_pos) | (mom_negative & short_pos)
    confidence[all_align] += 20

    # +15: Volume > 1.5x average (strong volume = volume_z > 0.5)
    high_volume = volume_z > 0.5
    confidence[high_volume] += 15

    # +10: RSI in favorable zone (40-60) -- room to run, not overbought/oversold
    rsi_favorable = (rsi_raw >= 40) & (rsi_raw <= 60)
    confidence[rsi_favorable] += 10

    # +10: Near 52-week extreme (handled via 52wk filter above, we approximate here)
    # Stocks with high absolute composite signal are likely near extremes
    near_extreme = composite_signal.abs() > composite_signal.abs().quantile(0.75, axis=1).values.reshape(-1, 1)
    confidence[near_extreme] += 10

    return confidence.clip(0, 85)   # max 85 from price data; +15 from FII in portfolio


# ---------------------------------------------
# POSITION ASSIGNMENT
# ---------------------------------------------

def assign_positions(composite_signal):
    """
    Convert continuous signal scores into discrete position directions.

    Method: Cross-sectional percentile ranking
        Top 20% of stocks by signal score  -> Long  (+1)
        Bottom 20% of stocks by signal score -> Short (-1)
        Middle 60%                           -> No position (0)

    This is called a "cross-sectional" approach because we rank
    stocks against each other, not against some absolute threshold.
    The portfolio is always market-neutral: equal dollar value long and short.

    Args:
        composite_signal: DataFrame (rows=dates, columns=stock tickers)

    Returns:
        positions: DataFrame with values in {-1, 0, +1}
    """
    def rank_row(row):
        """Rank one date's signals and assign positions."""
        valid = row.dropna()
        if len(valid) < 10:              # need enough stocks to rank
            return pd.Series(0, index=row.index)

        upper = valid.quantile(LONG_PERCENTILE)
        lower = valid.quantile(SHORT_PERCENTILE)

        pos = pd.Series(0, index=row.index)
        pos[row >= upper] = 1            # long: strongest momentum stocks
        pos[row <= lower] = -1           # short: weakest momentum stocks
        return pos

    print("[Signal] Assigning positions (long/short/neutral)...")
    positions = composite_signal.apply(rank_row, axis=1)

    return positions


# ---------------------------------------------
# ALPHA DECAY ANALYSIS
# ---------------------------------------------

def compute_alpha_decay(composite_signal, close_prices, horizons=[1, 5, 10, 21]):
    """
    Measure how quickly the signal loses predictive power.

    Method: Information Coefficient (IC)
        IC = rank correlation between signal on date T
             and actual return over next N days

        IC > 0  -> signal correctly predicts direction
        IC = 0  -> signal is random (no edge)
        IC < 0  -> signal predicts wrong direction

    We compute IC for horizons of 1, 5, 10, 21 days.
    A good signal has IC > 0.05 and decays slowly.

    Args:
        composite_signal: DataFrame of signal scores
        close_prices:     DataFrame of close prices
        horizons:         list of forward return horizons in days

    Returns:
        decay_df: DataFrame showing IC at each horizon
    """
    from scipy.stats import spearmanr

    results = {}

    for h in horizons:
        # Forward return: what actually happened next h days
        forward_return = close_prices.pct_change(h).shift(-h)

        ics = []
        for date in composite_signal.index:
            sig = composite_signal.loc[date].dropna()
            fwd = forward_return.loc[date].dropna()

            # Only use stocks present in both
            common = sig.index.intersection(fwd.index)
            if len(common) < 10:
                continue

            ic, _ = spearmanr(sig[common], fwd[common])
            ics.append(ic)

        results[f"IC_{h}d"] = {
            "mean_IC":  np.mean(ics),
            "std_IC":   np.std(ics),
            "IR":       np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
            "positive_pct": np.mean([ic > 0 for ic in ics])
        }

    decay_df = pd.DataFrame(results).T
    return decay_df


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------

def generate_signals(close_prices, volume, high_prices=None, low_prices=None,
                     output_dir="outputs/"):
    """
    Master function: run entire signal generation pipeline.

    Args:
        close_prices: DataFrame (rows=dates, columns=stock tickers)
                      Only closing prices. Already cleaned (no NaN gaps).
        volume:       DataFrame (rows=dates, columns=stock tickers)
                      Daily trading volume for each stock.
        high_prices:  DataFrame -- daily highs (optional, used for 52wk filter)
        low_prices:   DataFrame -- daily lows  (optional, used for 52wk filter)
        output_dir:   folder to save output CSV files

    Returns:
        positions:        DataFrame with {-1, 0, +1} for each stock x date
        composite_signal: DataFrame with raw signal scores
        factors:          dict of individual factor DataFrames (includes confidence)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SIGNAL GENERATION MODULE")
    print("Strategy: Multi-Factor Cross-Sectional (Nifty 500)")
    print("=" * 60)

    # Step 1: Compute composite signal (momentum + RSI + volume + volatility)
    composite_signal, factors = compute_composite_signal(close_prices, volume)

    # Step 2: Compute 52-week high/low filters
    print("[Signal] Computing 52-week high/low eligibility filters...")
    long_eligible, short_eligible, dist_high, dist_low = compute_52week_filter(
        close_prices, high_prices, low_prices
    )
    factors["long_eligible_52w"]  = long_eligible
    factors["short_eligible_52w"] = short_eligible
    factors["dist_from_52w_high"] = dist_high
    factors["dist_from_52w_low"]  = dist_low

    # Step 3: Assign positions (cross-sectional ranking)
    positions = assign_positions(composite_signal)

    # Step 4: Apply 52-week filter -- zero out ineligible positions
    print("[Signal] Applying 52-week high/low filter to positions...")
    positions_filtered = positions.copy()

    # Longs must be within 15% of 52-week high
    long_mask = (positions == 1) & (~long_eligible)
    positions_filtered[long_mask] = 0

    # Shorts must be within 15% of 52-week low
    short_mask = (positions == -1) & (~short_eligible)
    positions_filtered[short_mask] = 0

    pct_filtered = 1 - (positions_filtered.abs().sum().sum() /
                        max(positions.abs().sum().sum(), 1))
    print(f"[Signal] 52-week filter removed {pct_filtered:.1%} of raw positions")

    # Step 5: Compute confidence scores
    print("[Signal] Computing confidence scores...")
    rsi_raw    = factors.get("rsi_raw", pd.DataFrame(50, index=close_prices.index,
                                                       columns=close_prices.columns))
    volume_z   = factors.get("volume_z", pd.DataFrame(0, index=close_prices.index,
                                                        columns=close_prices.columns))
    momentum_z = factors.get("momentum_z", composite_signal)

    confidence = compute_confidence_score(
        composite_signal, momentum_z, rsi_raw, volume_z, close_prices
    )
    factors["confidence"] = confidence

    # Step 6: Alpha decay analysis
    print("[Signal] Running alpha decay analysis...")
    decay = compute_alpha_decay(composite_signal, close_prices)

    # Step 7: Save outputs
    print("[Signal] Saving outputs...")
    composite_signal.to_csv(f"{output_dir}signals.csv")
    positions_filtered.to_csv(f"{output_dir}positions.csv")
    positions.to_csv(f"{output_dir}positions_raw.csv")          # pre-filter
    confidence.to_csv(f"{output_dir}confidence.csv")
    long_eligible.to_csv(f"{output_dir}long_eligible.csv")
    short_eligible.to_csv(f"{output_dir}short_eligible.csv")
    decay.to_csv(f"{output_dir}alpha_decay.csv")

    for name, df in factors.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(f"{output_dir}factor_{name}.csv")

    # Step 8: Print summary
    print("\n" + "=" * 60)
    print("SIGNAL SUMMARY")
    print("=" * 60)
    total_dates   = len(positions_filtered)
    avg_long_day  = (positions_filtered == 1).sum(axis=1).mean()
    avg_short_day = (positions_filtered == -1).sum(axis=1).mean()
    avg_conf      = confidence[positions_filtered != 0].stack().mean()

    print(f"Date range        : {positions_filtered.index[0]} -> {positions_filtered.index[-1]}")
    print(f"Total dates       : {total_dates}")
    print(f"Avg longs/day     : {avg_long_day:.1f} stocks")
    print(f"Avg shorts/day    : {avg_short_day:.1f} stocks")
    print(f"Avg confidence    : {avg_conf:.1f} / 85")
    print(f"\nAlpha Decay Analysis:")
    print(decay.to_string())
    print("=" * 60)

    return positions_filtered, composite_signal, factors


# ---------------------------------------------
# RUN STANDALONE (for testing)
# ---------------------------------------------

if __name__ == "__main__":
    # Run via quick_test.py for end-to-end validation with real parquet data:
    #   python quick_test.py --stocks 50
    print("Run 'python quick_test.py --stocks 50' for full pipeline validation.")

    print("\nFiles saved to outputs/test_signals/")
