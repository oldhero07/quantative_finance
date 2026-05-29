"""
options_signal.py -- Options Market Signal Module
=================================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

What this module does:
    Uses the OPTIONS market to generate signals for EQUITY trading.

    Options traders are often more informed than equity traders.
    They trade on information before it shows up in stock prices.
    By reading the options market, we get an early warning system.

    Two signals extracted:
        1. Put-Call Ratio (PCR)   -- market sentiment indicator
        2. IV Skew                -- tail risk / crash risk indicator

Why this is original:
    Most student teams only use equity (price/volume) data.
    We use DERIVATIVES data to predict EQUITY direction.
    This is called "cross-market signal extraction."
    It directly uses your options knowledge from the playlist.

Data source:
    NSE India publishes daily option chain data for free.
    URL: https://www.nseindia.com/option-chain
    We also provide a simulation mode for testing without real data.

Output:
    pcr_signal.csv   -- daily PCR values and signal direction
    iv_skew.csv      -- daily IV skew values
    options_signal.csv -- combined options-derived signal
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

# PCR thresholds (contrarian interpretation)
PCR_EXTREME_BEARISH = 1.30   # PCR above this -> market very bearish -> contrarian BUY
PCR_BEARISH         = 1.00   # PCR above this -> mild bearishness -> slight buy
PCR_BULLISH         = 0.75   # PCR below this -> mild bullishness -> slight sell
PCR_EXTREME_BULLISH = 0.60   # PCR below this -> market very bullish -> contrarian SELL

# IV Skew thresholds
SKEW_HIGH_THRESHOLD = 0.08   # skew > 8% -> market fears crash -> reduce longs
SKEW_LOW_THRESHOLD  = 0.03   # skew < 3% -> complacency -> normal positioning

# Rolling window for smoothing
PCR_SMOOTH_WINDOW = 5        # 5-day moving average of PCR (reduce daily noise)


# ---------------------------------------------
# SIGNAL 1: PUT-CALL RATIO (PCR)
# ---------------------------------------------

def compute_pcr_signal(put_volume, call_volume):
    """
    Put-Call Ratio: measures relative demand for puts vs calls.

    Formula:
        PCR = Total Put Volume / Total Call Volume

    Interpretation (CONTRARIAN):
        High PCR (>1.2): Everyone is buying puts (bearish insurance).
                         When everyone is bearish, it often means the
                         market has already priced in the bad news.
                         -> Contrarian BUY signal.

        Low PCR (<0.7):  Everyone is buying calls (greedy, no fear).
                         When everyone is bullish, the upside is priced in.
                         -> Contrarian SELL / caution signal.

    Why contrarian?
        Options are used for hedging. When retail traders are scared,
        they buy lots of puts. Smart money is often on the other side.
        Extreme fear = opportunity. Extreme greed = danger.

    This connects to your playlist knowledge:
        - You understand call vs put from Lecture 1-4
        - You understand why high put buying signals fear
        - You understand why this is a contrarian, not a direct signal

    Args:
        put_volume  : pd.Series -- daily total put volume (Nifty options)
        call_volume : pd.Series -- daily total call volume (Nifty options)

    Returns:
        pcr          : pd.Series -- raw PCR values
        pcr_smoothed : pd.Series -- 5-day smoothed PCR
        pcr_signal   : pd.Series -- signal score (-1 to +1)
    """
    # Raw PCR
    pcr = put_volume / call_volume.replace(0, np.nan)

    # Smooth to reduce daily noise
    pcr_smoothed = pcr.rolling(PCR_SMOOTH_WINDOW).mean()

    # Convert PCR to signal score
    def pcr_to_signal(pcr_val):
        """
        Map PCR value to signal score.

        Score +1  = strong buy (extreme fear, contrarian opportunity)
        Score -1  = strong sell (extreme complacency, danger)
        Score 0   = neutral
        """
        if pd.isna(pcr_val):
            return 0.0
        elif pcr_val >= PCR_EXTREME_BEARISH:
            return +1.0    # extreme fear = strong contrarian buy
        elif pcr_val >= PCR_BEARISH:
            return +0.5    # mild fear = mild buy signal
        elif pcr_val <= PCR_EXTREME_BULLISH:
            return -1.0    # extreme greed = strong contrarian sell
        elif pcr_val <= PCR_BULLISH:
            return -0.5    # mild greed = mild sell signal
        else:
            return 0.0     # neutral zone

    pcr_signal = pcr_smoothed.apply(pcr_to_signal)

    return pcr, pcr_smoothed, pcr_signal


# ---------------------------------------------
# SIGNAL 2: IMPLIED VOLATILITY SKEW
# ---------------------------------------------

def compute_iv_skew_signal(iv_otm_puts, iv_atm_calls):
    """
    IV Skew: measures whether options market is pricing in crash risk.

    Formula:
        IV_skew = IV of OTM puts - IV of ATM calls

    Why does this matter?
        OTM puts protect against crash scenarios.
        When traders fear a crash, they buy OTM puts aggressively,
        driving up their implied volatility.

        High skew = options market is pricing in tail risk (crash risk)
                    This is a WARNING signal for long positions.

        Low skew  = market is calm, no crash fear
                    This is a GREEN LIGHT for momentum strategies.

    Connection to your playlist:
        - Vega (Lecture 10): IV changes affect option prices
        - Moneyness (Lecture 6): OTM vs ATM distinction
        - Greeks: IV skew is a direct application of Vega across strikes

    Academic basis:
        Yan (2011) showed that steeper IV skew predicts lower
        future stock returns. The options market is "smarter"
        about tail risks than the equity market.

    Args:
        iv_otm_puts  : pd.Series -- implied vol of ~5% OTM put options
        iv_atm_calls : pd.Series -- implied vol of ATM call options

    Returns:
        iv_skew        : pd.Series -- raw skew values
        skew_signal    : pd.Series -- signal score (negative = reduce risk)
        risk_flag      : pd.Series -- True when crash risk is elevated
    """
    iv_skew = iv_otm_puts - iv_atm_calls

    def skew_to_signal(skew_val):
        """
        High skew = options market fears crash = reduce position sizes.
        Low skew  = calm market = normal position sizes.
        """
        if pd.isna(skew_val):
            return 0.0
        elif skew_val > SKEW_HIGH_THRESHOLD:
            return -0.8    # high crash fear = reduce longs significantly
        elif skew_val > SKEW_LOW_THRESHOLD:
            return -0.3    # mild concern = slight reduction
        else:
            return 0.0     # calm = no adjustment needed

    skew_signal = iv_skew.apply(skew_to_signal)
    risk_flag   = iv_skew > SKEW_HIGH_THRESHOLD

    return iv_skew, skew_signal, risk_flag


# ---------------------------------------------
# SIMULATION MODE (when real options data unavailable)
# ---------------------------------------------

def simulate_options_data(nifty_close, seed=42):
    """
    Simulate realistic options market data from Nifty price data.

    Used for backtesting when historical options chain data is not available.
    The simulation captures the key stylised facts of options markets:
        - PCR tends to spike during market selloffs
        - IV skew rises when market drops sharply
        - Both are mean-reverting around long-term averages

    This is a principled simulation -- not random noise. Each simulated
    variable has a real relationship to price movements, which makes
    the backtest meaningful even without real data.

    How simulation works:
        PCR base    = 0.9 (long-run average for Nifty options)
        PCR shock   = increases when market drops (fear goes up)
        IV skew     = rises with realised volatility spikes

    Args:
        nifty_close : pd.Series of Nifty 50 closes
        seed        : random seed for reproducibility

    Returns:
        put_volume  : pd.Series (simulated)
        call_volume : pd.Series (simulated)
        iv_otm_puts : pd.Series (simulated IV of OTM puts)
        iv_atm_calls: pd.Series (simulated IV of ATM calls)
    """
    np.random.seed(seed)
    n = len(nifty_close)

    ret = nifty_close.pct_change().fillna(0)
    vol = ret.rolling(20).std().fillna(ret.std())

    # PCR simulation
    # Base PCR of 0.9, spikes when market falls, mean-reverting
    pcr_noise  = np.random.randn(n) * 0.08
    pcr_stress = -5 * ret.values               # PCR rises when returns negative
    pcr_base   = 0.90
    pcr_raw    = pcr_base + pcr_stress + pcr_noise
    pcr_raw    = np.clip(pcr_raw, 0.4, 2.0)   # realistic bounds

    # Convert PCR to put/call volumes (with realistic total volume)
    total_volume = 1_000_000 + np.random.randint(0, 500_000, n)
    call_volume  = (total_volume / (1 + pcr_raw)).astype(int)
    put_volume   = total_volume - call_volume

    # IV simulation
    # ATM call IV ~ realised vol + small premium
    iv_atm    = vol.values * np.sqrt(252) + 0.02 + np.random.randn(n) * 0.01
    iv_atm    = np.clip(iv_atm, 0.05, 0.80)

    # OTM put IV = ATM IV + skew
    # Skew rises with stress (high vol, negative returns)
    skew_base = 0.04
    skew_stress = 2 * vol.values * np.sqrt(252) * (-ret.values).clip(min=0)
    skew_noise  = np.random.randn(n) * 0.005
    iv_skew_sim = skew_base + skew_stress + skew_noise
    iv_otm      = iv_atm + iv_skew_sim.clip(min=0)

    # Wrap as Series with same index
    idx = nifty_close.index
    return (
        pd.Series(put_volume,  index=idx, name="put_volume"),
        pd.Series(call_volume, index=idx, name="call_volume"),
        pd.Series(iv_otm,      index=idx, name="iv_otm_puts"),
        pd.Series(iv_atm,      index=idx, name="iv_atm_calls"),
    )


# ---------------------------------------------
# COMBINED OPTIONS SIGNAL
# ---------------------------------------------

def compute_options_signal(put_volume, call_volume,
                           iv_otm_puts, iv_atm_calls):
    """
    Combine PCR signal and IV skew signal into one options-derived signal.

    The combined signal is used in two ways:
        1. As a CONFIRMATION filter for equity signals
           (only take long positions when PCR says market isn't too greedy)
        2. As a RISK OVERLAY
           (reduce position sizes when IV skew signals crash risk)

    Combined signal formula:
        options_signal = 0.6 x pcr_signal + 0.4 x skew_signal

    PCR gets higher weight because it's more reliable as a directional signal.
    IV skew is primarily a risk management tool.

    Args:
        put_volume   : pd.Series
        call_volume  : pd.Series
        iv_otm_puts  : pd.Series
        iv_atm_calls : pd.Series

    Returns:
        options_df   : pd.DataFrame with all computed signals
        combined     : pd.Series -- final combined options signal (-1 to +1)
    """
    print("[Options] Computing Put-Call Ratio signal...")
    pcr, pcr_smoothed, pcr_signal = compute_pcr_signal(put_volume, call_volume)

    print("[Options] Computing IV Skew signal...")
    iv_skew, skew_signal, risk_flag = compute_iv_skew_signal(iv_otm_puts, iv_atm_calls)

    # Combine
    combined = 0.6 * pcr_signal + 0.4 * skew_signal

    options_df = pd.DataFrame({
        "put_volume":    put_volume,
        "call_volume":   call_volume,
        "pcr_raw":       pcr,
        "pcr_smoothed":  pcr_smoothed,
        "pcr_signal":    pcr_signal,
        "iv_otm_puts":   iv_otm_puts,
        "iv_atm_calls":  iv_atm_calls,
        "iv_skew":       iv_skew,
        "skew_signal":   skew_signal,
        "crash_risk_flag": risk_flag.astype(int),
        "combined_signal": combined,
    })

    return options_df, combined


# ---------------------------------------------
# HOW THIS CONNECTS TO signals.py
# ---------------------------------------------

def apply_options_filter(equity_positions, options_signal,
                         iv_skew, risk_flag):
    """
    Use options signals to filter and scale equity positions.

    Three effects:
        1. CONFIRMATION: Only take strong positions when options market agrees
        2. RISK REDUCTION: Cut position sizes when crash risk flag is on
        3. REVERSAL FILTER: Don't go long when PCR says extreme greed

    This is the KEY CONNECTION between options module and the main strategy.
    The options signal does NOT replace the equity signal -- it modulates it.

    Think of it as:
        equity signal  = "which stocks to trade"
        options signal = "how aggressively to trade them right now"

    Args:
        equity_positions : pd.DataFrame -- positions (+1/0/-1) from signals.py
        options_signal   : pd.Series -- combined options signal
        iv_skew          : pd.Series -- IV skew values
        risk_flag        : pd.Series -- True when crash risk elevated

    Returns:
        filtered_positions : pd.DataFrame -- adjusted positions
        adjustment_log     : pd.DataFrame -- record of every adjustment made
    """
    filtered = equity_positions.copy()
    log_rows  = []

    for date in filtered.index:
        if date not in options_signal.index:
            continue

        opt_sig   = options_signal.loc[date]
        crash_risk = risk_flag.loc[date] if date in risk_flag.index else False

        # Rule 1: Crash risk -> cut all long positions by 50%
        if crash_risk:
            long_mask = filtered.loc[date] > 0
            filtered.loc[date, long_mask] *= 0.50
            log_rows.append({"date": date, "action": "CRASH_RISK_CUT",
                             "factor": 0.50, "reason": f"IV skew elevated"})

        # Rule 2: Extreme greed (PCR very low) -> don't add new longs
        elif opt_sig < -0.8:
            long_mask = filtered.loc[date] > 0
            filtered.loc[date, long_mask] *= 0.30
            log_rows.append({"date": date, "action": "EXTREME_GREED_CUT",
                             "factor": 0.30, "reason": f"PCR signal={opt_sig:.2f}"})

        # Rule 3: Extreme fear (PCR very high) -> slightly boost longs (contrarian)
        elif opt_sig > 0.8:
            long_mask = filtered.loc[date] > 0
            filtered.loc[date, long_mask] *= 1.20    # 20% larger long positions
            log_rows.append({"date": date, "action": "FEAR_BOOST",
                             "factor": 1.20, "reason": f"PCR signal={opt_sig:.2f}"})

    adjustment_log = pd.DataFrame(log_rows)
    if not adjustment_log.empty:
        adjustment_log.set_index("date", inplace=True)

    return filtered, adjustment_log


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------

def generate_options_signals(nifty_close, put_volume=None, call_volume=None,
                              iv_otm_puts=None, iv_atm_calls=None,
                              simulate=True, output_dir="outputs/"):
    """
    Master function: generate options-derived signals.

    Args:
        nifty_close  : pd.Series -- Nifty 50 closes (required always)
        put_volume   : pd.Series -- real put volume (if available)
        call_volume  : pd.Series -- real call volume (if available)
        iv_otm_puts  : pd.Series -- real OTM put IV (if available)
        iv_atm_calls : pd.Series -- real ATM call IV (if available)
        simulate     : bool -- if True, simulate options data from price data
        output_dir   : str -- where to save outputs

    Returns:
        options_df : pd.DataFrame -- all options metrics
        combined   : pd.Series -- final combined options signal
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("OPTIONS SIGNAL MODULE")
    print("Signals: Put-Call Ratio + IV Skew")
    print("=" * 60)

    # Use real data if provided, else simulate
    if simulate or put_volume is None:
        print("[Options] Simulating options data from price dynamics...")
        put_volume, call_volume, iv_otm_puts, iv_atm_calls = \
            simulate_options_data(nifty_close)
    else:
        print("[Options] Using real options chain data...")

    # Compute signals
    options_df, combined = compute_options_signal(
        put_volume, call_volume, iv_otm_puts, iv_atm_calls
    )

    # Save outputs
    options_df.to_csv(f"{output_dir}options_signals.csv")
    combined.to_csv(f"{output_dir}options_combined_signal.csv", header=True)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIONS SIGNAL SUMMARY")
    print("=" * 60)
    print(f"Date range     : {options_df.index[0]} -> {options_df.index[-1]}")
    print(f"Avg PCR        : {options_df['pcr_raw'].mean():.3f}")
    print(f"PCR range      : {options_df['pcr_raw'].min():.2f} -> {options_df['pcr_raw'].max():.2f}")
    print(f"Avg IV Skew    : {options_df['iv_skew'].mean():.3f}")
    print(f"Crash risk days: {options_df['crash_risk_flag'].sum()} days")
    print(f"Avg options sig: {combined.mean():+.3f}")
    print("=" * 60)

    return options_df, combined


# ---------------------------------------------
# RUN STANDALONE
# ---------------------------------------------

if __name__ == "__main__":
    import yfinance as yf

    print("Testing options_signal.py with Nifty 50 data...\n")

    nifty = yf.download("^NSEI", start="2020-01-01", end="2024-12-31",
                        progress=False)["Close"].squeeze()
    nifty.dropna(inplace=True)

    options_df, combined = generate_options_signals(
        nifty_close=nifty,
        simulate=True,
        output_dir="outputs/options/"
    )

    print("\nSample options signals (last 10 days):")
    print(options_df[["pcr_raw", "pcr_signal", "iv_skew",
                       "crash_risk_flag", "combined_signal"]].tail(10).to_string())
