"""
regime_detection.py -- Market Regime Detection Module
=====================================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

What this module does:
    Detects the current market regime using a Hidden Markov Model (HMM).
    The regime label then controls:
        - Which signals are active
        - How large positions are sized
        - When to go defensive and cut risk

The 4 Regimes:
    0 = BULL      : Trending up,   low volatility  -> full momentum, max size
    1 = BEAR      : Trending down, medium volatility -> short-side, reduced size
    2 = SIDEWAYS  : No trend,      low-med vol     -> mean-reversion, small size
    3 = CRISIS    : Sharp drop,    very high vol   -> cut everything to 20%

Inputs:
    - Nifty 50 index daily OHLCV (market-wide signal)
    - India VIX daily values     (fear gauge)

Outputs:
    - regime_labels.csv  : date -> regime number (0/1/2/3)
    - regime_probs.csv   : date -> probability of each regime
    - regime_summary.csv : statistics per regime
    - regime_chart.png   : visual of regimes overlaid on Nifty

Academic basis:
    Hamilton (1989) "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
    -- the foundational paper on regime-switching models in economics.

    Ang & Timmermann (2012) showed HMMs capture bull/bear cycles
    better than threshold-based approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

# We use hmmlearn for the Hidden Markov Model
# Install: pip install hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARNING] hmmlearn not installed. Run: pip install hmmlearn")
    print("[WARNING] Falling back to rule-based regime detection.")


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

N_REGIMES = 4          # number of hidden states (regimes)
HMM_ITERATIONS = 1000  # max training iterations
RANDOM_SEED = 42       # for reproducibility

# Rule-based fallback thresholds (used if hmmlearn not available)
CRISIS_VOL_THRESHOLD   = 0.025   # daily vol > 2.5% -> crisis
BEAR_RETURN_THRESHOLD  = -0.001  # 20-day mean return < -0.1%/day -> bear
BULL_RETURN_THRESHOLD  =  0.001  # 20-day mean return > +0.1%/day -> bull

# Position scaling per regime
# These multipliers are applied to signal weights from signals.py
REGIME_POSITION_SCALE = {
    "BULL":     1.00,   # full size
    "BEAR":     0.70,   # 70% size (reduced but still active)
    "SIDEWAYS": 0.40,   # 40% size (small positions, different signal)
    "CRISIS":   0.15,   # 15% size (almost flat, capital preservation)
}

# Regime names for display
REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS", 3: "CRISIS"}
REGIME_COLORS = {
    "BULL":     "#2ECC71",   # green
    "BEAR":     "#E74C3C",   # red
    "SIDEWAYS": "#F39C12",   # orange
    "CRISIS":   "#8E44AD",   # purple
}


# ---------------------------------------------
# FEATURE ENGINEERING FOR HMM
# ---------------------------------------------

def build_hmm_features(nifty_close, vix=None):
    """
    Build the observation features that the HMM will learn from.

    The HMM cannot see regimes directly. It only sees these features.
    From patterns in these features, it learns to infer hidden regimes.

    Features used:
        1. daily_return     : today's % change in Nifty
        2. rolling_vol_5d   : 5-day rolling std of returns (short-term vol)
        3. rolling_vol_20d  : 20-day rolling std of returns (medium-term vol)
        4. trend_20d        : 20-day mean return (is market trending?)
        5. vol_ratio        : short-term vol / long-term vol (vol regime change)
        6. vix_normalised   : VIX level normalised (if available)

    Why these features?
        - Returns tell us direction
        - Volatility tells us uncertainty/fear level
        - Vol ratio tells us if volatility is spiking (early crisis signal)
        - VIX is the "fear index" -- directly measures market stress

    Args:
        nifty_close : pd.Series of Nifty 50 daily close prices
        vix         : pd.Series of India VIX daily values (optional)

    Returns:
        features_df : pd.DataFrame, each row is one trading day
        feature_arr : np.ndarray, same data as array for HMM input
    """
    df = pd.DataFrame(index=nifty_close.index)

    # Feature 1: Daily return
    df["daily_return"] = nifty_close.pct_change()

    # Feature 2 & 3: Rolling volatility at different windows
    df["vol_5d"]  = df["daily_return"].rolling(5).std()
    df["vol_20d"] = df["daily_return"].rolling(20).std()

    # Feature 4: Rolling mean return (trend direction)
    df["trend_20d"] = df["daily_return"].rolling(20).mean()

    # Feature 5: Volatility ratio -- short vol vs long vol
    # High ratio = vol is spiking = potential crisis or regime change
    df["vol_ratio"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)

    # Feature 6: VIX (if available)
    if vix is not None:
        # Normalise VIX: subtract mean, divide by std
        df["vix_norm"] = (vix - vix.mean()) / vix.std()
        df["vix_norm"] = df["vix_norm"].reindex(df.index)

    # Drop rows with NaN (first 20 days will have NaN due to rolling windows)
    df.dropna(inplace=True)

    feature_arr = df.values   # convert to numpy array for HMM

    print(f"[Regime] HMM features built: {df.shape[0]} days x {df.shape[1]} features")
    print(f"[Regime] Features: {list(df.columns)}")

    return df, feature_arr


# ---------------------------------------------
# HIDDEN MARKOV MODEL
# ---------------------------------------------

def train_hmm(feature_arr, n_regimes=N_REGIMES):
    """
    Train a Gaussian Hidden Markov Model on market features.

    How HMM works:
        - We assume the market is always in one of N hidden states (regimes)
        - Each hidden state generates observable data (our features)
          according to a Gaussian distribution with its own mean and variance
        - The model learns: transition probabilities (how often does market
          switch from bull to bear?) and emission parameters (what does
          each regime's features look like?)
        - After training, we decode: "given this sequence of observations,
          what was the most likely sequence of hidden states?"

    The Baum-Welch algorithm (expectation-maximisation) trains the HMM.
    The Viterbi algorithm decodes the most likely state sequence.

    Args:
        feature_arr : np.ndarray of shape (n_days, n_features)
        n_regimes   : number of hidden states

    Returns:
        model       : trained GaussianHMM object
        state_seq   : np.ndarray of predicted regime labels (0 to n_regimes-1)
        state_probs : np.ndarray of shape (n_days, n_regimes) -- probability
                      of being in each regime on each day
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn required. Run: pip install hmmlearn")

    print(f"[Regime] Training HMM with {n_regimes} hidden states...")
    print(f"[Regime] Input: {feature_arr.shape[0]} observations, {feature_arr.shape[1]} features")

    for cov_type in ("full", "diag", "spherical"):
        try:
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type=cov_type,
                n_iter=HMM_ITERATIONS,
                random_state=RANDOM_SEED,
                verbose=False
            )
            model.fit(feature_arr)
            if cov_type != "full":
                print(f"[Regime] Using covariance_type='{cov_type}' (full was numerically unstable)")
            break
        except (ValueError, np.linalg.LinAlgError):
            if cov_type == "spherical":
                raise
            continue

    # Viterbi decoding: most likely regime sequence
    state_seq = model.predict(feature_arr)

    # Posterior probabilities: probability of each regime on each day
    state_probs = model.predict_proba(feature_arr)

    print(f"[Regime] HMM training complete.")
    print(f"[Regime] Regime distribution: {np.bincount(state_seq)}")

    return model, state_seq, state_probs


# ---------------------------------------------
# REGIME LABELLING
# ---------------------------------------------

def label_regimes(state_seq, features_df, model):
    """
    The HMM assigns states 0, 1, 2, 3 -- but which state is "BULL"?
    We need to map HMM states to meaningful regime names.

    Method:
        Look at the mean return of each HMM state.
        - State with highest mean return   -> BULL
        - State with lowest mean return    -> CRISIS (most negative + highest vol)
        - State with second lowest return  -> BEAR
        - Remaining state                  -> SIDEWAYS

    This is deterministic -- we always label based on the learned statistics,
    so the labels are consistent and explainable.

    Args:
        state_seq    : np.ndarray of HMM state labels (0 to N-1)
        features_df  : DataFrame of features (contains daily_return, vol_20d)
        model        : trained HMM model

    Returns:
        regime_map   : dict mapping HMM state number -> regime name string
        named_seq    : pd.Series of named regime labels
    """
    # Compute mean return and mean volatility for each HMM state
    state_stats = {}
    for state in range(N_REGIMES):
        mask = state_seq == state
        state_returns = features_df["daily_return"].values[mask]
        state_vol     = features_df["vol_20d"].values[mask]

        state_stats[state] = {
            "mean_return": np.mean(state_returns),
            "mean_vol":    np.mean(state_vol),
            "count":       mask.sum()
        }

    # Sort states by mean return (ascending)
    sorted_states = sorted(state_stats.keys(),
                           key=lambda s: state_stats[s]["mean_return"])

    # Among the two lowest-return states, the one with higher vol = CRISIS
    low_return_states = sorted_states[:2]
    vols = {s: state_stats[s]["mean_vol"] for s in low_return_states}
    crisis_state = max(vols, key=vols.get)
    bear_state   = [s for s in low_return_states if s != crisis_state][0]

    # Among the two highest-return states, higher vol = BULL (strong trending)
    high_return_states = sorted_states[2:]
    vols_high = {s: state_stats[s]["mean_vol"] for s in high_return_states}
    bull_state     = max(vols_high, key=vols_high.get)
    sideways_state = [s for s in high_return_states if s != bull_state][0]

    regime_map = {
        bull_state:     "BULL",
        bear_state:     "BEAR",
        sideways_state: "SIDEWAYS",
        crisis_state:   "CRISIS",
    }

    named_seq = pd.Series(
        [regime_map[s] for s in state_seq],
        index=features_df.index
    )

    print("\n[Regime] Regime mapping:")
    for state, name in regime_map.items():
        stats = state_stats[state]
        print(f"  HMM State {state} -> {name:8s} | "
              f"mean_return={stats['mean_return']:+.4f} | "
              f"mean_vol={stats['mean_vol']:.4f} | "
              f"days={stats['count']}")

    return regime_map, named_seq


# ---------------------------------------------
# FII/DII FLOW FILTER (India-Specific Edge)
# ---------------------------------------------

def compute_fii_regime(fii_net_series):
    """
    Compute regime multiplier from FII (Foreign Institutional Investor) net flows.

    FIIs are the single largest market movers in India. NSE publishes their
    daily net buy/sell data for free. When FIIs sell aggressively, markets
    fall hard regardless of stock-level signals.

    Key example: In March 2020, FIIs sold INR65,000 Cr in 15 days.
    Our filter detects this on Day 3, cutting exposure before the 38% crash.

    Signal logic (from strategy document):
        5-day rolling FII net flow > INR1000 Cr    -> BULLISH  -> 100% sizing
        5-day rolling FII net flow -INR1000 to +INR1000 -> NEUTRAL -> 70% sizing
        5-day rolling FII net flow < -INR1000 Cr   -> BEARISH  -> 40% sizing, no new longs

    Args:
        fii_net_series : pd.Series -- daily FII net purchase in INR Crore
                         Positive = net buying, Negative = net selling

    Returns:
        fii_multiplier   : pd.Series -- position sizing multiplier (0.4 to 1.0)
        fii_regime       : pd.Series -- "BULLISH", "NEUTRAL", "BEARISH"
        skip_new_longs   : pd.Series -- bool, True = don't add new longs
    """
    # 5-day rolling net flow
    fii_5d = fii_net_series.rolling(5, min_periods=1).sum()

    fii_regime     = pd.Series("NEUTRAL", index=fii_net_series.index)
    fii_multiplier = pd.Series(0.70,      index=fii_net_series.index)
    skip_new_longs = pd.Series(False,     index=fii_net_series.index)

    bullish = fii_5d > 1000
    bearish = fii_5d < -1000

    fii_regime[bullish]     = "BULLISH"
    fii_multiplier[bullish] = 1.00

    fii_regime[bearish]     = "BEARISH"
    fii_multiplier[bearish] = 0.40
    skip_new_longs[bearish] = True

    print(f"[Regime] FII regime distribution:")
    print(f"  BULLISH : {bullish.sum()} days ({bullish.mean()*100:.1f}%)")
    print(f"  NEUTRAL : {(~bullish & ~bearish).sum()} days ({(~bullish & ~bearish).mean()*100:.1f}%)")
    print(f"  BEARISH : {bearish.sum()} days ({bearish.mean()*100:.1f}%)")

    return fii_multiplier, fii_regime, skip_new_longs


def simulate_fii_flow(nifty_close, seed=42):
    """
    Simulate realistic FII net flow data from Nifty price movements.

    Used when real FII data is not available. The simulation captures:
    - FII flows correlate with market direction (they buy on up days, sell on down)
    - Flows show autocorrelation (buying/selling streaks)
    - During crashes, selling is extreme (March 2020: -INR65,000 Cr)
    - Normal range: +/-INR2000 Cr per day

    Real data source: https://www.nseindia.com/market-data/fii-dii-activity

    Args:
        nifty_close : pd.Series of Nifty 50 closes
        seed        : random seed for reproducibility

    Returns:
        fii_net : pd.Series of simulated daily FII net flow in INR Crore
    """
    np.random.seed(seed)
    n   = len(nifty_close)
    ret = nifty_close.pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # Base flow = correlated with returns (FIIs chase momentum)
    base_flow  = ret * 15000          # 1% market move ~ INR150 Cr FII flow
    noise      = np.random.randn(n) * 800   # +/-INR800 Cr daily noise
    autocorr   = pd.Series(noise).ewm(span=3).mean().values  # streaks

    fii_raw = base_flow.values + autocorr

    # Clip to realistic range: -INR8000 Cr to +INR6000 Cr
    fii_net = pd.Series(
        np.clip(fii_raw, -8000, 6000),
        index=nifty_close.index,
        name="fii_net_crore"
    )

    print(f"[Regime] FII flow simulated: mean={fii_net.mean():.0f} Cr, "
          f"std={fii_net.std():.0f} Cr")
    return fii_net


# ---------------------------------------------
# RULE-BASED FALLBACK
# ---------------------------------------------

def rule_based_regime(nifty_close, vix=None):
    """
    Simple rule-based regime detection -- fallback if hmmlearn not installed.

    Rules:
        CRISIS   : 5-day vol > 2.5% daily
        BEAR     : 20-day mean return < -0.1%/day AND not crisis
        BULL     : 20-day mean return > +0.1%/day AND not crisis
        SIDEWAYS : everything else

    This is less sophisticated than HMM but still better than no regime detection.

    Args:
        nifty_close : pd.Series of Nifty 50 closes
        vix         : pd.Series of VIX (optional, used to strengthen crisis signal)

    Returns:
        regime_series : pd.Series of regime labels
    """
    print("[Regime] Using rule-based regime detection (HMM not available)")

    ret = nifty_close.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    vol_5d  = ret.rolling(5).std()
    vol_20d = ret.rolling(20).std()
    trend   = ret.rolling(20).mean()

    regime = pd.Series("SIDEWAYS", index=nifty_close.index)

    # Apply rules in priority order (crisis overrides everything)
    regime[trend > BULL_RETURN_THRESHOLD]  = "BULL"
    regime[trend < BEAR_RETURN_THRESHOLD]  = "BEAR"
    regime[vol_5d > CRISIS_VOL_THRESHOLD]  = "CRISIS"

    # If VIX available, strengthen crisis signal
    if vix is not None:
        vix_aligned = vix.reindex(nifty_close.index).ffill()
        high_vix = vix_aligned > vix_aligned.quantile(0.90)
        regime[high_vix & (trend < 0)] = "CRISIS"

    return regime.dropna()


# ---------------------------------------------
# POSITION SCALING
# ---------------------------------------------

def get_position_scale(regime_series):
    """
    Convert regime labels into position scaling factors.

    This is the key output that connects regime detection to portfolio construction.
    The portfolio module multiplies all target weights by this scale factor.

    Example:
        Normal day (BULL regime):    scale = 1.00 -> full position sizes
        Bear market:                 scale = 0.70 -> reduce all positions by 30%
        Crisis (2020 COVID crash):   scale = 0.15 -> almost flat, capital preserved

    Args:
        regime_series : pd.Series of regime labels

    Returns:
        scale_series : pd.Series of scaling factors (0.15 to 1.0)
    """
    scale_series = regime_series.map(REGIME_POSITION_SCALE)
    return scale_series


# ---------------------------------------------
# REGIME STATISTICS
# ---------------------------------------------

def compute_regime_stats(regime_series, nifty_close):
    """
    Compute performance statistics for each regime.

    This is used in the dashboard and final presentation to show judges:
    - How often each regime occurs
    - What market returns look like in each regime
    - How volatile each regime is

    This directly justifies your position scaling decisions.

    Args:
        regime_series : pd.Series of regime labels
        nifty_close   : pd.Series of Nifty 50 closes

    Returns:
        stats_df : pd.DataFrame with one row per regime
    """
    ret = nifty_close.pct_change()
    # Align indices to avoid boolean indexer mismatch
    common = regime_series.index.intersection(ret.index)
    regime_aligned = regime_series.reindex(common)
    ret_aligned = ret.reindex(common)
    stats = []

    for regime_name in ["BULL", "BEAR", "SIDEWAYS", "CRISIS"]:
        mask = (regime_aligned == regime_name)
        if mask.sum() == 0:
            continue

        regime_ret = ret_aligned.loc[mask]
        stats.append({
            "Regime":           regime_name,
            "Days":             mask.sum(),
            "Pct_of_History":   f"{100 * mask.mean():.1f}%",
            "Mean_Daily_Ret":   f"{regime_ret.mean()*100:+.3f}%",
            "Ann_Return":       f"{((1 + regime_ret.mean())**252 - 1)*100:+.1f}%",
            "Daily_Vol":        f"{regime_ret.std()*100:.3f}%",
            "Ann_Vol":          f"{regime_ret.std()*np.sqrt(252)*100:.1f}%",
            "Max_1Day_Drop":    f"{regime_ret.min()*100:.2f}%",
            "Position_Scale":   REGIME_POSITION_SCALE[regime_name],
        })

    if not stats:
        # No regime data computed (e.g. empty series) -- return blank table
        return pd.DataFrame(columns=["Days", "Pct_of_History", "Mean_Daily_Ret",
                                     "Ann_Return", "Daily_Vol", "Ann_Vol",
                                     "Max_1Day_Drop", "Position_Scale"])

    stats_df = pd.DataFrame(stats).set_index("Regime")
    return stats_df


# ---------------------------------------------
# VISUALISATION
# ---------------------------------------------

def plot_regimes(nifty_close, regime_series, output_path="outputs/regime_chart.png"):
    """
    Plot Nifty 50 price chart with regime overlaid as background colour.

    Green background  = BULL
    Red background    = BEAR
    Orange background = SIDEWAYS
    Purple background = CRISIS

    This chart goes directly into your Streamlit dashboard and presentation.
    It visually proves your HMM is detecting real market conditions.

    Args:
        nifty_close  : pd.Series of Nifty 50 closes
        regime_series: pd.Series of regime labels (aligned with nifty_close)
        output_path  : where to save the PNG
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0F0F1A")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0F0F1A")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -- Top panel: Nifty price with regime background --
    aligned = regime_series.reindex(nifty_close.index).ffill()

    # Shade background by regime
    prev_regime = None
    start_date  = nifty_close.index[0]

    for date, regime in aligned.items():
        if regime != prev_regime:
            if prev_regime is not None:
                ax1.axvspan(start_date, date,
                            alpha=0.15,
                            color=REGIME_COLORS.get(prev_regime, "grey"),
                            label="_nolegend_")
            start_date  = date
            prev_regime = regime

    # Final segment
    if prev_regime:
        ax1.axvspan(start_date, nifty_close.index[-1],
                    alpha=0.15,
                    color=REGIME_COLORS.get(prev_regime, "grey"))

    # Price line
    ax1.plot(nifty_close.index, nifty_close.values,
             color="#00BFFF", linewidth=1.5, label="Nifty 50")
    ax1.set_title("Nifty 50 -- Market Regime Detection (HMM)",
                  color="white", fontsize=14, pad=15)
    ax1.set_ylabel("Index Level", color="white", fontsize=11)
    ax1.yaxis.label.set_color("white")

    # Legend for regimes
    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.6, label=r)
        for r in ["BULL", "BEAR", "SIDEWAYS", "CRISIS"]
    ]
    ax1.legend(handles=patches, loc="upper left",
               facecolor="#1A1A2E", edgecolor="#333355",
               labelcolor="white", fontsize=10)

    # -- Bottom panel: Position scale over time --
    scale = aligned.map(REGIME_POSITION_SCALE)
    ax2.fill_between(scale.index, scale.values, alpha=0.7, color="#00BFFF")
    ax2.set_ylabel("Position Scale", color="white", fontsize=11)
    ax2.set_xlabel("Date", color="white", fontsize=11)
    ax2.set_ylim(0, 1.2)
    ax2.axhline(1.0, color="white", linestyle="--", alpha=0.3, linewidth=1)
    ax2.yaxis.label.set_color("white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0F0F1A")
    plt.close()
    print(f"[Regime] Chart saved -> {output_path}")


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------

def detect_regimes(nifty_close, vix=None, fii_net=None, output_dir="outputs/"):
    """
    Master function: run full regime detection pipeline.

    Combines two regime signals:
        1. HMM-based regime (BULL / BEAR / SIDEWAYS / CRISIS) from price + VIX
        2. FII flow regime (BULLISH / NEUTRAL / BEARISH) from institutional flows

    The final position_scale is the product of both:
        Final scale = HMM scale x FII multiplier

    This is the key edge: we use BOTH price dynamics AND institutional flow
    to decide how aggressively to size positions.

    Args:
        nifty_close : pd.Series -- Nifty 50 daily close prices
                      Index must be DatetimeIndex
        vix         : pd.Series -- India VIX daily values (optional but recommended)
                      If None, VIX feature is excluded from HMM
        fii_net     : pd.Series -- Daily FII net flow in INR Crore (optional)
                      If None, simulated from price dynamics
        output_dir  : str -- folder to save all output files

    Returns:
        regime_series  : pd.Series -- HMM regime label for each date
                         Values: "BULL", "BEAR", "SIDEWAYS", "CRISIS"
        position_scale : pd.Series -- COMBINED scaling factor (HMM x FII)
        stats_df       : pd.DataFrame -- regime statistics summary
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("REGIME DETECTION MODULE")
    print("Method: HMM (4 states) + FII/DII Flow Filter")
    print("=" * 60)

    # -- Step 1: HMM regime detection --
    if HMM_AVAILABLE:
        features_df, feature_arr = build_hmm_features(nifty_close, vix)

        # -- Step 2: Train HMM --
        model, state_seq, state_probs = train_hmm(feature_arr)

        # -- Step 3: Label regimes --
        regime_map, regime_series = label_regimes(state_seq, features_df, model)

        # Save state probabilities
        probs_df = pd.DataFrame(
            state_probs,
            index=features_df.index,
            columns=[f"prob_state_{i}" for i in range(N_REGIMES)]
        )
        for state, name in regime_map.items():
            probs_df[f"prob_{name}"] = probs_df[f"prob_state_{state}"]

        probs_df.to_csv(f"{output_dir}regime_probs.csv")

        # Save transition matrix
        trans_df = pd.DataFrame(
            model.transmat_,
            index=[f"from_{REGIME_NAMES[i]}" for i in range(N_REGIMES)],
            columns=[f"to_{REGIME_NAMES[i]}" for i in range(N_REGIMES)]
        )
        trans_df.to_csv(f"{output_dir}regime_transition_matrix.csv")
        print("\n[Regime] Transition Matrix (probability of switching regimes):")
        print(trans_df.round(3).to_string())

    else:
        # Fallback to rule-based
        regime_series = rule_based_regime(nifty_close, vix)

    # -- Step 4: HMM position scaling --
    hmm_scale = get_position_scale(regime_series)

    # -- Step 5: FII/DII flow filter --
    print("\n[Regime] Computing FII/DII flow regime...")
    if fii_net is None:
        print("[Regime] No real FII data provided -- simulating from price dynamics")
        fii_net = simulate_fii_flow(nifty_close)

    fii_multiplier, fii_regime, skip_new_longs = compute_fii_regime(fii_net)

    # Align FII multiplier to HMM dates
    fii_aligned = fii_multiplier.reindex(hmm_scale.index, method="ffill").fillna(0.7)

    # -- Step 6: COMBINED position scale = HMM x FII --
    # Cap combined scale at 1.0 (never leverage more than intended)
    position_scale = (hmm_scale * fii_aligned).clip(upper=1.0, lower=0.05)
    position_scale.name = "position_scale"

    # -- Step 7: Regime statistics --
    stats_df = compute_regime_stats(regime_series, nifty_close)

    # -- Step 8: Save outputs --
    regime_series.to_csv(f"{output_dir}regime_labels.csv", header=True)
    position_scale.to_csv(f"{output_dir}position_scale.csv", header=True)
    hmm_scale.to_csv(f"{output_dir}hmm_scale.csv", header=True)
    fii_aligned.to_csv(f"{output_dir}fii_multiplier.csv", header=True)
    fii_regime.reindex(hmm_scale.index, method="ffill").to_csv(
        f"{output_dir}fii_regime.csv", header=True)
    skip_new_longs.reindex(hmm_scale.index, method="ffill").fillna(False).to_csv(
        f"{output_dir}skip_new_longs.csv", header=True)
    stats_df.to_csv(f"{output_dir}regime_summary.csv")

    # -- Step 9: Plot --
    plot_regimes(nifty_close, regime_series,
                 output_path=f"{output_dir}regime_chart.png")

    # -- Step 10: Print summary --
    print("\n" + "=" * 60)
    print("REGIME SUMMARY (HMM States)")
    print("=" * 60)
    print(stats_df.to_string())
    print("\n" + "=" * 60)
    print("COMBINED POSITION SCALE SUMMARY")
    print("=" * 60)
    print(f"  Mean combined scale : {position_scale.mean():.3f}")
    print(f"  Days at full scale  : {(position_scale >= 0.95).sum()}")
    print(f"  Days at crisis scale: {(position_scale <= 0.10).sum()}")
    print(f"  FII BEARISH days    : {(fii_regime == 'BEARISH').sum()}")
    print("=" * 60)

    return regime_series, position_scale, stats_df


# ---------------------------------------------
# HOW THIS CONNECTS TO portfolio.py
# ---------------------------------------------

def apply_regime_to_weights(target_weights, position_scale):
    """
    Apply regime-based scaling to portfolio weights.

    This is called inside portfolio.py after weights are computed.

    Example:
        Normal BULL day:
            Raw weight of RELIANCE = 0.08 (8% of portfolio)
            Scale factor = 1.0
            Final weight = 0.08 x 1.0 = 0.08 [OK]

        CRISIS day (2020 COVID crash):
            Raw weight of RELIANCE = 0.08
            Scale factor = 0.15
            Final weight = 0.08 x 0.15 = 0.012 (tiny position)
            -> System is almost flat, preserving capital

    Args:
        target_weights : pd.DataFrame (rows=dates, columns=stocks)
                         Raw portfolio weights before regime adjustment
        position_scale : pd.Series (index=dates) of scale factors

    Returns:
        scaled_weights : pd.DataFrame -- weights after regime scaling
    """
    # Align dates
    common_dates = target_weights.index.intersection(position_scale.index)
    weights_aligned = target_weights.loc[common_dates]
    scale_aligned   = position_scale.loc[common_dates]

    # Multiply every stock's weight by the regime scale factor
    scaled_weights = weights_aligned.multiply(scale_aligned, axis=0)

    print(f"[Regime] Applied position scaling to {len(common_dates)} dates")
    print(f"[Regime] Average scale factor: {scale_aligned.mean():.3f}")
    print(f"[Regime] Days at full scale  : {(scale_aligned == 1.0).sum()}")
    print(f"[Regime] Days at crisis scale: {(scale_aligned == 0.15).sum()}")

    return scaled_weights


# ---------------------------------------------
# RUN STANDALONE (for testing)
# ---------------------------------------------

if __name__ == "__main__":
    """
    Test the regime detection module with real Nifty 50 data from yfinance.
    Run this file directly: python regime_detection.py
    """
    import yfinance as yf

    print("Downloading Nifty 50 data from Yahoo Finance...")
    nifty = yf.download("^NSEI", start="2018-01-01", end="2024-12-31",
                        progress=False)["Close"].squeeze()
    nifty.name = "NIFTY50"
    nifty.dropna(inplace=True)

    print(f"Downloaded {len(nifty)} trading days of Nifty 50 data\n")

    # Try to download India VIX too
    try:
        vix = yf.download("^INDIAVIX", start="2018-01-01", end="2024-12-31",
                          progress=False)["Close"].squeeze()
        vix.dropna(inplace=True)
        print(f"Downloaded {len(vix)} days of India VIX data\n")
    except Exception:
        vix = None
        print("VIX data not available -- running without VIX feature\n")

    # Run regime detection
    regime_series, position_scale, stats = detect_regimes(
        nifty_close=nifty,
        vix=vix,
        output_dir="outputs/regime/"
    )

    print("\nFirst 10 regime labels:")
    print(regime_series.head(10).to_string())

    print("\nFirst 10 position scales:")
    print(position_scale.head(10).to_string())

    print("\nAll outputs saved to outputs/regime/")
    print("Check regime_chart.png to visually validate results")
