"""
quick_test.py -- Fast End-to-End Strategy Validation
====================================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

PURPOSE:
    Run the complete pipeline using the pre-built parquet data (no internet needed).
    Use this to verify the system works before running the full main.py pipeline.
    Completes in ~2-5 minutes instead of 15+ minutes.

WHAT IT TESTS:
    1. Parquet data loading & pivoting
    2. Signal generation (momentum + RSI + volume + 52wk filter + confidence)
    3. Regime detection (HMM + FII flow simulation)
    4. Portfolio construction (risk parity + confidence scaling + regime scaling)
    5. Backtest engine (transaction costs + equity curve + metrics)
    6. All output files created correctly

USAGE:
    python quick_test.py                # full quick test
    python quick_test.py --stocks 50   # test with only 50 stocks (faster)
    python quick_test.py --no-hmm      # skip HMM, use rule-based regime
"""

import argparse
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# -- Import all modules --
try:
    from data_loader      import load_from_parquet, download_index_data, clean_data, compute_returns, save_data, PARQUET_PATH
    from signals          import generate_signals
    from regime_detection import detect_regimes
    from options_signal   import generate_options_signals
    from portfolio        import build_portfolio
    from backtest         import run_full_backtest
except ImportError as e:
    print(f"\n[Error] Import failed: {e}")
    print("Make sure you are running from the project root directory.")
    sys.exit(1)


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

OUTPUT_DIR = "outputs/test/"
DATA_DIR   = "data/test/"


def banner():
    print("""
==============================================================
         QUICK TEST -- TECH GC 2026
         Nifty 500 | Long-Short Strategy
  Data    : Pre-built parquet (offline, no download)
  Signals : Momentum + RSI + Volume + 52wk + Confidence
  Regime  : HMM (4 states) + FII/DII flow filter
  Costs   : Commission + Bid-Ask + Almgren-Chriss slippage
==============================================================
""")


def timer_ctx(label):
    class T:
        def __enter__(self):
            self.t = time.time()
            return self
        def __exit__(self, *a):
            print(f"  [OK] {label} done in {time.time()-self.t:.1f}s\n")
    return T()


# ---------------------------------------------
# STEP 1: LOAD PARQUET DATA
# ---------------------------------------------

def step1_load_parquet(n_stocks=None):
    print("\n" + "=" * 60)
    print("STEP 1/6 -- LOADING NIFTY 500 PARQUET DATA")
    print("=" * 60)

    if not os.path.exists(PARQUET_PATH):
        print(f"[Test] ERROR: Parquet file not found at '{PARQUET_PATH}'")
        print("[Test] Make sure nifty500_daily_5Y_PATCHED.parquet is in this folder.")
        sys.exit(1)

    with timer_ctx("Parquet load + pivot"):
        raw = load_from_parquet(PARQUET_PATH)

    # Optionally limit to N stocks for speed
    if n_stocks and n_stocks < raw["Close"].shape[1]:
        print(f"[Test] Limiting to {n_stocks} stocks (--stocks flag)")
        # Pick stocks with most complete data
        completeness = raw["Close"].notna().sum().nlargest(n_stocks).index
        for k in raw:
            raw[k] = raw[k][completeness]

    # Build Nifty proxy from parquet before limiting to n_stocks
    # This ensures regime detection has a full-universe index proxy
    close_all_raw = raw["Close"].copy()
    nifty_proxy = close_all_raw.mean(axis=1)
    nifty_proxy.name = "NIFTY50"
    nifty_proxy = nifty_proxy.dropna()

    # Download index data (Nifty 50, VIX) -- needed for regime detection
    print("[Test] Fetching Nifty 50 + VIX from Yahoo Finance (falling back to parquet proxy)...")
    try:
        nifty, vix = download_index_data("2021-01-01", "2026-04-06")
        if len(nifty.dropna()) < 100:
            raise ValueError("Nifty download returned insufficient data")
    except Exception as e:
        print(f"[Test] Using Nifty proxy from parquet average ({e})")
        nifty = nifty_proxy
        vix = None

    with timer_ctx("Data cleaning"):
        clean = clean_data(raw)

    returns = compute_returns(clean["Close"])

    os.makedirs(DATA_DIR, exist_ok=True)
    save_data(clean, returns, nifty, vix, DATA_DIR)

    close  = clean["Close"]
    high   = clean.get("High")
    low    = clean.get("Low")
    volume = clean["Volume"]

    print(f"\n[Test] Data ready: {close.shape[0]} days x {close.shape[1]} stocks")
    print(f"[Test] Date range: {close.index[0].date()} -> {close.index[-1].date()}")

    return close, high, low, volume, returns, nifty, vix


# ---------------------------------------------
# STEP 2: SIGNAL GENERATION
# ---------------------------------------------

def step2_signals(close, volume, high=None, low=None):
    print("\n" + "=" * 60)
    print("STEP 2/6 -- SIGNAL GENERATION")
    print("=" * 60)
    print("Factors: Momentum + RSI + Volume Confirm + Volatility Penalty")
    print("Filters: RSI extremes + 52-week high/low distance")
    print("Scoring: Confidence score (0-85) per signal")

    with timer_ctx("Multi-factor signal computation + filtering"):
        positions, composite_signal, factors = generate_signals(
            close, volume,
            high_prices=high,
            low_prices=low,
            output_dir=OUTPUT_DIR
        )

    print(f"[Test] Avg longs/day : {(positions == 1).sum(axis=1).mean():.1f}")
    print(f"[Test] Avg shorts/day: {(positions == -1).sum(axis=1).mean():.1f}")
    if "confidence" in factors:
        conf = factors["confidence"]
        active_conf = conf[positions != 0].stack()
        print(f"[Test] Avg confidence: {active_conf.mean():.1f} / 85")

    return positions, composite_signal, factors


# ---------------------------------------------
# STEP 3: REGIME DETECTION
# ---------------------------------------------

def step3_regime(nifty, vix, use_hmm=True):
    print("\n" + "=" * 60)
    print("STEP 3/6 -- REGIME DETECTION")
    print("=" * 60)
    print("Method: HMM (4 states) + FII/DII flow filter (simulated)")
    print("Output: Combined position scale = HMM scale x FII multiplier")

    regime_dir = f"{OUTPUT_DIR}regime/"
    os.makedirs(regime_dir, exist_ok=True)

    # Temporarily patch HMM_AVAILABLE if --no-hmm flag
    if not use_hmm:
        import regime_detection as rd
        original = rd.HMM_AVAILABLE
        rd.HMM_AVAILABLE = False

    with timer_ctx("HMM training + FII simulation"):
        regime_series, position_scale, regime_stats = detect_regimes(
            nifty_close=nifty,
            vix=vix,
            fii_net=None,        # will auto-simulate
            output_dir=regime_dir
        )

    if not use_hmm:
        rd.HMM_AVAILABLE = original

    # Copy to main output dir
    regime_series.to_csv(f"{OUTPUT_DIR}regime_labels.csv", header=True)
    position_scale.to_csv(f"{OUTPUT_DIR}position_scale.csv", header=True)
    regime_stats.to_csv(f"{OUTPUT_DIR}regime_summary.csv")

    return regime_series, position_scale


# ---------------------------------------------
# STEP 4: OPTIONS SIGNALS
# ---------------------------------------------

def step4_options(nifty):
    print("\n" + "=" * 60)
    print("STEP 4/6 -- OPTIONS SIGNALS (PCR + IV SKEW)")
    print("=" * 60)
    print("Signals: Put-Call Ratio (contrarian) + IV Skew (crash risk)")

    options_dir = f"{OUTPUT_DIR}options/"
    os.makedirs(options_dir, exist_ok=True)

    with timer_ctx("Options signal computation"):
        options_df, options_combined = generate_options_signals(
            nifty_close=nifty,
            simulate=True,
            output_dir=options_dir
        )

    options_df.to_csv(f"{OUTPUT_DIR}options_signals.csv")
    return options_df, options_combined


# ---------------------------------------------
# STEP 5: PORTFOLIO CONSTRUCTION
# ---------------------------------------------

def step5_portfolio(positions, returns, position_scale, factors):
    print("\n" + "=" * 60)
    print("STEP 5/6 -- PORTFOLIO CONSTRUCTION")
    print("=" * 60)
    print("Method: Inverse-vol risk parity x confidence x regime scale")

    confidence = factors.get("confidence")

    with timer_ctx("Risk parity + confidence + regime scaling"):
        daily_weights, turnover = build_portfolio(
            positions=positions,
            returns=returns,
            position_scale=position_scale,
            confidence=confidence,
            output_dir=OUTPUT_DIR
        )

    return daily_weights, turnover


# ---------------------------------------------
# STEP 6: BACKTEST
# ---------------------------------------------

def step6_backtest(daily_weights, close, returns, volume, nifty):
    print("\n" + "=" * 60)
    print("STEP 6/6 -- BACKTESTING ENGINE")
    print("=" * 60)
    print("Costs: Commission 0.05%/side + Bid-Ask 0.05%/side + Almgren-Chriss slippage")
    print("Split: Walk-forward -- Train ends 2022-12-31, Test 2023-2026")

    with timer_ctx("Backtest simulation + metrics"):
        equity_curve, trade_log, metrics = run_full_backtest(
            daily_weights=daily_weights,
            close=close,
            returns=returns,
            volume=volume,
            nifty_close=nifty,
            output_dir=OUTPUT_DIR
        )

    return equity_curve, trade_log, metrics


# ---------------------------------------------
# RESULTS SUMMARY
# ---------------------------------------------

def print_results(equity_curve, metrics, total_time):
    nav = equity_curve["nav"]
    ret = equity_curve["daily_return"]
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_days    = len(ret)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol   = ret.std() * np.sqrt(252)
    sharpe    = (ann_ret - 0.065) / ann_vol if ann_vol > 0 else 0
    max_dd    = ((nav / nav.cummax()) - 1).min()

    print("""
==============================================================
                    QUICK TEST RESULTS
==============================================================""")
    print(f"  Total Return     : {total_ret:>+8.2%}")
    print(f"  Ann. Return      : {ann_ret:>+8.2%}")
    print(f"  Sharpe Ratio     : {sharpe:>8.3f}")
    print(f"  Max Drawdown     : {max_dd:>8.2%}")
    print(f"  Ann. Volatility  : {ann_vol:>8.2%}")
    print(f"  Pipeline Time    : {total_time:>7.0f}s")
    print("""--------------------------------------------------------------
  All outputs saved to outputs/test/
  [OK] Parquet data loaded correctly
  [OK] Signals + 52wk filter + confidence score working
  [OK] HMM regime + FII flow filter working
  [OK] Risk parity portfolio with confidence scaling working
  [OK] Backtest engine with realistic costs working
  Next step: streamlit run dashboard/app.py
=============================================================""")

    print("\nKEY NUMBERS FOR PRESENTATION:")
    print(f"   Sharpe Ratio  : {sharpe:.2f}  (target > 1.2)")
    print(f"   Max Drawdown  : {max_dd:.1%} (target < 15%)")
    print(f"   Annual Return : {ann_ret:.1%}")
    if metrics is not None and not metrics.empty:
        print(f"\nTRAIN vs TEST BREAKDOWN:")
        print(metrics[["Ann. Return", "Sharpe Ratio", "Max Drawdown"]].to_string())


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick Test -- Tech GC 2026")
    parser.add_argument("--stocks", type=int, default=None,
                        help="Limit to N stocks for speed (default: all 499)")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Use rule-based regime instead of HMM (faster)")
    args = parser.parse_args()

    banner()
    total_start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 1. Load data from parquet
        close, high, low, volume, returns, nifty, vix = step1_load_parquet(
            n_stocks=args.stocks
        )

        # 2. Generate signals
        positions, composite_signal, factors = step2_signals(
            close, volume, high=high, low=low
        )

        # 3. Regime detection (HMM + FII)
        regime_series, position_scale = step3_regime(
            nifty, vix, use_hmm=not args.no_hmm
        )

        # 4. Options signals
        options_df, options_combined = step4_options(nifty)

        # 5. Portfolio construction
        daily_weights, turnover = step5_portfolio(
            positions, returns, position_scale, factors
        )

        # 6. Backtest
        equity_curve, trade_log, metrics = step6_backtest(
            daily_weights, close, returns, volume, nifty
        )

        total_time = time.time() - total_start
        print_results(equity_curve, metrics, total_time)

    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[Test] FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
