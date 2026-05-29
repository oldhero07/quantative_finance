"""
main.py -- Master Pipeline Runner
==================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

This file connects all 7 modules in order.
Run this ONE file to execute the entire system end-to-end.

Usage:
    python main.py                  # full pipeline
    python main.py --skip-download  # skip data download (use cached data)
    python main.py --module data    # run only one module

Output:
    data/          -> cleaned OHLCV CSV files
    outputs/       -> signals, weights, equity curve, trade log, metrics
    outputs/regime -> regime labels, charts
    outputs/options-> PCR, IV skew signals

Then launch dashboard:
    streamlit run dashboard/app.py
"""

import argparse
import sys
import os
import pandas as pd
import time

# ---------------------------------------------
# IMPORTS -- all modules
# ---------------------------------------------

from data_loader        import run_data_pipeline, load_data
from signals            import generate_signals
from regime_detection   import detect_regimes
from options_signal     import generate_options_signals
from portfolio          import build_portfolio
from backtest           import run_full_backtest


# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

DATA_DIR    = "data/"
OUTPUT_DIR  = "outputs/"
START_DATE  = "2018-01-01"
END_DATE    = "2024-12-31"


def print_banner():
    print("""
+==============================================================+
|         QUANTITATIVE TRADING SYSTEM -- TECH GC 2026          |
|         High Prep Problem Solving | IIT Roorkee              |
|                                                              |
|  Strategy  : Multi-Factor Equity Long-Short                  |
|  Universe  : Nifty 100 (50 stocks)                           |
|  Signal    : Momentum + RSI + Volume + Volatility            |
|  Regime    : Hidden Markov Model (4 states)                  |
|  Options   : Put-Call Ratio + IV Skew                        |
|  Costs     : Commission + Spread + Almgren-Chriss Slippage   |
+==============================================================+
""")


def timer(label):
    """Simple timing context manager for profiling each step."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"  [OK] {label} completed in {elapsed:.1f}s\n")
    return Timer()


# ---------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------

def step1_data(skip_download=False):
    """
    MODULE 1: Data Download & Cleaning
    Downloads Nifty 100 OHLCV data from Yahoo Finance.
    Cleans, validates, saves to data/ folder.
    Skip with --skip-download if you already have the data.
    """
    print("\n" + "?" * 60)
    print("STEP 1/6 -- DATA LOADER")
    print("?" * 60)

    if skip_download and os.path.exists(f"{DATA_DIR}close.csv"):
        print("[Main] Skipping download -- loading cached data...")
    else:
        with timer("Data download + cleaning"):
            run_data_pipeline(start=START_DATE, end=END_DATE,
                              output_dir=DATA_DIR)

    close, open_, high, low, volume, returns, nifty, vix = load_data(DATA_DIR)
    print(f"[Main] Data loaded: {close.shape[0]} days x {close.shape[1]} stocks")
    return close, open_, high, low, volume, returns, nifty, vix


def step2_signals(close, volume, high=None, low=None):
    """
    MODULE 2+3: Feature Engineering + Signal Generation
    Computes 4-factor composite signal for all stocks.
    Applies 52-week high/low filter and confidence scoring.
    Saves signals.csv, positions.csv, confidence.csv, alpha_decay.csv
    """
    print("\n" + "?" * 60)
    print("STEP 2/6 -- SIGNAL GENERATION")
    print("?" * 60)

    with timer("Multi-factor signal computation"):
        positions, composite_signal, factors = generate_signals(
            close, volume,
            high_prices=high,
            low_prices=low,
            output_dir=OUTPUT_DIR
        )

    return positions, composite_signal, factors


def step3_regime(nifty, vix):
    """
    MODULE 4: Regime Detection
    Trains HMM on Nifty returns + VIX features.
    Outputs regime labels and position scale factors.
    """
    print("\n" + "?" * 60)
    print("STEP 3/6 -- REGIME DETECTION (HMM)")
    print("?" * 60)

    with timer("Hidden Markov Model training + decoding"):
        regime_series, position_scale, regime_stats = detect_regimes(
            nifty_close=nifty,
            vix=vix,
            output_dir=f"{OUTPUT_DIR}regime/"
        )

    # Copy regime outputs to main outputs dir for dashboard
    regime_series.to_csv(f"{OUTPUT_DIR}regime_labels.csv", header=True)
    position_scale.to_csv(f"{OUTPUT_DIR}position_scale.csv", header=True)
    regime_stats.to_csv(f"{OUTPUT_DIR}regime_summary.csv")

    return regime_series, position_scale


def step4_options(nifty):
    """
    MODULE 5: Options Market Signals
    Computes PCR and IV Skew signals.
    Uses simulation mode (real NSE data requires scraping).
    """
    print("\n" + "?" * 60)
    print("STEP 4/6 -- OPTIONS SIGNALS (PCR + IV SKEW)")
    print("?" * 60)

    with timer("Options signal computation"):
        options_df, options_combined = generate_options_signals(
            nifty_close=nifty,
            simulate=True,          # set False if you have real NSE option chain data
            output_dir=f"{OUTPUT_DIR}options/"
        )

    # Copy to main outputs for dashboard
    options_df.to_csv(f"{OUTPUT_DIR}options_signals.csv")

    return options_df, options_combined


def step5_portfolio(positions, returns, position_scale, factors=None):
    """
    MODULE 6: Portfolio Construction
    Converts signals to risk-parity weights x confidence scaling.
    Applies HMM + FII regime scaling.
    Enforces constraints (max position, turnover limit).
    """
    print("\n" + "?" * 60)
    print("STEP 5/6 -- PORTFOLIO CONSTRUCTION")
    print("?" * 60)

    confidence = factors.get("confidence") if factors else None

    with timer("Risk parity weights + confidence + regime scaling"):
        daily_weights, turnover = build_portfolio(
            positions=positions,
            returns=returns,
            position_scale=position_scale,
            confidence=confidence,
            output_dir=OUTPUT_DIR
        )

    return daily_weights, turnover


def step6_backtest(daily_weights, close, returns, volume, nifty):
    """
    MODULE 7: Backtesting Engine
    Simulates strategy with realistic transaction costs.
    Computes all performance metrics.
    Walk-forward validation (train 2018-2021, test 2022-2024).
    """
    print("\n" + "?" * 60)
    print("STEP 6/6 -- BACKTESTING ENGINE")
    print("?" * 60)

    with timer("Backtest simulation + metrics computation"):
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
# FINAL SUMMARY
# ---------------------------------------------

def print_summary(equity_curve, metrics):
    """Print final results summary after full pipeline run."""
    nav = equity_curve["nav"]
    ret = equity_curve["daily_return"]

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret   = (1 + total_ret) ** (252 / len(ret)) - 1
    ann_vol   = ret.std() * np.sqrt(252)
    sharpe    = (ann_ret - 0.065) / ann_vol
    max_dd    = ((nav / nav.cummax()) - 1).min()

    print("""
+==============================================================+
|                    FINAL RESULTS SUMMARY                     |
+==============================================================+""")
    print(f"|  Total Return     : {total_ret:>8.2%}                              |")
    print(f"|  Ann. Return      : {ann_ret:>8.2%}                              |")
    print(f"|  Sharpe Ratio     : {sharpe:>8.3f}                              |")
    print(f"|  Max Drawdown     : {max_dd:>8.2%}                              |")
    print(f"|  Ann. Volatility  : {ann_vol:>8.2%}                              |")
    print("""+==============================================================+
|  All outputs saved to outputs/                               |
|                                                              |
|  Launch dashboard:                                           |
|     streamlit run dashboard/app.py                           |
+==============================================================+""")


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Quant Trading System -- Tech GC 2026")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use cached data/ folder")
    parser.add_argument("--module", type=str, default="all",
                        choices=["all", "data", "signals", "regime",
                                 "options", "portfolio", "backtest"],
                        help="Run only a specific module (for testing)")
    args = parser.parse_args()

    print_banner()
    total_start = time.time()

    try:
        # Always need data first
        close, open_, high, low, volume, returns, nifty, vix = step1_data(
            skip_download=args.skip_download
        )

        if args.module == "data":
            print("[Main] --module data: stopping after data step.")
            return

        positions, composite_signal, factors = step2_signals(close, volume,
                                                              high=high, low=low)

        if args.module == "signals":
            return

        regime_series, position_scale = step3_regime(nifty, vix)

        if args.module == "regime":
            return

        options_df, options_combined = step4_options(nifty)

        if args.module == "options":
            return

        daily_weights, turnover = step5_portfolio(
            positions, returns, position_scale, factors=factors
        )

        if args.module == "portfolio":
            return

        equity_curve, trade_log, metrics = step6_backtest(
            daily_weights, close, returns, volume, nifty
        )

        total_time = time.time() - total_start
        print(f"\n[Main] Total pipeline time: {total_time:.0f}s")

        print_summary(equity_curve, metrics)

    except FileNotFoundError as e:
        print(f"\n[Error] Missing file: {e}")
        print("Run without --skip-download first to generate data.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[Error] Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
