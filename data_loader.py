"""
data_loader.py -- Data & Feature Engineering Module
===================================================
Tech GC 2026 | High Prep Problem Solving | IIT Roorkee

What this does:
    PRIMARY: Loads Nifty 500 data from the pre-built parquet file (fast, offline).
    FALLBACK: Downloads a curated set of large-cap Nifty 500 stocks from Yahoo Finance (free).
    Cleans it: fills gaps, removes bad data, adjusts for splits.
    Saves clean OHLCV CSVs that every other module reads from.

Run standalone:
    python data_loader.py

Output files (saved to data/ folder):
    close.csv   -- daily closing prices for all stocks
    open.csv    -- daily open prices
    high.csv    -- daily high prices
    low.csv     -- daily low prices
    volume.csv  -- daily trading volume
    returns.csv -- daily % returns (computed from close)
    nifty.csv   -- Nifty 50 index close prices
    vix.csv     -- India VIX daily values
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------
# CONFIGURATION -- edit these as needed
# ---------------------------------------------

START_DATE = "2021-01-01"
END_DATE   = "2026-04-06"
DATA_DIR   = "data/"

# Path to the pre-built Nifty 500 parquet file (5 years, 499 stocks)
# This is the primary data source -- much faster than downloading
PARQUET_PATH = "nifty500_daily_5Y_PATCHED.parquet"

# Fallback tickers (large-cap Nifty 500 stocks) on Yahoo Finance (NSE format: ticker.NS)
# Used ONLY if the parquet file is not found
NIFTY50_FALLBACK_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS", "NESTLEIND.NS", "WIPRO.NS", "ADANIPORTS.NS",
    "TECHM.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "GRASIM.NS", "BAJAJFINSV.NS",
    "INDUSINDBK.NS", "HEROMOTOCO.NS", "BRITANNIA.NS", "EICHERMOT.NS", "BPCL.NS",
    "SHREECEM.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "PIDILITIND.NS", "DABUR.NS",
    "GODREJCP.NS", "HAVELLS.NS", "BERGEPAINT.NS", "MCDOWELL-N.NS", "COLPAL.NS",
]

# Minimum data quality thresholds
MIN_HISTORY_DAYS = 200      # drop stocks with less than this many trading days
MAX_MISSING_PCT  = 0.05     # drop stocks with more than 5% missing values


# ---------------------------------------------
# PRIMARY: LOAD FROM PARQUET (Nifty 500, 5Y)
# ---------------------------------------------

def load_from_parquet(parquet_path=PARQUET_PATH):
    """
    Load Nifty 500 OHLCV data from the pre-built parquet file.

    The parquet has long format: Date, Open, High, Low, Close, Volume, Symbol.
    We pivot it to wide format: rows=dates, columns=stock tickers.
    This matches the format expected by all downstream modules.

    Args:
        parquet_path : str -- path to the .parquet file

    Returns:
        raw : dict with keys "Close", "Open", "High", "Low", "Volume"
              each value is a DataFrame (rows=dates, columns=tickers)
    """
    print(f"[Data] Loading Nifty 500 data from parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Parse dates -- remove timezone info for consistency
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()   # strip time component -> date only

    print(f"[Data] Raw parquet: {len(df):,} rows, {df['Symbol'].nunique()} symbols")
    print(f"[Data] Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    raw = {}
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        pivoted = df.pivot_table(index="Date", columns="Symbol", values=col, aggfunc="last")
        pivoted.index = pd.DatetimeIndex(pivoted.index)
        raw[col] = pivoted

    print(f"[Data] Pivoted shape: {raw['Close'].shape[0]} days x {raw['Close'].shape[1]} stocks")
    return raw


# ---------------------------------------------
# FALLBACK: DOWNLOAD FROM YAHOO FINANCE
# ---------------------------------------------

def download_stock_data(tickers, start, end):
    """
    Download OHLCV data for all tickers from Yahoo Finance.

    Uses yfinance batch download for speed.
    auto_adjust=True handles stock splits and dividends automatically.

    Args:
        tickers : list of ticker strings
        start   : start date string "YYYY-MM-DD"
        end     : end date string "YYYY-MM-DD"

    Returns:
        raw : dict with keys "Close", "Open", "High", "Low", "Volume"
              each value is a DataFrame (rows=dates, columns=tickers)
    """
    print(f"[Data] Downloading {len(tickers)} stocks: {start} -> {end}")
    print("[Data] Source: Yahoo Finance (free, no API key needed)")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,     # adjusts for splits + dividends automatically
        progress=True,
        group_by="column",
        threads=True
    )

    # Extract each price type
    raw = {}
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col in data.columns.get_level_values(0):
            raw[col] = data[col]
        elif col in data.columns:
            raw[col] = data[[col]]

    print(f"[Data] Downloaded: {raw['Close'].shape[0]} days x {raw['Close'].shape[1]} stocks")
    return raw


def download_index_data(start, end):
    """
    Download Nifty 50 index and India VIX separately.
    These are used by the regime detection module.

    Returns:
        nifty : pd.Series of Nifty 50 close prices
        vix   : pd.Series of India VIX values (or None if unavailable)
    """
    print("[Data] Downloading Nifty 50 index...")
    nifty_data = yf.download("^NSEI", start=start, end=end,
                              auto_adjust=True, progress=False)
    nifty = nifty_data["Close"].squeeze()
    nifty.name = "NIFTY50"

    print("[Data] Downloading India VIX...")
    try:
        vix_data = yf.download("^INDIAVIX", start=start, end=end,
                               auto_adjust=True, progress=False)
        vix = vix_data["Close"].squeeze()
        vix.name = "INDIAVIX"
        print(f"[Data] VIX downloaded: {len(vix)} days")
    except Exception as e:
        print(f"[Data] VIX download failed ({e}). Continuing without VIX.")
        vix = None

    return nifty, vix


# ---------------------------------------------
# CLEANING
# ---------------------------------------------

def clean_data(raw):
    """
    Clean raw downloaded data.

    Steps:
        1. Remove stocks with too little history
        2. Remove stocks with too many missing values
        3. Forward-fill small gaps (up to 3 days -- e.g. trading halts)
        4. Drop remaining NaNs
        5. Remove obvious outliers (returns > 50% in one day)

    Why forward-fill and not just drop NaN?
        If a stock doesn't trade on one day (trading halt, holiday),
        its price didn't change -- carry forward the last known price.
        But we cap at 3 days to avoid stale prices from real delistings.

    Args:
        raw : dict from download_stock_data()

    Returns:
        clean : dict with same structure, cleaned DataFrames
    """
    print("[Data] Cleaning data...")

    close = raw["Close"].copy()
    n_days = len(close)

    # Step 1: Remove stocks with insufficient history
    valid_days = close.notna().sum()
    enough_history = valid_days[valid_days >= MIN_HISTORY_DAYS].index
    print(f"[Data] Stocks with enough history (>{MIN_HISTORY_DAYS} days): {len(enough_history)}")

    # Step 2: Remove stocks with too many missing values
    missing_pct = close[enough_history].isna().mean()
    low_missing = missing_pct[missing_pct <= MAX_MISSING_PCT].index
    print(f"[Data] Stocks passing quality filter: {len(low_missing)}")

    # Apply filter to all price types
    clean = {}
    for col, df in raw.items():
        filtered = df[low_missing] if all(t in df.columns for t in low_missing) else df
        # Step 3: Forward fill gaps (max 3 consecutive days)
        filled = filtered.ffill(limit=3)
        # Step 4: Drop remaining NaN rows (early dates where some stocks didn't exist)
        clean[col] = filled

    # Step 5: Remove extreme outliers in returns
    ret = clean["Close"].pct_change()
    # Flag cells where return > 75% or < -75% as bad data
    bad_mask = ret.abs() > 0.75
    for col in ["Close", "Open", "High", "Low"]:
        if col in clean:
            clean[col][bad_mask.shift(-1).fillna(False)] = np.nan
            clean[col] = clean[col].ffill(limit=1)

    print(f"[Data] Final dataset: {clean['Close'].shape[0]} days x {clean['Close'].shape[1]} stocks")
    print(f"[Data] Date range: {clean['Close'].index[0]} -> {clean['Close'].index[-1]}")

    return clean


# ---------------------------------------------
# COMPUTE RETURNS
# ---------------------------------------------

def compute_returns(close):
    """
    Compute daily percentage returns from closing prices.

    Formula: return_t = (close_t - close_{t-1}) / close_{t-1}

    This is the fundamental unit of analysis in quantitative finance.
    All risk metrics (volatility, Sharpe, VaR) are computed from returns.

    Args:
        close : pd.DataFrame of close prices

    Returns:
        returns : pd.DataFrame of daily returns (first row is NaN)
    """
    returns = close.pct_change()
    print(f"[Data] Returns computed: mean={returns.mean().mean()*100:.3f}%/day")
    return returns


# ---------------------------------------------
# SAVE DATA
# ---------------------------------------------

def save_data(clean, returns, nifty, vix, output_dir=DATA_DIR):
    """
    Save all cleaned data to CSV files.

    Files saved:
        close.csv, open.csv, high.csv, low.csv, volume.csv
        returns.csv
        nifty.csv
        vix.csv (if available)

    CSV format: rows=dates (DatetimeIndex), columns=stock tickers.
    This format is directly readable by all other modules.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, df in clean.items():
        path = f"{output_dir}{name.lower()}.csv"
        df.to_csv(path)
        print(f"[Data] Saved -> {path}")

    returns.to_csv(f"{output_dir}returns.csv")
    print(f"[Data] Saved -> {output_dir}returns.csv")

    nifty.to_csv(f"{output_dir}nifty.csv", header=True)
    print(f"[Data] Saved -> {output_dir}nifty.csv")

    if vix is not None:
        vix.to_csv(f"{output_dir}vix.csv", header=True)
        print(f"[Data] Saved -> {output_dir}vix.csv")

    print(f"\n[Data] All data saved to {output_dir}")


# ---------------------------------------------
# LOAD DATA (called by other modules)
# ---------------------------------------------

def load_data(data_dir=DATA_DIR):
    """
    Load previously downloaded and cleaned data from CSV files.

    Call this from other modules instead of re-downloading every time.
    Much faster than downloading -- use after first run.

    Returns:
        close, open_, high, low, volume, returns : DataFrames
        nifty : pd.Series
        vix   : pd.Series or None
    """
    def read(name):
        path = f"{data_dir}{name}.csv"
        return pd.read_csv(path, index_col=0, parse_dates=True)

    close   = read("close")
    open_   = read("open")
    high    = read("high")
    low     = read("low")
    volume  = read("volume")
    returns = read("returns")
    nifty   = pd.read_csv(f"{data_dir}nifty.csv",
                          index_col=0, parse_dates=True).squeeze()

    vix_path = f"{data_dir}vix.csv"
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).squeeze() \
          if os.path.exists(vix_path) else None

    print(f"[Data] Loaded: {close.shape[0]} days x {close.shape[1]} stocks")
    return close, open_, high, low, volume, returns, nifty, vix


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def run_data_pipeline(tickers=None, start=START_DATE, end=END_DATE,
                      output_dir=DATA_DIR, use_parquet=True):
    """
    Run full data download + clean + save pipeline.

    Priority:
        1. If parquet file exists and use_parquet=True -> load from parquet (fast)
        2. Otherwise -> download from Yahoo Finance (slow, requires internet)

    Call from main.py or run this file directly.
    """
    print("=" * 60)
    print("DATA LOADER MODULE")
    print("=" * 60)

    # -- Try parquet first --
    if use_parquet and os.path.exists(PARQUET_PATH):
        print(f"[Data] Using pre-built parquet: {PARQUET_PATH}")
        print(f"[Data] Universe : Nifty 500 (499 stocks)")
        raw = load_from_parquet(PARQUET_PATH)
    else:
        # Fallback to Yahoo Finance download
        if tickers is None:
            tickers = NIFTY50_FALLBACK_TICKERS
        print(f"[Data] Parquet not found -> downloading from Yahoo Finance")
        print(f"[Data] Universe : {len(tickers)} stocks")
        print(f"[Data] Period   : {start} -> {end}")
        raw = download_stock_data(tickers, start, end)

    nifty, vix = download_index_data(start, end)
    clean      = clean_data(raw)
    returns    = compute_returns(clean["Close"])
    save_data(clean, returns, nifty, vix, output_dir)

    print("\n[Data] Pipeline complete.")
    return clean, returns, nifty, vix


if __name__ == "__main__":
    run_data_pipeline()
