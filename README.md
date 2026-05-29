# Nifty 500 Long-Short Quantitative Strategy

A regime-aware, multi-factor long-short equity trading system built for the Indian market. The strategy systematically trades the Nifty 500 universe by first identifying what kind of market we are in, then sizing positions accordingly — rather than applying the same signal regardless of conditions.

---

## Links

| | |
|---|---|
| **Live Dashboard** | [quantativefinance-lpmrajgmgj6zqvkqt4bf9b.streamlit.app](https://quantativefinance-lpmrajgmgj6zqvkqt4bf9b.streamlit.app) |
| **Demo Video** | [Google Drive](https://drive.google.com/drive/folders/15iqAevku7K1luzZhgJHyYbkZT28rwyv0?usp=sharing) |
| **Technical Report (DOCX)** | [Google Drive](https://drive.google.com/drive/folders/1wrQ46KpxIA-EC_Ue1V-bZQVIcB4lRxlQ?usp=sharing) |
| **Dataset** | Included in repo — `nifty500_daily_5Y_PATCHED.parquet` (499 stocks, 5Y daily) |
| **Source Code** | [github.com/oldhero07/quantative_finance](https://github.com/oldhero07/quantative_finance) |

---

## What Makes This Different

Most quant systems apply a fixed signal in all market conditions. This one doesn't.

### 1. Regime-Aware Position Sizing (HMM + FII/DII Overlay)

A 4-state Gaussian Hidden Markov Model is trained on six daily market features — return, short and long-term volatility, trend, volatility ratio, and VIX. It learns to identify when the market is in a Bull, Bear, Sideways, or Crisis state, and position sizes scale automatically:

| Regime | % of History | Position Scale |
|--------|-------------|----------------|
| Bull | 35.9% | 1.00x — full exposure |
| Sideways | 45.3% | 0.40x — signal quality drops in choppy markets |
| Crisis | 18.8% | 0.15x — capital preservation mode |
| Bear | 0.1% | 0.70x — directional risk elevated |

On top of the HMM, an FII/DII institutional flow overlay adds a second scaling layer. When foreign institutional buying is strong, positions scale up; when net flow is bearish, positions scale down further. The final size = HMM scale × FII multiplier.

This two-layer regime filter is what separates this system from a naive momentum strategy. During the 2022 drawdown, the Crisis classification automatically cut gross exposure to 15% while trend-following systems stayed fully invested.

### 2. 52-Week Proximity Filter

Before any signal becomes a position, it must pass a structural filter:
- **Longs** are only taken when a stock is within 15% of its 52-week high — confirming structural strength
- **Shorts** are only taken when a stock is within 15% of its 52-week low — confirming structural weakness

This removes roughly 62% of raw signals and keeps only setups where the market structure agrees with the factor score. It significantly reduces false positives compared to a pure factor-rank approach.

### 3. Per-Signal Confidence Scoring

Every signal that survives the 52-week filter receives a confidence score from 0 to 85 based on:
- Momentum rank percentile across the universe
- Agreement across all four factors (not just one firing)
- Volume confirmation (are traders acting on this move?)
- RSI zone quality
- Proximity to the 52-week extreme

Position size then scales as `0.5 + 0.5 × (confidence / 85)` — high-conviction signals get up to 2× the allocation of marginal ones.

### 4. Almgren-Chriss Slippage Model

Transaction costs use the square-root market impact model rather than a flat bps assumption:

```
slippage = σ × √(order_size / ADV) × 0.10
```

Larger orders in less-liquid stocks get penalised proportionally. This matters for Indian mid and small caps where liquidity is thin and flat-cost assumptions significantly underestimate real drag.

### 5. Options Market Overlay (PCR + IV Skew)

Put-Call Ratio and IV Skew from Nifty options are computed as confirmation signals:
- High PCR (> 1.2) signals market fear — contrarian buy confirmation
- High IV Skew (> 0.08) signals crash risk pricing — reduce long exposure

These are used as filters on equity signals rather than standalone trades.

---

## Strategy Pipeline

```
data_loader.py        →  499 stocks, 5Y daily OHLCV from parquet
        ↓
signals.py            →  4-factor composite + 52wk filter + confidence score
        ↓
regime_detection.py   →  HMM (4 states) × FII/DII overlay → position scale
        ↓
options_signal.py     →  PCR + IV skew confirmation signals
        ↓
portfolio.py          →  Inverse-vol risk parity × confidence × regime scale
        ↓
backtest.py           →  Walk-forward simulation + realistic costs + metrics
        ↓
dashboard/app.py      →  Streamlit interactive dashboard
```

---

## Signal Factors

| Factor | Weight | Logic |
|--------|--------|-------|
| Momentum | 40% | 12-month return minus 1-month reversal, cross-sectionally z-scored |
| RSI Mean-Reversion | 25% | Long bias RSI < 45, short bias RSI > 55 |
| Volume Confirmation | 20% | 5-day vs 20-day average volume ratio |
| Volatility Penalty | 15% | Negative weight on realised volatility |

All four factors are z-scored cross-sectionally each day before combining.

---

## Backtest Results

Walk-forward validation — no look-ahead bias. Train: 2021–2022, Test OOS: 2023–2026.

| Metric | Full Period | OOS Test |
|--------|------------|----------|
| Total Return | +10.3% | +10.3% |
| Annual Return | +2.0% | +2.4% |
| Annual Volatility | 8.54% | 9.25% |
| Sharpe Ratio | -0.53 | -0.45 |
| Max Drawdown | -22.7% | -22.7% |
| Win Rate | 44.8% | 52.6% |

Transaction costs account for ~3.7%/year drag (commission + bid-ask + market impact). The Information Coefficient is positive at all horizons from 5 to 21 days — the gross alpha is real; net Sharpe is compressed by costs at the 50-stock subset used in testing.

---

## Project Structure

```
.
├── main.py                  # Full pipeline entry point
├── quick_test.py            # End-to-end validation — runs in ~15 seconds
├── data_loader.py           # Data ingestion from parquet, cleaning, returns
├── signals.py               # Multi-factor signal generation + filters + scoring
├── regime_detection.py      # HMM regime model + FII/DII overlay
├── options_signal.py        # PCR and IV skew signals
├── portfolio.py             # Risk parity portfolio construction
├── backtest.py              # Backtesting engine with realistic cost model
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
├── outputs/test/            # Pre-computed results (powers the live dashboard)
├── data/test/               # Pre-computed market data
└── nifty500_daily_5Y_PATCHED.parquet   # 499 stocks, 5Y daily OHLCV (11MB)
```

---

## Quickstart

```bash
git clone https://github.com/ZethetaIntern/quantative_finance
cd quantative_finance
pip install -r requirements.txt

# Validate full pipeline end-to-end (~15 seconds)
python quick_test.py --stocks 50

# Launch interactive dashboard
streamlit run dashboard/app.py
```

---

## Dependencies

```
pandas, numpy, scipy, scikit-learn
hmmlearn
yfinance, pyarrow
streamlit, plotly
```

Python 3.10+ recommended. Full list in `requirements.txt`.
