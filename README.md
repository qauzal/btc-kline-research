# BTC/USDT 1-second klines: data pipeline and research

Take-home project: retrieve BTC/USDT 1-second kline data since January 2023 with Python and public APIs, then build and backtest a strategy on it.

## Results at a glance

| | |
|---|---|
| Data | 122,160,829 one-second bars, 1 Jan 2023 – 14 Nov 2026, every file SHA-256 verified, 99.993% coverage |
| Modelling frequency | 2,036,080 clean 1-minute bars; target = next-minute log return |
| Method | Weekly walk-forward; 27 candidates (elastic net at 3 penalties and elastic net on PCA factors, each on 5 training windows; boosted trees at 2 depths on 3 windows; a zero forecast) chosen each week by ATOMS (Capponi et al., 2026) on purged, embargoed validation blocks |
| Strategy (2024-01-01 onward, out of sample) | Daily-P&L Sharpe 6.5 with sign positions, 10.1 with volatility targeting; buy-and-hold 0.3 |
| Leakage audit | 11 checks pass, including a full truncation test (forecasts identical after deleting all later data) and a 500-draw block-permutation test (p = 0.002) |
| What the edge is | About 0.07 bp per minute against 6.7 bp of per-minute volatility. It lives inside the next minute and is gone with a one-minute execution delay. |

## Notebooks

Read them in order; each one opens with what it does and ends with a summary of its results.

| Notebook | What it does |
|---|---|
| [`01_data_pipeline`](notebooks/01_data_pipeline.ipynb) | Downloads every month from Binance's public archive, verifies SHA-256 checksums, stores Parquet, audits data quality, builds clean 1-minute bars |
| [`02_one_second_patterns`](notebooks/02_one_second_patterns.ipynb) | Intraday seasonality, 1-second autocorrelation, the volatility signature, fat tails, volatility clustering and order-flow imbalance |
| [`03_strategy_atoms`](notebooks/03_strategy_atoms.ipynb) | Baseline: stationarity tests, HMM regimes, 13 features, walk-forward ATOMS model selection with purged validation, sign positions |
| [`04_features_pca_sizing`](notebooks/04_features_pca_sizing.ipynb) | Extension: deseasonalised regimes, 64 causal features, Bonferroni screening, PCA, time-of-day test, execution delay, position sizing |
| [`05_leakage_audit`](notebooks/05_leakage_audit.ipynb) | 13 checks, layer by layer, that the result is not look-ahead or a backtest artefact |

## Run it

```bash
pip install -r requirements.txt
jupyter lab            # open the notebooks in order
```

Notebook 01 downloads about 3.6 GB of zips and needs about 10 GB of free disk. It can be interrupted and re-run: files already downloaded and verified are skipped. The saved outputs use data up to 2026-11-14; re-running notebook 01 later extends the data, which changes every downstream number slightly.

Notebooks 03 and 04 cache their walk-forward results in `data/processed/`; set `RERUN = True` to recompute (a few hours per seed on a laptop). Notebook 05 reads those caches and writes nothing.

## Layout

```
src/btc/data.py           download, checksum, parse, clean, resample
src/btc/quality.py        data-quality checks and gap finder
src/btc/patterns.py       month-by-month 1-second statistics
src/btc/features.py       baseline features (13)
src/btc/features_full.py  extended features (64)
src/btc/splits.py         weekly periods and the purged, embargoed validation split
src/btc/models.py         candidate pipelines (scaler, clip, optional PCA, model)
src/btc/atoms.py          ATOMS tournament and the fixed-horizon benchmarks
src/btc/walkforward.py    the walk-forward loop (resumable)
src/btc/regimes.py        forward-filtered HMM regimes
src/btc/diagnostics.py    feature screening, PCA, Diebold-Mariano test
src/btc/sizing.py         position sizing, execution delay, performance metrics
src/btc/evaluate.py       OOS R², bootstrap Sharpe interval
src/btc/plotting.py       chart style
scripts/leakage_audit.py  the leakage checks as a standalone script
notebooks/                the analysis, in order
data/                     created on first run (not committed)
```

## Data sources

All public, no API key.

- Bulk archive: `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/`
- Daily archive: `https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/` (days after the last monthly file)
- REST API: `https://data-api.binance.vision/api/v3/klines` (only for days whose daily file is not published yet)

## Reference

ATOMS (Adaptive Tournament Model Selection): Capponi, Huang, Sidaoui, Wang & Zou (2026).
