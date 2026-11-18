"""Leakage audit for the notebook-04 walk-forward results.

Run from the project root after notebook 04 has produced data/processed/wf_v2_*_seed0.parquet:

    python scripts/leakage_audit.py            # fast checks (about 2 minutes)
    python scripts/leakage_audit.py --truncate # also re-run one test week with all later data deleted (about 5 minutes)

Every check prints what it compares and the result. Nothing is written to data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btc import features_full, models, regimes, sizing, splits, walkforward  # noqa: E402

P = ROOT / "data" / "processed"
REPORT_FROM = pd.Timestamp("2024-01-01", tz="UTC")


def sharpe(pnl: pd.Series) -> float:
    d = pnl.groupby(pnl.index.floor("D")).sum()
    return d.mean() / d.std() * np.sqrt(365)


def fast_checks(m1: pd.DataFrame, fc: pd.DataFrame, raw_month: str = "2025-03") -> None:
    print("== 1. bar alignment (one month of raw 1-second bars vs the 1-minute bars)")
    s = pd.read_parquet(ROOT / "data" / "parquet" / f"BTCUSDT-1s-{raw_month}.parquet")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    mm = m1[(m1.index >= s.index.min().floor("D")) & (m1.index <= s.index.max())]
    chk = s.groupby(s.index.floor("1min")).agg(o=("open", "first"), c=("close", "last"), h=("high", "max"), l=("low", "min"))
    j = mm.join(chk, how="inner")
    print("   1-minute close == last 1-second close:", np.allclose(j["close"], j["c"]))
    print("   1-minute open  == first 1-second open:", np.allclose(j["open"], j["o"]))
    print("   high/low match:", np.allclose(j["high"], j["h"]) and np.allclose(j["low"], j["l"]))

    print("== 2. cached target vs the 1-minute bars")
    grid = pd.date_range(m1.index.min(), m1.index.max(), freq="1min")
    lg = np.log(m1["close"]).reindex(grid)
    rf = (lg.shift(-1) - lg).reindex(fc.index)
    print("   y == 1e4*(log close[i+1] - log close[i]):", np.allclose(fc["y"], 1e4 * rf, atol=1e-6))
    print("   duplicate timestamps:", int(fc.index.duplicated().sum()))
    print("   period label == timestamp's week:", bool((fc.index.tz_convert(None).to_period("W-SUN").astype(str) == fc["period"]).all()))

    print("== 3. where the P&L comes from (report period, ATOMS sign positions)")
    rep = fc[fc.index >= REPORT_FROM]
    pos = sizing.sign_pos(rep["atoms"])
    op = np.log(m1["open"]).reindex(grid)
    cl, nxt_cl, nxt_op = lg.reindex(rep.index), lg.shift(-1).reindex(rep.index), op.shift(-1).reindex(rep.index)
    gap, rest, full = pos * (nxt_op - cl), pos * (nxt_cl - nxt_op), pos * (nxt_cl - cl)
    print(f"   close(i)->close(i+1), the backtest:              Sharpe {sharpe(full):.2f}")
    print(f"   close(i)->open(i+1), what last-print bounce earns: Sharpe {sharpe(gap):.2f}, {gap.sum() / full.sum():.1%} of P&L")
    print(f"   open(i+1)->close(i+1), entry at the next first trade: Sharpe {sharpe(rest):.2f}")
    q = pd.qcut(rep["atoms"].abs(), 5, labels=False, duplicates="drop")
    print("   mean P&L (bp) by |forecast| quintile:", (pos * rep["y"]).groupby(q).mean().round(4).tolist())

    print("== 4. inside minute i+1 on raw 1-second bars: cumulative P&L from the first trade to second k")
    lc = np.log(s["close"]); mins = s.index.floor("1min")
    secs = [0, 2, 5, 10, 20, 30, 45, 59]
    tab = pd.DataFrame({k: lc[s.index.second == k].set_axis(mins[s.index.second == k]) for k in secs})
    tab["open"] = np.log(s["open"])[s.index.second == 0].set_axis(mins[s.index.second == 0])
    p = pos.reindex(tab.index - pd.Timedelta(minutes=1)).set_axis(tab.index).fillna(0)
    print("   " + "  ".join(f"s{k}: {1e4 * (p * (tab[k] - tab['open'])).mean():+.4f}" for k in secs) + "  (bp/min, " + raw_month + ")")

    print("== 5. volatility targeting with the volatility estimate lagged")
    r = lg.diff(); rv60 = np.sqrt(r.pow(2).rolling(60, min_periods=48).mean())
    for lag in (0, 1, 60):
        res = sizing.run(sizing.vol_target_pos(rep["atoms"], rv60.shift(lag).reindex(rep.index), 0.40, 2.0), rep["ret_fwd"])
        print(f"   vol lagged {lag:>2} min: Sharpe {sizing.perf(res)['sharpe_daily']:.2f}")


def permutation_test(fc: pd.DataFrame, n_boot: int = 500, block: int = 10080, seed: int = 42) -> None:
    """Shuffle the forecast series in week-long blocks (preserving its own autocorrelation,
    destroying its alignment with returns) and rebuild the sign backtest each time.
    If shuffled signals also earned a high Sharpe, the evaluation itself would be biased."""
    print("== 6. block-permutation test")
    rep = fc[fc.index >= REPORT_FROM]
    f, r = rep["atoms"].to_numpy(), rep["ret_fwd"].to_numpy()
    n = len(f)
    real = sharpe(pd.Series(np.sign(f) * r, index=rep.index))
    rng = np.random.default_rng(seed)
    days = rep.index.floor("D")
    null = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        d = pd.Series(np.sign(f[idx]) * r, index=rep.index).groupby(days).sum()
        null[b] = d.mean() / d.std() * np.sqrt(365)
    print(f"   real Sharpe {real:.2f}; null mean {null.mean():+.3f}, sd {null.std():.3f}, "
          f"max of {n_boot} draws {null.max():.2f}; p-value {(null >= real).mean():.4f}; "
          f"z = {(real - null.mean()) / null.std():.1f} sd above the null")


def truncation_test(m1: pd.DataFrame, fc: pd.DataFrame, week: str = "2024-06-10/2024-06-16") -> None:
    """Delete every bar after the test week, rebuild regimes, features, split and all
    candidate fits for that week only, and compare with the cached forecasts."""
    print(f"== 7. truncation test, week {week}")
    w_end = pd.Timestamp(week.split("/")[1], tz="UTC") + pd.Timedelta(days=1)
    m1t = m1[m1.index < w_end]
    h = regimes.hourly_inputs(m1t, design_end=REPORT_FROM, smooth_hours=24)
    hmm = regimes.fit_hmm(h[h.index < REPORT_FROM])
    F, groups = features_full.build_full(m1t, horizon=1, p_storm=regimes.lagged(regimes.regime_series(hmm, h)), spectral=True)
    cols = [c for v in groups.values() for c in v]
    frame = splits.split_all(F[cols + ["y", "ret_fwd"]], "W-SUN", 360, 0.20, 60, 15, 0)
    settings = [{"kind": "enet", "alpha": a, "l1_ratio": 0.5} for a in (0.001, 0.01, 0.1)] + \
               [{"kind": "enet", "alpha": 0.01, "l1_ratio": 0.5, "pca": 0.95},
                {"kind": "gbt", "depth": 2, "lr": 0.05, "n_iter": 150, "min_leaf": 1000}]
    cands = models.candidate_grid(settings, [1, 4, 16, 64, 0], 16)
    out = walkforward.run(frame, cands, min_history=4, max_lookback=64, fh_short=4, atoms_delta=0.1, atoms_M=None,
                          seed=0, verbose=False, start_period=pd.Period(week, freq="W-SUN"), feature_cols=cols)
    new = out["forecasts"]; new = new[new["period"] == week]
    old = fc.loc[new.index]
    worst = max((new[c.label] - old[c.label]).abs().max() for c in cands if c.kind != "zero")
    print(f"   {len(new)} test rows, {len(cands) - 1} candidates; largest |forecast difference| vs cached: {worst:.2e}")


if __name__ == "__main__":
    m1 = pd.read_parquet(P / "BTCUSDT-1m.parquet")
    fc = pd.read_parquet(P / "wf_v2_forecasts_seed0.parquet")
    fast_checks(m1, fc)
    permutation_test(fc)
    if "--truncate" in sys.argv:
        truncation_test(m1, fc)
