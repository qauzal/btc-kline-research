"""The extended feature set: ~60 causal features in 11 groups, from 1-minute bars.

Rule for every feature: the value at minute i uses bars up to and including i.
Rolling windows tolerate up to 20% missing minutes, so a short exchange outage
does not wipe out a day of rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

GROUP_DOC = {
    "momentum":    "log returns over the last 1 minute to 1 day",
    "volatility":  "realised and high-low (Parkinson) volatility, short vs long volatility ratio",
    "scaled":      "returns divided by recent volatility (how unusual the move is)",
    "location":    "distance from moving averages and position in the recent high-low range",
    "moments":     "rolling skewness and kurtosis of 1-minute returns",
    "dependence":  "rolling lag-1 autocorrelation and 5-minute variance ratio",
    "activity":    "volume, trade count and average trade size vs their recent norm",
    "order_flow":  "taker-buy imbalance, volume-weighted imbalance, flow-return correlation, illiquidity",
    "time":        "hour of day (3 harmonics: 24h, 12h, 8h cycles), day of week, weekend",
    "spectral":    "spectral entropy and low-frequency power share of the last 64 returns",
    "regime":      "lagged HMM storm probability and its interactions with key signals",
}


def _roll(x: pd.Series, k: int):
    return x.rolling(k, min_periods=max(2, int(0.8 * k)))


def _moments(x: pd.Series, k: int):
    """Rolling skewness and excess kurtosis from rolling raw moments.

    pandas' own rolling().skew() / .kurt() return NaN for every window after a
    block of missing values (seen in pandas 3.0), so they are computed here
    from rolling means of x, x^2, x^3, x^4, which handle gaps correctly.
    """
    m1, m2, m3, m4 = (_roll(x ** j, k).mean() for j in (1, 2, 3, 4))
    var = m2 - m1 ** 2
    sd = np.sqrt(var.where(var > 0))
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / sd ** 3
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / var ** 2 - 3
    return skew, kurt


def _spectral(r: np.ndarray, w: int = 64, chunk: int = 200_000):
    """Spectral entropy (0 = one dominant cycle, 1 = white noise) and the share of
    power in the lowest quarter of frequencies, over the trailing w returns."""
    n = len(r)
    valid = pd.Series(~np.isnan(r)).rolling(w, min_periods=1).sum().to_numpy() >= 0.8 * w
    x = np.nan_to_num(r, nan=0.0)
    ent, low = np.full(n, np.nan), np.full(n, np.nan)
    nb = w // 2
    for s in range(w - 1, n, chunk):
        e = min(n, s + chunk)
        win = sliding_window_view(x[s - w + 1:e], w)            # window j ends at minute s + j
        win = win - win.mean(1, keepdims=True)
        pw = np.abs(np.fft.rfft(win, axis=1))[:, 1:] ** 2
        tot = pw.sum(1, keepdims=True)
        tot[tot == 0] = np.nan
        q = pw / tot
        ent[s:e] = -(q * np.log(q + 1e-12)).sum(1) / np.log(nb)
        low[s:e] = q[:, : nb // 4].sum(1)
    ent[~valid], low[~valid] = np.nan, np.nan
    return ent, low


def build_full(m1: pd.DataFrame, horizon: int = 1, min_obs: int = 50,
               p_storm: pd.Series | None = None, spectral: bool = True):
    """Return (frame, groups). `frame` has every feature plus `y` (target, bps) and
    `ret_fwd`. `groups` maps each group name to its column names.

    `p_storm` must already be lagged so its timestamps are when it became known
    (see regimes.lagged). If None, the regime group is skipped.
    """
    m1 = m1[m1["n_obs"] >= min_obs]
    grid = pd.date_range(m1.index.min(), m1.index.max(), freq="1min", name="time")
    d = m1.reindex(grid)
    logp = np.log(d["close"]); r = logp.diff(); r2 = r ** 2
    v, q, n, tb = d["volume"], d["quote_volume"], d["trades"], d["taker_buy_volume"]
    f, g = {}, {k: [] for k in GROUP_DOC}

    def add(group, name, values):
        f[name] = values; g[group].append(name)

    # momentum / reversal
    for k in (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 720, 1440):
        add("momentum", f"ret_{k}", logp.diff(k))
    # volatility
    rv = {k: np.sqrt(_roll(r2, k).mean() * k) for k in (5, 15, 60, 240, 1440)}
    for k, x in rv.items():
        add("volatility", f"rv_{k}", x)
    hl2 = np.log(d["high"] / d["low"]) ** 2 / (4 * np.log(2))
    for k in (15, 60):
        add("volatility", f"pk_{k}", np.sqrt(_roll(hl2, k).mean() * k))
    add("volatility", "vol_ratio_15_240", np.log((rv[15] / np.sqrt(15)) / (rv[240] / np.sqrt(240))))
    add("volatility", "range_1", np.log(d["high"] / d["low"]))
    # scaled returns
    sig = rv[60] / np.sqrt(60)                                    # per-minute volatility
    for k in (5, 15, 60):
        add("scaled", f"zret_{k}", logp.diff(k) / (sig * np.sqrt(k)))
    add("scaled", "zret_240", logp.diff(240) / rv[240])
    # location
    for k in (60, 240, 1440):
        add("location", f"dist_ma_{k}", (logp - _roll(logp, k).mean()) / (sig * np.sqrt(k)))
    for k in (60, 1440):
        hi, lo = _roll(logp, k).max(), _roll(logp, k).min()
        add("location", f"range_pos_{k}", ((logp - lo) / (hi - lo)).where(hi > lo) - 0.5)
    # moments
    sk, ku = _moments(1e4 * r, 240)
    add("moments", "skew_240", sk)
    add("moments", "kurt_240", ku)
    # dependence
    add("dependence", "ac1_240", _roll(r, 240).corr(r.shift(1)))
    add("dependence", "vr5_240", np.log(_roll(logp.diff(5) ** 2, 240).mean() / (5 * _roll(r2, 240).mean())))
    # activity
    lv, lt = np.log1p(v), np.log1p(n)
    lsz = np.log((v / n).where(n > 0))
    for name, x, k in (("vol_z", lv, 1440), ("vol_z_60", lv, 60), ("trades_z", lt, 1440), ("size_z", lsz, 1440)):
        add("activity", name, (x - _roll(x, k).mean()) / _roll(x, k).std())
    add("activity", "vol_mom_15_240", np.log(_roll(v, 15).mean() / _roll(v, 240).mean()))
    # order flow
    imb = (2 * tb / v - 1).where(v > 0)
    sv = 2 * tb - v                                               # signed BTC volume
    add("order_flow", "imb_1", imb)
    for k in (5, 15, 60):
        add("order_flow", f"imb_{k}", _roll(imb, k).mean())
    for k in (15, 60):
        add("order_flow", f"vw_imb_{k}", _roll(sv, k).sum() / _roll(v, k).sum())
    add("order_flow", "flow_corr_240", _roll(r, 240).corr(sv))
    add("order_flow", "amihud_60", np.log(_roll((r.abs() / q).where(q > 0), 60).mean() + 1e-18))
    # time
    hrs = grid.hour + grid.minute / 60
    for k in (1, 2, 3):
        add("time", f"hour_sin_{k}", pd.Series(np.sin(2 * np.pi * k * hrs / 24), index=grid))
        add("time", f"hour_cos_{k}", pd.Series(np.cos(2 * np.pi * k * hrs / 24), index=grid))
    dow = grid.dayofweek + hrs / 24
    add("time", "dow_sin", pd.Series(np.sin(2 * np.pi * dow / 7), index=grid))
    add("time", "dow_cos", pd.Series(np.cos(2 * np.pi * dow / 7), index=grid))
    add("time", "weekend", pd.Series((grid.dayofweek >= 5).astype(float), index=grid))
    # spectral
    if spectral:
        ent, low = _spectral(r.to_numpy())
        add("spectral", "spec_entropy_64", pd.Series(ent, index=grid))
        add("spectral", "low_band_64", pd.Series(low, index=grid))
    # regime
    if p_storm is not None:
        ps = p_storm.reindex(grid.union(p_storm.index)).ffill().reindex(grid)
        add("regime", "p_storm", ps)
        for c in ("ret_1", "ret_15", "zret_60", "imb_1", "imb_15"):
            add("regime", f"{c}_x_storm", f[c] * ps)

    out = pd.DataFrame(f, index=grid)
    out["ret_fwd"] = logp.shift(-horizon) - logp
    out["y"] = 1e4 * out["ret_fwd"]
    cols = [c for k in g for c in g[k]]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out[cols] = out[cols].astype("float32")
    return out, {k: v for k, v in g.items() if v}
