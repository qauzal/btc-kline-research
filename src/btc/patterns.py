"""Pattern statistics on 1-second data, accumulated month by month.

The full history is ~120 million rows, so nothing here holds more than one
month in memory. Each month is reduced to small sums; the sums add up across
months and are turned into the final statistics at the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEC_PER_YEAR = 365 * 24 * 3600
ACF_LAGS = 30
SAMPLING_S = [1, 2, 5, 10, 30, 60, 300, 900, 3600]


def month_stats(df: pd.DataFrame) -> dict:
    """Reduce one month of CLEAN 1-second bars to additive sums."""
    grid = pd.date_range(df.index[0], df.index[-1], freq="1s")
    logp = np.log(df["close"].reindex(grid))
    r = logp.diff().to_numpy()          # NaN wherever either second is missing
    ok = ~np.isnan(r)
    r0 = np.where(ok, r, 0.0)

    # 1. intraday / weekly seasonality
    g = pd.DataFrame({
        "dow": grid.dayofweek, "hour": grid.hour, "r2": r0 ** 2, "n": ok.astype("int64"),
        "volume": df["volume"].reindex(grid).fillna(0).to_numpy(),
        "trades": df["trades"].reindex(grid).fillna(0).to_numpy(),
    }).groupby(["dow", "hour"]).sum()

    # 2. autocorrelation of 1-second returns (bid-ask bounce shows up as negative lag 1)
    prod = np.array([np.sum(r0[k:] * r0[:-k]) for k in range(1, ACF_LAGS + 1)])

    # 3. volatility signature: realised variance when sampling every s seconds
    lp = logp.ffill().to_numpy()
    sig = {s: (float(np.sum(np.diff(lp[::s]) ** 2)), (len(lp[::s]) - 1) * s) for s in SAMPLING_S}

    # 4. moments
    rv = r[ok]
    mom = {"n": rv.size, "s2": float(np.sum(rv ** 2)), "s4": float(np.sum(rv ** 4)),
           "zero": int(np.sum(rv == 0))}
    return {"hourly": g, "acf_prod": prod, "acf_s2": float(np.sum(r0 ** 2)), "sig": sig, "mom": mom}


def combine(stats: list[dict]) -> dict:
    hourly = stats[0]["hourly"]
    for s in stats[1:]:
        hourly = hourly.add(s["hourly"], fill_value=0)
    return {
        "hourly": hourly,
        "acf": pd.Series(sum(s["acf_prod"] for s in stats) / sum(s["acf_s2"] for s in stats),
                         index=range(1, ACF_LAGS + 1), name="acf"),
        "signature": pd.Series({
            s: np.sqrt(sum(x["sig"][s][0] for x in stats) / sum(x["sig"][s][1] for x in stats) * SEC_PER_YEAR)
            for s in SAMPLING_S}, name="annualised_vol"),
        "mom": {k: sum(s["mom"][k] for s in stats) for k in ("n", "s2", "s4", "zero")},
    }


def ann_vol(r2_sum, n, seconds_per_obs: float = 1.0):
    """Annualised volatility from a sum of squared returns over n observations."""
    return np.sqrt(r2_sum / n * SEC_PER_YEAR / seconds_per_obs)


def kurtosis(s2: float, s4: float, n: int) -> float:
    """Raw kurtosis assuming a zero mean (3 for a normal distribution)."""
    return (s4 / n) / (s2 / n) ** 2


def acf(x: np.ndarray, lags) -> pd.Series:
    """Sample autocorrelation at the given lags."""
    x = np.asarray(x, dtype="float64")
    x = x - x.mean()
    v = np.dot(x, x)
    return pd.Series({k: np.dot(x[k:], x[:-k]) / v for k in lags}, name="acf")


def decile_table(signal: pd.Series, outcome: pd.Series, q: int = 10) -> pd.DataFrame:
    """Mean outcome (in basis points) per quantile bucket of the signal, with standard errors."""
    d = pd.DataFrame({"s": signal, "y": outcome}).dropna()
    d["bucket"] = pd.qcut(d["s"], q, labels=False, duplicates="drop") + 1
    out = d.groupby("bucket")["y"].agg(["mean", "sem", "count"])
    out[["mean", "sem"]] *= 1e4
    return out.rename(columns={"mean": "mean_bps", "sem": "se_bps"})
