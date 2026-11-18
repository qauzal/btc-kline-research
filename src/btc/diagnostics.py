"""Feature screening, PCA analysis, time-of-day test and the Diebold-Mariano test.
Everything here that chooses or fits anything is meant to run on the DESIGN period only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def daily_ic(frame: pd.DataFrame, cols: list[str], target: str = "y") -> pd.DataFrame:
    """Spearman information coefficient (IC) of each feature with the target, one value per day."""
    day = frame.index.floor("D")
    ranks = frame[cols + [target]].groupby(day).rank()
    z = (ranks - ranks.groupby(day).transform("mean")) / ranks.groupby(day).transform("std")
    prod = z[cols].mul(z[target], axis=0)
    return prod.groupby(day).mean()


def screen(frame: pd.DataFrame, groups: dict[str, list[str]], alpha: float = 0.05) -> pd.DataFrame:
    """Mean daily IC, its t-statistic across days, sign stability across months, and
    whether it clears a Bonferroni bar for the number of features tested."""
    cols = [c for v in groups.values() for c in v]
    ic = daily_ic(frame, cols)
    mean, se = ic.mean(), ic.std() / np.sqrt(ic.count())
    monthly = ic.groupby(ic.index.tz_convert(None).to_period("M")).mean()
    same_sign = (np.sign(monthly) == np.sign(mean)).mean()
    bar = stats.norm.ppf(1 - alpha / 2 / len(cols))
    group_of = {c: k for k, v in groups.items() for c in v}
    out = pd.DataFrame({"group": [group_of[c] for c in cols], "mean_ic": mean[cols], "t_stat": (mean / se)[cols],
                        "months_same_sign": same_sign[cols]})
    out["passes_bonferroni"] = out["t_stat"].abs() > bar
    out.attrs["bar"] = bar
    return out.sort_values("t_stat", key=np.abs, ascending=False)


def pca_fit(X: np.ndarray):
    """Standardise, clip at +/-5 sd, eigendecompose. Returns (explained share, loadings, transform fn)."""
    mu, sd = X.mean(0), X.std(0)
    Z = np.clip((X - mu) / sd, -5, 5)
    vals, vecs = np.linalg.eigh(np.cov(Z, rowvar=False))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vals / vals.sum(), vecs, lambda A: np.clip((A - mu) / sd, -5, 5) @ vecs


def hour_profile(frame: pd.DataFrame, design_end) -> pd.DataFrame:
    """Mean 1-minute return (bps) and its standard error by UTC hour, design period vs after."""
    d = pd.DataFrame({"y": frame["y"], "hour": frame.index.hour,
                      "period": np.where(frame.index < design_end, "design", "after")})
    g = d.groupby(["period", "hour"])["y"]
    return pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count())}).unstack("period")


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """DM test on squared-error loss with a Newey-West variance. Negative = model 1 more accurate."""
    d = e1 ** 2 - e2 ** 2
    n = len(d); lag = max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    dm = d - d.mean()
    gam = [np.dot(dm[k:], dm[:-k] if k else dm) / n for k in range(lag + 1)]
    var = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, lag + 1))
    s = d.mean() / np.sqrt(var / n)
    return {"dm_stat": s, "p_value": 2 * (1 - stats.norm.cdf(abs(s)))}
