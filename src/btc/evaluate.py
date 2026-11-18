"""Out-of-sample metrics, sign strategy, and Diebold-Mariano testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_PER_YEAR = 365 * 24 * 60


def oos_r2(y: np.ndarray, f: np.ndarray) -> float:
    """R-squared against a zero forecast."""
    return 1 - np.sum((f - y) ** 2) / np.sum(y ** 2)


def sign_strategy(forecast: pd.Series, ret_fwd: pd.Series) -> pd.DataFrame:
    """Long for positive forecasts, short for negative forecasts, otherwise flat."""
    pos = np.sign(forecast).fillna(0)
    pnl = pos * ret_fwd
    return pd.DataFrame({"pos": pos, "pnl": pnl})


def summarize(strat: pd.DataFrame, horizon: int = 1) -> dict:
    """Summarize a strategy using daily log P&L and compounded equity."""
    pnl, pos = strat["pnl"], strat["pos"]
    daily = pnl.resample("1D").sum()
    mu, sd = daily.mean(), daily.std()
    log_equity = pnl.cumsum()
    equity = np.exp(log_equity)
    drawdown = equity / equity.cummax() - 1
    return {"ann_return_%": 100 * np.expm1(mu * 365), "ann_vol_%": 100 * sd * np.sqrt(365),
            "sharpe": mu / sd * np.sqrt(365) if sd > 0 else np.nan,
            "max_drawdown_%": 100 * drawdown.min(), "hit_rate_%": 100 * (pnl > 0).mean(),
            "time_in_market_%": 100 * (pos != 0).mean()}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """DM test on squared-error loss with a Newey-West variance estimate."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    lag = max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    dm = d - d.mean()
    gamma = [np.dot(dm[k:], dm[:-k] if k else dm) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / n)
    return {"dm_stat": stat, "p_value": 2 * (1 - stats.norm.cdf(abs(stat))), "n": n}


def block_bootstrap_sharpe(pnl: pd.Series, horizon: int, block: int = 5,
                           n_boot: int = 500, seed: int = 0):
    """Block-bootstrap interval for daily-P&L annualised Sharpe."""
    rng = np.random.default_rng(seed)
    x = pnl.resample("1D").sum().to_numpy()
    n = len(x)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) % n
                              for s in rng.integers(0, n, size=n // block + 1)])[:n]
        b = x[idx]
        out.append(b.mean() / b.std() * np.sqrt(365))
    return np.percentile(out, [2.5, 97.5])
