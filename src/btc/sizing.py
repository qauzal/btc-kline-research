"""Position sizing, execution delay and performance metrics.

Positions are in units of notional: 1 = fully long, -1 = fully short.
`ret_fwd` at minute i is the log return from the close of i to the close of i+H.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PER_YEAR = 365 * 1440


def sign_pos(forecast: pd.Series) -> pd.Series:
    return np.sign(forecast).fillna(0.0)


def vol_target_pos(forecast: pd.Series, vol_per_min: pd.Series, target_ann: float = 0.40,
                   cap: float = 2.0) -> pd.Series:
    """Direction from the forecast, size inversely to recent volatility (known at minute i),
    so expected risk is roughly constant. Capped at `cap` times notional."""
    target = target_ann / np.sqrt(MIN_PER_YEAR)
    return sign_pos(forecast) * (target / vol_per_min.reindex(forecast.index)).clip(upper=cap).fillna(0.0)


def proportional_pos(forecast: pd.Series, band: float = 0.5, lookback: int = 10080, cap: float = 1.0) -> pd.Series:
    """Size in proportion to the forecast, relative to its own trailing typical size.
    Forecasts smaller than `band` times that typical size are ignored (no-trade band)."""
    scale = forecast.abs().rolling(lookback, min_periods=1440).mean()
    pos = (forecast / scale).clip(-cap, cap)
    return pos.where(forecast.abs() >= band * scale, 0.0).fillna(0.0)


def run(pos: pd.Series, ret_fwd: pd.Series, delay: int = 0) -> pd.DataFrame:
    """P&L of a position series. `delay` = minutes between decision and execution."""
    p = pos.shift(delay).fillna(0.0) if delay else pos
    return pd.DataFrame({"pos": p, "pnl": p * ret_fwd})


def perf(res: pd.DataFrame, horizon: int = 1) -> dict:
    pnl = res["pnl"]; per_year = MIN_PER_YEAR / horizon
    daily = pnl.groupby(pnl.index.floor("D")).sum()
    eq = (1 + pnl).cumprod()
    return {"ann_return_%": 100 * pnl.mean() * per_year,
            "ann_vol_%": 100 * pnl.std() * np.sqrt(per_year),
            "sharpe_daily": daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan,
            "max_drawdown_%": 100 * (eq / eq.cummax() - 1).min(),
            "hit_rate_%": 100 * (pnl[res["pos"] != 0] > 0).mean(),
            "avg_|position|": res["pos"].abs().mean()}


def equal_risk(pnl: pd.Series, target_ann: float = 0.20) -> pd.Series:
    """Rescale a P&L stream to a common annual volatility. Uses the whole sample's
    volatility, so it is for DISPLAY only; it does not change the Sharpe ratio."""
    return pnl * target_ann / (pnl.std() * np.sqrt(MIN_PER_YEAR))


def kelly_leverage(pnl: pd.Series) -> float:
    """Growth-optimal leverage mu / sigma^2 (both annualised). Only as good as the edge estimate."""
    return (pnl.mean() * MIN_PER_YEAR) / (pnl.var() * MIN_PER_YEAR)
