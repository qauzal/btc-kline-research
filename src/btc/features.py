"""Features and target from clean 1-minute bars.

Every feature at minute i uses only bars up to and including i. The target is
the return from the close of minute i to the close of minute i + H, which is
what a trade placed at the close of minute i would earn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["ret_1", "ret_5", "ret_15", "ret_60", "rv_15", "rv_60", "range_1",
            "vol_z", "trades_z", "imb_1", "imb_15", "hour_sin", "hour_cos"]


def build(m1: pd.DataFrame, horizon: int = 1, min_obs: int = 50) -> pd.DataFrame:
    """Return a frame with FEATURES, `y` (target, in basis points) and `ret_fwd`.

    Bars are placed on a complete 1-minute grid first, so every rolling window
    sees gaps as NaN and any feature touching a gap is dropped.
    """
    m1 = m1[m1["n_obs"] >= min_obs]
    grid = pd.date_range(m1.index.min(), m1.index.max(), freq="1min", name="time")
    d = m1.reindex(grid)

    logp = np.log(d["close"])
    r = logp.diff()
    f = pd.DataFrame(index=grid)
    for k in (1, 5, 15, 60):
        f[f"ret_{k}"] = logp.diff(k)
    for k in (15, 60):
        f[f"rv_{k}"] = np.sqrt(r.pow(2).rolling(k, min_periods=k).sum())
    f["range_1"] = np.log(d["high"] / d["low"])
    for col, name in (("volume", "vol_z"), ("trades", "trades_z")):
        x = np.log1p(d[col])
        mu = x.rolling(1440, min_periods=720).mean()
        sd = x.rolling(1440, min_periods=720).std()
        f[name] = (x - mu) / sd
    imb = (2 * d["taker_buy_volume"] / d["volume"] - 1).where(d["volume"] > 0)
    f["imb_1"] = imb
    f["imb_15"] = imb.rolling(15, min_periods=15).mean()
    h = grid.hour + grid.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)

    f["ret_fwd"] = logp.shift(-horizon) - logp
    f["y"] = 1e4 * f["ret_fwd"]
    f = f.dropna()
    f = f[(f[FEATURES].abs() < 1e6).all(axis=1)]
    return f
