"""Period structure and the purged, embargoed validation split inside each period."""
from __future__ import annotations

import numpy as np
import pandas as pd


def period_ids(index: pd.DatetimeIndex, freq: str) -> pd.PeriodIndex:
    """Assign each timestamp to a period: 'W-SUN' (weeks ending Sunday), 'M', 'D', ..."""
    return index.tz_convert(None).to_period(freq)


def purged_split(index: pd.DatetimeIndex, block_minutes: int, val_frac: float,
                 purge: int, embargo: int, rng: np.random.Generator) -> np.ndarray:
    """Label each row of one period: 0=train, 1=validation, -1=purge/embargo."""
    minutes = ((index - index[0]).total_seconds() // 60).astype(int)
    block = minutes // block_minutes
    blocks = np.unique(block)
    n_val = max(1, int(round(val_frac * len(blocks))))
    val_blocks = rng.choice(blocks, size=n_val, replace=False)
    is_val = np.isin(block, val_blocks)
    label = np.where(is_val, 1, 0)
    starts = np.flatnonzero(is_val & ~np.r_[False, is_val[:-1]])
    ends = np.flatnonzero(is_val & ~np.r_[is_val[1:], False])
    for s, e in zip(starts, ends):
        before = (minutes >= minutes[s] - purge) & (minutes < minutes[s])
        after = (minutes > minutes[e]) & (minutes <= minutes[e] + embargo)
        label[(before | after) & (label == 0)] = -1
    return label


def split_all(frame: pd.DataFrame, freq: str, block_minutes: int, val_frac: float,
              purge: int, embargo: int, seed: int) -> pd.DataFrame:
    """Add `period` and `role` columns to a feature frame."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["period"] = period_ids(out.index, freq)
    role = np.empty(len(out), dtype="int8")
    for _, idx in out.groupby("period", sort=True).indices.items():
        role[idx] = purged_split(out.index[idx], block_minutes, val_frac, purge, embargo, rng)
    out["role"] = role
    return out
