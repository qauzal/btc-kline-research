"""Data-quality checks on raw (uncleaned) 1-second klines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_gaps(df: pd.DataFrame, month: str | None = None, step_s: int = 1) -> pd.DataFrame:
    """Every run of missing seconds: first missing, last missing, length.

    If `month` is given, seconds missing before the first row or after the last
    row of that calendar month are counted too.
    """
    idx = df.index.unique().sort_values()
    edges = [idx]
    if month is not None:
        p = pd.Period(month, freq="M")
        lo = p.start_time.tz_localize("UTC") - pd.Timedelta(seconds=step_s)
        hi = (p + 1).start_time.tz_localize("UTC")
        edges = [pd.DatetimeIndex([lo]), idx, pd.DatetimeIndex([hi])]
    t = edges[0].append(edges[1:]) if len(edges) > 1 else idx
    d = np.asarray((t[1:] - t[:-1]).total_seconds())
    j = np.flatnonzero(d > step_s)
    step = pd.Timedelta(seconds=step_s)
    return pd.DataFrame({
        "first_missing": t[j] + step,
        "last_missing": t[j + 1] - step,
        "missing_seconds": (d[j] / step_s - 1).astype("int64"),
    })


def check_month(df: pd.DataFrame, month: str, partial: bool = False) -> dict:
    """One row of the quality report. `partial=True` for a month still in progress:
    expected seconds are then counted up to the last row instead of month end."""
    p = pd.Period(month, freq="M")
    idx = df.index
    start = p.start_time.tz_localize("UTC")
    end = (idx.max() + pd.Timedelta(seconds=1)) if partial else (p + 1).start_time.tz_localize("UTC")
    expected = int((end - start).total_seconds())
    unique = idx.nunique()
    gaps = find_gaps(df, None if partial else month)
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    return {
        "month": month,
        "rows": len(df),
        "expected_seconds": expected,
        "missing_seconds": expected - unique,
        "coverage_pct": 100 * unique / expected,
        "duplicate_rows": len(df) - unique,
        "out_of_order_rows": int((np.diff(idx.asi8) < 0).sum()),
        "outside_month_rows": int(((idx < start) | (idx >= end)).sum()),
        "null_values": int(df.isna().sum().sum()),
        "nonpositive_prices": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "ohlc_inconsistent": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum()),
        "taker_gt_total": int((df["taker_buy_volume"] > df["volume"] * (1 + 1e-9)).sum()),
        "zero_volume_pct": 100 * float((df["volume"] == 0).mean()),
        "n_gaps": len(gaps),
        "longest_gap_s": int(gaps["missing_seconds"].max()) if len(gaps) else 0,
    }
