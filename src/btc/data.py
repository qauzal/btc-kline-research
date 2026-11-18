"""Download, verify, store and clean Binance spot kline data.

Two public sources are used:

* the bulk archive (data.binance.vision): one zip per month, each with a
  SHA-256 checksum file. Used for all complete months.
* the REST market-data API (/api/v3/klines): 1,000 rows per request. Used only
  for the days after the last published monthly file.
"""
from __future__ import annotations

import hashlib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BULK_URL = "https://data.binance.vision/data/spot/monthly/klines"
API_URL = "https://data-api.binance.vision/api/v3/klines"

RAW_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def month_list(start: str, end: str) -> list[str]:
    """['2023-01', '2023-02', ...] inclusive of both ends."""
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def last_complete_month() -> str:
    return str(pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1)


def _get(url: str, retries: int = 5, stream: bool = False, **kw) -> requests.Response:
    """GET with exponential backoff. A 404 is not retried: the file does not exist."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, stream=stream, **kw)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            if r.status_code in (418, 429):  # rate limited: honour Retry-After
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {url}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# bulk archive
# ----------------------------------------------------------------------------
def fetch_month(month: str, raw_dir: Path, symbol: str = "BTCUSDT", interval: str = "1s") -> dict:
    """Download one monthly zip and verify it against Binance's checksum.

    Idempotent: a zip that is already on disk and matches its checksum is skipped.
    Returns a status row for the download log.
    """
    return _fetch_archive(month, "monthly", raw_dir, symbol, interval)


def fetch_day(day: str, raw_dir: Path, symbol: str = "BTCUSDT", interval: str = "1s") -> dict:
    """Same as `fetch_month`, for one daily zip ('2026-11-01'). Binance publishes
    each day's file shortly after that day ends, with its own checksum."""
    return _fetch_archive(day, "daily", raw_dir, symbol, interval)


def _fetch_archive(period: str, freq: str, raw_dir: Path, symbol: str, interval: str) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}-{interval}-{period}.zip"
    zpath, cpath = raw_dir / name, raw_dir / f"{name}.CHECKSUM"
    url = f"{BULK_URL.replace('/monthly/', f'/{freq}/')}/{symbol}/{interval}/{name}"
    row = {"month" if freq == "monthly" else "day": period, "file": name}
    try:
        if not cpath.exists():
            cpath.write_text(_get(url + ".CHECKSUM").text)
        expected = cpath.read_text().split()[0]
        if zpath.exists() and sha256(zpath) == expected:
            return row | {"status": "cached", "mb": zpath.stat().st_size / 1e6, "sha256_ok": True}
        for _ in range(2):  # one re-download if the checksum does not match
            tmp = zpath.with_suffix(".part")
            with _get(url, stream=True) as r, open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(zpath)
            if sha256(zpath) == expected:
                return row | {"status": "downloaded", "mb": zpath.stat().st_size / 1e6, "sha256_ok": True}
        return row | {"status": "checksum mismatch", "mb": zpath.stat().st_size / 1e6, "sha256_ok": False}
    except FileNotFoundError:
        return row | {"status": "not published", "mb": np.nan, "sha256_ok": False}
    except Exception as e:  # keep going; the log shows what failed
        return row | {"status": f"error: {type(e).__name__}", "mb": np.nan, "sha256_ok": False}


def fetch_all(start: str, end: str, raw_dir: Path, workers: int = 4, **kw) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda m: fetch_month(m, raw_dir, **kw), month_list(start, end)))
    return pd.DataFrame(rows).set_index("month")


def fetch_days(start: str, end: str, raw_dir: Path, workers: int = 4, **kw) -> pd.DataFrame:
    """Daily zips from `start` to `end` (inclusive, 'YYYY-MM-DD')."""
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda d: fetch_day(d, raw_dir, **kw), days))
    return pd.DataFrame(rows).set_index("day")


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
def to_utc(open_time: pd.Series) -> pd.DatetimeIndex:
    """Epoch integers -> UTC timestamps.

    Binance's spot archive uses milliseconds in older files and microseconds in
    newer ones. The unit is detected from the magnitude, not from the file date.
    """
    v = open_time.to_numpy(dtype="int64")
    unit = "us" if np.median(v) > 1e14 else "ms"
    return pd.DatetimeIndex(pd.to_datetime(v, unit=unit, utc=True), name="time")


def _frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[KEEP_COLS].astype("float64")
    df["trades"] = raw["trades"].astype("int32")
    df.index = to_utc(raw["open_time"])
    return df


def read_zip(zpath: Path) -> pd.DataFrame:
    """Read one archive zip exactly as published: no sorting, no de-duplication."""
    with zipfile.ZipFile(zpath) as z, z.open(z.namelist()[0]) as f:
        has_header = not f.read(1).isdigit()
    with zipfile.ZipFile(zpath) as z, z.open(z.namelist()[0]) as f:
        raw = pd.read_csv(f, header=0 if has_header else None, names=RAW_COLS)
    return _frame(raw)


def zip_to_parquet(zpath: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (zpath.stem + ".parquet")
    # Rebuild if the zip is newer: a month first built from daily files gets
    # replaced once its full monthly zip is downloaded.
    if not out.exists() or out.stat().st_mtime < zpath.stat().st_mtime:
        read_zip(zpath).to_parquet(out, compression="zstd")
    return out


# ----------------------------------------------------------------------------
# REST API (fallback for days not yet in the daily archive)
# ----------------------------------------------------------------------------
def fetch_api(start: pd.Timestamp, end: pd.Timestamp, symbol: str = "BTCUSDT",
              interval: str = "1s", pause: float = 0.05) -> pd.DataFrame:
    """Page through /api/v3/klines from `start` (inclusive) to `end` (exclusive)."""
    cur, stop = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    parts = []
    while cur < stop:
        rows = _get(API_URL, params={"symbol": symbol, "interval": interval,
                                     "startTime": cur, "endTime": stop - 1, "limit": 1000}).json()
        if not rows:
            break
        parts.append(pd.DataFrame(rows, columns=RAW_COLS))
        cur = int(rows[-1][0]) + 1
        time.sleep(pause)
    if not parts:
        return pd.DataFrame(columns=KEEP_COLS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    raw = pd.concat(parts, ignore_index=True)
    raw["open_time"] = raw["open_time"].astype("int64")
    return _frame(raw)


# ----------------------------------------------------------------------------
# cleaning and resampling
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """The only edits made to the published data:

    1. sort by time;
    2. where a timestamp appears more than once, keep the last row;
    3. drop rows with a non-positive price or an impossible bar (high < low).

    Missing seconds are NOT filled. A gap stays a gap.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["high"] >= df["low"])
    return df[ok]


AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
       "quote_volume": "sum", "trades": "sum", "taker_buy_volume": "sum"}


def resample_bars(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Aggregate clean 1-second bars. Bars with no underlying data are dropped,
    and `n_obs` records how many seconds each bar was built from."""
    out = df.resample(rule).agg(AGG)
    out["n_obs"] = df["close"].resample(rule).count().astype("int32")
    return out[out["n_obs"] > 0]


def parquet_files(pq_dir: Path, symbol: str = "BTCUSDT", interval: str = "1s") -> list[Path]:
    return sorted(pq_dir.glob(f"{symbol}-{interval}-*.parquet"))
