"""Two-state Gaussian HMM for market regimes, with a look-ahead-free forward filter.

hmmlearn's `predict_proba` runs forward-backward, so each point's state uses
FUTURE data. For a regime that could be acted on in real time we need the
forward filter only: P(state_t | data up to t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def hourly_inputs(m1: pd.DataFrame, design_end=None, smooth_hours: int = 24) -> pd.DataFrame:
    """Hourly return (bps) and log realised variance from 1-minute bars.

    With `design_end` given, the hour-of-day pattern in log variance (estimated
    on hours before `design_end` only) is subtracted, and the result is averaged
    over a TRAILING `smooth_hours` window. Without this, BTC's strong intraday
    volatility cycle dominates and a 2-state HMM simply learns "US hours vs
    Asian hours" instead of calm vs stormy markets. Both steps are causal.
    """
    r = np.log(m1["close"]).diff()
    h = pd.DataFrame({"ret": 1e4 * r.resample("1h").sum(), "rv": r.pow(2).resample("1h").sum()})   # return in bps
    h = h[r.resample("1h").count() >= 30]
    h["log_rv"] = np.log(h["rv"] + 1e-12)
    if design_end is not None:
        mask = h.index < design_end
        season = h.loc[mask, "log_rv"].groupby(h.index[mask].hour).mean()
        h["log_rv"] = h["log_rv"] - season.reindex(h.index.hour).to_numpy()
        if smooth_hours:
            h["log_rv"] = h["log_rv"].rolling(smooth_hours, min_periods=max(1, smooth_hours // 2)).mean()
    return h[["ret", "log_rv"]].dropna()


def lagged(p_high: pd.Series, hours: int = 1) -> pd.Series:
    """Shift hourly probabilities so each value is stamped when it becomes KNOWN.

    resample("1h") labels an hour by its start: the value at 14:00 uses data up
    to 14:59. As a feature for minute 14:37 it would look ahead, so it is moved
    to 15:00 here and then forward-filled onto minutes.
    """
    out = p_high.copy()
    out.index = out.index + pd.Timedelta(hours=hours)
    return out


def fit_hmm(x: pd.DataFrame, n_states: int = 2, seed: int = 0) -> GaussianHMM:
    m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=300, random_state=seed)
    m.fit(x.to_numpy())
    return m


def forward_filter(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """Filtered state probabilities, shape (T, K). Uses only past and current observations."""
    means = model.means_
    var = np.array([np.diag(c) for c in model.covars_])   # hmmlearn returns full matrices even for "diag"
    A, pi = model.transmat_, model.startprob_
    # log emission densities (diagonal Gaussian)
    z = (x[:, None, :] - means[None]) ** 2 / var[None]
    logb = -0.5 * (z.sum(2) + np.log(2 * np.pi * var).sum(1)[None])
    b = np.exp(logb - logb.max(1, keepdims=True))
    out = np.empty_like(b)
    a = pi * b[0]; out[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (out[t - 1] @ A) * b[t]
        out[t] = a / a.sum()
    return out


def regime_series(model: GaussianHMM, x: pd.DataFrame) -> pd.Series:
    """Probability of the HIGH-volatility state at each hour, forward-filtered."""
    hi = int(np.argmax(model.means_[:, 1]))          # state with the larger mean log RV
    p = forward_filter(model, x.to_numpy())[:, hi]
    return pd.Series(p, index=x.index, name="p_high_vol")
