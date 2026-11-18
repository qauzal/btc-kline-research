"""Candidate models. A candidate = (model setting, training window in periods)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Zero(BaseEstimator, RegressorMixin):
    """Always forecasts 0. If nothing beats it, the strategy stays flat."""
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))


class Clip(BaseEstimator):
    """Clip standardised features to +/- c so single extreme minutes cannot dominate a fit."""
    def __init__(self, c=5.0): self.c = c
    def fit(self, X, y=None): return self
    def transform(self, X): return np.clip(X, -self.c, self.c)


class CovPCA(BaseEstimator):
    """PCA via the eigendecomposition of the feature covariance matrix.

    Fitted inside each candidate's pipeline, so the components are estimated
    on that candidate's training rows only. `var` keeps the fewest components
    explaining that share of variance (e.g. 0.95); an integer keeps that many.
    Cheaper than sklearn's PCA when rows >> features.
    """
    def __init__(self, var=0.95): self.var = var

    def fit(self, X, y=None):
        X = np.asarray(X, dtype="float64")
        self.mean_ = X.mean(0)
        vals, vecs = np.linalg.eigh(np.cov(X - self.mean_, rowvar=False))
        order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
        share = np.cumsum(vals) / vals.sum()
        k = int(self.var) if self.var >= 1 else int(np.searchsorted(share, self.var) + 1)
        self.components_, self.explained_ = vecs[:, :k], share[:k]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype="float64") - self.mean_) @ self.components_


def make_model(kind: str, **p):
    if kind == "zero":
        return Zero()
    pre = [StandardScaler(), Clip()] + ([CovPCA(p["pca"]), StandardScaler()] if p.get("pca") else [])
    if kind == "enet":
        return make_pipeline(*pre, ElasticNet(alpha=p["alpha"], l1_ratio=p.get("l1_ratio", 0.5),
                                              max_iter=5000, precompute=True))
    if kind == "gbt":
        return make_pipeline(*pre, HistGradientBoostingRegressor(
            max_depth=p["depth"], learning_rate=p.get("lr", 0.05), max_iter=p.get("n_iter", 150),
            min_samples_leaf=p.get("min_leaf", 1000), l2_regularization=p.get("l2", 1.0),
            early_stopping=False, random_state=0))
    raise ValueError(kind)


@dataclass(frozen=True)
class Candidate:
    name: str      # e.g. "enet(a=0.01)"
    kind: str      # zero | enet | gbt
    params: tuple  # hashable parameter pairs
    window: int    # training window in periods; 0 = all history; -1 = n/a (zero)

    @property
    def label(self): return f"{self.name} | {'all' if self.window == 0 else self.window}p" if self.window >= 0 else self.name

    def build(self): return make_model(self.kind, **dict(self.params))


def candidate_grid(settings: list[dict], windows: list[int], gbt_max_window: int | None) -> list[Candidate]:
    """Cross model settings with training windows. GBT can be capped to shorter
    windows because fitting it on the full history is slow."""
    out = [Candidate("zero", "zero", (), -1)]
    for s in settings:
        kind, p = s["kind"], {k: v for k, v in s.items() if k != "kind"}
        name = kind + "(" + ",".join(f"{k}={v}" for k, v in p.items()) + ")"
        for w in windows:
            if kind == "gbt" and gbt_max_window and (w == 0 or w > gbt_max_window):
                continue
            out.append(Candidate(name, kind, tuple(sorted(p.items())), w))
    return out
