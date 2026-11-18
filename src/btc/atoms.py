"""Adaptive Tournament Model Selection (Capponi, Huang, Sidaoui, Wang & Zou, 2026)."""
from __future__ import annotations

import numpy as np


class ATOMS:
    def __init__(self, delta: float = 0.1, M: float | None = None, seed: int = 0):
        """Create an ATOMS selector.

        With M=None, calibrate the loss-gap bound per candidate pair from the
        90th percentile of past absolute validation-loss gaps.
        """
        self.delta, self.M = delta, M
        self.rng = np.random.default_rng(seed)
        self.log2d = np.log(2 / delta)

    def compare_pivot(self, L: list[np.ndarray], p: int):
        """Compare pivot p with every candidate and choose adaptive look-backs."""
        T = len(L)
        S1 = np.array([(loss[:, p:p + 1] - loss).sum(0) for loss in L])
        S2 = np.array([((loss[:, p:p + 1] - loss) ** 2).sum(0) for loss in L])
        n = np.array([len(loss) for loss in L], dtype=float)
        M = self.M if self.M is not None else np.quantile(
            np.concatenate([np.abs(loss[:, p:p + 1] - loss) for loss in L]), 0.9, axis=0)
        M = np.maximum(M, 1e-12)
        c1, c2, cn = S1[::-1].cumsum(0), S2[::-1].cumsum(0), n[::-1].cumsum()[:, None]
        delta = c1 / cn
        var = np.clip((c2 - cn * delta ** 2) / np.maximum(cn - 1, 1), 0, None)
        psi = np.sqrt(var * 2 * self.log2d / cn) + 8 * M * self.log2d / (3 * np.maximum(cn - 1, 1))
        psi[cn[:, 0] < 2] = M if np.isscalar(M) else M[None, :].repeat((cn[:, 0] < 2).sum(), 0)
        gap = np.abs(delta[:, None, :] - delta[None, :, :]) - psi[:, None, :] - psi[None, :, :]
        gap = np.clip(gap, 0, None)
        tri = np.tril(np.ones((T, T), dtype=bool))
        phi = np.where(tri[:, :, None], gap, 0).max(1)
        ell = np.argmin(phi + psi, axis=0)
        d_hat = delta[ell, np.arange(delta.shape[1])]
        return d_hat > 0, ell + 1

    def select(self, L: list[np.ndarray]) -> dict:
        remaining = list(range(L[0].shape[1]))
        rounds, lookbacks = 0, []
        while len(remaining) > 1:
            rounds += 1
            p = int(self.rng.choice(remaining))
            beats, ell = self.compare_pivot(L, p)
            others = [j for j in remaining if j != p]
            lookbacks = [int(ell[j]) for j in others]
            survivors = [j for j in others if beats[j]]
            if not survivors:
                return {"winner": p, "rounds": rounds, "lookback": float(np.mean(lookbacks))}
            remaining = survivors
        return {"winner": remaining[0], "rounds": rounds,
                "lookback": float(np.mean(lookbacks)) if lookbacks else np.nan}


def fixed_horizon(L: list[np.ndarray], ell: int | None) -> int:
    """Select the lowest mean validation loss over a fixed look-back."""
    use = L if ell is None else L[-ell:]
    tot = sum(loss.sum(0) for loss in use) / sum(len(loss) for loss in use)
    return int(np.argmin(tot))
