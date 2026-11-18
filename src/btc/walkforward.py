"""The walk-forward loop: train candidates, score them on past validation rows,
let ATOMS (and the benchmarks) pick, test on the next period. One period = one step."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .atoms import ATOMS, fixed_horizon
from .features import FEATURES
from .models import Candidate


def run(frame: pd.DataFrame, cands: list[Candidate], *, min_history: int, max_lookback: int,
        fh_short: int, atoms_delta: float, atoms_M: float, seed: int = 0, verbose: bool = True,
        start_period=None, feature_cols: list[str] | None = None,
        checkpoint_dir: str | Path | None = None) -> dict:
    """`frame` needs the feature columns (FEATURES unless `feature_cols` is given),
    y, ret_fwd, period, role (from splits.split_all).

    Returns per-period selections and per-row test forecasts for every candidate
    and every selector (atoms, fh_short, fh_long).

    With `checkpoint_dir`, each finished period is saved there and skipped on the
    next call, so an interrupted run resumes instead of starting over.
    """
    ck = Path(checkpoint_dir) if checkpoint_dir else None
    if ck: ck.mkdir(parents=True, exist_ok=True)
    periods = sorted(frame["period"].unique())
    cols = list(feature_cols) if feature_cols is not None else FEATURES
    X, y = frame[cols].to_numpy(dtype="float32"), frame["y"].to_numpy()
    pidx = frame.groupby("period", sort=True).indices
    tr_rows = {p: idx[frame["role"].to_numpy()[idx] == 0] for p, idx in pidx.items()}
    va_rows = {p: idx[frame["role"].to_numpy()[idx] == 1] for p, idx in pidx.items()}

    atoms = ATOMS(delta=atoms_delta, M=atoms_M, seed=seed)
    sel_rows, fc_parts = [], []
    t0 = time.time()
    first = max(min_history, next((i for i, p in enumerate(periods) if start_period is None or p >= start_period), 0))
    for ti in range(first, len(periods)):
        p = periods[ti]
        if ck and (ck / (str(p).replace("/", "_") + ".pkl")).exists():
            row, fc = pickle.loads((ck / (str(p).replace("/", "_") + ".pkl")).read_bytes())
            sel_rows.append(row); fc_parts.append(fc)
            if verbose: print(f"[{ti - first + 1:>4}/{len(periods) - first}] {p}  loaded from checkpoint", flush=True)
            continue
        hist = periods[max(0, ti - max_lookback):ti]            # look-back periods, oldest first
        va_idx = [va_rows[q] for q in hist]
        te_idx = pidx[p]
        L = [np.empty((len(v), len(cands))) for v in va_idx]     # validation squared errors
        F = np.empty((len(te_idx), len(cands)))                  # test forecasts
        for j, c in enumerate(cands):
            if c.kind == "zero":
                for k, v in enumerate(va_idx): L[k][:, j] = y[v] ** 2
                F[:, j] = 0.0
                continue
            win = periods[:ti] if c.window == 0 else periods[max(0, ti - c.window):ti]
            rows = np.concatenate([tr_rows[q] for q in win])
            m = c.build().fit(X[rows], y[rows])
            for k, v in enumerate(va_idx):
                L[k][:, j] = (m.predict(X[v]) - y[v]) ** 2
            F[:, j] = m.predict(X[te_idx])
        res = atoms.select(L)
        pick = {"atoms": res["winner"], "fh_short": fixed_horizon(L, fh_short), "fh_long": fixed_horizon(L, None)}
        test_loss = ((F - y[te_idx, None]) ** 2).mean(0)
        row = ({"period": p, **{k: cands[v].label for k, v in pick.items()},
                         "atoms_kind": cands[pick["atoms"]].kind, "atoms_window": cands[pick["atoms"]].window,
                         "atoms_lookback": res["lookback"], "atoms_rounds": res["rounds"],
                         "hindsight_best": cands[int(np.argmin(test_loss))].label,
                         "hindsight_kind": cands[int(np.argmin(test_loss))].kind,
                         "hindsight_window": cands[int(np.argmin(test_loss))].window,
                         "n_train_periods": ti, "n_test_rows": len(te_idx)})
        sel_rows.append(row)
        fc = pd.DataFrame(F, index=frame.index[te_idx], columns=[c.label for c in cands])
        for k, v in pick.items(): fc[k] = F[:, v]
        fc["y"], fc["ret_fwd"], fc["period"] = y[te_idx], frame["ret_fwd"].to_numpy()[te_idx], str(p)
        fc_parts.append(fc)
        if ck: (ck / (str(p).replace("/", "_") + ".pkl")).write_bytes(pickle.dumps((row, fc)))
        if verbose:
            el = time.time() - t0
            print(f"[{ti - first + 1:>4}/{len(periods) - first}] {p}  atoms -> {cands[res['winner']].label:<28} "
                  f"lookback {res['lookback']:.1f}  ({el/60:.1f} min)", flush=True)
    return {"selections": pd.DataFrame(sel_rows).set_index("period"), "forecasts": pd.concat(fc_parts)}
