"""Ability profiles with bootstrap confidence intervals.

    python src/profiles.py                 # all models in results/correctness/
    python src/profiles.py --n-boot 50     # quicker

Method (ported from the ADeLe paper's notebook / toolkit, MIT licence,
Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation, src/adele/analysis/ability.py):
  for each dimension d:
    1. dominant slice: keep items whose other 17 demands are all <= the item's level on d
    2. re-weight demand levels 1-5 so well-populated bins count equally (dup_threshold)
    3. add a synthetic failure anchor at level 20
    4. weighted logistic regression P(success) ~ level
    5. ability = integral of the fitted curve over [0, 100]   (the paper's scale; ~level of 50% success)
Items with unguessability < 75 are dropped first, as in the paper.

Outputs:
  results/profiles.csv        model, dim, ability, lo, hi (95% bootstrap CI), n_slice
  results/profiles_boot.csv   every bootstrap replicate (feeds the CIs in evolution / redlines)
  results/sensitivity.csv     MC-only vs MC+open-ended point estimates for our models
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

import common
from common import DEMANDS, log

_trapz = getattr(np, "trapezoid", None) or np.trapz


def anchor_weights(x: np.ndarray, y: np.ndarray, dup_threshold: int):
    """Bin-balancing weights + synthetic failure anchor (faithful port of the toolkit helper)."""
    levels = [1, 2, 3, 4, 5]
    counts = {lv: int(np.sum(x == lv)) for lv in levels}
    total = sum(counts.values())
    if total == 0:
        return x.reshape(-1, 1), y, np.ones(len(y))
    eligible = [lv for lv in levels if counts[lv] >= dup_threshold]
    majority = max(counts[lv] for lv in eligible) if eligible else max(counts.values())
    targets = {lv: (float(majority) if counts[lv] >= dup_threshold else None) for lv in levels}
    inelig = [lv for lv in levels if targets[lv] is None]
    E = sum(targets[lv] for lv in levels if targets[lv] is not None)
    s_in = sum(counts[lv] for lv in inelig)
    T = E / (1 - s_in / total) if (s_in > 0 and total > s_in) else (E if E > 0 else 1.0)
    for lv in inelig:
        targets[lv] = counts[lv] / total * T
    w = np.zeros(len(x))
    for lv in levels:
        if counts[lv]:
            w[x == lv] = targets[lv] / counts[lv]
    anchor_w = max(sum(targets.values()), 1.0)
    X = np.concatenate([x, [20.0]]).reshape(-1, 1)
    Y = np.concatenate([y, [0.0]])
    W = np.concatenate([w, [anchor_w]])
    return X, Y, W


def fit_profile(D: np.ndarray, y: np.ndarray, pc: dict, n_grid: int = 10001) -> np.ndarray:
    """D: items x 18 demand levels, y: 0/1. Returns 18 abilities (NaN if a slice is empty)."""
    grid = np.linspace(0, pc["grid_max"], n_grid).reshape(-1, 1)
    out = np.full(D.shape[1], np.nan)
    for j in range(D.shape[1]):
        others = np.delete(D, j, axis=1).max(axis=1)
        mask = others <= D[:, j] + pc["level_diff"]
        if not mask.any():
            continue
        X, Y, W = anchor_weights(D[mask, j].astype(float), y[mask].astype(float), pc["dup_threshold"])
        if len(np.unique(Y)) < 2:      # always wrong -> only failures: ability ~ 0
            out[j] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(random_state=42, max_iter=10000).fit(X, Y, sample_weight=W)
        out[j] = float(_trapz(lr.predict_proba(grid)[:, 1], grid.ravel()))
    return out


def slice_sizes(D: np.ndarray, pc: dict) -> np.ndarray:
    return np.array([(np.delete(D, j, axis=1).max(axis=1) <= D[:, j] + pc["level_diff"]).sum()
                     for j in range(D.shape[1])])


def model_profile(name: str, merged: pd.DataFrame, pc: dict, n_boot: int, seed: int):
    D = merged[DEMANDS].to_numpy(float)
    y = merged["correct"].to_numpy(float)
    point = fit_profile(D, y, pc)
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, len(DEMANDS)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots[b] = fit_profile(D[idx], y[idx], pc, n_grid=2001)
    return name, point, boots, slice_sizes(D, pc)


def load_correctness(cfg: dict) -> dict[str, pd.DataFrame]:
    cdir = common.path_of(cfg, "results_dir") / "correctness"
    files = sorted(cdir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No correctness files in {cdir}. Run collect.py first.")
    return {p.stem: pd.read_csv(p, dtype={"custom_id": str}) for p in files}


def merge_with_demands(df: pd.DataFrame, ann: pd.DataFrame, min_ug: float) -> pd.DataFrame:
    m = ann.merge(df[["custom_id", "correct"]], on="custom_id", how="inner")
    m = m.dropna(subset=DEMANDS + ["UG", "correct"])
    return m[m["UG"] >= min_ug]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n-boot", type=int)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    pc = cfg["profiles"]
    n_boot = args.n_boot if args.n_boot is not None else pc["n_boot"]
    min_ug = 75  # fixed by the method (paper/toolkit), independent of the subset setting

    ann = common.load_battery(cfg, columns=["instance_id", "answer_format", "UG"] + DEMANDS)
    data = load_correctness(cfg)
    merged = {n: merge_with_demands(d, ann, min_ug) for n, d in data.items()}
    for n, m in merged.items():
        log.info("[%s] %d items enter the fit", n, len(m))

    log.info("Fitting %d models x %d bootstrap replicates ...", len(merged), n_boot)
    results = Parallel(n_jobs=args.jobs)(
        delayed(model_profile)(n, m, pc, n_boot, cfg["seed"] + i)
        for i, (n, m) in enumerate(merged.items()) if len(m))

    rows, brows = [], []
    for name, point, boots, sizes in results:
        lo = np.nanpercentile(boots, 2.5, axis=0) if n_boot else np.full(len(DEMANDS), np.nan)
        hi = np.nanpercentile(boots, 97.5, axis=0) if n_boot else np.full(len(DEMANDS), np.nan)
        for j, d in enumerate(DEMANDS):
            rows.append({"model": name, "dim": d, "ability": point[j], "lo": lo[j], "hi": hi[j],
                         "n_slice": int(sizes[j])})
        for b in range(boots.shape[0]):
            for j, d in enumerate(DEMANDS):
                brows.append({"model": name, "rep": b, "dim": d, "ability": boots[b, j]})

    rdir = common.path_of(cfg, "results_dir")
    prof = pd.DataFrame(rows)
    prof.to_csv(rdir / "profiles.csv", index=False)
    pd.DataFrame(brows).to_csv(rdir / "profiles_boot.csv", index=False)
    wide = prof.pivot(index="model", columns="dim", values="ability")[DEMANDS].round(2)
    print(wide.to_string())

    # Sensitivity: does adding judged open-ended items move our models' profiles?
    sens = []
    for n, m in merged.items():
        if n.startswith("ref__"):
            continue
        fmt = data[n].set_index("custom_id")["answer_format"]
        mc = m[m["custom_id"].map(fmt) == "MC"]
        if len(mc) == len(m) or mc.empty:
            continue
        p_all = fit_profile(m[DEMANDS].to_numpy(float), m["correct"].to_numpy(float), pc)
        p_mc = fit_profile(mc[DEMANDS].to_numpy(float), mc["correct"].to_numpy(float), pc)
        for j, d in enumerate(DEMANDS):
            sens.append({"model": n, "dim": d, "ability_all": p_all[j], "ability_mc_only": p_mc[j],
                         "diff": p_all[j] - p_mc[j]})
    if sens:
        s = pd.DataFrame(sens)
        s.to_csv(rdir / "sensitivity.csv", index=False)
        log.info("Sensitivity (mean |all - MC-only| per model):\n%s",
                 s.groupby("model")["diff"].apply(lambda x: x.abs().mean()).round(2).to_string())
    log.info("Wrote %s and profiles_boot.csv", rdir / "profiles.csv")


if __name__ == "__main__":
    main()
