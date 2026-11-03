"""Sanity check on synthetic data: do fitted abilities track the hidden ones?"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "_synthetic"
true = np.load(OUT / "true_abilities.npy", allow_pickle=True).item()
prof = pd.read_csv(OUT / "results_dir" / "profiles.csv")
wide = prof.pivot(index="model", columns="dim", values="ability")
dims = pd.read_csv(OUT / "results_dir" / "profiles.csv").dim.unique().tolist()

rows = []
for name, ab in true.items():
    if name not in wide.index:
        continue
    est = wide.loc[name, dims].to_numpy(float)
    ok = np.isfinite(est)
    rho = spearmanr(ab[ok], est[ok]).statistic
    rows.append({"model": name, "spearman_true_vs_fit": rho, "mean_true": ab.mean(), "mean_fit": np.nanmean(est)})
r = pd.DataFrame(rows)
print(r.round(2).to_string(index=False))
lvl = spearmanr(r.mean_true, r.mean_fit).statistic
print(f"Rank correlation of overall level across models: {lvl:.2f}")
assert r.spearman_true_vs_fit.median() > 0.5, "per-dimension recovery too weak"
assert lvl > 0.8, "model levels not recovered"
