"""Q1 - how do lineages' profiles evolve across generations? Diversifying or converging?

    python src/evolution.py

Each profile is split into
  level = mean ability over the core dimensions (11 primordial + 5 knowledge; AT/VO excluded)
  shape = ability - level  (what the model is relatively good or bad at)
Everything rises over time, so convergence is judged on shape, not level.

Outputs:
  results/model_meta.csv     lineage, generation, release, label for every profiled model
  results/growth.csv         per-lineage change in ability per dimension between generations
  results/convergence.csv    mean pairwise shape distance between lineages, per generation (95% CI)
  results/pca.csv            2-D projection of all shape vectors (for the main figure)
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

import common
from common import CORE, log


def model_meta(registry: dict, names: list[str]) -> pd.DataFrame:
    by_name = {m["name"]: m for m in registry["models"]}
    refs = {r["name"]: r for r in registry.get("reference_models", [])}
    rows = []
    for n in names:
        if n.startswith("ref__"):
            r = refs.get(n[5:], {"label": n[5:], "family": "reference", "release": None})
            rows.append({"model": n, "kind": "reference", "lineage": r.get("family"),
                         "generation": np.nan, "release": r.get("release"), "label": r["label"]})
        else:
            m = by_name.get(n, {})
            rows.append({"model": n, "kind": "own", "lineage": m.get("lineage", "unknown"),
                         "generation": m.get("generation", np.nan), "release": m.get("release"),
                         "label": n})
    df = pd.DataFrame(rows)
    df["release_date"] = pd.to_datetime(df["release"], format="%Y-%m", errors="coerce")
    return df


def usable_dims(prof: pd.DataFrame) -> list[str]:
    """Core dimensions estimable (non-NaN) for every model, so all vectors are comparable."""
    wide = prof.pivot(index="model", columns="dim", values="ability")
    return [d for d in CORE if d in wide.columns and wide[d].notna().all()]


def shape_level(vec: np.ndarray) -> tuple[np.ndarray, float]:
    level = float(np.nanmean(vec))
    return vec - level, level


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    registry = common.load_models()
    rdir = common.path_of(cfg, "results_dir")

    prof = pd.read_csv(rdir / "profiles.csv")
    boot = pd.read_csv(rdir / "profiles_boot.csv")
    meta = model_meta(registry, sorted(prof.model.unique()))
    meta.to_csv(rdir / "model_meta.csv", index=False)
    dims = usable_dims(prof)
    log.info("Core dimensions usable for every model (%d): %s", len(dims), dims)
    if len(dims) < 3:
        raise SystemExit("Fewer than 3 dimensions are estimable for every model; enlarge the subset.")

    wide = prof.pivot(index="model", columns="dim", values="ability")[dims]
    lineages = [l for l in registry["lineages"] if l in set(meta.lineage)]
    own = meta[(meta.kind == "own") & meta.lineage.isin(lineages)]

    # ---- growth per dimension between consecutive generations
    g_rows = []
    for lin in lineages:
        seq = own[own.lineage == lin].sort_values("generation")
        for (_, a), (_, b) in zip(seq.iterrows(), seq.iloc[1:].iterrows()):
            for d in dims:
                g_rows.append({"lineage": lin, "from": a.model, "to": b.model,
                               "transition": f"G{int(a.generation)}->G{int(b.generation)}",
                               "dim": d, "delta": wide.loc[b.model, d] - wide.loc[a.model, d]})
    growth = pd.DataFrame(g_rows)
    growth.to_csv(rdir / "growth.csv", index=False)
    if len(growth):
        mean_growth = growth.groupby("dim")["delta"].mean().sort_values(ascending=False)
        log.info("Mean growth per generation step (all lineages):\n%s", mean_growth.round(2).to_string())

    # ---- convergence of shapes between lineages, per generation (with bootstrap CI)
    bwide = boot[boot.dim.isin(dims)].pivot_table(index=["model", "rep"], columns="dim",
                                                   values="ability")[dims]
    reps = sorted(boot.rep.unique())
    c_rows = []
    for g in sorted(own.generation.dropna().unique()):
        members = own[own.generation == g].set_index("lineage")["model"]
        if len(members) < 2:
            continue
        pairs = list(combinations(members.index, 2))
        point_shapes = {l: shape_level(wide.loc[m].to_numpy())[0] for l, m in members.items()}
        point_level = {l: shape_level(wide.loc[m].to_numpy())[1] for l, m in members.items()}
        d_point = np.mean([rms(point_shapes[a], point_shapes[b]) for a, b in pairs])
        lv_point = np.mean([abs(point_level[a] - point_level[b]) for a, b in pairs])
        d_boot = []
        for r in reps:
            try:
                sh = {l: shape_level(bwide.loc[(m, r)].to_numpy())[0] for l, m in members.items()}
            except KeyError:
                continue
            vals = [rms(sh[a], sh[b]) for a, b in pairs]
            if np.all(np.isfinite(vals)):
                d_boot.append(np.mean(vals))
        c_rows.append({"generation": int(g), "n_lineages": len(members),
                       "shape_distance": d_point,
                       "lo": np.percentile(d_boot, 2.5) if d_boot else np.nan,
                       "hi": np.percentile(d_boot, 97.5) if d_boot else np.nan,
                       "level_gap": lv_point})
    conv = pd.DataFrame(c_rows)
    conv.to_csv(rdir / "convergence.csv", index=False)
    if len(conv):
        log.info("Shape distance between lineages by generation (RMS, ability units):\n%s",
                 conv.round(3).to_string(index=False))
        if len(conv) >= 2:
            trend = "converging" if conv.shape_distance.iloc[-1] < conv.shape_distance.iloc[0] else "diverging"
            log.info("First vs last generation: %s (check whether the CIs overlap before claiming it)", trend)

    # ---- 2-D projection of shapes (own + reference models)
    shapes = np.vstack([shape_level(wide.loc[m].to_numpy())[0] for m in wide.index])
    X = shapes - shapes.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    levels = [shape_level(wide.loc[m].to_numpy())[1] for m in wide.index]
    pca = pd.DataFrame({"model": wide.index, "pc1": coords[:, 0], "pc2": coords[:, 1], "level": levels})
    pca.attrs["explained"] = var[:2]
    pca.to_csv(rdir / "pca.csv", index=False)
    loadings = pd.DataFrame(Vt[:2].T, index=dims, columns=["pc1", "pc2"])
    loadings.to_csv(rdir / "pca_loadings.csv")
    with open(rdir / "pca_variance.txt", "w") as f:
        f.write(f"{var[0]:.4f} {var[1]:.4f}\n")
    log.info("PCA on shapes: PC1 %.0f%%, PC2 %.0f%% of variance", 100 * var[0], 100 * var[1])


if __name__ == "__main__":
    main()
