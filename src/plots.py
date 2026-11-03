"""Figures for the write-up (PNG, light background).

    python src/plots.py

  figures/fig1_main.png        A: lineage trajectories in profile-shape space; B: RL1 gap over time
  figures/fig2_radars.png      ability profiles per lineage, generations light -> dark
  figures/fig3_growth.png      change in ability per dimension between generations
  figures/fig4_validation.png  our run of Llama-3.2-3B vs the paper's run (if available)
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

import common
from common import DIM_NAMES, log

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
REF = "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def shade(hex_colour: str, t: float) -> tuple:
    """Blend a hue towards the surface: t=1 full colour, t=0 surface."""
    c, s = np.array(to_rgb(hex_colour)), np.array(to_rgb(SURFACE))
    return tuple(s + (c - s) * t)


def load(rdir):
    meta = pd.read_csv(rdir / "model_meta.csv", parse_dates=["release_date"])
    prof = pd.read_csv(rdir / "profiles.csv")
    return meta, prof


def fig_main(rdir, fdir, registry, cfg):
    meta, _ = load(rdir)
    pca = pd.read_csv(rdir / "pca.csv").merge(meta, on="model")
    var = [float(v) for v in open(rdir / "pca_variance.txt").read().split()]
    rl = pd.read_csv(rdir / "redlines.csv", parse_dates=["release_date"])
    rl1 = rl[rl.redline == "RL1"]
    colours = {k: v["colour"] for k, v in registry["lineages"].items()}
    labels = {k: v["label"] for k, v in registry["lineages"].items()}

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.05, 1]})

    # --- A: trajectories in shape space
    ref = pca[pca.kind == "reference"]
    a.scatter(ref.pc1, ref.pc2, s=26, color=REF, zorder=2, edgecolor=SURFACE, linewidth=1)
    for _, r in ref.iterrows():
        if r.label in {"GPT-4o", "o1 (low effort)", "Llama-3.1-405B", "Babbage-002", "R1-Distill-Qwen-32B"}:
            a.annotate(r.label, (r.pc1, r.pc2), xytext=(4, 3), textcoords="offset points",
                       fontsize=7, color=INK2)
    for lin, col in colours.items():
        seq = pca[(pca.kind == "own") & (pca.lineage == lin)].sort_values("generation")
        if seq.empty:
            continue
        a.plot(seq.pc1, seq.pc2, color=col, lw=2, zorder=3)
        for (_, p), (_, q) in zip(seq.iterrows(), seq.iloc[1:].iterrows()):
            a.annotate("", xy=(q.pc1, q.pc2), xytext=(p.pc1, p.pc2),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=12), zorder=4)
        a.scatter(seq.pc1, seq.pc2, s=[40 + 25 * g for g in seq.generation], color=col, zorder=5,
                  edgecolor=SURFACE, linewidth=1.5)
        last = seq.iloc[-1]
        a.annotate(labels[lin], (last.pc1, last.pc2), xytext=(6, -10), textcoords="offset points",
                   fontsize=8, color=INK, fontweight="bold")
    a.set_xlabel(f"Shape PC1 ({100 * var[0]:.0f}% of variance)")
    a.set_ylabel(f"Shape PC2 ({100 * var[1]:.0f}%)")
    a.set_title("A  Profile shape by lineage, generation 1 → 3")
    a.scatter([], [], s=26, color=REF, label="Nature-paper models")
    a.legend(loc="lower left", fontsize=7)

    # --- B: RL1 gap over release date
    thr = cfg["redlines"]["rl1_knowledge_minus_mcu"]
    top = max(thr + 0.3, np.nanmax(rl1.hi.to_numpy(float)) + 0.2)
    b.axhspan(thr, top, color=shade("#e34948", 0.12), zorder=0)
    b.axhline(thr, color="#e34948", lw=1, ls="--", zorder=1)
    b.text(rl1.release_date.min(), thr, "  illustrative red line", va="bottom", fontsize=7, color="#b8302f")
    refr = rl1[rl1.kind == "reference"]
    b.scatter(refr.release_date, refr.gap, s=22, color=REF, zorder=2, edgecolor=SURFACE, linewidth=1)
    for lin, col in colours.items():
        seq = rl1[(rl1.kind == "own") & (rl1.lineage == lin)].sort_values("release_date")
        if seq.empty:
            continue
        b.errorbar(seq.release_date, seq.gap, yerr=[seq.gap - seq.lo, seq.hi - seq.gap], color=col,
                   lw=2, marker="o", ms=6, capsize=0, elinewidth=1.2, zorder=3, label=labels[lin])
    b.set_ylabel("Knowledge ability − MCu ability")
    b.set_ylim(top=top)
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    b.set_title("B  RL1: knows more than it can tell it knows")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(fdir / "fig1_main.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_radars(rdir, fdir, registry):
    meta, prof = load(rdir)
    dims = [d for d in common.CORE if prof[prof.dim == d].ability.notna().all()]
    wide = prof.pivot(index="model", columns="dim", values="ability")
    lins = [l for l in registry["lineages"] if l in set(meta[meta.kind == "own"].lineage)]
    if not lins:
        return
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    vmax = np.nanmax(wide[dims].to_numpy()) * 1.05
    fig, axes = plt.subplots(1, len(lins), figsize=(4 * len(lins), 4.8), subplot_kw={"polar": True})
    axes = np.atleast_1d(axes)
    for ax, lin in zip(axes, lins):
        col = registry["lineages"][lin]["colour"]
        seq = meta[(meta.kind == "own") & (meta.lineage == lin)].sort_values("generation")
        n = len(seq)
        for k, (_, r) in enumerate(seq.iterrows()):
            v = wide.loc[r.model, dims].to_numpy()
            c = shade(col, 0.35 + 0.65 * (k + 1) / n)
            ax.plot(np.r_[ang, ang[:1]], np.r_[v, v[:1]], color=c, lw=2, label=r.model)
        ax.set_xticks(ang)
        ax.set_xticklabels(dims, fontsize=7, color=INK2)
        ax.set_ylim(0, vmax)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(registry["lineages"][lin]["label"], pad=22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7)
    fig.tight_layout()
    fig.savefig(fdir / "fig2_radars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_growth(rdir, fdir):
    g = pd.read_csv(rdir / "growth.csv")
    if g.empty:
        return
    g["col"] = g.lineage + " " + g.transition
    mat = g.pivot(index="dim", columns="col", values="delta")
    mat = mat.reindex([d for d in common.CORE if d in mat.index])
    lim = np.nanmax(np.abs(mat.to_numpy())) or 1
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#efeeea", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(1.1 * mat.shape[1] + 2.5, 0.32 * mat.shape[0] + 1.4))
    im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"{d}  {DIM_NAMES[d]}" for d in mat.index], fontsize=7)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if np.isfinite(v) and abs(v) >= 0.5 * lim:
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Change in ability (paper scale)", color=INK2)
    ax.set_title("Ability change between consecutive generations")
    fig.tight_layout()
    fig.savefig(fdir / "fig3_growth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_validation(rdir, fdir, registry):
    prof = pd.read_csv(rdir / "profiles.csv")
    pairs = [(m["name"], "ref__" + m["validates"]) for m in registry["models"] if m.get("validates")]
    pairs = [(a, b) for a, b in pairs if a in set(prof.model) and b in set(prof.model)]
    if not pairs:
        return
    ours, theirs = pairs[0]
    dims = [d for d in common.DEMANDS
            if prof[(prof.model.isin([ours, theirs])) & (prof.dim == d)].ability.notna().all()]
    p = prof.set_index(["model", "dim"])
    x = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for off, name, col, lab in [(-0.18, ours, "#2a78d6", "our run"), (0.18, theirs, REF, "paper's run")]:
        v = np.array([p.loc[(name, d), "ability"] for d in dims])
        lo = np.array([p.loc[(name, d), "lo"] for d in dims])
        hi = np.array([p.loc[(name, d), "hi"] for d in dims])
        ax.bar(x + off, v, width=0.34, color=col, label=lab, zorder=2, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(x + off, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2, lw=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=7)
    ax.set_ylabel("Ability")
    ax.set_title(f"Pipeline check: {ours} vs the paper's instance-level results (same items)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fdir / "fig4_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    registry = common.load_models()
    rdir = common.path_of(cfg, "results_dir")
    fdir = common.path_of(cfg, "figures_dir")
    fdir.mkdir(parents=True, exist_ok=True)
    fig_main(rdir, fdir, registry, cfg)
    fig_radars(rdir, fdir, registry)
    fig_growth(rdir, fdir)
    fig_validation(rdir, fdir, registry)
    log.info("Figures written to %s", fdir)


if __name__ == "__main__":
    main()
