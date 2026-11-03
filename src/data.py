"""Get the ADeLe battery and draw the stratified item subset.

    python src/data.py fetch      # download the gated battery CSV (needs HF_TOKEN + accepted terms)
    python src/data.py subset     # write data/subset_ids.csv (IDs only) + coverage report
    python src/data.py coverage   # print dominant-slice coverage of the current subset

The subset file contains item IDs, answer format and benchmark name only - no item text -
so it can be committed and used on Kaggle.
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

import common
from common import DEMANDS, log


def fetch(cfg: dict) -> None:
    from huggingface_hub import hf_hub_download

    dest = common.path_of(cfg, "battery_csv")
    if dest.exists():
        log.info("Battery already at %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    last_err = None
    for repo in common.HF_REPOS:
        try:
            p = hf_hub_download(repo_id=repo, filename=common.HF_FILE, repo_type="dataset", token=token)
            shutil.copy(p, dest)
            log.info("Downloaded battery from %s -> %s", repo, dest)
            return
        except Exception as e:  # gated / renamed repo
            last_err = e
            log.warning("Could not download from %s: %s", repo, e)
    raise SystemExit(f"Battery download failed. Accept the dataset terms on Hugging Face "
                     f"and check HF_TOKEN. Last error: {last_err}")


def dominant_mask(df: pd.DataFrame, dim: str, level: int, level_diff: int = 0) -> pd.Series:
    """Items whose `dim` demand equals `level` and no other demand exceeds it (paper's slice)."""
    others = [d for d in DEMANDS if d != dim]
    return (df[dim] == level) & (df[others].max(axis=1) <= level + level_diff)


def coverage_table(df: pd.DataFrame, level_diff: int = 0) -> pd.DataFrame:
    rows = []
    for d in DEMANDS:
        row = {"dim": d}
        for lv in range(1, 6):
            row[f"L{lv}"] = int(dominant_mask(df, d, lv, level_diff).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("dim")


def draw_subset(cfg: dict) -> pd.DataFrame:
    s = cfg["subset"]
    rng = np.random.default_rng(cfg["seed"])
    bat = common.load_battery(cfg, with_text=True)
    n0 = len(bat)
    bat["prompt_chars"] = bat["prompt"].astype(str).str.len()
    bat = bat.drop(columns=["prompt", "target"])  # text not needed past this point

    keep = bat["answer_format"].isin(["MC", "Open-ended"])
    keep &= bat["UG"] >= s["min_ug"]
    keep &= bat["prompt_chars"] <= s["max_prompt_chars"]
    keep &= bat[DEMANDS].notna().all(axis=1)
    pool = bat[keep].copy()
    log.info("Eligible items: %d of %d (UG >= %s, prompt <= %s chars)",
             len(pool), n0, s["min_ug"], s["max_prompt_chars"])

    targets = {"MC": s["n_mc"], "Open-ended": s["n_open"]}
    total = sum(targets.values())
    chosen: list[pd.DataFrame] = []
    for fmt, n in targets.items():
        fp = pool[pool["answer_format"] == fmt]
        if len(fp) <= n:
            log.warning("%s: only %d eligible items, taking all", fmt, len(fp))
            chosen.append(fp)
            continue
        per_cell = max(1, round(s["min_per_cell"] * n / total))
        picked: set[str] = set()
        # 1) coverage pass: guarantee items in every (dimension, level) dominant-slice cell
        for d in DEMANDS:
            for lv in range(1, 6):
                cell = fp[dominant_mask(fp, d, lv) & ~fp["custom_id"].isin(picked)]
                if len(cell):
                    take = cell.sample(n=min(per_cell, len(cell)), random_state=int(rng.integers(1e9)))
                    picked.update(take["custom_id"])
        if len(picked) > n:
            log.warning("%s: coverage pass picked %d > target %d; keeping all of them", fmt, len(picked), n)
        # 2) fill the rest at random (proportional to the battery's benchmark mix in expectation)
        rest = fp[~fp["custom_id"].isin(picked)]
        k = max(0, n - len(picked))
        if k:
            fill = rest.sample(n=min(k, len(rest)), random_state=int(rng.integers(1e9)))
            picked.update(fill["custom_id"])
        chosen.append(fp[fp["custom_id"].isin(picked)])

    sub = pd.concat(chosen, ignore_index=True)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "subset", "coverage"])
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = common.load_config(args.config)

    if args.command == "fetch":
        fetch(cfg)
        return

    if args.command == "subset":
        sub = draw_subset(cfg)
        out = common.path_of(cfg, "subset_ids")
        out.parent.mkdir(parents=True, exist_ok=True)
        sub[["custom_id", "answer_format", "benchmark"]].to_csv(out, index=False)
        log.info("Wrote %d item IDs to %s (MC=%d, open-ended=%d)", len(sub), out,
                 (sub.answer_format == "MC").sum(), (sub.answer_format == "Open-ended").sum())
        cov = coverage_table(sub, cfg["profiles"]["level_diff"])
        cov.to_csv(out.with_name("subset_coverage.csv"))
        sparse = cov[(cov[["L1", "L2", "L3", "L4", "L5"]] > 0).sum(axis=1) < 3]
        print("\nDominant-slice items per (dimension, level) in the subset:\n", cov.to_string())
        if len(sparse):
            print("\nWARNING - fewer than 3 populated levels (ability fit will be weak):",
                  ", ".join(sparse.index))
        return

    if args.command == "coverage":
        ids = common.load_subset_ids(cfg)
        bat = common.load_battery(cfg)
        sub = bat[bat["custom_id"].isin(ids["custom_id"])]
        print(coverage_table(sub, cfg["profiles"]["level_diff"]).to_string())


if __name__ == "__main__":
    main()
