"""Merge MC + judged scores into one correctness file per model, add the paper's models.

    python src/collect.py                 # reference models restricted to our item subset
    python src/collect.py --ref-full      # reference models on the full battery instead

Outputs (no item text):
  results/correctness/<model>.csv          custom_id, answer_format, correct
  results/correctness/ref__<name>.csv      same, for the Nature-paper models
  results/validation.json                  item-level agreement of our run with the paper's run
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import common
from common import log


def own_models(cfg: dict) -> dict[str, pd.DataFrame]:
    sdir = common.path_of(cfg, "scores_dir")
    names = sorted({p.stem.rsplit("_", 1)[0] for p in sdir.glob("*_mc.csv")} |
                   {p.stem.rsplit("_", 1)[0] for p in sdir.glob("*_open.csv")})
    out = {}
    for n in names:
        parts = []
        for suffix, fmt in [("mc", "MC"), ("open", "Open-ended")]:
            p = sdir / f"{n}_{suffix}.csv"
            if p.exists():
                d = pd.read_csv(p, dtype={"custom_id": str})[["custom_id", "correct"]]
                parts.append(d.assign(answer_format=fmt))
        df = pd.concat(parts, ignore_index=True).dropna(subset=["correct"])
        if (df.answer_format == "Open-ended").sum() == 0:
            log.warning("[%s] no judged open-ended items yet - profile will be MC only", n)
        out[n] = df[["custom_id", "answer_format", "correct"]]
    return out


def read_reference(path, fmt_lookup: dict[str, str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_col = next((c for c in ["instance_id", "custom_id"] if c in header), None)
    y_col = next((c for c in ["verification_final", "correct"] if c in header), None)
    if id_col is None or y_col is None:
        raise SystemExit(f"{path.name}: expected an id column (instance_id) and a correctness "
                         f"column (verification_final); found {header[:12]}...")
    d = pd.read_csv(path, usecols=[id_col, y_col], dtype={id_col: str})
    d = d.rename(columns={id_col: "custom_id", y_col: "correct"})
    d["correct"] = pd.to_numeric(d["correct"], errors="coerce")
    d["answer_format"] = d["custom_id"].map(fmt_lookup)
    return d.dropna(subset=["correct"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ref-full", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    registry = common.load_models()

    out_dir = common.path_of(cfg, "results_dir") / "correctness"
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = set(common.load_subset_ids(cfg)["custom_id"])
    bat = common.load_battery(cfg, columns=["instance_id", "answer_format"])
    fmt_lookup = dict(zip(bat.custom_id, bat.answer_format))

    own = own_models(cfg)
    for n, df in own.items():
        df = own[n] = df[df.custom_id.isin(subset)]
        df.to_csv(out_dir / f"{n}.csv", index=False)
        log.info("[%s] %d scored items (accuracy %.3f)", n, len(df), df.correct.mean())

    ref_dir = common.path_of(cfg, "reference_dir")
    refs = {}
    for rm in registry.get("reference_models", []):
        p = ref_dir / f"{rm['name']}.csv"
        if not p.exists():
            continue
        d = read_reference(p, fmt_lookup)
        if not args.ref_full:
            d = d[d.custom_id.isin(subset)]
        d.to_csv(out_dir / f"ref__{rm['name']}.csv", index=False)
        refs[rm["name"]] = d
        log.info("[ref %s] %d items", rm["name"], len(d))
    if not refs:
        log.warning("No reference CSVs in %s (run scripts/fetch_reference.sh) - continuing without", ref_dir)

    # Item-level validation: our run of a paper model vs the paper's run of it.
    report = {}
    for m in registry["models"]:
        target = m.get("validates")
        if not target or m["name"] not in own or target not in refs:
            continue
        mine = own[m["name"]].set_index("custom_id")["correct"]
        theirs = refs[target].set_index("custom_id")["correct"]
        common_ids = mine.index.intersection(theirs.index)
        if len(common_ids) == 0:
            continue
        rep = {"n_items": int(len(common_ids)),
               "item_agreement": round(float((mine[common_ids] == theirs[common_ids]).mean()), 3),
               "accuracy_ours": round(float(mine[common_ids].mean()), 3),
               "accuracy_paper": round(float(theirs[common_ids].mean()), 3)}
        for fmt in ["MC", "Open-ended"]:
            ids = [i for i in common_ids if fmt_lookup.get(i) == fmt]
            if ids:
                rep[f"item_agreement_{fmt}"] = round(float((mine[ids] == theirs[ids]).mean()), 3)
        report[f"{m['name']} vs {target}"] = rep
        log.info("Validation %s vs paper %s: %s", m["name"], target, rep)
    if report:
        with open(common.path_of(cfg, "results_dir") / "validation.json", "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
