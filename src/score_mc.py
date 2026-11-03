"""Score multiple-choice responses by option-letter match (deterministic, free).

    python src/score_mc.py              # every responses/*.jsonl
    python src/score_mc.py --models llama

Writes scores/<model>_mc.csv with custom_id, pred, gold, correct.
Items skipped for length are left out (correct = NaN), so they never count as failures.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common
from common import log


def score_model(name: str, cfg: dict, gold: dict[str, str]) -> pd.DataFrame:
    rows = common.read_jsonl(common.path_of(cfg, "responses_dir") / f"{name}.jsonl")
    out = []
    for r in rows:
        if r["answer_format"] != "MC":
            continue
        g = common.choice_letter(gold.get(r["custom_id"], ""))
        if r.get("status") != "ok":
            out.append({"custom_id": r["custom_id"], "pred": None, "gold": g, "correct": np.nan})
            continue
        p = common.choice_letter(r["response"])
        out.append({"custom_id": r["custom_id"], "pred": p, "gold": g,
                    "correct": float(p is not None and g is not None and p == g)})
    return pd.DataFrame(out, columns=["custom_id", "pred", "gold", "correct"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="names/lineages (default: every responses file)")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = common.load_config(args.config)

    resp_dir = common.path_of(cfg, "responses_dir")
    if args.models:
        names = [m["name"] for m in common.select_models(common.load_models(), args.models)]
    else:
        names = sorted(p.stem for p in resp_dir.glob("*.jsonl"))
    bat = common.load_battery(cfg, with_text=True, columns=["instance_id", "answer_format"])
    gold = dict(zip(bat.loc[bat.answer_format == "MC", "custom_id"],
                    bat.loc[bat.answer_format == "MC", "target"].astype(str)))

    out_dir = common.path_of(cfg, "scores_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        df = score_model(name, cfg, gold)
        if df.empty:
            log.info("[%s] no MC responses", name)
            continue
        df.to_csv(out_dir / f"{name}_mc.csv", index=False)
        no_letter = df["pred"].isna() & df["correct"].notna()
        log.info("[%s] MC accuracy %.3f on %d items (%d with no extractable letter)",
                 name, df["correct"].mean(), df["correct"].notna().sum(), int(no_letter.sum()))


if __name__ == "__main__":
    main()
