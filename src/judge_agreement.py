"""Check the cheap judge against a stronger one on a random sample (default 200 items).

    python src/judge_agreement.py           # GPT-4o-mini vs GPT-4o, ~$2-3

Reports raw agreement, Cohen's kappa and the confusion table, and writes
results/judge_agreement.json. Run judge.py first (the mini verdicts are reused from cache).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import common
import judge
from common import log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    jc = cfg["judge"]
    n = args.n or jc["agreement_n"]

    names = sorted(p.stem for p in common.path_of(cfg, "responses_dir").glob("*.jsonl"))
    items = judge.open_ended_items(names, cfg)
    if items.empty:
        raise SystemExit("No open-ended responses to compare.")
    sample = items.sample(n=min(n, len(items)), random_state=cfg["seed"])

    a = judge.judge_items(sample, cfg, jc["model"]).rename(columns={"grade": "cheap"})
    b = judge.judge_items(sample, cfg, jc["agreement_model"]).rename(columns={"grade": "strong"})
    m = a.merge(b, on=["model", "custom_id"]).dropna(subset=["cheap", "strong"])
    if m.empty:
        raise SystemExit("No overlapping verdicts.")

    agree = float((m.cheap == m.strong).mean())
    kappa = float(cohen_kappa_score(m.strong, m.cheap)) if m.strong.nunique() > 1 else float("nan")
    confusion = pd.crosstab(m.strong, m.cheap, rownames=[jc["agreement_model"]], colnames=[jc["model"]])
    rate_cheap = float((m.cheap == "C").mean())
    rate_strong = float((m.strong == "C").mean())
    report = {"cheap_judge": jc["model"], "strong_judge": jc["agreement_model"], "n": int(len(m)),
              "agreement": round(agree, 3), "cohen_kappa": round(kappa, 3),
              "pass_rate_cheap": round(rate_cheap, 3), "pass_rate_strong": round(rate_strong, 3),
              "confusion": confusion.to_dict()}
    out = common.path_of(cfg, "results_dir") / "judge_agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(confusion.to_string())
    log.info("Agreement %.1f%%, kappa %.2f on %d items (pass rate %s %.1f%% vs %s %.1f%%) -> %s",
             100 * agree, kappa, len(m), jc["model"], 100 * rate_cheap, jc["agreement_model"],
             100 * rate_strong, out)
    if kappa == kappa and kappa < 0.6:
        log.warning("Kappa below 0.6: inspect the disagreements in the judge caches before "
                    "trusting the cheap judge.")


if __name__ == "__main__":
    main()
