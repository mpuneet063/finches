"""Build a fake battery + fake model outputs so the whole CPU pipeline can be tested
without the gated data, a GPU or an API key.

    python tests/make_synthetic.py            # writes tests/_synthetic/ and its run.yaml
    bash tests/run_synthetic.sh               # runs every CPU stage on it

Each fake model has a hidden ability per dimension; P(correct) falls with the demand of the
item's dominant dimension. A working pipeline should roughly recover those abilities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402

OUT = ROOT / "tests" / "_synthetic"
rng = np.random.default_rng(0)
D = common.DEMANDS
N = 6000


def battery() -> pd.DataFrame:
    rows = []
    for i in range(N):
        fmt = "MC" if rng.random() < 0.65 else "Open-ended"
        main = rng.integers(len(D))
        lv = int(rng.integers(1, 6))
        dem = np.minimum(rng.integers(0, lv + 1, len(D)), lv)
        dem[main] = lv
        letter = "ABCD"[rng.integers(4)]
        rows.append({
            "source": "synthetic", "benchmark": f"bench{rng.integers(6)}", "task": "t",
            "answer_format": fmt, "instance_id": f"syn-{i}", "original_id": str(i),
            "question": f"q{i}", "prompt": f"Synthetic question {i}. Thus, the correct answer is: ?",
            "groundtruth": letter if fmt == "MC" else f"answer {i}",
            "UG": 75 if fmt == "MC" else 100, **dict(zip(D, dem.tolist())), "verification_final": np.nan,
        })
    return pd.DataFrame(rows)


def p_correct(bat: pd.DataFrame, ability: np.ndarray) -> np.ndarray:
    dem = bat[D].to_numpy(float)
    j = dem.argmax(axis=1)
    return 1 / (1 + np.exp(-1.6 * (ability[j] - dem[np.arange(len(dem)), j])))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bat = battery()
    bat.to_csv(OUT / "battery.csv", index=False)

    cfg = common.load_config()
    for k in cfg["paths"]:
        cfg["paths"][k] = str(OUT / k)
    cfg["paths"]["battery_csv"] = str(OUT / "battery.csv")
    cfg["subset"].update({"n_mc": 1500, "n_open": 700})
    cfg["profiles"]["n_boot"] = 20
    with open(OUT / "run.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    registry = common.load_models()
    base = rng.uniform(2.0, 3.0, len(D))
    own = [m for m in registry["models"] if m["lineage"] in registry["lineages"] or m.get("validates")]
    resp_dir = Path(cfg["paths"]["responses_dir"])
    resp_dir.mkdir(parents=True, exist_ok=True)
    abilities = {}
    for m in own:
        g = m.get("generation", 1) or 1
        ab = base + 0.6 * g + rng.normal(0, 0.3, len(D))
        ab[D.index("MCu")] -= 0.4 * g          # make the RL1 gap widen with generation
        abilities[m["name"]] = ab
    # The validation model behaves like its paper counterpart
    abilities["ref__llama3d2-3b"] = abilities["llama-3.2-3b-instruct"]
    np.save(OUT / "true_abilities.npy", abilities, allow_pickle=True)

    subset_placeholder = bat  # the subset is drawn later by data.py; simulate every item
    for m in own:
        pc = p_correct(subset_placeholder, abilities[m["name"]])
        ok = rng.random(len(pc)) < pc
        rows = []
        for (_, r), c in zip(subset_placeholder.iterrows(), ok):
            if r.answer_format == "MC":
                letter = r.groundtruth if c else "ABCD".replace(r.groundtruth, "")[rng.integers(3)]
                text = f"Reasoning... Thus, the correct answer is: {letter}"
            else:
                text = f"<think>hmm</think>Final: {'answer' if c else 'wrong'}"
            ans, think = common.split_thinking(text)
            rows.append({"custom_id": r.instance_id, "model": m["name"], "answer_format": r.answer_format,
                         "response": ans, "thinking": think, "n_prompt_tokens": 20,
                         "n_output_tokens": 10, "finish_reason": "stop", "status": "ok",
                         "_true_correct": bool(c)})
        with open(resp_dir / f"{m['name']}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # Fake judge output for open-ended items (judge.py itself needs an API key)
    sdir = Path(cfg["paths"]["scores_dir"])
    sdir.mkdir(parents=True, exist_ok=True)
    for m in own:
        rows = [json.loads(l) for l in open(resp_dir / f"{m['name']}.jsonl")]
        oe = pd.DataFrame([{"custom_id": r["custom_id"], "grade": "C" if r["_true_correct"] else "I",
                            "correct": float(r["_true_correct"])}
                           for r in rows if r["answer_format"] == "Open-ended"])
        oe.to_csv(sdir / f"{m['name']}_open.csv", index=False)

    # Fake reference models
    ref_dir = Path(cfg["paths"]["reference_dir"])
    ref_dir.mkdir(parents=True, exist_ok=True)
    for k, rm in enumerate(registry["reference_models"]):
        ab = abilities.get("ref__" + rm["name"], base + rng.uniform(0, 3) + rng.normal(0, 0.4, len(D)))
        pc = p_correct(bat, ab)
        pd.DataFrame({"instance_id": bat.instance_id,
                      "verification_final": (rng.random(len(pc)) < pc).astype(int)}).to_csv(
            ref_dir / f"{rm['name']}.csv", index=False)
    print(f"Synthetic data in {OUT}; config at {OUT / 'run.yaml'}")


if __name__ == "__main__":
    main()
