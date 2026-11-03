"""Grade open-ended responses with an LLM judge (default GPT-4o-mini). Runs on CPU (Codespaces).

    export OPENAI_API_KEY=...
    python src/judge.py                    # every responses/*.jsonl
    python src/judge.py --models qwen
    python src/judge.py --dry-run          # count calls and estimate cost, no API calls

* Template modelled on Inspect AI's model_graded_qa (what the ADeLe toolkit uses):
  task + submission + criterion (= the battery's reference answer) -> GRADE: C / I.
* The judge sees only the final answer; any <think> block was already split off in generate.py.
* Every verdict is cached in cache/judge_<judge>.jsonl keyed by (model, custom_id): reruns
  are free and an interrupted run resumes.
* Hard budget stop: cumulative spend (from cached token counts x configured prices) may not
  exceed judge.budget_usd.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import common
from common import log

TEMPLATE = """You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

Does the submission meet the criterion? The criterion is the reference answer: the submission is correct if its final answer is equivalent to the reference answer, regardless of wording or extra explanation. Ignore the quality of any reasoning; judge only whether the final answer matches.

Reply with at most two sentences of justification, then on the last line write 'GRADE: C' if the submission is correct or 'GRADE: I' if it is incorrect."""

_GRADE = re.compile(r"GRADE:\s*([CI])\b")


def cache_path(cfg: dict, judge_model: str):
    return common.path_of(cfg, "cache_dir") / f"judge_{judge_model}.jsonl"


def load_cache(cfg: dict, judge_model: str) -> dict[tuple[str, str], dict]:
    rows = common.read_jsonl(cache_path(cfg, judge_model))
    return {(r["model"], r["custom_id"]): r for r in rows}


def cost_usd(cfg: dict, judge_model: str, n_in: int, n_out: int) -> float:
    p = cfg["judge"]["prices_per_mtok"][judge_model]
    return n_in / 1e6 * p["input"] + n_out / 1e6 * p["output"]


def spent(cfg: dict, judge_model: str, cache: dict) -> float:
    return sum(cost_usd(cfg, judge_model, r.get("in_tokens", 0), r.get("out_tokens", 0))
               for r in cache.values())


def call_judge(client, judge_model: str, prompt: str, temperature: float, retries: int = 6):
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model, temperature=temperature, max_tokens=250,
                messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content or ""
            u = resp.usage
            return text, u.prompt_tokens, u.completion_tokens
        except Exception as e:  # rate limits, timeouts, 5xx
            if attempt == retries - 1:
                raise
            log.warning("Judge call failed (%s); retrying in %.0fs", type(e).__name__, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def build_prompt(question: str, answer: str, criterion: str, max_chars: int) -> str:
    answer = answer or "(empty response)"
    if len(answer) > max_chars:
        answer = "..." + answer[-max_chars:]
    return TEMPLATE.format(question=question, answer=answer, criterion=criterion)


def judge_items(items: pd.DataFrame, cfg: dict, judge_model: str, dry_run: bool = False) -> pd.DataFrame:
    """items: columns model, custom_id, question, answer, criterion. Returns model, custom_id, grade."""
    jc = cfg["judge"]
    cache = load_cache(cfg, judge_model)
    todo = items[[(m, c) not in cache for m, c in zip(items["model"], items["custom_id"])]]
    already = spent(cfg, judge_model, cache)
    log.info("[judge %s] %d items, %d cached, %d to call. Spent so far: $%.2f / $%.2f",
             judge_model, len(items), len(items) - len(todo), len(todo), already, jc["budget_usd"])

    if dry_run:
        est_in = todo.apply(lambda r: len(build_prompt(r.question, r.answer, r.criterion,
                                                       jc["max_answer_chars"])) / 4, axis=1).sum()
        est = cost_usd(cfg, judge_model, int(est_in), 120 * len(todo))
        print(f"Dry run: {len(todo)} calls, ~{est_in/1e6:.2f}M input tokens, estimated ${est:.2f}")
        return pd.DataFrame(columns=["model", "custom_id", "grade"])

    if len(todo):
        from openai import OpenAI
        client = OpenAI()
        lock = threading.Lock()
        state = {"spent": already, "stopped": already >= jc["budget_usd"]}
        path = cache_path(cfg, judge_model)

        def work(r):
            if state["stopped"]:
                return
            prompt = build_prompt(r.question, r.answer, r.criterion, jc["max_answer_chars"])
            text, n_in, n_out = call_judge(client, judge_model, prompt, jc["temperature"])
            m = _GRADE.findall(text)
            rec = {"model": r.model, "custom_id": r.custom_id,
                   "grade": (m[-1] if m else None), "in_tokens": n_in, "out_tokens": n_out,
                   "judge_text": text}
            with lock:
                common.append_jsonl(path, [rec])
                cache[(r.model, r.custom_id)] = rec
                state["spent"] += cost_usd(cfg, judge_model, n_in, n_out)
                if state["spent"] >= jc["budget_usd"]:
                    state["stopped"] = True

        with ThreadPoolExecutor(max_workers=jc["max_workers"]) as ex:
            futs = [ex.submit(work, r) for r in todo.itertuples(index=False)]
            for i, f in enumerate(as_completed(futs), 1):
                f.result()
                if i % 200 == 0:
                    log.info("[judge %s] %d/%d done, $%.2f spent", judge_model, i, len(todo), state["spent"])
        if state["stopped"]:
            log.error("Budget of $%.2f reached - stopped early. Raise judge.budget_usd to continue.",
                      jc["budget_usd"])

    out = [{"model": m, "custom_id": c, "grade": cache[(m, c)]["grade"]}
           for m, c in zip(items["model"], items["custom_id"]) if (m, c) in cache]
    return pd.DataFrame(out, columns=["model", "custom_id", "grade"])


def open_ended_items(names: list[str], cfg: dict) -> pd.DataFrame:
    bat = common.load_battery(cfg, with_text=True, columns=["instance_id", "answer_format"])
    bat = bat[bat.answer_format == "Open-ended"].set_index("custom_id")
    subset = set(common.load_subset_ids(cfg)["custom_id"])
    rows = []
    for name in names:
        for r in common.read_jsonl(common.path_of(cfg, "responses_dir") / f"{name}.jsonl"):
            if (r["answer_format"] != "Open-ended" or r.get("status") != "ok"
                    or r["custom_id"] not in subset):
                continue
            b = bat.loc[r["custom_id"]]
            rows.append({"model": name, "custom_id": r["custom_id"], "question": str(b["prompt"]),
                         "answer": r["response"], "criterion": str(b["target"])})
    return pd.DataFrame(rows, columns=["model", "custom_id", "question", "answer", "criterion"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--judge", default=None, help="override judge.model")
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    judge_model = args.judge or cfg["judge"]["model"]

    resp_dir = common.path_of(cfg, "responses_dir")
    names = ([m["name"] for m in common.select_models(common.load_models(), args.models)]
             if args.models else sorted(p.stem for p in resp_dir.glob("*.jsonl")))
    items = open_ended_items(names, cfg)
    if items.empty:
        log.info("No open-ended responses found.")
        return
    graded = judge_items(items, cfg, judge_model, dry_run=args.dry_run)
    if args.dry_run:
        return

    out_dir = common.path_of(cfg, "scores_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, g in graded.groupby("model"):
        g = g.assign(correct=g["grade"].map({"C": 1.0, "I": 0.0}).astype(float))
        g[["custom_id", "grade", "correct"]].to_csv(out_dir / f"{name}_open.csv", index=False)
        log.info("[%s] open-ended accuracy %.3f on %d items (%d unparsable verdicts)",
                 name, g["correct"].mean(), g["correct"].notna().sum(), int(g["grade"].isna().sum()))


if __name__ == "__main__":
    main()
