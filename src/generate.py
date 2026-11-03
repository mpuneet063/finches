"""Run open-weight models on the item subset and save raw responses (GPU step, run on Kaggle).

    python src/generate.py --models llama                  # one lineage
    python src/generate.py --models qwen,mistral           # several lineages
    python src/generate.py --models all
    python src/generate.py --models smoke --limit 5 --backend hf   # CPU smoke test

Design:
  * Each model runs in its own subprocess, so GPU memory is fully released between models.
  * Responses are appended to responses/<model>.jsonl after every chunk; a rerun skips
    items that already have a response (resume after a Kaggle timeout or crash).
  * Every model gets the identical input: the battery's own prompt as a single user turn,
    the model's chat template, greedy decoding (temperature 0), same max_new_tokens, fp16.
  * Weights are deleted from the HF cache after each model (Kaggle disk is small).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import common
from common import log


# --------------------------------------------------------------------------- helpers
def hf_cache_dir() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def delete_weights(hf_id: str) -> None:
    d = hf_cache_dir() / ("models--" + hf_id.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        log.info("Deleted cached weights for %s (%.1f GB free now)", hf_id, free_gb(hf_cache_dir()))


def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if gpu_count() == 0:
        return "hf"
    try:
        import vllm  # noqa: F401
        return "vllm"
    except Exception as e:
        log.warning("vLLM not importable (%s); falling back to transformers", e)
        return "hf"


# --------------------------------------------------------------------------- one model
def build_inputs(tokenizer, prompts: list[str], template_kwargs: dict) -> list[list[int]]:
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("This tokenizer has no chat template; add one before running.")
    ids = []
    for p in prompts:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True, **template_kwargs,
        )
        # Newer transformers can return a BatchEncoding/dict instead of a list.
        if hasattr(out, "keys") and "input_ids" in out:
            out = out["input_ids"]
        ids.append(list(out))
    return ids


def run_vllm(model: dict, gcfg: dict, inputs: list[list[int]], max_new: int):
    from vllm import LLM, SamplingParams

    vc = gcfg["vllm"]
    tp = max(1, min(vc["tensor_parallel_size"], gpu_count()))
    llm = LLM(model=model["hf_id"], dtype=gcfg["dtype"], tensor_parallel_size=tp,
              gpu_memory_utilization=vc["gpu_memory_utilization"],
              max_model_len=vc["max_model_len"], seed=0, trust_remote_code=False)
    sp = SamplingParams(temperature=gcfg["temperature"], max_tokens=max_new, seed=0)

    def gen(batch_ids):
        outs = llm.generate([{"prompt_token_ids": x} for x in batch_ids], sp, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]

    return gen, vc["chunk_size"]


def run_hf(model: dict, gcfg: dict, tokenizer, max_new: int):
    import torch
    from transformers import AutoModelForCausalLM

    use_gpu = gpu_count() > 0
    dtype = getattr(torch, gcfg["dtype"]) if use_gpu else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(
        model["hf_id"], torch_dtype=dtype, device_map="auto" if use_gpu else None)
    lm.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad = tokenizer.pad_token_id
    device = next(lm.parameters()).device

    @torch.inference_mode()
    def gen(batch_ids):
        width = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(batch_ids), width), pad, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, x in enumerate(batch_ids):            # left padding
            input_ids[i, width - len(x):] = torch.tensor(x)
            attn[i, width - len(x):] = 1
        out = lm.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                          max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        res = []
        for row in out[:, width:]:
            toks = row.tolist()
            eos = tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            cut = next((i for i, t in enumerate(toks) if t in eos_ids or t == pad), len(toks))
            finish = "stop" if cut < len(toks) else "length"
            res.append((tokenizer.decode(toks[:cut], skip_special_tokens=True), cut, finish))
        return res

    return gen, gcfg["hf"]["batch_size"]


def run_one(model: dict, cfg: dict, backend: str, limit: int | None) -> None:
    from transformers import AutoTokenizer

    gcfg = cfg["generation"]
    out_path = common.path_of(cfg, "responses_dir") / f"{model['name']}.jsonl"
    done = {r["custom_id"] for r in common.read_jsonl(out_path)}

    ids = common.load_subset_ids(cfg)
    bat = common.load_battery(cfg, with_text=True, columns=["instance_id", "answer_format"])
    bat = bat[bat["custom_id"].isin(ids["custom_id"])]
    if limit:
        bat = bat.groupby("answer_format", group_keys=False).head(limit)
    todo = bat[~bat["custom_id"].isin(done)].reset_index(drop=True)
    log.info("[%s] %d items in subset, %d already done, %d to run",
             model["name"], len(bat), len(done & set(bat["custom_id"])), len(todo))
    if todo.empty:
        return

    need = gcfg["min_free_disk_gb"]
    if free_gb(hf_cache_dir()) < need:
        raise SystemExit(f"Less than {need} GB free in {hf_cache_dir()}; free space first.")

    max_new = int(model.get("max_new_tokens", gcfg["max_new_tokens"]))
    tok = AutoTokenizer.from_pretrained(model["hf_id"])
    inputs = build_inputs(tok, todo["prompt"].astype(str).tolist(), model.get("chat_template_kwargs", {}))
    max_len = gcfg["vllm"]["max_model_len"]

    gen, chunk = (run_vllm(model, gcfg, inputs, max_new) if backend == "vllm"
                  else run_hf(model, gcfg, tok, max_new))

    t0 = time.time()
    n_done = 0
    for start in range(0, len(todo), chunk):
        rows_idx = list(range(start, min(start + chunk, len(todo))))
        fits = [i for i in rows_idx if len(inputs[i]) + max_new <= max_len]
        too_long = [i for i in rows_idx if i not in fits]
        results = dict(zip(fits, gen([inputs[i] for i in fits]))) if fits else {}
        records = []
        for i in rows_idx:
            r = todo.iloc[i]
            if i in too_long:
                records.append({"custom_id": r["custom_id"], "model": model["name"],
                                "answer_format": r["answer_format"], "response": "", "thinking": "",
                                "n_prompt_tokens": len(inputs[i]), "n_output_tokens": 0,
                                "finish_reason": "skipped", "status": "too_long"})
                continue
            text, n_out, finish = results[i]
            answer, thinking = common.split_thinking(text)
            records.append({"custom_id": r["custom_id"], "model": model["name"],
                            "answer_format": r["answer_format"], "response": answer,
                            "thinking": thinking, "n_prompt_tokens": len(inputs[i]),
                            "n_output_tokens": n_out, "finish_reason": finish, "status": "ok"})
        common.append_jsonl(out_path, records)
        n_done += len(records)
        rate = n_done / max(time.time() - t0, 1e-6)
        log.info("[%s] %d/%d (%.2f items/s, ETA %.0f min)", model["name"], n_done, len(todo),
                 rate, (len(todo) - n_done) / max(rate, 1e-6) / 60)

    meta = {"model": model["name"], "hf_id": model["hf_id"], "backend": backend,
            "dtype": gcfg["dtype"], "temperature": gcfg["temperature"], "max_new_tokens": max_new,
            "chat_template_kwargs": model.get("chat_template_kwargs", {}),
            "minutes": round((time.time() - t0) / 60, 1)}
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="names/lineages, 'all' or 'smoke'")
    ap.add_argument("--config")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"])
    ap.add_argument("--limit", type=int, help="max items per answer format (smoke tests)")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--one", help=argparse.SUPPRESS)   # internal: run a single model in-process
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    registry = common.load_models()
    backend = pick_backend(args.backend or cfg["generation"]["backend"])

    if args.one:
        model = next(m for m in registry["models"] if m["name"] == args.one)
        run_one(model, cfg, backend, args.limit)
        return

    models = common.select_models(registry, args.models)
    log.info("Backend: %s | GPUs: %d | models: %s", backend, gpu_count(), [m["name"] for m in models])
    failed = []
    for m in models:
        cmd = [sys.executable, __file__, "--models", m["name"], "--one", m["name"], "--backend", backend]
        if args.config:
            cmd += ["--config", args.config]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        rc = subprocess.call(cmd, env=os.environ.copy())
        if rc != 0:
            log.error("[%s] failed with exit code %d (continuing with the next model)", m["name"], rc)
            failed.append(m["name"])
        if cfg["generation"]["delete_weights_after"] and not args.keep_weights:
            delete_weights(m["hf_id"])
    if failed:
        log.error("Failed models: %s - rerun them; finished items are kept.", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
