"""Shared helpers: config, paths, model registry, battery loading, answer extraction.

Every other script does `import common` (scripts are run as `python src/<name>.py`,
which puts src/ on sys.path).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

# The 18 DeLeAn v1.0 demand dimensions, in the paper's order.
DEMANDS = [
    "AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS", "QLl", "QLq",
    "SNs", "KNa", "KNc", "KNf", "KNn", "KNs", "AT", "VO",
]
PRIMORDIAL = ["AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS", "QLl", "QLq", "SNs"]
KNOWLEDGE = ["KNa", "KNc", "KNf", "KNn", "KNs"]
EXTRANEOUS = ["AT", "VO"]
CORE = PRIMORDIAL + KNOWLEDGE          # used for profile level/shape (extraneous excluded)

DIM_NAMES = {
    "AS": "Attention & scan", "CEc": "Verbal comprehension", "CEe": "Verbal expression",
    "CL": "Conceptualisation & abstraction", "MCr": "Identifying relevant info",
    "MCt": "Critical thinking", "MCu": "Calibrating knowns/unknowns",
    "MS": "Mind modelling", "QLl": "Logical reasoning", "QLq": "Quantitative reasoning",
    "SNs": "Spatio-physical reasoning", "KNa": "Applied sciences", "KNc": "Everyday knowledge",
    "KNf": "Formal sciences", "KNn": "Natural sciences", "KNs": "Social sciences",
    "AT": "Atypicality", "VO": "Volume",
}

# Battery columns that are safe to keep in memory without the item text.
META_COLS = ["instance_id", "answer_format", "benchmark", "source", "task", "UG"]

HF_REPOS = ["adele-ai-evaluations/ADeLe_battery_v1dot0", "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"]
HF_FILE = "ADeLe_batterry_v1dot0.csv"   # upstream spelling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("adele-lineages")


# --------------------------------------------------------------------------- config
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "run.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def path_of(cfg: dict, key: str) -> Path:
    """Resolve a configured path relative to the repo root (env override for the battery)."""
    if key == "battery_csv" and os.environ.get("ADELE_BATTERY_CSV"):
        return Path(os.environ["ADELE_BATTERY_CSV"])
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def load_models(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "models.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def select_models(registry: dict, spec: str) -> list[dict]:
    """Resolve a --models spec.

    spec is a comma-separated list of model names and/or lineage names, or one of:
      all   - every non-optional, non-smoke model
      smoke - the smoke-test model
    """
    models = registry["models"]
    out: list[dict] = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token == "all":
            hits = [m for m in models if not m.get("optional") and not m.get("smoke")]
        elif token == "smoke":
            hits = [m for m in models if m.get("smoke")]
        else:
            hits = [m for m in models if m["name"] == token or m["lineage"] == token]
        if not hits:
            raise SystemExit(f"--models: nothing matches {token!r}")
        out.extend(h for h in hits if h not in out)
    return out


# --------------------------------------------------------------------------- battery
def load_battery(cfg: dict, *, with_text: bool = False, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the ADeLe battery. Text columns (prompt, groundtruth) only when with_text=True.

    Always returns `custom_id` (= the battery's instance_id).
    """
    path = path_of(cfg, "battery_csv")
    if not path.exists():
        raise SystemExit(f"Battery not found at {path}. Run `python src/data.py fetch` "
                         "(needs HF_TOKEN and the accepted dataset terms) or set ADELE_BATTERY_CSV.")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    want = list(columns) if columns else META_COLS + DEMANDS
    if with_text:
        want = want + ["prompt", "groundtruth"]
    usecols = [c for c in dict.fromkeys(want) if c in header]
    df = pd.read_csv(path, usecols=usecols)
    df = df.rename(columns={"instance_id": "custom_id", "groundtruth": "target"})
    df["custom_id"] = df["custom_id"].astype(str)
    for c in DEMANDS + ["UG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_subset_ids(cfg: dict) -> pd.DataFrame:
    p = path_of(cfg, "subset_ids")
    if not p.exists():
        raise SystemExit(f"{p} not found. Run `python src/data.py subset` first.")
    df = pd.read_csv(p, dtype={"custom_id": str})
    return df


# --------------------------------------------------------------------------- answer extraction
# Ported from the ADeLe toolkit (MIT, Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation,
# src/adele/evaluation/scoring.py). v1.0 prompts ask the model to end with
# "Thus, the correct answer is: X".
_MARKER = re.compile(r"correct answer is:?\s*", re.IGNORECASE)
_LETTER = re.compile(r"^\s*\(?([A-Za-z])\)?[\.\):]")
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def final_segment(text: str) -> str:
    matches = list(_MARKER.finditer(text or ""))
    return text[matches[-1].end():].strip() if matches else (text or "").strip()


def choice_letter(text: str) -> str | None:
    segment = final_segment(text)
    m = _LETTER.match(segment)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Za-z])\b", segment)
    return m.group(1).upper() if m else None


def split_thinking(text: str) -> tuple[str, str]:
    """Return (answer_without_think_blocks, thinking_text)."""
    text = text or ""
    thinking = "\n".join(m.group(1).strip() for m in _THINK.finditer(text))
    answer = _THINK.sub("", text)
    # An unterminated <think> (generation cut off mid-thought) leaves no usable answer.
    if "<think>" in answer.lower():
        idx = answer.lower().index("<think>")
        thinking = (thinking + "\n" + answer[idx + 7:]).strip()
        answer = answer[:idx]
    return answer.strip(), thinking


# --------------------------------------------------------------------------- jsonl
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping a corrupt line in %s (likely an interrupted write)", path)
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
