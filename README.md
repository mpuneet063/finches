# ADeLe lineages: how do open-model ability profiles evolve?

**Question.** The *diversity* standpoint of Hernández-Orallo et al., "The next Turing tests"
(*Science* 393, 2026) says AI progress should be read as an evolving ecosystem of capability
profiles, not a ladder. This repo asks, empirically:

1. **Evolution.** Across three generations of three open lineages (Llama, Qwen, Mistral), which
   abilities grow fastest, and are the lineages' profile *shapes* converging (monoculture) or
   diverging (niches)?
2. **Red lines.** What does a profile-based red line look like in practice? Example: "knowledge
   ability well above the ability to calibrate knowns and unknowns (MCu)".

**Method.** Ability profiles on the general scales of Zhou et al., "General scales unlock AI
evaluation with explanatory and predictive power" (*Nature* 652, 2026), using the public ADeLe v1.0
battery and the paper's own fitting procedure. The 15 models from that paper serve as reference
points, and Llama-3.2-3B is re-run as a pipeline check against the paper's instance-level results.

## Pipeline

| Step | Script | Where | Output |
|---|---|---|---|
| 0 | `scripts/setup_cpu.sh`, `scripts/fetch_reference.sh` | Codespaces | env, `data/reference/` |
| 1 | `src/data.py fetch` then `src/data.py subset` | Codespaces | `data/subset_ids.csv` (IDs only, commit it) |
| 2 | `notebooks/kaggle_generate.ipynb` -> `src/generate.py` | Kaggle, GPU T4 x2 | `responses/<model>.jsonl` |
| 3 | `src/score_mc.py` | Codespaces | `scores/<model>_mc.csv` |
| 4 | `src/judge.py`, `src/judge_agreement.py` | Codespaces + OpenAI | `scores/<model>_open.csv`, `results/judge_agreement.json` |
| 5 | `src/collect.py` | Codespaces | `results/correctness/`, `results/validation.json` |
| 6 | `src/profiles.py` | Codespaces | `results/profiles.csv` (+ bootstrap CIs), `results/sensitivity.csv` |
| 7 | `src/evolution.py` | Codespaces | `results/growth.csv`, `convergence.csv`, `pca.csv` |
| 8 | `src/redlines.py` | Codespaces | `results/redlines.csv`, `redline_trends.csv`, `redline_instances.csv` |
| 9 | `src/plots.py` | Codespaces | `figures/fig1_main.png` ... `fig4_validation.png` |

Steps 3-9 in one go: `bash scripts/run_analysis.sh`. Settings live in `configs/run.yaml`;
models in `configs/models.yaml`.

## Quickstart

**Once, in Codespaces**
```bash
bash scripts/setup_cpu.sh
bash tests/run_synthetic.sh          # whole CPU pipeline on fake data: checks your environment
export HF_TOKEN=...                  # after accepting the ADeLe terms on Hugging Face
python src/data.py fetch
python src/data.py subset            # prints dominant-slice coverage per dimension
bash scripts/fetch_reference.sh      # Nature-paper instance-level results (git LFS)
git add data/subset_ids.csv data/subset_coverage.csv && git commit -m "Item subset" && git push
```

**On Kaggle.** Import `notebooks/kaggle_generate.ipynb`, set `REPO_URL` and `BATCH`, and run it
once per batch (`llama,validation`, then `qwen`, then `mistral`). Download each
`responses_<batch>.zip` and unzip it into `responses/` in Codespaces.

**Back in Codespaces**
```bash
export OPENAI_API_KEY=...
bash scripts/run_analysis.sh
```

## Design choices

* **Same input for every model.** The battery's own prompt as one user turn, the model's chat
  template, greedy decoding, fp16, 768 new tokens. Qwen3 runs with thinking off; a thinking-on
  variant is optional (`--models qwen3-8b-thinking`).
* **Paper-faithful ability fit.** For each dimension: dominant slice, bin re-weighting, failure
  anchor at level 20, weighted logistic fit, ability = area under the curve on [0, 100]. This is
  ported from CFI's MIT-licensed toolkit (see `NOTICE`). Items with unguessability < 75 are dropped.
  95% CIs come from 200 item bootstraps.
* **Level vs shape.** Abilities all rise over time, so convergence is measured on profile shape
  (ability minus the model's mean over the 16 core dimensions), not on level.
* **Judge.** GPT-4o-mini with a template modelled on Inspect's `model_graded_qa`. Agreement with
  GPT-4o on 200 items is reported. The judge sees only the final answer, never reasoning traces.
  A hard budget cap is set in `configs/run.yaml`.
* **Red lines are illustrative.** The thresholds in `configs/run.yaml` show how the method
  works; they are not proposed real-world red lines.

## Data handling

The ADeLe battery is released on condition that its examples are not posted online as text
or images, to limit training-data leakage. This repo respects that: it only contains item IDs, scores
and profiles. Item text, model responses (which can echo the question) and judge caches are
gitignored and never committed or plotted.

## Limitations

* ADeLe v1.0 has no items annotated above level 5, so estimates for strong models are weaker.
* Text-only, single-turn evaluation; no agentic, multimodal or tool-use dimensions.
* Only 7-8B open models (three per lineage), so trends are descriptive and the confidence
  intervals matter more than the point estimates.
* Open-ended items are graded by GPT-4o-mini rather than the paper's GPT-4o. See the agreement report
  and the MC-only sensitivity check.
* Release dates in `configs/models.yaml` are approximate.

## Credits

ADeLe battery, DeLeAn rubrics and the ability-fitting procedure come from Zhou, Pacchiardi,
Martínez-Plumed, ..., Cheke, Xie & Hernández-Orallo (Nature, 2026), released by the Kinds of
Intelligence team at the Leverhulme Centre for the Future of Intelligence:
https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation
