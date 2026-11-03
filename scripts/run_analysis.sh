#!/usr/bin/env bash
# Everything after generation, on CPU (Codespaces). Needs responses/*.jsonl and OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/score_mc.py
python src/judge.py --dry-run
read -r -p "Run the judge now? [y/N] " ok
if [[ "$ok" =~ ^[Yy]$ ]]; then
  python src/judge.py
  python src/judge_agreement.py
fi
python src/collect.py
python src/profiles.py
python src/evolution.py
python src/redlines.py
python src/plots.py
echo "Done: see results/ and figures/"
