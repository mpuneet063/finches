#!/usr/bin/env bash
# End-to-end test of every CPU stage on synthetic data (no GPU, no API key, no gated data).
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/make_synthetic.py
C=tests/_synthetic/run.yaml
python src/data.py subset --config $C
python src/score_mc.py --config $C
python src/judge.py --config $C --dry-run
python src/collect.py --config $C
python src/profiles.py --config $C
python src/evolution.py --config $C
python src/redlines.py --config $C
python src/plots.py --config $C
python tests/check_recovery.py
echo "Synthetic pipeline OK"
