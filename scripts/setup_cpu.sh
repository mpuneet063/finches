#!/usr/bin/env bash
# Codespaces / CPU setup.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -r requirements-cpu.txt
python -c "import pandas, sklearn, yaml, openai; print('CPU environment ready')"
