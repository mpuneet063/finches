#!/usr/bin/env bash
# Install generation dependencies on a Kaggle GPU notebook ("GPU T4 x2", Internet on).
# vLLM is optional: if it fails to install or import, generate.py falls back to transformers.
set -uo pipefail
cd "$(dirname "$0")/.."
pip install -q -r requirements-gpu.txt
if [ "${SKIP_VLLM:-0}" != "1" ]; then
  # Pin with VLLM_SPEC (e.g. VLLM_SPEC="vllm==<version>") if the latest clashes with Kaggle's torch.
  pip install -q "${VLLM_SPEC:-vllm}" || echo "vLLM install failed -> transformers backend will be used"
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "| GPUs", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", torch.cuda.get_device_name(i), "compute capability", torch.cuda.get_device_capability(i))
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm unavailable:", e)
PY
