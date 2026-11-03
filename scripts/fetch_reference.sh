#!/usr/bin/env bash
# Download the Nature paper's per-model instance-level results (git LFS, CFI's public repo)
# into data/reference/. These files hold item-level data: they stay out of git (see .gitignore).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v git-lfs >/dev/null || { echo "Install git-lfs first (sudo apt-get install -y git-lfs)"; exit 1; }
TMP=$(mktemp -d)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation.git "$TMP/adele"
( cd "$TMP/adele" && git lfs pull --include "ADeLe_battery_data/subject_specific_instance_level_data/*" )
mkdir -p data/reference
cp "$TMP"/adele/ADeLe_battery_data/subject_specific_instance_level_data/*.csv data/reference/
rm -rf "$TMP"
ls -lh data/reference
