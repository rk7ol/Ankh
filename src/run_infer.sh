#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${SM_CODE_DIR:-/opt/ml/processing/input/code}"

# shellcheck source=/dev/null
source "${CODE_DIR}/templates/bash/sm_infer.sh"

sm_pip_bootstrap

# Install Ankh from PyPI once to pull dependencies, then uninstall it so the
# local `${CODE_DIR}/ankh` package is used at runtime (same pattern as ESM).
python -m pip install --no-cache-dir ankh
python -m pip uninstall -y ankh

sm_install_requirements

infer_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  infer_args+=(--hf_token "${HF_TOKEN}")
fi

exec python "${CODE_DIR}/infer.py" "${infer_args[@]}" "$@"
