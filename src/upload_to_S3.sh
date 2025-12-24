#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${REPO_ROOT}/templates/bash/s3_upload.sh"

S3_BASE_URI_DEFAULT="s3://plm-ml/code/ankh"

UPLOAD_SYNC_REL=("ankh")
UPLOAD_CP_REL=("infer.py" "requirements.txt" "run_infer.sh")
UPLOAD_SYNC_EXCLUDES=(--exclude "*__pycache__*" --exclude "*.pyc")

# Make templates available inside the SageMaker container at: ${CODE_DIR}/templates/...
UPLOAD_SYNC_BASE=("${REPO_ROOT}/templates::templates")

s3_upload_main "$@"
