#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$project_dir/.cache/modules}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
cd "$project_dir"
exec "${PYTHON:-python}" -u "$@"
