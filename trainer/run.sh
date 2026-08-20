#!/usr/bin/env bash
# Launch the Yale QC companion train/infer server via uv.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

if [ -z "${QC_TRAIN_TOKEN:-}" ]; then
    echo "ERROR: QC_TRAIN_TOKEN is not set (must match the QC app's TRAIN_SERVER_TOKEN)." >&2
    exit 1
fi
export QC_TRAIN_WORKDIR="${QC_TRAIN_WORKDIR:-$PWD/work}"

# uv run syncs the venv from pyproject on first launch, then runs.
exec uv run python train_server.py --host 0.0.0.0 --port "${QC_TRAIN_PORT:-8050}"
