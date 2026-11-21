#!/usr/bin/env bash
set -euo pipefail

REVISION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${REVISION_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REVISION_DIR}/.venv}"
export XDG_CACHE_HOME="${REVISION_DIR}/.cache"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TMPDIR="${REVISION_DIR}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON:-python3}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
if [[ "$(uname -s)" == "Linux" && "${REVISION_DEVICE:-cuda}" == "cuda" ]]; then
  # Match the archived slurm_102076 run (torch=2.14.0+cu130).
  "${VENV_DIR}/bin/python" -m pip install "torch==2.14.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130
fi
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}[experiment]"
"${VENV_DIR}/bin/python" - <<'PY'
import os
import torch
from minGRU_pytorch import minGRU
from linear_attention_transformer import LinearAttentionTransformer
from performer_pytorch import Performer
if os.uname().sysname == "Linux" and os.environ.get("REVISION_DEVICE", "cuda") == "cuda":
    if torch.__version__ != "2.14.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected archived PyTorch 2.14.0+cu130; got {torch.__version__}")
print(f"Revision environment ready: torch={torch.__version__}, cuda={torch.version.cuda}")
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "${VENV_DIR}/.revision_deps_ready"
