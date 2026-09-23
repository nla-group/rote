#!/usr/bin/env bash
set -euo pipefail

ensure_dir() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    return 0
  fi
  if [[ -e "${dir}" ]]; then
    echo "Path exists but is not a directory: ${dir}" >&2
    ls -ld "${dir}" >&2 || true
    return 1
  fi
  mkdir -p "${dir}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPS_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-${EXPS_DIR}/.venv}"
READY_FILE="${VENV_DIR}/.rote_experiment_deps_ready"

EXPS_CACHE_DIR="${EXPS_CACHE_DIR:-${EXPS_DIR}/.cache}"
EXPS_TMPDIR="${EXPS_TMPDIR:-${EXPS_DIR}/tmp}"
export XDG_CACHE_HOME="${EXPS_CACHE_DIR}"
export PIP_CACHE_DIR="${EXPS_CACHE_DIR}/pip"
export TORCH_EXTENSIONS_DIR="${EXPS_CACHE_DIR}/torch_extensions"
export TMPDIR="${EXPS_TMPDIR}"
for runtime_dir in "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"; do
  echo "Ensuring runtime directory: ${runtime_dir}"
  ensure_dir "${runtime_dir}"
done

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Existing ${VENV_DIR} has a broken pip; rebuilding it."
    rm -rf "${VENV_DIR}"
  fi
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  rm -rf "${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO_ROOT}[all]"

python - <<'PYCHECK'
import torch
from minGRU_pytorch import minGRU
from linear_attention_transformer import LinearAttentionTransformer
from performer_pytorch import Performer

print(f"Experiment environment ready. torch={torch.__version__}")
print("Verified: minGRU_pytorch, linear_attention_transformer, performer_pytorch")
PYCHECK

date -u +%Y-%m-%dT%H:%M:%SZ > "${READY_FILE}"
