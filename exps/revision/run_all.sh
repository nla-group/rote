#!/usr/bin/env bash
set -euo pipefail

REVISION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${REVISION_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SMOKE=0
DEVICE="${REVISION_DEVICE:-cuda}"
OUTPUT_ROOT="${REVISION_OUTPUT_ROOT:-${REVISION_DIR}/results/${SLURM_JOB_ID:+slurm_${SLURM_JOB_ID}}}"
if [[ "${OUTPUT_ROOT}" == "${REVISION_DIR}/results/" ]]; then
  OUTPUT_ROOT="${REVISION_DIR}/results/local_$(date -u +%Y%m%dT%H%M%SZ)"
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
export REVISION_DEVICE="${DEVICE}"

if [[ "${REVISION_SKIP_INSTALL:-0}" != "1" ]]; then
  VENV_DIR="${VENV_DIR:-${REVISION_DIR}/.venv}"
  if [[ ! -f "${VENV_DIR}/.revision_deps_ready" ]]; then
    bash "${REVISION_DIR}/install_deps.sh"
  fi
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
echo "Revision output: ${OUTPUT_ROOT}"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
if [[ "$(uname -s)" == "Linux" && "${DEVICE}" == "cuda" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
if torch.__version__ != "2.14.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError(f"Expected archived PyTorch 2.14.0+cu130; got {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("The pinned PyTorch build cannot see the allocated CUDA GPU")
print(f"CUDA preflight: torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}")
PY
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

for STAGE in baseline fixed_length context exposure convergence sensitivity tuned capacity text_bridge; do
  echo "Starting revision stage: ${STAGE}"
  STAGE_ARGS=(--stage "${STAGE}" --output-dir "${OUTPUT_ROOT}/${STAGE}" --device "${DEVICE}")
  if [[ "${SMOKE}" == "1" ]]; then STAGE_ARGS+=(--smoke); fi
  if [[ "${STAGE}" == "tuned" && "${SMOKE}" == "0" ]]; then
    STAGE_ARGS+=(--tuning-results "${OUTPUT_ROOT}/sensitivity/results.csv")
  fi
  "${PYTHON_BIN}" -m exps.revision.run_revision "${STAGE_ARGS[@]}"
done
"${PYTHON_BIN}" -m exps.revision.summarize "${OUTPUT_ROOT}"
echo "All revision stages completed: ${OUTPUT_ROOT}"
