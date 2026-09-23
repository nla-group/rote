#!/bin/bash
#SBATCH --job-name=matched-size-symbolic
#SBATCH --partition=convergence
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=36:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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

echo "Starting matched-size symbolic Slurm job."

RAW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/matched_size_symbolic_benchmark.py" ]]; then
  EXPS_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/exps/matched_size_symbolic_benchmark.py" ]]; then
  EXPS_DIR="$(cd "${SLURM_SUBMIT_DIR}/exps" && pwd)"
else
  EXPS_DIR="$(cd "${RAW_SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${EXPS_DIR}/scripts"
REPO_ROOT="$(cd "${EXPS_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Raw script dir: ${RAW_SCRIPT_DIR}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-unset}"
echo "Resolved script dir: ${SCRIPT_DIR}"
echo "Resolved exps dir: ${EXPS_DIR}"
echo "Resolved repo root: ${REPO_ROOT}"

for output_dir in "${EXPS_DIR}/results_symbolic_matched_size" "${EXPS_DIR}/logs"; do
  echo "Ensuring output directory: ${output_dir}"
  ensure_dir "${output_dir}"
done

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

for required_dir in "${EXPS_DIR}" "${EXPS_DIR}/logs" "${EXPS_DIR}/results_symbolic_matched_size" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"; do
  if [[ ! -w "${required_dir}" ]]; then
    echo "Required directory is not writable: ${required_dir}" >&2
    ls -ld "${required_dir}" >&2 || true
    exit 1
  fi
done

echo "Node: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Repo root: ${REPO_ROOT}"
echo "Exps dir: ${EXPS_DIR}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "TMPDIR=${TMPDIR}"
echo "Job: ${SLURM_JOB_ID:-NA}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

VENV_DIR="${VENV_DIR:-${EXPS_DIR}/.venv}"
READY_FILE="${VENV_DIR}/.slearn_experiment_deps_ready"
LOCK_DIR="${VENV_DIR}.lock"

if [[ ! -f "${READY_FILE}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
  LOCK_ERR="${EXPS_DIR}/logs/env-lock-${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}.err"
  if mkdir "${LOCK_DIR}" 2>"${LOCK_ERR}"; then
    trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
    echo "Preparing experiment environment in ${VENV_DIR}."
    bash "${SCRIPT_DIR}/install_experiment_deps.sh"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    trap - EXIT
  else
    lock_status=$?
    if [[ ! -d "${LOCK_DIR}" ]]; then
      echo "Failed to create environment lock directory: ${LOCK_DIR}" >&2
      cat "${LOCK_ERR}" >&2 || true
      ls -ld "${EXPS_DIR}" "${VENV_DIR}" 2>&1 >&2 || true
      exit "${lock_status}"
    fi
    echo "Another array task is preparing ${VENV_DIR}; waiting for ${READY_FILE}."
    for _ in $(seq 1 180); do
      if [[ -f "${READY_FILE}" && -f "${VENV_DIR}/bin/activate" ]]; then
        break
      fi
      sleep 20
    done
  fi
fi

if [[ ! -f "${READY_FILE}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Experiment environment is still incomplete after waiting." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python: $(which python)"
python --version
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

MODELS="${MODELS:-LSTM GRU Transformer LinearAttention Performer RWKV}"
TARGET_SIZES_M="${TARGET_SIZES_M:-0.2}"
SYMBOLS="${SYMBOLS:-4 8}"
COMPLEXITIES="${COMPLEXITIES:-90}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-3500}"
LAYERS="${LAYERS:-2}"
RECURRENT_UNITS="${RECURRENT_UNITS:-32 48 64 80 96 112 128 160 192 224 256 320 384 448 512 640 768}"
D_MODELS="${D_MODELS:-32 48 64 80 96 112 128 160 192 224 256}"
FF_MULT="${FF_MULT:-4}"
BATCH_SIZES="${BATCH_SIZES:-128}"
LEARNING_RATES="${LEARNING_RATES:-0.0003}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.01}"
RUNS="${RUNS:-2}"
SEED_COUNT="${SEED_COUNT:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-10}"
STOPPING_LOSS="${STOPPING_LOSS:-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPS_DIR}/results_symbolic_matched_size/slurm_${SLURM_ARRAY_JOB_ID:-manual}}"

python "${EXPS_DIR}/matched_size_symbolic_benchmark.py" \
  --models ${MODELS} \
  --target-sizes-m ${TARGET_SIZES_M} \
  --symbols ${SYMBOLS} \
  --complexities ${COMPLEXITIES} \
  --sequence-lengths ${SEQUENCE_LENGTHS} \
  --window-size 100 \
  --forecast-horizon 100 \
  --seed-count "${SEED_COUNT}" \
  --runs "${RUNS}" \
  --layers ${LAYERS} \
  --recurrent-units ${RECURRENT_UNITS} \
  --d-models ${D_MODELS} \
  --ff-mult "${FF_MULT}" \
  --batch-sizes ${BATCH_SIZES} \
  --optimizers AdamW \
  --learning-rates ${LEARNING_RATES} \
  --weight-decays ${WEIGHT_DECAYS} \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --stopping-loss "${STOPPING_LOSS}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --task-count "${SLURM_ARRAY_TASK_COUNT:-1}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
printf -v shard "%03d" "${task_id}"
echo "Finished. Results shard: ${OUTPUT_DIR}/results_task_${shard}.csv"
echo "Merge after completion with:"
echo "  cd ${EXPS_DIR} && bash scripts/merge_symbolic_results.sh ${OUTPUT_DIR}"
echo "Visualize after merging with:"
echo "  cd ${EXPS_DIR} && bash scripts/run_symbolic_visualizations.sh ${OUTPUT_DIR}/results_merged.csv ${OUTPUT_DIR}/figures"
