#!/bin/bash
#SBATCH --job-name=rote-revision
#SBATCH --partition=convergence
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Slurm executes a spooled copy of this script; BASH_SOURCE is not the repo path.
REVISION_DIR="${SLURM_SUBMIT_DIR:?Submit from the exps/revision directory}"
if [[ ! -f "${REVISION_DIR}/run_all.sh" ]]; then
  echo "Submit from the repository's exps/revision directory." >&2
  exit 1
fi
cd "${REVISION_DIR}"
echo "Node: $(hostname)"
echo "Revision directory: ${REVISION_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
export REVISION_OUTPUT_ROOT="${REVISION_OUTPUT_ROOT:-${REVISION_DIR}/results/slurm_${SLURM_JOB_ID}}"
RUN_ARGS=(--device "${REVISION_DEVICE:-cuda}" --output-root "${REVISION_OUTPUT_ROOT}")
if [[ "${REVISION_SMOKE:-0}" == "1" ]]; then RUN_ARGS+=(--smoke); fi
bash "${REVISION_DIR}/run_all.sh" "${RUN_ARGS[@]}"
