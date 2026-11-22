#!/bin/bash
#SBATCH --job-name=rote-tuning-followup
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

# Slurm runs a spooled copy of the batch script.
revision_dir="${SLURM_SUBMIT_DIR:?Submit from the exps/revision directory}"
if [[ ! -f "${revision_dir}/tuning_followup/run.sh" ]]; then
  echo "Submit from the repository's exps/revision directory." >&2
  exit 1
fi
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec bash "${revision_dir}/tuning_followup/run.sh" "$@"
