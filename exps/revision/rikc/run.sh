#!/usr/bin/env bash
set -euo pipefail

revision_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${revision_dir}/../.." && pwd)"
device="${REVISION_DEVICE:-cuda}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="rikc_slurm_${SLURM_JOB_ID}"
else
  run_name="rikc_local_$(date -u +%Y%m%dT%H%M%SZ)"
fi
output_root="${REVISION_OUTPUT_ROOT:-${revision_dir}/results/${run_name}}"
[[ "${output_root}" == /* ]] || output_root="${PWD}/${output_root}"

python_bin="${PYTHON:-python3}"
if [[ "${REVISION_SKIP_INSTALL:-0}" != 1 ]]; then
  venv_dir="${VENV_DIR:-${revision_dir}/.venv}"
  if [[ ! -x "${venv_dir}/bin/python" || ! -f "${venv_dir}/.revision_deps_ready" ]]; then
    REVISION_DEVICE="${device}" VENV_DIR="${venv_dir}" bash "${revision_dir}/install_deps.sh"
  fi
  python_bin="${venv_dir}/bin/python"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
args=(--output-dir "${output_root}" --device "${device}")
if [[ "${REVISION_SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
cd "${repo_root}"
echo "Rikc experiment output: ${output_root}"
exec "${python_bin}" -m exps.revision.rikc.runner "${args[@]}" "$@"
