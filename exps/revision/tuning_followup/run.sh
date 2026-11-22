#!/usr/bin/env bash
set -euo pipefail

revision_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${revision_dir}/../.." && pwd)"
input_run="${REVISION_INPUT_RUN:-${revision_dir}/results/slurm_118058}"
device="${REVISION_DEVICE:-cuda}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="tuning_followup_slurm_${SLURM_JOB_ID}"
else
  run_name="tuning_followup_local_$(date -u +%Y%m%dT%H%M%SZ)"
fi
output_root="${REVISION_OUTPUT_ROOT:-${revision_dir}/results/${run_name}}"

if [[ ! -f "${input_run}/sensitivity/config.json" ]]; then
  echo "Missing archived sensitivity configuration under ${input_run}" >&2
  exit 1
fi
input_run="$(cd "${input_run}" && pwd)"
[[ "${output_root}" == /* ]] || output_root="${PWD}/${output_root}"

python_bin="${PYTHON:-python3}"
if [[ "${REVISION_SKIP_INSTALL:-0}" != 1 ]]; then
  venv_dir="${VENV_DIR:-${revision_dir}/.venv}"
  if [[ ! -x "${venv_dir}/bin/python" || ! -f "${venv_dir}/.revision_deps_ready" ]]; then
    if [[ "${device}" == cuda ]]; then
      constraints="$(mktemp)"
      trap 'rm -f "${constraints}"' EXIT
      "${python_bin}" - "${input_run}/sensitivity/config.json" > "${constraints}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    archived = json.load(stream)
if sys.version.split()[0].split(".")[:2] != archived["python"].split(".")[:2]:
    raise SystemExit(f"Use Python {archived['python']} or the same Python X.Y for a new venv.")
for name, version in sorted(archived["packages"].items()):
    print(f"{name}=={version}")
PY
      PIP_CONSTRAINT="${constraints}" REVISION_DEVICE="${device}" VENV_DIR="${venv_dir}" \
        bash "${revision_dir}/install_deps.sh"
      rm -f "${constraints}"
      trap - EXIT
    else
      REVISION_DEVICE="${device}" VENV_DIR="${venv_dir}" bash "${revision_dir}/install_deps.sh"
    fi
  fi
  python_bin="${venv_dir}/bin/python"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
args=(--revision-run "${input_run}" --output-dir "${output_root}" --device "${device}")
if [[ "${REVISION_SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
cd "${repo_root}"
echo "Tuning follow-up archive: ${input_run}"
echo "Tuning follow-up output: ${output_root}"
exec "${python_bin}" -m exps.revision.tuning_followup.runner "${args[@]}" "$@"
