#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

latest_results_csv() {
  local base_dir="$1"
  local latest_dir
  latest_dir="$(find "${base_dir}" -maxdepth 1 -type d -name 'slurm_*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${latest_dir}" && -f "${latest_dir}/results_merged.csv" ]]; then
    printf '%s\n' "${latest_dir}/results_merged.csv"
    return 0
  fi
  if [[ -f "${base_dir}/results_merged.csv" ]]; then
    printf '%s\n' "${base_dir}/results_merged.csv"
    return 0
  fi
  if [[ -f "${base_dir}/results.csv" ]]; then
    printf '%s\n' "${base_dir}/results.csv"
    return 0
  fi
  return 1
}

FIXED_RESULTS="${1:-}"
MATCHED_RESULTS="${2:-}"
OUTPUT_DIR="${3:-}"

if [[ -z "${FIXED_RESULTS}" ]]; then
  FIXED_RESULTS="$(latest_results_csv "${EXPS_DIR}/results_symbolic")" || {
    echo "Could not find fixed-budget results automatically." >&2
    exit 1
  }
fi

if [[ -z "${MATCHED_RESULTS}" ]]; then
  MATCHED_RESULTS="$(latest_results_csv "${EXPS_DIR}/results_symbolic_matched_size")" || {
    echo "Could not find matched-size results automatically." >&2
    exit 1
  }
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="$(dirname "${MATCHED_RESULTS}")/figures"
fi

PYTHON_BIN="${PYTHON:-python}"
if [[ -x "${EXPS_DIR}/.venv/bin/python" && -z "${PYTHON:-}" ]]; then
  PYTHON_BIN="${EXPS_DIR}/.venv/bin/python"
fi

echo "Fixed-budget results: ${FIXED_RESULTS}"
echo "Matched-size results: ${MATCHED_RESULTS}"
echo "Output:               ${OUTPUT_DIR}"
echo "Python:               ${PYTHON_BIN}"

"${PYTHON_BIN}" "${EXPS_DIR}/visualize_matched_size_comparison.py" \
  --fixed-budget-results "${FIXED_RESULTS}" \
  --matched-size-results "${MATCHED_RESULTS}" \
  --output-dir "${OUTPUT_DIR}"

for suffix in pdf png; do
  figure_path="${OUTPUT_DIR}/matched_size_comparison.${suffix}"
  if [[ -f "${figure_path}" ]]; then
    echo "Wrote: ${figure_path}"
  fi
done
