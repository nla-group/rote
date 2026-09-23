#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "${RESULTS}" ]]; then
    if [[ -f "${EXPS_DIR}/results_symbolic/results_merged.csv" ]]; then
        RESULTS="${EXPS_DIR}/results_symbolic/results_merged.csv"
    elif [[ -f "${EXPS_DIR}/results_symbolic/results.csv" ]]; then
        RESULTS="${EXPS_DIR}/results_symbolic/results.csv"
    else
        latest_slurm_dir="$(find "${EXPS_DIR}/results_symbolic" -maxdepth 1 -type d -name 'slurm_*' 2>/dev/null | sort | tail -n 1 || true)"
        if [[ -n "${latest_slurm_dir}" && -f "${latest_slurm_dir}/results_merged.csv" ]]; then
            RESULTS="${latest_slurm_dir}/results_merged.csv"
        else
            echo "Could not find a result CSV automatically." >&2
            echo "Usage: bash exps/scripts/run_symbolic_visualizations.sh <results.csv|results_dir> [output_dir]" >&2
            exit 1
        fi
    fi
fi

if [[ -d "${RESULTS}" ]]; then
    if [[ -f "${RESULTS}/results_merged.csv" ]]; then
        RESULTS="${RESULTS}/results_merged.csv"
    elif [[ -f "${RESULTS}/results.csv" ]]; then
        RESULTS="${RESULTS}/results.csv"
    else
        echo "Could not find results_merged.csv or results.csv in ${RESULTS}." >&2
        exit 1
    fi
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ "${RESULTS}" == *"/slurm_"* ]]; then
        OUTPUT_DIR="$(dirname "${RESULTS}")/figures"
    else
        OUTPUT_DIR="${EXPS_DIR}/figures_symbolic"
    fi
fi

PYTHON_BIN="${PYTHON:-python}"
if [[ -x "${EXPS_DIR}/.venv/bin/python" && -z "${PYTHON:-}" ]]; then
    PYTHON_BIN="${EXPS_DIR}/.venv/bin/python"
fi

echo "Results: ${RESULTS}"
echo "Output:  ${OUTPUT_DIR}"
echo "Python:  ${PYTHON_BIN}"

"${PYTHON_BIN}" "${EXPS_DIR}/visualize_symbolic_results.py" \
    --results "${RESULTS}" \
    --output-dir "${OUTPUT_DIR}"
