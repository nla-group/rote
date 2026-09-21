#!/bin/bash
#SBATCH --job-name=debug-mkdir
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -x
echo "hello from job"
pwd
whoami
id
echo "HOME=$HOME"
echo "TMPDIR=${TMPDIR:-unset}"
mkdir -pv "$HOME/slearn/exps/tmp/debug_$SLURM_JOB_ID"
ls -ld "$HOME" "$HOME/slearn" "$HOME/slearn/exps" "$HOME/slearn/exps/tmp"
