# ROTE experiment entry points

This directory contains thin repository entry points and Slurm/merge/plot scripts. The implementation lives in `rote_benchmark/`; installing the package exposes `rote-bench` (`rote-benchmark` is a compatibility alias), `rote-matched-size`, `rote-plot`, and `rote-plot-matched`.

```bash
python -m pip install -e '.[all]'
python exps/symbolic_sequence_benchmark.py --smoke --device cpu --output-dir /tmp/rote-smoke

cd exps
sbatch scripts/run_symbolic_benchmark_slurm.sh
sbatch scripts/run_matched_size_symbolic_slurm.sh
```

From the repository root, merge and plot completed jobs with:

```bash
cd ..
bash exps/scripts/merge_symbolic_results.sh exps/results_symbolic/slurm_<job_id>
bash exps/scripts/run_symbolic_visualizations.sh exps/results_symbolic/slurm_<job_id>/results_merged.csv
bash exps/scripts/merge_symbolic_results.sh exps/results_symbolic_matched_size/slurm_<matched_job_id>
bash exps/scripts/run_matched_size_comparison_visualization.sh \
  exps/results_symbolic/slurm_<job_id>/results_merged.csv \
  exps/results_symbolic_matched_size/slurm_<matched_job_id>/results_merged.csv
```

The manuscript's completed CSVs and figures remain under `legacy/exps/`. Pass those CSVs explicitly to the plotting scripts to recreate archived figures. For exact historical seed replay, pass the corresponding archived `results_merged.csv` to a benchmark with `--seed-manifest`, or set `SEED_MANIFEST` for a Slurm submission. The archived generator used a process-global Python random stream; the current generator seeds each string independently. Its `config.json` alone therefore does not restore those historical strings. New runs write beneath this directory only.
