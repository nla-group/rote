# Symbolic Sequence Benchmark Experiments

This directory is self-contained for manuscript experiments. Run the benchmark, Slurm jobs, shard merging, and local figure generation from `exps/`; the scripts locate the repository root automatically and install the package in editable mode from the parent directory. The experiment implementation and default benchmark settings are unchanged by this layout.

### Example Paper Run

From the repository root:

```bash
python exps/symbolic_sequence_benchmark.py \
  --models LSTM GRU minGRU minLSTM Transformer LinearAttention Performer RWKV \
  --symbols 2 4 6 8 \
  --complexities 10 30 50 70 90 \
  --sequence-lengths 3500 \
  --window-size 100 \
  --forecast-horizon 100 \
  --layers 1 2 3 \
  --units 64 128 256 \
  --d-models 256 \
  --optimizers AdamW \
  --learning-rates 0.0001 0.0003 \
  --weight-decays 0.0 0.01 \
  --runs 3
```

For scaling-law-style sweeps, add token ratios. The script converts each ratio into a model-specific sequence length using `sequence_length = min(max_sequence_length, ceil(ratio * number_of_parameters))` and also keeps any fixed values passed through `--sequence-lengths`:

```bash
python exps/symbolic_sequence_benchmark.py \
  --models LSTM GRU minGRU minLSTM LinearAttention Performer RWKV \
  --token-ratios 1 5 20 \
  --max-sequence-length 20000
```

Single-process results are written incrementally to `exps/results_symbolic/results.csv`, with the exact run configuration saved as `exps/results_symbolic/config.json`.

### Slurm on [LIP6 Convergence](https://front.convergence.lip6.fr/) 

Submit from `exps/` so that Slurm logs and experiment outputs stay inside the experiment directory:

```bash
cd exps
sbatch scripts/run_symbolic_benchmark_slurm.sh
```

The script requests one `a100_3g.40gb` GPU, 12 CPU threads, 64 GB RAM, and runs an eight-way array with at most five tasks active at once. It writes one CSV shard per array task under `results_symbolic/slurm_<array_job_id>/`. The array job uses a filesystem lock so that only one task creates or repairs `.venv`; the other tasks wait until `.venv/.slearn_experiment_deps_ready` exists.

If an older job failed while creating `.venv`, resubmitting with the updated script is usually enough. To clean manually before resubmission, remove the incomplete environment and stale lock once from the login node:

```bash
rm -rf .venv .venv.lock
```

Merge the shards after completion with:

```bash
bash scripts/merge_symbolic_results.sh results_symbolic/slurm_<array_job_id>
```

Generate publication-style figures locally after the merged CSV is available:

```bash
bash scripts/run_symbolic_visualizations.sh results_symbolic/slurm_<array_job_id>/results_merged.csv
```

If `results_symbolic/results_merged.csv`, `results_symbolic/results.csv`, or the newest `results_symbolic/slurm_*/results_merged.csv` exists, the visualization script can also infer the input:

```bash
bash scripts/run_symbolic_visualizations.sh
```

The visualization script uses a fixed publication-style encoding for each model: color, marker shape, hollow/filled marker state, and line style are consistent across all generated figures. Legends are placed outside the axes at the bottom of each figure.

For figure-specific layout tuning, edit `FIGURE_LAYOUTS` in `visualize_symbolic_results.py`; entries such as `rollout_error_vs_horizon`, `compute_performance_pareto`, `test_loss_vs_model_params`, and `dl_vs_model_params` can each set independent `figsize`, `legend_y`, `bottom`, and `legend_ncol` values.

The Slurm script can be configured through environment variables:

```bash
MODELS="LSTM GRU minGRU minLSTM Transformer LinearAttention Performer RWKV BERT GPT" \
MAX_EPOCHS=300 \
RUNS=3 \
SEED_COUNT=3 \
sbatch scripts/run_symbolic_benchmark_slurm.sh
```

### Local Visualization

Visualization is intentionally kept outside the Slurm job so that figures can be regenerated locally after inspecting or filtering the CSV. After a single-machine run from the repository root, use:

```bash
python exps/visualize_symbolic_results.py \
  --results exps/results_symbolic/results.csv \
  --output-dir exps/figures_symbolic
```

From `exps/`, the same command is:

```bash
python visualize_symbolic_results.py \
  --results results_symbolic/results.csv \
  --output-dir figures_symbolic
```

After a Slurm array run, merge first and then plot from `exps/`:

```bash
bash scripts/merge_symbolic_results.sh results_symbolic/slurm_<array_job_id>
python visualize_symbolic_results.py \
  --results results_symbolic/slurm_<array_job_id>/results_merged.csv \
  --output-dir figures_symbolic/slurm_<array_job_id>
```

The script writes each analysis as a separate figure in PNG and PDF by default:

- next-token accuracy, cross-entropy, DL, and JW versus LZW complexity;
- cumulative rollout error versus forecast horizon;
- compute-performance Pareto plot;
- model-size and sequence-length scaling plots when the corresponding sweep exists;
- context-window sensitivity when multiple window sizes exist;
- model-by-complexity heatmaps.

All plots use shared font-size constants for axis labels, ticks, titles, annotations, and legends. Multi-model legends are placed outside the axes at the bottom center.


### Matched-Size Robustness Check

The matched-size experiment is a small high-complexity robustness check for the manuscript. It does not change the main benchmark defaults. Instead, for each model family it searches a width grid and selects the candidate whose trainable parameter count is closest to a target size. The default run uses the hardest LZW setting, two alphabet sizes, and the six model families most relevant to the model-size concern:

```text
models: LSTM GRU Transformer LinearAttention Performer RWKV
target size: 0.2M parameters
symbols: 4 8
complexity: 90
seeds: 2
runs: 2
```

This gives `6 models x 2 alphabet sizes x 2 seeds x 2 runs = 48` model fits, sharded across a six-way Slurm array by default.

Submit from `exps/`:

```bash
cd exps
sbatch scripts/run_matched_size_symbolic_slurm.sh
```

Each task writes one shard under `results_symbolic_matched_size/slurm_<array_job_id>/`. After completion, merge the shards:

```bash
bash scripts/merge_symbolic_results.sh results_symbolic_matched_size/slurm_<array_job_id>
```

Keep the main benchmark figures and the matched-size robustness figure separate. Main benchmark figures should be generated from `results_symbolic/slurm_<array_job_id>/results_merged.csv` with `run_symbolic_visualizations.sh`; the matched-size track is used for the paired high-complexity robustness comparison below.

The matched-size script also writes `matched_configs.csv`, which records the selected width, `d_model`, target size, achieved parameter count, and matching error for each model and alphabet size. This file is useful for reporting how closely each family could be matched to the requested target.

To regenerate the appendix comparison figure between the fixed-budget and matched-size tracks, run:

```bash
bash scripts/run_matched_size_comparison_visualization.sh \
  results_symbolic/slurm_102076/results_merged.csv \
  results_symbolic_matched_size/slurm_104393/results_merged.csv \
  results_symbolic_matched_size/slurm_104393/figures
```

When called without arguments, the helper selects the newest `results_symbolic/slurm_*/results_merged.csv` and newest `results_symbolic_matched_size/slurm_*/results_merged.csv`. It writes `matched_size_comparison.pdf`, `matched_size_comparison.png`, and `matched_size_fixed_budget_comparison.csv`. The matched-size comparison plot uses `MATCHED_COMPARISON_FONT_SIZE` in `visualize_symbolic_results.py` for panel titles, axis labels, tick labels, and legend text.

Common overrides:

```bash
TARGET_SIZES_M="0.2 0.5" \
SYMBOLS="8" \
RUNS=1 \
SEED_COUNT=1 \
sbatch scripts/run_matched_size_symbolic_slurm.sh
```

For a dry run that only computes the matched configurations after installing dependencies, use the Python entry point directly:

```bash
python matched_size_symbolic_benchmark.py \
  --models LSTM GRU Transformer LinearAttention Performer RWKV \
  --target-sizes-m 0.2 \
  --symbols 4 8 \
  --complexities 90 \
  --dry-run
```

## Legacy Experiments

The older `it_*_scale.py` and `test_low_*.py` scripts are retained for reproducibility and now recognize the added model names. For new manuscript runs, prefer `symbolic_sequence_benchmark.py` because it avoids target leakage in the Transformer decoder path and initializes rollouts from the observed prefix instead of the withheld target.
