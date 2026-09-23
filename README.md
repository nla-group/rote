# ROTE

[![Tests](https://github.com/nla-group/slearn/actions/workflows/tests.yml/badge.svg)](https://github.com/nla-group/slearn/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/nla-group/slearn/blob/main/pyproject.toml)
[![Distribution: rote-bench](https://img.shields.io/badge/distribution-rote--bench-006C70.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-008080.svg)](https://github.com/nla-group/slearn/blob/main/LICENSE)

**ROTE** (Rollout Testing of Exact memorization) is a reproducible benchmark for learning and extending controlled symbolic sequences. It generates seeds at specified Lempel-Ziv-Welch (LZW) complexity, trains finite-context next-symbol predictors, and evaluates both teacher-forced prediction and closed-loop rollout. The package includes the fixed-budget comparison and a high-complexity matched-size check used in the accompanying manuscript.

## Install

ROTE is currently installed from source; a PyPI release has not yet been published. The intended PyPI distribution name is `rote-bench` and the import name is `rote_benchmark`. The shorter [`rote`](https://pypi.org/project/rote/) name belongs to an unrelated PyPI project. `rote-bench` is not yet published or reserved on PyPI.

```bash
git clone https://github.com/nla-group/rote.git
cd slearn
python -m pip install -e .                 # seed generation and rollout metrics
python -m pip install -e '.[neural,plot]'   # custom PyTorch models and figures
python -m pip install -e '.[all]'           # full paper architecture set
```

or simply via
``
pip install rote-bench
``
The current clone URL is `slearn`; when the repository is renamed, use `https://github.com/nla-group/rote.git` for new clones. The package import and commands will stay the same. Python 3.10 or newer is required. The core install uses NumPy and pandas. The optional experiment dependencies include PyTorch, `minGRU-pytorch`, `linear-attention-transformer`, and `performer-pytorch`. The RWKV-style baseline is implemented in PyTorch and needs no separate RWKV package.

## Quick Start

```python
from rote_benchmark import lzw_string_seeds, normalized_damerau_levenshtein_distance

seeds = lzw_string_seeds(symbols=[4, 8], complexity=[30, 90], iterations=2, random_state=3407)
seed = seeds.loc[0, "string"]
print(seeds[["nr_symbols", "LZW_complexity", "length"]])
print(normalized_damerau_levenshtein_distance(seed, seed))  # 0.0
```

The generated table has `nr_symbols`, `LZW_complexity`, `length`, and `string` columns. Complexity is the number of LZW output codes after reducing a seed to its shortest period; it is not a compressed byte count. For complete experiment settings and CLI options, see the [documentation](docs/source/index.rst).

## Evaluate Your Own Model

Install `.[neural,plot]`, then run a complete single-model example:

```bash
python examples/evaluate_your_model.py --output-dir /tmp/rote-single
```

The model receives one-hot windows `(batch, window, alphabet)` and returns raw next-symbol logits `(batch, alphabet)`. ROTE trains it on the observed prefix, measures teacher-forced test loss and accuracy, then generates the withheld suffix recursively and reports DL/JW distances. It saves configuration JSON, incremental CSV and JSONL records, an event log, and available figures. For three self-contained MLP/GRU/Transformer examples with repeated runs:

```bash
python examples/compare_toy_models.py --runs 2 --output-dir /tmp/rote-toys
python examples/plot_saved_results.py /tmp/rote-toys --font-size 12 --uncertainty sem
python examples/complexity_sweep.py --runs 2 --output-dir /tmp/rote-sweep
python examples/score_checkpoint.py --output-dir /tmp/rote-checkpoint
```

The example writes to `/tmp/rote-toys` only because its `--output-dir` is explicit. Omit the option to create a unique directory beneath `./rote_runs` (or set `ROTE_RUNS_DIR`). An explicit run directory must not already exist. Figures default to `<run>/figures`; use `--output-dir` with the plotting script to choose another location. Existing figures are protected unless `--overwrite` is passed.

Use the Python API directly for your own network:

```python
from rote_benchmark.evaluation import EvaluationConfig, run_benchmark
from rote_benchmark.visualization import PlotStyle, visualize_benchmark

saved = run_benchmark(
    {"MyModel": lambda n_symbols: MyModel(n_symbols)}, seed_string,
    config=EvaluationConfig(window_size=12, device="cpu"), runs=3,
    output_dir="my_rote_run", metadata={"study": "symbolic prototype"},
)
report = visualize_benchmark(saved, style=PlotStyle(font_size=12))
print(saved.results[["model", "test_loss", "DL"]], report.figures)
```

Here `MyModel` and `seed_string` are provided by your code; the [complete runnable toy example](examples/compare_toy_models.py) shows their definitions. The figure API creates quality, rollout, compute-tradeoff, and complexity-sweep plots when the input table has the required columns. For the last figure, pass a mapping of seed IDs to generated strings as the second argument of `run_benchmark`; it records one measured complexity per seed. `load_run("my_rote_run")` restores its configuration, results, and run status without retraining.

See [Evaluating Your Own Model](docs/source/custom_models.rst) for the complete scripts, the `EvaluationConfig`/`evaluate_model`/`evaluate_models`/`score_model` API, and the model contract. [Data and Metrics](docs/source/methodology.rst) explains the split, LZW count, rollout metrics and measurement limits; [Results and Visualization](docs/source/visualization.rst) explains the figures and CSV schema. These toy runs demonstrate the API and are not manuscript results.

## Reproduce the Benchmark

```bash
rote-bench --smoke --device cpu --output-dir /tmp/rote-smoke
rote-bench --help
rote-matched-size --help
```

The main track compares LSTM, GRU, minGRU, minLSTM, Transformer, LinearAttention, Performer, and a compact RWKV-style model. The matched-size track selects widths close to a target parameter count. Both write configuration JSON and incremental CSV results. Install `.[all]` before running the full model set. The smoke test uses tiny settings and is an installation check, not a paper run.

On the Convergence cluster, submit from `exps/`:

```bash
cd exps
sbatch scripts/run_symbolic_benchmark_slurm.sh
sbatch scripts/run_matched_size_symbolic_slurm.sh
```

Merge completed array shards and plot locally from the repository root:

```bash
cd ..
bash exps/scripts/merge_symbolic_results.sh exps/results_symbolic/slurm_<job_id>
bash exps/scripts/run_symbolic_visualizations.sh exps/results_symbolic/slurm_<job_id>/results_merged.csv
bash exps/scripts/run_matched_size_comparison_visualization.sh \
  exps/results_symbolic/slurm_<job_id>/results_merged.csv \
  exps/results_symbolic_matched_size/slurm_<matched_job_id>/results_merged.csv
```

Published experiment outputs remain archived in [`legacy/exps`](legacy/exps); the new scripts write only to root `exps/`. Historical seeds used a process-global Python random stream before the generator adopted an independently seeded stream for each seed. To replay those exact strings, pass the archived `results_merged.csv` as `--seed-manifest` (or set `SEED_MANIFEST` for Slurm). See the [experiment guide](docs/source/experiments.rst) for commands and reproducibility limits.

## Documentation and Tests

```bash
python -m pip install -e '.[docs,test]'
python -m pytest
sphinx-build -b html -W docs/source docs/build/html
```

## Citation and License

The benchmark builds on Cahuantzi, Chen, and Guettel, [*A Comparison of LSTM and GRU Networks for Learning Symbolic Sequences*](https://doi.org/10.1007/978-3-031-37963-5_53) (2023). For the ROTE manuscript, cite the final published version when available. ROTE is distributed under the [MIT License](LICENSE). The historical software and results are preserved in `legacy/`.
