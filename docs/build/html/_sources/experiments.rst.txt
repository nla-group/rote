Experiments and Reproducibility
===============================

Protocol
--------

For each alphabet size :math:`n` and target LZW code count :math:`c`, ROTE
generates a seed :math:`u`, repeats it to length :math:`T`, and forms
chronologically ordered context-target pairs
:math:`(x_{t-w:t-1}, x_t)` of window size :math:`w`. A model predicts the next
symbol from a one-hot context. The experiment keeps a withheld suffix of length
:math:`H` for free-running rollout. Teacher-forced test loss and accuracy use
observed contexts; rollout feeds each prediction back into the next context.

The rollout distances are normalized DL and JW. A rollout is exact when its
DL distance is zero. The benchmark also records model parameters, training
time, epochs, time per epoch, and peak GPU memory where available. These are
separate measurements: low one-step loss does not imply exact rollout.

Models
------

The fixed-budget track includes LSTM, GRU, minGRU, minLSTM, Transformer,
LinearAttention, Performer, and a compact RWKV-style PyTorch model. The
matched-size check includes LSTM, GRU, Transformer, LinearAttention, Performer,
and RWKV. The latter searches a width grid for the closest trainable parameter
count to a target size; its ``matched_configs.csv`` records the achieved sizes.

Local Commands
--------------

.. code-block:: bash

   python -m pip install -e '.[all]'
   rote-bench --smoke --device cpu --output-dir /tmp/rote-smoke
   rote-bench --help
   rote-matched-size --help
   rote-plot --help
   rote-plot-matched --help

The default fixed-budget run sweeps four alphabet sizes, five LZW complexity
targets, two generated seeds, and two training runs. The default matched-size
check uses :math:`c=90`, two alphabet sizes, two seeds, and two runs. Both
write ``config.json`` and incremental ``results.csv``. Defaults and all
overrides are included in the saved configuration. The smoke run is only an
installation check.

Slurm Workflow
--------------

Submit from the repository's ``exps/`` directory so that array logs resolve
correctly:

.. code-block:: bash

   cd exps
   sbatch scripts/run_symbolic_benchmark_slurm.sh
   sbatch scripts/run_matched_size_symbolic_slurm.sh

Each task writes ``results_task_*.csv``. After the array completes, merge its
shards from the repository root:

.. code-block:: bash

   cd ..
   bash exps/scripts/merge_symbolic_results.sh exps/results_symbolic/slurm_<job_id>
   bash exps/scripts/merge_symbolic_results.sh exps/results_symbolic_matched_size/slurm_<matched_job_id>

The Slurm scripts request one A100 40 GB GPU, 12 CPU threads and 64 GB RAM on
the Convergence partition. Runtime statistics describe this environment and
should not be interpreted as hardware-independent architecture properties.
Performer may use a slower, more memory-intensive non-CUDA autoregressive
fallback when its optional CUDA kernel is unavailable.

Figures
-------

Generate main and matched-size figures separately, on a machine with the
``plot`` extra installed:

.. code-block:: bash

   bash exps/scripts/run_symbolic_visualizations.sh \
     exps/results_symbolic/slurm_<job_id>/results_merged.csv

   bash exps/scripts/run_matched_size_comparison_visualization.sh \
     exps/results_symbolic/slurm_<job_id>/results_merged.csv \
     exps/results_symbolic_matched_size/slurm_<matched_job_id>/results_merged.csv

The first command generates one PDF and PNG per analysis. The second generates
``matched_size_comparison.pdf`` and ``.png``. The archived manuscript data are
in ``legacy/exps/results_symbolic/slurm_102076`` and
``legacy/exps/results_symbolic_matched_size/slurm_104393``; pass those CSV paths
to the same scripts to reproduce figures without retraining. No files in
``legacy/`` are written by the new workflow.

Historical Seed Replay
----------------------

The archived runs used the process-global Python random stream inside the
LZW generator. A later change made that stream independent for each seed.
Consequently, ``config.json`` alone with the current generator does not
recreate the original strings. The result CSVs record every
``seed_string``. Pass the corresponding CSV as ``--seed-manifest`` to replay
those exact strings in a new run:

.. code-block:: bash

   rote-bench --seed-manifest legacy/exps/results_symbolic/slurm_102076/results_merged.csv
   rote-matched-size --seed-manifest legacy/exps/results_symbolic_matched_size/slurm_104393/results_merged.csv

For Slurm, export ``SEED_MANIFEST`` with the corresponding CSV path before
submitting the job. The loader validates the alphabet size, LZW complexity,
and unique seed index; it accepts either a results CSV or a seed table. The
archived main track contains 40 distinct seeds, and the matched-size track
contains four.

Reproducibility Limits
----------------------

The training scripts seed Python, NumPy and PyTorch, save parsed configuration,
and write each completed fit. Exact floating-point trajectories and timing may
vary with PyTorch, CUDA, cuDNN, optional kernels, and hardware. LZW target
complexity is checked against the generated seed, but it is distinct from
finite-window identifiability. Use the archived CSV and configuration files
for the exact results reported in the manuscript.
