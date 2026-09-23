Symbolic Benchmark Experiments
==============================

Purpose
-------

The benchmark in ``exps/`` evaluates neural architectures as finite-context
conditional predictors on controlled symbolic sequences. A seed string is
generated with a requested alphabet size and LZW complexity, repeated to the
selected sequence length, and split chronologically. Models are trained on
one-step prediction from observed context windows and then evaluated both by
teacher forcing and by closed-loop rollout.

The protocol is designed to separate several quantities that are often
confounded in sequence-model comparisons: alphabet size, symbolic
compressibility, finite-context identifiability, model size, and compute budget.

Model Families
--------------

Supported model names are:

.. code-block:: text

   LSTM GRU minGRU minLSTM Transformer BERT GPT LinearAttention Performer RWKV

The benchmark currently includes gated recurrent models, minimal recurrent
models, standard attention baselines, efficient-attention baselines, and a compact
RWKV-style time-mixing model implemented in PyTorch.

Local Smoke Test
----------------

Run this first after installation:

.. code-block:: bash

   python exps/symbolic_sequence_benchmark.py --smoke --device cpu

The smoke test uses one alphabet, one complexity, one seed, one run, small model
widths, and a short sequence. It verifies the installation rather than producing
paper-quality numbers.

Default Run
-----------

The default configuration is pilot-sized but nontrivial:

.. code-block:: bash

   python exps/symbolic_sequence_benchmark.py

Defaults include eight models, four alphabet sizes, five LZW complexity targets,
two generated seeds, two runs, context window 100, forecast horizon 100, sequence
length 3500, AdamW, learning rate ``3e-4``, weight decay ``0.01``, at most 200
epochs, and patience 10.

Results are written incrementally to:

.. code-block:: text

   exps/results_symbolic/results.csv
   exps/results_symbolic/config.json

Slurm Workflow
--------------

On the LIP6 Convergence cluster, submit from ``exps/`` so logs, virtual
environment files, result shards, and merged outputs remain inside the experiment
directory:

.. code-block:: bash

   cd exps
   sbatch scripts/run_symbolic_benchmark_slurm.sh

The Slurm script requests one node, 12 CPU threads, 64 GB RAM, one
``a100_3g.40gb`` GPU, 48 hours, and an eight-task array with at most five tasks
active at once. Each array task writes one CSV shard under:

.. code-block:: text

   exps/results_symbolic/slurm_<array_job_id>/results_task_<task_id>.csv

The script creates ``exps/.venv`` if needed. A filesystem lock ensures that only
one array task installs dependencies while the others wait for the ready marker
``exps/.venv/.slearn_experiment_deps_ready``.

After the job finishes, merge shards from ``exps/``:

.. code-block:: bash

   bash scripts/merge_symbolic_results.sh results_symbolic/slurm_<array_job_id>

The merged file is:

.. code-block:: text

   exps/results_symbolic/slurm_<array_job_id>/results_merged.csv

Configurable Slurm Variables
----------------------------

The Slurm script exposes common sweep settings as environment variables:

.. code-block:: bash

   MODELS="LSTM GRU minGRU minLSTM Transformer LinearAttention Performer RWKV" \
   SYMBOLS="2 4 6 8" \
   COMPLEXITIES="10 30 50 70 90" \
   MAX_EPOCHS=200 \
   RUNS=2 \
   SEED_COUNT=2 \
   sbatch scripts/run_symbolic_benchmark_slurm.sh

Visualization
-------------

Visualization is intentionally separate from the Slurm job. After merging, run:

.. code-block:: bash

   bash scripts/run_symbolic_visualizations.sh results_symbolic/slurm_<array_job_id>/results_merged.csv

When a merged result file can be inferred automatically, this shorter command is
enough:

.. code-block:: bash

   bash scripts/run_symbolic_visualizations.sh

The plotting script writes one PNG and one PDF for each analysis by default. It
uses consistent model colors, markers, marker fill states, and line styles across
figures. Legends are placed outside the axes, below the plot region. Figure-level
layout controls such as ``legend_y`` and ``bottom`` are collected in
``FIGURE_LAYOUTS`` in ``exps/visualize_symbolic_results.py``.

Output Columns
--------------

The benchmark records configuration fields such as model name, alphabet size,
LZW complexity, sequence length, window size, forecast horizon, layer count,
hidden width, optimizer, learning rate, weight decay, batch size, run index, and
seed index. It also records trainable parameter count, training time, epoch
count, time per epoch, peak GPU memory, teacher-forced test loss and accuracy,
and rollout distances ``dl`` and ``jw``.

Reproducibility Notes
---------------------

The script fixes Python, NumPy, and PyTorch random seeds, disables cuDNN's
non-deterministic fast path, saves the exact parsed configuration to
``config.json``, and writes one result row per completed model fit. Hardware,
CUDA, PyTorch, and optional attention-kernel versions can still affect runtime
and low-level floating-point behavior, so exact wall-clock timing should be
reported with the cluster resource description and software environment.

Performer Note
--------------

``performer-pytorch`` may print a warning when its optional CUDA kernel for
auto-regressive attention is unavailable. With the benchmark's causal Performer
configuration, the model still runs; the fallback is less memory efficient and
can be slower. For strict compute comparisons, report whether the CUDA kernel was
available in the experiment environment.
