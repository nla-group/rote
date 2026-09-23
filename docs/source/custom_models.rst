Evaluating Your Own Model
=========================

The Python adapter accepts any ``torch.nn.Module`` that predicts the next
symbol from a fixed-length one-hot context. You do not need to modify ROTE's
model registry or its paper experiment runner. Use ``evaluate_model`` for an
already constructed network, ``evaluate_models`` for in-memory replicated
runs, and ``run_benchmark`` for persistent runs with configuration, results,
and event logs.

Model contract
--------------

For alphabet :math:`\mathcal A` with :math:`n=|\mathcal A|` symbols and window
length :math:`w`, the forward method must implement

.. math::

   f_\theta: \mathbb{R}^{B\times w\times n}\longrightarrow
   \mathbb{R}^{B\times n}.

The input is a floating-point, one-hot tensor; the output is **unnormalized
logits** for the next symbol. The last dimension must correspond to ROTE's
sorted alphabet order. Do not return probabilities, class indices, a sequence
of per-token logits, or a tuple such as ``(logits, hidden_state)``. The adapter
checks the shape and finiteness before training. Cross-entropy applies the
softmax internally. ROTE moves the module and data to ``config.device``.

Single prebuilt model
---------------------

This complete script constructs an MLP, generates a controlled seed, trains
on the chronological prefix, evaluates the withheld suffix, saves a run
directory with its configuration and records, and creates available figures. Run it with
``python examples/evaluate_your_model.py --output-dir /tmp/rote-single``.

.. literalinclude:: ../../examples/evaluate_your_model.py
   :language: python

Replace the ``nn.Sequential`` module with your own network, keeping the
forward contract. A prebuilt module is fitted **in place**; its initialization
is not reset by ROTE. For repeated evaluations, pass a factory instead.

Three toy architectures
-----------------------

The next runnable example defines an MLP, a GRU, and a small causal
Transformer encoder without downloading pretrained weights or installing the
optional research architectures. Factories receive the alphabet size and
create fresh weights for each run. All three models see the same generated
seed, window length, training settings, and rollout horizon. Save and plot
their records with:

.. code-block:: bash

   python examples/compare_toy_models.py --runs 2 --epochs 20 \
     --output-dir /tmp/rote-toys

.. literalinclude:: ../../examples/compare_toy_models.py
   :language: python

The example is a CPU integration demonstration, **not** a reproduction of
the manuscript's model-size, token, hyperparameter, or seed grid. A lower DL
on one short seed is not evidence that an architecture is generally superior.
For an actual comparison, fix the protocol in advance, report all runs, and
consider parameter counts and compute alongside accuracy.

Using a factory directly
------------------------

For your own class ``MyNetwork``, a minimal adapter looks like this:

.. code-block:: python

   from rote_benchmark import lzw_string_generator
   from rote_benchmark.evaluation import EvaluationConfig, run_benchmark

   seed, _ = lzw_string_generator(4, 30, random_state=19)
   config = EvaluationConfig(window_size=12, sequence_length=320,
                             forecast_horizon=24, max_epochs=20,
                             device="cuda")
   saved = run_benchmark(
       {"MyNetwork": lambda n_symbols: MyNetwork(n_symbols)},
       seed, config=config, runs=3,
       output_dir="my_network_run",
       metadata={"dataset": "synthetic LZW seed", "study": "prototype"},
   )
   print(saved.path, saved.results[["model", "test_loss", "DL"]])

Here ``MyNetwork`` is a placeholder for your own imported ``nn.Module``.
Change ``device`` to ``"cpu"`` if no CUDA device is available. The run index
increments the random seed before each factory call; train/validation/test
windows and the withheld target remain fixed for that seed. If you need
multiple generated strings, pass a mapping of seed IDs to strings to
``run_benchmark``. Each row then includes ``seed_id``, and measured
``complexity`` is computed from that seed. Preserve the requested generator
settings in ``metadata``; the measured complexity can differ from the
target.

Saved run and figure APIs
-------------------------

``run_benchmark`` saves ``config.json``, incremental ``records.jsonl``
and ``results.csv``, a progress ``manifest.json``, and ``events.jsonl``.
If ``output_dir`` is omitted, it creates a unique directory beneath
``./rote_runs`` (or ``ROTE_RUNS_DIR``). Explicit paths must be new.
Completed records remain readable if a later model fails. Reload and
visualize a run without fitting the models again:

.. code-block:: python

   from rote_benchmark.artifacts import load_run, save_benchmark
   from rote_benchmark.visualization import PlotStyle, visualize_benchmark

   saved = load_run("my_network_run")
   report = visualize_benchmark(
       saved, style=PlotStyle(font_size=12, uncertainty="std"),
   )
   print(report.figures)

For results already computed in memory, ``save_benchmark(results, config,
output_dir=...)`` writes the same artifact layout. Pass a JSON-serializable
configuration mapping (or a dataclass). The single-prebuilt-model example
shows this route. See :doc:`visualization` for figure selection, style,
output formats and overwrite behavior. ``visualize_benchmark`` makes
quality, rollout, cost, and complexity plots when the required columns and
multiple complexity levels are present.

Two-complexity toy sweep
------------------------

A complete example evaluates two tiny models at two LZW targets in a single
saved run. The report automatically includes a ``complexity`` figure in
addition to quality, rollout, and available compute trade-offs.

.. code-block:: bash

   python examples/complexity_sweep.py --runs 2 --epochs 5 \\
     --output-dir /tmp/rote-sweep

.. literalinclude:: ../../examples/complexity_sweep.py
   :language: python

Training versus evaluation
--------------------------

``evaluate_model`` trains from the supplied initial weights using the same
objective and stopping logic as the fixed-budget experiment. It does not
merely score an already trained checkpoint. Use ``score_model`` for a
checkpoint that is already trained: it calculates the same held-out test and
rollout metrics without calling an optimizer. Its training-related config
fields are ignored, and its record intentionally has no training-time or
epoch fields. The checkpoint must have been trained with the same alphabet
order and window size, and the withheld suffix must not have appeared in its
training data.

This complete example saves and reloads a PyTorch state dictionary, scores
the reloaded model, checks that its metrics agree with the fitted instance,
and saves a comparison figure:

.. code-block:: bash

   python examples/score_checkpoint.py --output-dir /tmp/rote-checkpoint

.. literalinclude:: ../../examples/score_checkpoint.py
   :language: python

Common mistakes
---------------

* ``model(x)`` must return one logit vector per window, not one per time step.
* Pass raw logits; ``nn.CrossEntropyLoss`` handles normalization.
* Match the model's expected window length to ``config.window_size``.
* A prebuilt model retains its existing weights; use factories to avoid
  training the same instance again across runs.
* Generated seeds use ASCII letters. The model receives one-hot tensors, not
  characters, and ROTE maps them back to characters for rollout.
