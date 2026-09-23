API Reference
=============

ROTE separates its lightweight seed and metric API from optional neural
evaluation and plotting. Importing ``rote_benchmark`` does not import
PyTorch. The examples below use the public Python API; paper reproduction
uses the command-line interface described in :doc:`experiments`.

Seed generation
---------------

.. automodule:: rote_benchmark.generation
   :members: lzw_complexity, lzw_string_generator, lzw_string_seeds

The output of ``lzw_string_seeds`` is a DataFrame with ``nr_symbols``,
``LZW_complexity``, ``length``, and ``string`` columns. A requested code count
is a target; use the returned measured count in analyses.

Rollout distances
-----------------

.. automodule:: rote_benchmark.metrics
   :members: normalized_damerau_levenshtein_distance, normalized_jaro_winkler_distance

Custom neural evaluation
------------------------

Install ``.[neural]`` first. Import these names from
``rote_benchmark.evaluation``. The complete model contract and runnable code
are in :doc:`custom_models`.

.. py:class:: EvaluationConfig(sequence_length=320, window_size=12, forecast_horizon=24, batch_size=32, max_epochs=12, patience=4, stopping_loss=0.05, optimizer="AdamW", learning_rate=0.001, weight_decay=0.0, lr_step_size=100, lr_gamma=0.5, device="cpu", seed=3407)

   Immutable dataclass of training, split and rollout settings. A sequence
   must have room for the context, withheld suffix and chronological splits.
   ``optimizer`` is ``"Adam"`` or ``"AdamW"``. The default device is CPU.

.. py:function:: evaluate_model(model, seed_string, *, config=None, name=None, run=0)

   Train one prebuilt ``nn.Module`` or a fresh ``factory(n_symbols)`` and
   evaluate teacher-forced test windows plus closed-loop rollout. Returns a
   CSV-ready dict. A prebuilt module is fitted in place. The factory is called
   after setting ``config.seed``. Raises ``ValueError`` for invalid model
   output shape or nonfinite logits.

.. py:function:: score_model(model, seed_string, *, config=None, name=None)

   Score a fitted ``nn.Module`` with no optimization. Returns a CSV-ready
   dict with test metrics, target and forecast strings, rollout distances and
   parameter count; no train-time or epoch fields are reported. The module
   is moved to ``config.device`` and left in eval mode. Use this for a
   checkpoint whose alphabet order and window size match the ROTE sequence.

.. py:function:: evaluate_models(models, seed_string, *, config=None, runs=1)

   Evaluate a mapping of model names to fresh factories using a shared seed
   and settings. Returns one pandas DataFrame row per model/run pair. The
   random seed for run :math:`r` is ``config.seed + r``.

Saved benchmark runs
--------------------

``rote_benchmark.artifacts`` uses pandas and the lightweight core only.
``run_benchmark`` additionally requires PyTorch and is imported from
``rote_benchmark.evaluation``.

.. py:function:: run_benchmark(models, seed_string, *, config=None, runs=1, output_dir=None, metadata=None)

   Evaluate fresh model factories under one protocol, writing each completed
   record immediately. The second argument is one seed string or a mapping
   from seed ID to string for a complexity sweep. Return a ``SavedRun``. ``output_dir=None`` creates
   a unique path beneath ``./rote_runs`` or ``ROTE_RUNS_DIR``; an explicit
   directory must not exist. ``metadata`` is a JSON-serializable mapping.

.. py:function:: save_benchmark(results, config, *, output_dir=None)

   Save an existing DataFrame and configuration mapping or dataclass in a
   new run directory. Return a ``SavedRun``.

.. py:function:: load_run(path)

   Reload a run directory. Prefer the append-only JSONL records if present,
   so completed records survive interruption before CSV replacement.

.. py:class:: SavedRun(path, config, results, manifest)

   Immutable container for the run directory, configuration dict, result
   DataFrame and progress manifest. Properties ``results_csv``,
   ``records_jsonl``, and ``events_jsonl`` give artifact paths.

Custom-model figures
--------------------

Install ``.[plot]`` and import from ``rote_benchmark.visualization``.
All plot functions accept a result DataFrame, return a Matplotlib ``Figure``,
and optionally save an output path. The high-level function accepts a
``SavedRun``, run directory, CSV path, or DataFrame.

.. py:class:: PlotStyle

   Frozen dataclass controlling typography, colors, markers, line styles,
   figure size, uncertainty summaries, selected metrics, and file formats.
   See :doc:`visualization` for settings and a complete example.

.. py:function:: visualize_benchmark(source, *, output_dir=None, style=None, figures=None, overwrite=False)

   Save a coordinated report and ``visualization.json``. Available figure
   names are ``quality``, ``rollout``, ``tradeoffs``, and ``complexity``.
   Existing files are protected unless ``overwrite=True``. Returns a
   ``VisualizationReport`` with ``.path``, ``.figures``, and
   ``.config_path``.

.. py:function:: plot_model_comparison(results, output=None, *, style=None)

   Point-and-error panels for configured teacher-forced and rollout metrics.

.. py:function:: plot_rollout_error(results, output=None, *, style=None)

   Prefix DL over horizon, with model styles and uncertainty bands.

.. py:function:: plot_tradeoffs(results, output=None, *, style=None)

   Rollout distance against available compute and resource measurements.

.. py:function:: plot_complexity_sweep(results, output=None, *, style=None)

   Mean metrics and variation against measured LZW code count; requires at
   least two distinct complexities.

Experiment commands
-------------------

.. code-block:: text

   rote-bench
   rote-matched-size
   rote-plot
   rote-plot-matched

``rote-benchmark`` remains a compatibility alias for ``rote-bench``. Use
``--help`` on each command for all options. The fixed-budget and matched-size
CLI runners preserve the manuscript protocol and archived CSV schema. Their
implementations live in ``rote_benchmark.benchmark`` and
``rote_benchmark.matched_size``; specialized publication plots live in
``rote_benchmark.plotting`` and ``rote_benchmark.plot_matched``. The generic
custom-model plots are separate and do not alter publication figures.
