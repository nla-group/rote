Data and Metrics
================

Controlled symbolic streams
---------------------------

ROTE starts from a seed string :math:`u\in\mathcal A^*` over an alphabet of
:math:`n` ASCII letters. ``lzw_string_generator(n, c)`` extends a shuffled
alphabet until the shortest period of the string yields at least :math:`c`
Lempel--Ziv--Welch output codes. ``lzw_complexity`` counts emitted codes with
the 52 ASCII letters as the initial dictionary. It is **not** byte length,
entropy, Kolmogorov complexity, or a guarantee that a window identifies the
next symbol. The generator returns both the seed and its measured count;
always record that measured count, because a requested target and measured
value need not be assumed interchangeable.

The benchmark repeats :math:`u` and truncates it to :math:`T` symbols. Its
final :math:`H` symbols form the withheld target :math:`y_{1:H}`. The
preceding :math:`w` observed symbols form the initial rollout context. From
the prefix before the withheld target, ROTE forms ordered next-token examples
:math:`(x_{t-w:t-1},x_t)`. The examples are split chronologically into roughly
80% training, 10% validation, and 10% test windows. Windows overlap near
boundaries; this is a **within-sequence** test, not an independent-string
generalization test. The validation set selects the best checkpoint and can
trigger early stopping. The test windows do not include withheld suffix
targets.

One-step and closed-loop evaluation
-----------------------------------

Teacher-forced test loss is mean cross-entropy on observed test contexts;
test accuracy is the fraction of correct next-symbol predictions. During
closed-loop evaluation, the model instead sees its own previous predictions:

.. math::

   \hat y_h = \mathop{\mathrm{argmax}}_{a\in\mathcal A}
   f_\theta(\hat x_{h-w:h-1})_a, \qquad h=1,\ldots,H.

The initial :math:`\hat x` consists of the last :math:`w` observed prefix
symbols. These two evaluations answer different questions: teacher-forced
accuracy concerns local prediction under correct context, while rollout
measures the stability of iterating the learned rule after errors feed back.

For target :math:`y` and forecast :math:`\hat y` of length :math:`H`, the
reported DL distance is the optimal-string-alignment edit distance divided
by :math:`H`. Despite the historical name, the implementation uses the
restricted adjacent-transposition variant of Damerau--Levenshtein, not the
unrestricted edit distance. ``JW`` is **one minus** Jaro--Winkler similarity;
zero is best for both metrics. An exact rollout has DL equal to zero.

Metric fields and resource measurements
---------------------------------------

.. list-table:: One row returned by ``evaluate_model``
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``test_loss``, ``test_accuracy``
     - Teacher-forced next-symbol performance on the chronological test windows.
   * - ``DL``, ``JW``
     - Distances between full withheld target and closed-loop forecast; lower is better.
   * - ``target_string``, ``forecast``
     - Exact strings needed to inspect failures and recompute horizon curves.
   * - ``model_params``, ``model_size_m``, ``trainable_params``
     - Total parameter count, its value in millions, and the trainable subset.
   * - ``train_time``, ``epochs``, ``time_per_epoch``
     - Wall-clock fit/evaluation duration, epochs used, and their quotient.
   * - ``memory_mb``
     - Peak allocated CUDA memory in MiB; zero on CPU, **not** a CPU-RAM estimate.
   * - ``seed_string``, ``complexity``, ``symbols``
     - Source seed, measured shortest-period LZW code count, and alphabet size.

The benchmark's test loss is the mean of per-batch cross-entropies, matching
the experiment implementation. When the final batch is smaller, this is not
exactly the token-weighted average. For comparisons, keep ``batch_size``
fixed and report it. Timing includes test evaluation and rollout; CUDA timing
may also be affected by asynchronous execution and the software stack. Do
not interpret single-run runtime differences as hardware-independent laws.

Reproducibility scope
---------------------

``evaluate_models`` constructs a fresh model for every ``(name, run)`` pair
with seed ``config.seed + run``. ROTE restores Python, NumPy and PyTorch
CPU/CUDA random states after each call, but it cannot guarantee bit-identical
training across devices or backend versions. Persist the entire CSV, package
versions, hardware, your model source, and the config used. Archived paper
seeds require the seed manifest described in :doc:`experiments`.
