Applications
============

Controlled Symbolic Sequence Experiments
----------------------------------------

The LZW generator creates symbolic strings with a prescribed alphabet size and a
prescribed compression-complexity target. This is useful when a study needs
synthetic sequences that are simple enough to interpret but structured enough to
stress sequence models. A seed can be repeated to form a long periodic sequence,
then split chronologically into training, validation, test, and rollout regions.

Typical uses include memorization studies, finite-context prediction, curriculum
construction, sequence-complexity sweeps, and controlled ablations where natural
language corpora would confound data complexity with vocabulary, semantics, and
corpus preprocessing choices.

Symbolic Time-Series Representation
-----------------------------------

``slearn.symbols`` provides symbolic encoders that turn a real-valued time
series into a sequence of discrete symbols. These methods can reduce dimension,
regularize noisy signals, and make downstream tasks compatible with string and
categorical models.

Available transforms include:

* ``SAX`` for Piecewise Aggregate Approximation followed by Gaussian breakpoints.
* ``SAXTD`` for SAX with trend-direction suffixes.
* ``ESAX``, ``MSAX``, and ``ASAX`` for extended, modified, and adaptive SAX-style
  encodings.
* ``ABBA`` and ``fABBA`` for aggregation-based symbolic approximation with an
  inverse transform back to a real-valued signal.

Scikit-Learn Forecasting On Symbols
-----------------------------------

``symbolicML`` converts a symbolic sequence into supervised pairs
:math:`(x_{t-w+1:t}, x_{t+1})`, where :math:`w` is the context window. Any
supported scikit-learn classifier can then be trained as a next-symbol model and
rolled forward autoregressively. The higher-level ``slearn`` wrapper combines a
symbolic transform with a classifier and returns either symbolic or numeric
forecasts.

This workflow is useful for fast baselines, interpretable symbolic pipelines,
and time-series forecasting experiments where symbolic compression is part of the
model design.

String Metrics For Evaluation
-----------------------------

The distance module contains edit, alignment, token-overlap, and similarity
metrics for predicted symbolic sequences. The normalized variants are especially
useful for comparing forecasts across different horizons because their values
are scaled to a common range.

Neural Architecture Benchmarks
------------------------------

The ``exps/`` directory contains a self-contained benchmark that compares
recurrent, minimal recurrent, attention-based, efficient-attention, and
RWKV-style models on the same finite-context next-token task. It records local
teacher-forced metrics, closed-loop rollout distances, parameter counts, timing,
and peak GPU memory. The scripts are designed so that Slurm runs, result shards,
merged CSV files, and generated figures can all remain under ``exps/``.
