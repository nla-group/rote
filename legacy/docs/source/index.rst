slearn: Symbolic Sequence Learning
==================================

.. rst-class:: slearn-hero

``slearn`` is a Python package for symbolic sequence generation, symbolic
representations of real-valued time series, string-distance evaluation, and
controlled neural sequence-learning experiments. It combines classic symbolic
methods such as SAX and ABBA-style representations with LZW-controlled synthetic
strings and a modern benchmark for recurrent, attention-based, and hybrid
architectures.

The package is intended for researchers who want symbolic sequences with a known
alphabet size and compression complexity, practitioners who want to convert time
series into symbolic forms before applying machine learning, and authors who need
a reproducible benchmark for finite-context sequence prediction.

.. rst-class:: slearn-card-list

* **Controlled symbolic data.** Generate strings with prescribed alphabet size
  and Lempel-Ziv-Welch (LZW) compression complexity.
* **Symbolic time-series representation.** Encode real-valued series with SAX,
  SAX-TD, eSAX, mSAX, aSAX, ABBA, and fABBA-style transforms.
* **String-level evaluation.** Compare symbolic predictions with normalized
  Damerau-Levenshtein, Jaro-Winkler, Hamming, cosine, LCS, Dice, and
  Smith-Waterman metrics.
* **Neural symbolic benchmark.** Train LSTM, GRU, minGRU, minLSTM, Transformer,
  linear-attention, Performer, and RWKV-style predictors under one data protocol.

Where To Start
--------------

* New users should begin with :doc:`installation` and :doc:`quick_start`.
* Time-series users should read :doc:`predict_with_symbols_representation`.
* Benchmark users should read :doc:`experiments` and ``exps/README.md``.
* API users can jump directly to :doc:`api`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quick_start
   applications
   examples
   symbol_seeds
   symbols_machine_learning
   predict_with_symbols_representation
   dmetric
   experiments

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   citations
   license
