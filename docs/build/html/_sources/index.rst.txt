ROTE
====

ROTE (Rollout Testing of Exact memorization) evaluates neural models on
controlled symbolic sequences. The benchmark varies alphabet size and LZW
complexity, measures teacher-forced next-symbol prediction, and tests whether
the model can extend a withheld suffix under closed-loop rollout.

The Python API provides seed generation, custom PyTorch model evaluation, rollout metrics, and generic visualization. Command-line entry points reproduce the fixed-budget benchmark, matched-size robustness check, and manuscript figures.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quick_start
   custom_models
   methodology
   visualization
   experiments

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   citations
   license
