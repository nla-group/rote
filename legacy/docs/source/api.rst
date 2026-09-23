API Reference
=============

The public API is organized around four modules: symbolic string generation,
string metrics, symbolic time-series representations, and scikit-learn-based
symbolic predictors. The historical module name ``slearn.classifer`` is kept for
backward compatibility.

String Generation
-----------------

.. automodule:: slearn.sgenerate
   :members: lzwcompress, lzwdecompress, reduce, lzw_string_generator, lzw_string_seeds, random_generate, mkc_gererate
   :undoc-members:
   :show-inheritance:

String Metrics
--------------

.. automodule:: slearn.dmetric
   :members:
   :undoc-members:
   :show-inheritance:

Symbolic Time-Series Representations
------------------------------------

The main classes are ``SAX``, ``SAXTD``, ``ESAX``, ``MSAX``, ``ASAX``, ``fABBA``,
and ``ABBA``.

.. automodule:: slearn.symbols
   :members: SAX, SAXTD, ESAX, MSAX, ASAX, symbolsAssign, aggregate, compress, fABBA, ABBA
   :undoc-members:
   :show-inheritance:

Scikit-Learn Wrappers
---------------------

The public wrappers are ``symbolicML`` for symbolic strings and ``slearn`` for
symbolic time-series forecasting.

.. automodule:: slearn.classifer
   :members: symbolicML, slearn
   :undoc-members:
   :show-inheritance:

Experiment Entry Points
-----------------------

The neural benchmark is intentionally documented as a command-line workflow
rather than an imported public API. The primary commands are:

.. code-block:: bash

   python exps/symbolic_sequence_benchmark.py --help
   python exps/visualize_symbolic_results.py --help
   bash exps/scripts/install_experiment_deps.sh
   bash exps/scripts/run_symbolic_benchmark_slurm.sh
   bash exps/scripts/merge_symbolic_results.sh exps/results_symbolic/slurm_<job_id>
   bash exps/scripts/run_symbolic_visualizations.sh
