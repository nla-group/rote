Quick Start
===========

Generate controlled seeds:

.. code-block:: python

   from rote_benchmark import lzw_string_generator, lzw_string_seeds

   seed, measured = lzw_string_generator(4, 30, random_state=7)
   table = lzw_string_seeds(
       symbols=[4, 8], complexity=[30, 90], iterations=2, random_state=3407
   )
   print(measured, table[["nr_symbols", "LZW_complexity", "length"]])

``LZW_complexity`` counts LZW output codes over the shortest period of a
generated seed. It does not measure compressed file size. Seeds have a fixed
alphabet size and deterministic random state. Repeating each seed creates the
symbolic stream used in the benchmark.

Evaluate a generated suffix:

.. code-block:: python

   from rote_benchmark import (
       normalized_damerau_levenshtein_distance,
       normalized_jaro_winkler_distance,
   )

   target, generated = "ABAC", "ABCA"
   print(normalized_damerau_levenshtein_distance(target, generated))
   print(normalized_jaro_winkler_distance(target, generated))

Both distances are zero for identical strings. The DL function uses adjacent
transpositions in addition to insertion, deletion, and substitution. This is
the optimal-string-alignment variant used for the reported benchmark results.

To train and score your own ``torch.nn.Module``, continue with
:doc:`custom_models`. For the split, LZW definition and interpretation of
rollout distances, see :doc:`methodology`.

Check the neural environment:

.. code-block:: bash

   python -m pip install -e '.[all]'
   rote-bench --smoke --device cpu --output-dir /tmp/rote-smoke
