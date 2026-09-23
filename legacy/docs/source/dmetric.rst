String Distances And Similarities
=================================

The ``slearn.dmetric`` module collects edit, alignment, and token-overlap
metrics for symbolic strings. Distances are useful for rollout evaluation, while
similarities are useful for nearest-neighbor search and descriptive analysis.

Basic Distances
---------------

.. code-block:: python

   from slearn.dmetric import (
       damerau_levenshtein_distance,
       normalized_damerau_levenshtein_distance,
       normalized_jaro_winkler_distance,
   )

   reference = 'ABACABAD'
   predicted = 'ABADABAC'

   print(damerau_levenshtein_distance(reference, predicted))
   print(normalized_damerau_levenshtein_distance(reference, predicted))
   print(normalized_jaro_winkler_distance(reference, predicted))

Metric Guide
------------

.. list-table:: Public functions
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Output type
     - Use case
   * - ``damerau_levenshtein_distance``
     - distance
     - Edit distance allowing insertion, deletion, substitution, and adjacent transposition.
   * - ``levenshtein_distance``
     - distance
     - Edit distance without transposition.
   * - ``hamming_distance``
     - distance
     - Position-wise mismatch count for equal-length strings.
   * - ``jaro_similarity``
     - similarity
     - Matching and transposition-aware similarity for short strings.
   * - ``jaro_winkler_distance``
     - distance
     - Prefix-weighted Jaro-Winkler distance.
   * - ``cosine_similarity``
     - similarity
     - Character-frequency cosine similarity.
   * - ``cosine_bigram_similarity``
     - similarity
     - Bigram-frequency cosine similarity.
   * - ``lcs_distance``
     - distance
     - Longest common subsequence distance.
   * - ``dice_coefficient``
     - similarity
     - Bigram overlap coefficient.
   * - ``smith_waterman_distance``
     - distance
     - Local alignment-based distance.

Normalized Variants
-------------------

The module also provides normalized forms such as
``normalized_levenshtein_distance``, ``normalized_hamming_distance``,
``normalized_jaro_similarity``, ``normalized_jaro_winkler_similarity``,
``normalized_jaro_winkler_distance``, ``normalized_cosine_similarity``,
``normalized_cosine_bigram_similarity``, ``normalized_lcs_distance``,
``normalized_dice_coefficient``, ``normalized_smith_waterman_distance``, and
``normalized_damerau_levenshtein_distance``.

For benchmark rollout, the recommended pair is:

* normalized Damerau-Levenshtein distance (``DL``): lower is better;
* normalized Jaro-Winkler distance (``JW``): lower is better.

Using both gives complementary views: ``DL`` emphasizes edit operations over the
whole forecast, while ``JW`` is sensitive to ordered matches and prefix
agreement.
