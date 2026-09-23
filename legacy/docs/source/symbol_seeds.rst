LZW-Controlled Symbolic Seeds
=============================

Why LZW Complexity?
-------------------

For a finite string :math:`s = s_1, \ldots, s_n`, Lempel-Ziv-Welch compression
builds a dictionary of substrings while scanning the sequence from left to
right. The number of emitted dictionary codes is a practical proxy for symbolic
regularity: highly repetitive strings compress into fewer codes, while strings
with more novel local patterns require more codes.

``slearn`` uses this quantity as a controllable data-generation axis. A user can
request an alphabet size :math:`A` and a target complexity :math:`C`; the
generator then returns a seed string whose reduced form has approximately that
LZW code length. Repeating the seed gives a long symbolic sequence whose global
period is known while its local compressibility can be swept systematically.

Canonical Reduction
-------------------

The function ``reduce`` maps symbols to a canonical order based on first
appearance. For example, two strings that differ only by a relabeling of symbols
have the same reduced form. This makes the complexity target depend on pattern
structure rather than on arbitrary character names.

.. code-block:: python

   from slearn import reduce, lzwcompress

   pattern = reduce('ZXZZYZX')
   codes = lzwcompress(pattern)
   print(pattern)
   print(len(codes))

Single Seed Generation
----------------------

.. code-block:: python

   from slearn import lzw_string_generator

   seed, measured_complexity = lzw_string_generator(
       nr_symbols=6,
       target_complexity=40,
       priorise_complexity=True,
       random_state=12,
   )

   assert len(set(seed)) == 6
   print(seed, measured_complexity)

``priorise_complexity=True`` stops once the requested compression level is
reached while maintaining the requested alphabet. With ``priorise_complexity``
set to ``False``, generation prioritizes reaching the requested alphabet before
stopping.

Seed Libraries
--------------

``lzw_string_seeds`` serializes many generation calls into a pandas DataFrame.
The output columns are:

* ``nr_symbols``: number of distinct symbols in the generated seed.
* ``LZW_complexity``: measured LZW code length of the reduced seed.
* ``length``: seed length.
* ``string``: generated symbolic seed.

.. code-block:: python

   from slearn import lzw_string_seeds

   seeds = lzw_string_seeds(
       symbols=[2, 4, 6],
       complexity=[10, 30, 50],
       iterations=3,
       save_csv=False,
       random_state=3407,
   )

   print(seeds.head())

The arguments ``symbols`` and ``complexity`` accept either scalars or explicit
lists. When used with ``symbols_range_distribution`` or
``complexity_range_distribution``, two- or three-value ranges can be expanded
linearly or geometrically.

Practical Guidance
------------------

Use small alphabets and low complexity for debugging. For neural benchmarks,
keep the seed length comfortably shorter than the final sequence length so that
repetition creates many supervised windows. Cases where ``nr_symbols`` is larger
than the target complexity are skipped because every symbol must appear at least
once.
