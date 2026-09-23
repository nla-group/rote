Symbolic Time-Series Representation
===================================

Overview
--------

The ``slearn.symbols`` module converts real-valued sequences into discrete
symbol strings. This can reduce dimension, make local structure easier to
compare, and allow time-series workflows to use string metrics or categorical
sequence models.

SAX Example
-----------

.. code-block:: python

   import numpy as np
   from slearn.symbols import SAX

   rng = np.random.default_rng(0)
   t = np.linspace(0, 8 * np.pi, 400)
   series = np.sin(t) + 0.1 * rng.normal(size=t.size)

   sax = SAX(window_size=32, alphabet_size=8)
   symbols = sax.fit_transform(series)
   reconstruction = sax.inverse_transform()

   print(symbols[:8])
   print(reconstruction.shape)

SAX normalizes the input, partitions it into ``window_size`` aggregate segments,
computes each segment mean, and assigns a symbol according to Gaussian
breakpoints. The inverse transform expands the stored aggregate values back to
the original segment lengths.

Trend-Aware SAX
---------------

``SAXTD`` augments SAX symbols with a trend suffix. Suffix ``u`` denotes an
upward local slope, ``d`` a downward local slope, and ``f`` a flat segment under
the selected threshold.

.. code-block:: python

   from slearn.symbols import SAXTD

   saxtd = SAXTD(window_size=24, alphabet_size=6, slope_threshold=0.01)
   trend_symbols = saxtd.fit_transform(series)

Adaptive And Aggregation-Based Methods
--------------------------------------

``ESAX``, ``MSAX``, and ``ASAX`` provide alternative SAX-style encodings that
retain additional shape information or adapt segment boundaries. ``ABBA`` and
``fABBA`` construct symbolic pieces from approximately linear increments and can
reconstruct an approximate numeric signal from the symbolic representation.

.. code-block:: python

   from slearn.symbols import fABBA

   encoder = fABBA(tol=0.1, alpha=0.5, sorting='2-norm', verbose=0)
   symbols = encoder.fit_transform(series)
   recovered = encoder.inverse_transform(symbols, start=series[0])

Choosing A Representation
-------------------------

Use ``SAX`` when a fast, stable, Gaussian-breakpoint discretization is enough.
Use ``SAXTD`` when local trend direction matters. Use adaptive SAX variants when
fixed equal-size segments are too restrictive. Use ``ABBA`` or ``fABBA`` when
piecewise-linear reconstruction is part of the workflow.
