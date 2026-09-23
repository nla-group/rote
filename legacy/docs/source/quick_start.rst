Quick Start
===========

Generate A Controlled Symbolic String
-------------------------------------

``slearn`` can generate a seed string whose alphabet size and LZW complexity are
specified by the user. The returned complexity is measured after reducing the
string to the canonical order in which new symbols first appear.

.. code-block:: python

   from slearn import lzw_string_generator

   seed, complexity = lzw_string_generator(
       nr_symbols=4,
       target_complexity=30,
       priorise_complexity=True,
       random_state=7,
   )

   print(seed)
   print(complexity)

Build A Library Of Seeds
------------------------

Use ``lzw_string_seeds`` when experiments require several alphabets,
complexities, and random restarts.

.. code-block:: python

   from slearn import lzw_string_seeds

   library = lzw_string_seeds(
       symbols=[2, 4, 6, 8],
       complexity=[10, 30, 50],
       iterations=2,
       random_state=42,
   )

   print(library[['nr_symbols', 'LZW_complexity', 'length']])

Train A Symbolic Predictor
--------------------------

``symbolicML`` turns a string into fixed-window examples and trains a
scikit-learn classifier to forecast future symbols recursively.

.. code-block:: python

   from slearn import symbolicML

   model = symbolicML(classifier_name='MLPClassifier', ws=4, random_seed=0)
   X, y = model.encode('ABACABADABACABAD')
   prediction = model.forecast(X, y, step=6, hidden_layer_sizes=(32,), max_iter=500)

   print(''.join(prediction))

Transform A Time Series To Symbols
----------------------------------

Low-level transform classes live in ``slearn.symbols``.

.. code-block:: python

   import numpy as np
   from slearn.symbols import SAX

   t = np.linspace(0, 4 * np.pi, 200)
   series = np.sin(t) + 0.05 * np.random.default_rng(0).normal(size=t.size)

   sax = SAX(window_size=20, alphabet_size=6)
   symbols = sax.fit_transform(series)
   reconstruction = sax.inverse_transform()

Run A Benchmark Smoke Test
--------------------------

The neural benchmark includes a tiny configuration that checks imports, data
preparation, training, and rollout without launching the full workload.

.. code-block:: bash

   python exps/symbolic_sequence_benchmark.py --smoke --device cpu

Results are written to ``exps/results_symbolic/results.csv`` unless another
``--output-dir`` is supplied.
