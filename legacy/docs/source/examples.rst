Examples
========

LZW Complexity Sweep
--------------------

.. code-block:: python

   from slearn import lzw_string_seeds

   seeds = lzw_string_seeds(
       symbols=[2, 4, 8],
       complexity=[10, 30, 50, 70],
       iterations=5,
       random_state=123,
   )

   summary = seeds.groupby(['nr_symbols', 'LZW_complexity'])['length'].mean()
   print(summary)

Compare Two Symbolic Forecasts
------------------------------

.. code-block:: python

   from slearn.dmetric import (
       normalized_damerau_levenshtein_distance,
       normalized_jaro_winkler_distance,
   )

   target = 'ABCABCABCABC'
   model_a = 'ABCABCABBABC'
   model_b = 'ACBACBACBABC'

   for name, forecast in [('A', model_a), ('B', model_b)]:
       dl = normalized_damerau_levenshtein_distance(target, forecast)
       jw = normalized_jaro_winkler_distance(target, forecast)
       print(name, {'DL': dl, 'JW': jw})

Build A SAX Pipeline
--------------------

.. code-block:: python

   import numpy as np
   from slearn.symbols import SAX
   from slearn import symbolicML

   rng = np.random.default_rng(4)
   t = np.linspace(0, 10, 300)
   series = np.cos(t) + 0.05 * rng.normal(size=t.size)

   sax = SAX(window_size=30, alphabet_size=6)
   symbols = ''.join(sax.fit_transform(series))

   predictor = symbolicML(classifier_name='LogisticRegression', ws=3, random_seed=4)
   X, y = predictor.encode(symbols)
   future_symbols = predictor.forecast(X, y, step=4, max_iter=1000)

   print(future_symbols)

Run A Small Neural Benchmark
----------------------------

.. code-block:: bash

   python exps/symbolic_sequence_benchmark.py \
     --models LSTM GRU minGRU minLSTM LinearAttention Performer RWKV \
     --symbols 2 4 \
     --complexities 10 30 \
     --sequence-lengths 800 \
     --seed-count 1 \
     --runs 1 \
     --layers 1 \
     --units 64 \
     --d-models 128 \
     --max-epochs 30 \
     --patience 5 \
     --device cpu

Generate Figures From A Result CSV
----------------------------------

.. code-block:: bash

   python exps/visualize_symbolic_results.py \
     --results exps/results_symbolic/results.csv \
     --output-dir exps/figures_symbolic

The visualization script creates separate figures for complexity sweeps,
teacher-forced prediction, rollout stability, compute-performance trade-offs,
model-size trends, sequence-length trends, context-window trends, and heatmaps
when the required columns are present.
