Symbolic Machine Learning
=========================

Fixed-Window Prediction
-----------------------

``symbolicML`` treats a symbolic sequence as a supervised next-token problem. For
window length :math:`w`, the string is converted into samples
:math:`(s_t, \ldots, s_{t+w-1})` with target :math:`s_{t+w}`. Symbols are encoded
as integer labels before fitting a scikit-learn classifier.

.. code-block:: python

   from slearn import symbolicML

   sequence = 'ABACABADABACABAD'
   predictor = symbolicML(classifier_name='MLPClassifier', ws=4, random_seed=0)
   X, y = predictor.encode(sequence)

   future = predictor.forecast(
       X,
       y,
       step=5,
       hidden_layer_sizes=(32,),
       max_iter=500,
   )

   print(''.join(future))

Supported Classifier Names
--------------------------

The public constructor selects a scikit-learn estimator by name. Common options
include ``MLPClassifier``, ``KNeighborsClassifier``,
``GaussianProcessClassifier``, ``QuadraticDiscriminantAnalysis``,
``DecisionTreeClassifier``, ``LogisticRegression``, ``AdaBoostClassifier``,
``GaussianNB``, and ``SVC``. Parameters not consumed by ``symbolicML`` are passed
to the underlying estimator.

Forecasting Behavior
--------------------

After fitting on the observed sequence, ``forecast`` predicts one symbol at a
time and appends each prediction to the context used for the next step. This is a
closed-loop rollout. It is therefore stricter than teacher-forced one-step
accuracy: early errors can change the future inputs seen by the classifier.

Time-Series Wrapper
-------------------

The higher-level ``slearn`` class combines a symbolic representation method with
a classifier. It first transforms a numeric time series into symbols, trains the
classifier on the symbolic sequence, forecasts future symbols, and optionally
maps the result back to the numeric domain.

.. code-block:: python

   import numpy as np
   from slearn import slearn

   t = np.linspace(0, 12, 240)
   series = np.sin(t)

   model = slearn(method='fABBA', classifier_name='MLPClassifier', ws=3, step=20)
   model.set_symbols(series, tol=0.1, alpha=0.5)
   forecast = model.predict(hidden_layer_sizes=(64,), max_iter=500)

   print(forecast.shape)

For new code that needs explicit control over the symbolic transform, use the
low-level classes in ``slearn.symbols`` and then pass the resulting string to
``symbolicML``.
