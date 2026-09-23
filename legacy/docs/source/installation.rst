Installation
============

Core Package
------------

Install the released package from PyPI:

.. code-block:: bash

   pip install slearn

or from conda-forge:

.. code-block:: bash

   conda install -c conda-forge slearn

For development, clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/chenxinye/slearn.git
   cd slearn
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e .

The core package depends on NumPy, pandas, scikit-learn, requests, and PyTorch.
The symbolic representation and string-distance utilities are lightweight. The
neural benchmark has additional optional dependencies described below.

Experiment Dependencies
-----------------------

The manuscript experiments live under ``exps/``. They use a separate virtual
environment by default so that experimental PyTorch and sequence-model packages
do not interfere with the core package installation.

.. code-block:: bash

   git clone https://github.com/chenxinye/slearn.git
   cd slearn
   bash exps/scripts/install_experiment_deps.sh
   source exps/.venv/bin/activate

The script installs ``slearn`` in editable mode and then installs
``requirements-experiments.txt``. The benchmark dependencies include
``minGRU-pytorch``, ``linear-attention-transformer``, ``performer-pytorch``,
``transformers``, matplotlib, seaborn, SciPy, pandas, scikit-learn, and
``textdistance``.

Set ``INSTALL_RWKV_TRAINER=1`` only if you also want the separate
``rwkv-trainer`` package:

.. code-block:: bash

   INSTALL_RWKV_TRAINER=1 bash exps/scripts/install_experiment_deps.sh

The default RWKV baseline used by the benchmark is implemented directly in
``exps/models.py`` and does not require ``rwkv-trainer``.

Documentation Dependencies
--------------------------

Build the documentation locally with Sphinx:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html

The documentation uses ``furo`` for a clean modern scientific Python style and
falls back to ``sphinx_rtd_theme`` if Furo is not installed.

Version Notes
-------------

The current package version is read from ``slearn.__version__``. The package
metadata currently supports Python 3.7 and later. The experiment environment is
best tested with a recent Python and PyTorch stack, especially on GPU clusters.
