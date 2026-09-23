Installation
============

ROTE requires Python 3.10 or newer. Its distribution name is
``rote-bench`` and its import name is ``rote_benchmark``. It is currently
distributed from source; the PyPI release is not published yet.

.. code-block:: bash

   git clone https://github.com/nla-group/rote.git
   cd slearn
   python -m pip install -e .

The core install contains LZW seed generation and rollout metrics without
importing PyTorch. Choose only the optional dependencies needed for your task:

.. code-block:: bash

   python -m pip install -e '.[neural]'        # custom PyTorch evaluation
   python -m pip install -e '.[neural,plot]'   # evaluation plus figures
   python -m pip install -e '.[experiment]'
   python -m pip install -e '.[plot]'
   # or install both groups
   python -m pip install -e '.[all]'

The ``experiment`` extra installs the optional architecture packages used by
the manuscript; ``neural`` installs only PyTorch and is sufficient for the
self-contained toy examples. ``plot`` installs Matplotlib and seaborn. The
distribution is named ``rote-bench``, but Python imports remain
``rote_benchmark``; this avoids colliding with an unrelated ``rote`` package
on PyPI. ``pip install rote-bench`` is not a working release instruction
until this project has actually been published.

The current repository URL is ``https://github.com/nla-group/rote.git``.
When the repository is renamed to ``rote``, use
``https://github.com/chenxinye/rote.git`` for new clones. The distribution
name, import path and console commands do not need to change. Repository
links in ``pyproject.toml``, ``CITATION.cff``, README badges and Furo source
links should be switched as part of the repository rename; they continue
to point to the live location until then.

Verify a source install with ``python -c "import rote_benchmark"``. For the
neural adapter, run ``python examples/evaluate_your_model.py``. A GPU is not
required for these examples. To run on CUDA, install a PyTorch build compatible
with the local driver and set ``EvaluationConfig(device="cuda")``.

For the Slurm workflow, ``bash exps/scripts/install_experiment_deps.sh``
creates ``exps/.venv`` and installs the same ``all`` extra. The Slurm array
scripts call this installer automatically when needed. A GPU-enabled PyTorch
installation should be checked against the local CUDA environment before
submitting compute-intensive jobs.

Build the Furo documentation locally with:

.. code-block:: bash

   python -m pip install -e '.[docs]'
   sphinx-build -b html -W docs/source docs/build/html
