slearn: learning symbolic sequences
===================================

.. image:: https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master
    :target: https://app.travis-ci.com/github/nla-group/slearn
    :alt: Build Status
.. image:: https://badge.fury.io/py/slearn.svg
    :target: https://badge.fury.io/py/slearn
    :alt: PyPI version
.. image:: https://img.shields.io/pypi/pyversions/slearn.svg
    :target: https://pypi.python.org/pypi/slearn/
    :alt: Python versions
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/nla-group/slearn/blob/master/LICENSE
    :alt: License
.. image:: https://readthedocs.org/projects/slearn/badge/?version=latest
    :target: https://slearn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

``slearn`` is a research package for symbolic sequence generation, symbolic
time-series representation, string-distance evaluation, and controlled
sequence-learning experiments. It was originally developed around
LZW-controlled symbolic strings and LSTM/GRU forecasting; the current experiment
suite also compares minimal recurrent, Transformer, efficient-attention, and
RWKV-style models under one finite-context prediction protocol.

Install
-------

Core package:

.. code-block:: bash

    pip install slearn

or:

.. code-block:: bash

    conda install -c conda-forge slearn

Manuscript experiment environment:

.. code-block:: bash

    git clone https://github.com/chenxinye/slearn.git
    cd slearn
    bash exps/scripts/install_experiment_deps.sh
    source exps/.venv/bin/activate

Core Features
-------------

* LZW-controlled symbolic string generation with ``lzw_string_generator`` and
  ``lzw_string_seeds``.
* Symbolic time-series transforms including SAX, SAX-TD, eSAX, mSAX, aSAX,
  ABBA, and fABBA-style representations.
* String distances and similarities including Damerau-Levenshtein,
  Jaro-Winkler, Hamming, cosine, LCS, Dice, and Smith-Waterman variants.
* A reproducible neural benchmark for finite-context symbolic prediction and
  recursive rollout.

Quick Example
-------------

.. code-block:: python

    from slearn import lzw_string_generator, symbolicML
    from slearn.dmetric import normalized_damerau_levenshtein_distance

    seed, complexity = lzw_string_generator(
        nr_symbols=4,
        target_complexity=30,
        random_state=7,
    )

    model = symbolicML(classifier_name="MLPClassifier", ws=4, random_seed=0)
    X, y = model.encode(seed * 4)
    pred = model.forecast(X, y, step=10, hidden_layer_sizes=(32,), max_iter=500)

    target = (seed * 5)[len(seed * 4):len(seed * 4) + 10]
    print(complexity)
    print(normalized_damerau_levenshtein_distance(target, "".join(pred)))

Benchmark Smoke Test
--------------------

.. code-block:: bash

    python exps/symbolic_sequence_benchmark.py --smoke --device cpu

For Slurm runs, submit from ``exps/``:

.. code-block:: bash

    cd exps
    sbatch scripts/run_symbolic_benchmark_slurm.sh

After completion, merge shards and generate figures:

.. code-block:: bash

    bash scripts/merge_symbolic_results.sh results_symbolic/slurm_<array_job_id>
    bash scripts/run_symbolic_visualizations.sh results_symbolic/slurm_<array_job_id>/results_merged.csv

Documentation
-------------

The Furo-styled documentation covers installation, quick start examples,
application workflows, experiment reproduction, API references, license, and
citations. Build it locally with:

.. code-block:: bash

    python -m pip install -r docs/requirements.txt
    sphinx-build -b html docs/source docs/build/html

Citation
--------

If you use ``slearn`` or the LZW symbolic string library, please cite:

.. code-block:: bibtex

    @inproceedings{cahuantzi2023comparison,
      title = {A Comparison of LSTM and GRU Networks for Learning Symbolic Sequences},
      author = {Cahuantzi, Roberto and Chen, Xinye and Guettel, Stefan},
      booktitle = {Intelligent Computing},
      pages = {771--785},
      year = {2023},
      publisher = {Springer Nature Switzerland}
    }

License
-------

This project is licensed under the MIT License.
