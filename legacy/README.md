<div align="center">

# slearn: learning symbolic sequences
[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![Codecov Workflow](https://github.com/nla-group/slearn/actions/workflows/unittests.yml/badge.svg)](https://github.com/nla-group/slearn/actions/workflows/unittests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Conda Version](https://anaconda.org/conda-forge/slearn/badges/version.svg)](https://anaconda.org/conda-forge/slearn)
[![Download Status](https://static.pepy.tech/badge/slearn)](https://pypi.org/project/slearn/)
[![Documentation Status](https://readthedocs.org/projects/slearn/badge/?version=latest)](https://slearn.readthedocs.io/en/latest/)


</div>


`slearn` is a research package for symbolic sequence generation, symbolic time-series representation, string-distance evaluation, and controlled sequence-learning experiments. It connects classic symbolic representations such as SAX and ABBA-style encodings with LZW-controlled synthetic strings and a modern benchmark for finite-context neural prediction.

## Install

Core package:

```bash
pip install slearn
# or
conda install -c conda-forge slearn
```

Experiment environment:

```bash
git clone https://github.com/chenxinye/slearn.git
cd slearn
bash exps/scripts/install_experiment_deps.sh
source exps/.venv/bin/activate
```

Set `INSTALL_RWKV_TRAINER=1` only if you also want the separate `rwkv-trainer` package. The benchmark's default RWKV baseline is implemented directly in `exps/models.py`.

## What slearn Provides

- LZW-controlled symbolic string generation with `lzw_string_generator` and `lzw_string_seeds`.
- Symbolic time-series transforms, including SAX, SAX-TD, eSAX, mSAX, aSAX, ABBA, and fABBA-style representations.
- String distances and similarities, including Damerau-Levenshtein, Jaro-Winkler, Hamming, cosine, LCS, Dice, and Smith-Waterman variants.
- Scikit-learn-based symbolic next-token forecasting through `symbolicML`.
- A self-contained experiment suite for LSTM, GRU, minGRU, minLSTM, Transformer, BERT, GPT, LinearAttention, Performer, and RWKV-style models.

## Quick Start

```python
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
```

## LZW String Libraries

```python
from slearn import lzw_string_seeds

library = lzw_string_seeds(
    symbols=[2, 4, 6, 8],
    complexity=[10, 30, 50],
    iterations=3,
    random_state=42,
)
print(library.head())
```

The output contains `nr_symbols`, `LZW_complexity`, `length`, and `string`. Repeating these seeds produces controlled symbolic sequences for memorization, finite-context prediction, and rollout studies.

## Symbolic Time-Series Representation

```python
import numpy as np
from slearn.symbols import SAX

t = np.linspace(0, 10, 100)
series = np.sin(t) + np.random.default_rng(0).normal(0, 0.1, 100)

sax = SAX(window_size=10, alphabet_size=8)
symbols = sax.fit_transform(series)
reconstruction = sax.inverse_transform()
```

## Neural Symbolic Benchmark

Run a quick installation check:

```bash
python exps/symbolic_sequence_benchmark.py --smoke --device cpu
```

Run the default pilot benchmark:

```bash
python exps/symbolic_sequence_benchmark.py
```

Submit a Slurm array job from `exps/`:

```bash
cd exps
sbatch scripts/run_symbolic_benchmark_slurm.sh
```

Merge result shards and generate figures:

```bash
bash scripts/merge_symbolic_results.sh results_symbolic/slurm_<array_job_id>
bash scripts/run_symbolic_visualizations.sh results_symbolic/slurm_<array_job_id>/results_merged.csv
```

The benchmark evaluates teacher-forced test loss and accuracy, recursive rollout distance (`DL` and `JW`), trainable parameters, training time, time per epoch, epoch count, and peak GPU memory.

## Documentation

The Furo-styled Sphinx documentation covers installation, quick start examples, application workflows, experiment reproduction, API references, license, and citations. Build it locally with:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

## Citation

If you use `slearn` or the LZW symbolic string library, please cite:

[1] Cahuantzi, R., Chen, X. and Güttel, S. (2023) ‘A comparison of LSTM and GRU networks for learning symbolic sequences’, in Intelligent Computing. Springer Nature Switzerland, pp. 771–785. Available at: https://doi.org/10.1007/978-3-031-37963-5_53


If you use the ABBA/fABBA symbolic time-series tools, please cite:

[2] Chen, X. (2024) Fast aggregation-based algorithms for knowledge discovery. PhD thesis. The University of Manchester.

[3]  Chen, X. and Güttel, S. (2023) ‘An efficient aggregation method for the symbolic representation of temporal data’, ACM Transactions on Knowledge Discovery from Data, 17(1), article 22. Available at: https://doi.org/10.1145/3532622




## License

This project is licensed under the [MIT License](https://github.com/nla-group/slearn/blob/master/LICENSE).
