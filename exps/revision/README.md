# TMLR reviewer experiments

These experiments are **additional controls**, not replacements for the archived
fixed-recipe and matched-size results in `legacy/exps/`. No result from this
directory should be described as complete until its stage `manifest.json` says
`"status": "complete"` and its row count equals `total`.

## One-command cluster run

From the repository's `exps/revision` directory:

```bash
cd exps/revision
sbatch run_revision_slurm.sh
```

The Slurm job installs the optional architecture dependencies in
`exps/revision/.venv` on first use, runs all stages in the order below, and
builds `summary_by_configuration.csv`, `summary_by_seed.csv`, and
`run_status.json`. It uses one
`a100_3g.40gb` allocation, 12 CPU cores, 64 GB RAM, and a 48-hour limit.
For the CUDA job, installation pins PyTorch to `2.14.0+cu130`, matching the
archived `slurm_102076` log, and checks the installed CUDA runtime before
training. The wheel source follows the [official PyTorch previous-version
instructions](https://pytorch.org/get-started/previous-versions/). The
optional model packages are installed from the project's `[experiment]` extra,
and their resolved versions are recorded per stage.
The complete design has 1,172 evaluations across nine stages, including 320
inexpensive nonneural baseline evaluations. It can take substantial GPU time;
the 48-hour limit does not guarantee completion in a single allocation.
Outputs default to `exps/revision/results/slurm_<job_id>/`. If a job is
interrupted, submit again with
`REVISION_OUTPUT_ROOT=/absolute/path/to/old/output sbatch run_revision_slurm.sh`;
the stage store checks configuration and seed manifests and skips completed
case IDs. Do not use the same output root for a changed experiment design.

## Stages

| Stage | Question | Design |
|---|---|---|
| `baseline` | Is a neural model needed? | Train-prefix lookup and smoothed n-gram on all 40 archived seeds at windows around predictive order and 100. |
| `fixed_length` | Does LZW matter beyond period length? | For each alphabet of 4 or 8 symbols, generate six independent seeds at each measured score 70/90 while fixing length and symbol counts within that alphabet. Four neural models, two initializations. |
| `context` | Does finite-context ambiguity predict error? | Eight archived seeds; windows `r* - 1`, `r*`, `r* + 2`, and 100; three neural models, two initializations. Sequence length stays at 3500, so the number of training windows varies slightly with `w` and is reported. Ambiguity, Bayes error, and conditional entropy refer to uniform cyclic phases; the finite chronological test block can have different phase frequencies. |
| `exposure` | Does seeing more periods matter? | On four fixed archived seeds, use 4, 8, or 16 complete periods in the observed prefix before the chronological train/validation/test split. Three models, two initializations. The amount of training data changes by design. |
| `convergence` | Did the original shared recipe converge? | Refit the eight main architectures on four high-complexity seeds and record train/validation loss, accuracy, epochs, and exact rollout from the best-validation checkpoint. |
| `sensitivity` | Which optimization recipe works for each model? | Three learning rates on tuning seeds (seed index 0); for Transformer, Performer, and LinearAttention also compare 0 versus 10 warmup epochs. Selection uses **validation loss only**. |
| `tuned` | Does the ranking survive tuning? | Fix one configuration per architecture from the sensitivity stage; evaluate on the separate archived seeds with index 1. Do not choose on test loss or rollout. |
| `capacity` | What happens for longer periods? | Exact LZW scores 90/150/250, 4/8 symbols, two seeds, three models, two widths, two initializations; the observed prefix contains eight periods before its chronological split. This is a stress test, not a formal capacity bound. |
| `text_bridge` | Does a related behavior appear on real text? | Three disjoint 3500-character Tiny Shakespeare excerpts; four models, two initializations, chronological train/validation/test and an unseen 100-character suffix. This is a **nonperiodic, exploratory character-level diagnostic**, not evidence of transfer to large language models. |

The fixed-length generator also fixes each alphabet's symbol counts, but it
does not hold predictive order or other motif statistics constant. A residual
gap between LZW levels is therefore an association under these controls, not
an isolated causal effect of the code count. The natural-text alphabet sizes
and entropy differ from the synthetic cells, so absolute losses across tracks
are not directly comparable.

The text corpus is the public-domain Shakespeare sample distributed with
[Karpathy's char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
The bundled file has SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Its exact bytes are checked before use. Nonperiodic text is never interpreted
using the periodic ambiguity or predictive-order diagnostics.

## Local verification and individual stages

```bash
REVISION_SKIP_INSTALL=1 PYTHON=python3 bash exps/revision/run_all.sh \
  --smoke --device cpu --output-root /tmp/rote-revision-smoke
python3 -m exps.revision.run_revision --stage baseline \
  --device cpu --output-dir /tmp/rote-baseline
python3 -m exps.revision.summarize /tmp/rote-revision-smoke
```

Install `pip install -e '.[experiment]'` before running these commands with
non-smoke models. Each stage directory contains `config.json`, `seeds.csv`,
`records.jsonl`, `results.csv`, `events.jsonl`, and `manifest.json`. The
configuration and seed manifest are checked on resume; results are written
after every completed case. `config.json` records a SHA-256 fingerprint of
the experiment source and the installed package versions; changing source code
or dependencies prevents accidental continuation into an existing output root.
`summary_by_seed.csv` averages repeated initializations of each symbolic
seed. `summary_by_configuration.csv` then averages those seed means and
computes standard deviations across independent seeds. Its standard-deviation
field is blank when only one seed is available. These are descriptive
statistics, not confidence intervals.

The original archived experiment used a maximum of 200 epochs with early
stopping, so it did **not** equalize realized optimization steps. For a
reviewer response, report both the shared-recipe and validation-selected
results, and state explicitly when a control changes data volume or model
parameter count. Keep claims about real-language relevance conditional on
the observed `text_bridge` results.
