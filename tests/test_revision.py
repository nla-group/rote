"""Focused checks for reviewer-experiment design and resumable outputs."""

import csv
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from exps.revision.baselines import evaluate_baseline
from exps.revision.common import Protocol, effective_hidden_size
from exps.revision.run_revision import select_tuning
from exps.revision.seeds import (archived_seeds, controlled_seeds,
                                 phase_statistics, predictive_order)
from exps.revision.storage import StageStore, append_jsonl, read_jsonl
from exps.revision.summarize import summarize
from exps.revision.text_bridge import text_excerpts
from rote_benchmark.generation import lzw_complexity
from rote_benchmark.benchmark import prepare_data
from rote_benchmark.models import get_model


def test_controlled_seeds_match_length_composition_and_score():
    frame = controlled_seeds(count=2)
    assert len(frame) == 8
    for n, group in frame.groupby("symbols"):
        assert group.length.nunique() == 1
        histograms = {tuple(sorted(pd.Series(seed).value_counts().items()))
                      for seed in group.seed_string.map(list)}
        assert len(histograms) == 1
        for row in group.itertuples(index=False):
            assert lzw_complexity(row.seed_string) == row.complexity


def test_ambiguity_and_lookup_on_archived_seed():
    frame = archived_seeds()
    row = frame.iloc[0].to_dict()
    order = predictive_order(row["seed_string"])
    assert phase_statistics(row["seed_string"], order)["alpha"] == 0
    if order > 1:
        assert phase_statistics(row["seed_string"], order - 1)["alpha"] > 0
    result = evaluate_baseline(row, window=100)
    assert result["test_accuracy"] == 1.0
    assert result["exact"] == 1


def test_phase_uniform_bayes_values_and_chronological_split():
    import math

    values = phase_statistics("AAB", 1)
    assert values["alpha"] == pytest.approx(2 / 3)
    assert values["bayes_error"] == pytest.approx(1 / 3)
    assert values["conditional_entropy"] == pytest.approx(2 * math.log(2) / 3)
    assert predictive_order("AAB") == 2
    data = prepare_data("AAB", 3500, 100, 100, "cpu")
    assert (len(data["X_train"]), len(data["X_valid"]),
            len(data["X_test"])) == (2640, 330, 330)


def test_stage_store_resumes_without_duplicate_case():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "baseline"
        config = {"stage": "baseline"}
        seeds = [{"symbols": 2, "complexity": 10, "seed_index": 0,
                  "seed_string": "ABBA"}]
        first = StageStore(path, config, 2, seeds)
        first.append({"case_id": "one", "model": "Lookup"})
        second = StageStore(path, config, 2, seeds)
        assert second.completed == {"one"}
        second.append({"case_id": "two", "model": "NGram"})
        second.finish()
        with (path / "results.csv").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 2
        with pytest.raises(ValueError):
            StageStore(path, {"stage": "changed"}, 2, seeds)


def test_stage_store_repairs_interrupted_jsonl_and_csv(tmp_path):
    path = tmp_path / "baseline"
    seeds = [{"symbols": 2, "complexity": 10, "seed_index": 0,
              "seed_string": "ABBA"}]
    first = StageStore(path, {"stage": "baseline"}, 2, seeds)
    first.append({"case_id": "one", "model": "Lookup"})
    (path / "results.csv").unlink()
    with (path / "records.jsonl").open("ab") as stream:
        stream.write(b'{"case_id":"partial"')
    second = StageStore(path, {"stage": "baseline"}, 2, seeds)
    assert second.completed == {"one"}
    with (path / "results.csv").open(newline="") as stream:
        assert [row["case_id"] for row in csv.DictReader(stream)] == ["one"]
    second.append({"case_id": "two", "model": "NGram"})
    second.finish()
    assert [row["case_id"] for row in read_jsonl(path / "records.jsonl")] == [
        "one", "two"
    ]


def test_valid_jsonl_without_final_newline_can_be_extended(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"case_id":"one"}', encoding="utf-8")
    assert len(read_jsonl(path)) == 1
    append_jsonl(path, {"case_id": "two"})
    assert [row["case_id"] for row in read_jsonl(path)] == ["one", "two"]


def test_tuning_uses_validation_only_and_covers_models(tmp_path):
    from exps.revision.run_revision import (ATTENTION_MODELS, MAIN_MODELS,
                                            TUNING_RATES)

    rows = []
    for model in MAIN_MODELS:
        for rate in TUNING_RATES:
            for warmup in ((0, 10) if model in ATTENTION_MODELS else (0,)):
                for n in (4, 8):
                    for run in (0, 1):
                        loss = 0.1 if rate == 0.001 else 0.3
                        rows.append({"stage": "sensitivity", "model": model,
                                     "symbols": n, "complexity": 90,
                                     "seed_index": 0, "run": run,
                                     "learning_rate": rate,
                                     "warmup_epochs": warmup, "valid_loss": loss,
                                     "test_loss": 1 - loss})
    path = tmp_path / "sensitivity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert all(choice["learning_rate"] == 0.001
               for choice in select_tuning(path).values())
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        select_tuning(path)


def test_text_bridge_is_nonperiodic_and_reproducible():
    frame = text_excerpts(count=2)
    assert len(frame) == 2
    assert all(frame.length == 3500)
    assert frame.seed_string.nunique() == 2
    assert all(frame.source_type == "nonperiodic_natural_text")


def test_capacity_summary_keeps_model_scales_separate(tmp_path):
    stage = tmp_path / "capacity"
    stage.mkdir()
    (stage / "manifest.json").write_text(
        '{"status":"complete","completed":2,"total":2}', encoding="utf-8"
    )
    rows = [
        {"stage": "capacity", "model": "GRU", "model_scale": scale,
         "symbols": 4, "complexity": 90, "seed_index": 0,
         "seed_string": "ABCD", "test_loss": loss}
        for scale, loss in (("small", 0.1), ("large", 0.2))
    ]
    pd.DataFrame(rows).to_csv(stage / "results.csv", index=False)
    summarize(tmp_path)
    summary = pd.read_csv(tmp_path / "summary_by_configuration.csv")
    assert set(summary.model_scale) == {"small", "large"}
    assert len(summary) == 2


def test_configuration_variation_uses_independent_seed_means(tmp_path):
    stage = tmp_path / "fixed_length"
    stage.mkdir()
    (stage / "manifest.json").write_text(
        '{"status":"complete","completed":4,"total":4}', encoding="utf-8"
    )
    rows = [
        {"stage": "fixed_length", "model": "GRU", "symbols": 4,
         "complexity": 90, "seed_index": seed_index,
         "seed_string": seed, "run": run, "test_loss": loss}
        for seed_index, seed, losses in ((0, "ABCD", (0.0, 1.0)),
                                         (1, "ACBD", (2.0, 3.0)))
        for run, loss in enumerate(losses)
    ]
    pd.DataFrame(rows).to_csv(stage / "results.csv", index=False)
    summarize(tmp_path)
    summary = pd.read_csv(tmp_path / "summary_by_configuration.csv").iloc[0]
    by_seed = pd.read_csv(tmp_path / "summary_by_seed.csv")
    assert summary.independent_seeds == 2
    assert summary.test_loss_mean == 1.5
    assert summary.test_loss_sd == pytest.approx(2 ** 0.5)
    assert sorted(by_seed.test_loss_mean) == [0.5, 2.5]


def test_model_widths_match_archived_main_benchmark():
    archive = pd.read_csv(
        Path(__file__).resolve().parents[1] /
        "legacy/exps/results_symbolic/slurm_102076/results_merged.csv"
    )
    protocol = Protocol(device="cpu")
    for name in ("LSTM", "GRU", "minLSTM", "Transformer", "RWKV"):
        archived = archive[(archive.model == name) & (archive.symbols == 4)]
        assert not archived.empty
        model = get_model(name, 4, effective_hidden_size(name, protocol), 4,
                          num_layers=protocol.layers, d_model=protocol.d_model,
                          window_size=100)
        assert sum(parameter.numel() for parameter in model.parameters()) == int(
            archived.iloc[0].model_params
        )
