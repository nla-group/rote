"""Regression tests for validation-only selection and paired confirmation."""

import pytest

from exps.revision.tuning_followup.report import METRICS, pair_results, summarize_seed
from exps.revision.tuning_followup.runner import choose_recipes, confirmation_cases


def test_selection_requires_all_validation_trials_and_ignores_test_loss():
    rates = (0.0003, 0.003, 0.01)
    rows = [
        {"model": "GRU", "symbols": 4, "complexity": 90, "seed_index": 0,
         "seed_string": "ABCD", "run": run, "warmup_epochs": 0, "learning_rate": rate,
         "valid_loss": loss, "test_loss": test, "status": "ok" if loss is not None else "numerical_failure"}
        for rate, losses, test in zip(rates, [(0.05, 0.07), (0.03, 0.05), (0.001, None)], [0.01, 9, 0])
        for run, loss in enumerate(losses)
    ]
    selected, grid = choose_recipes(rows, ("GRU",), rates, 2)
    assert selected["GRU"]["learning_rate"] == 0.003
    assert selected["GRU"]["mean_valid_loss"] == pytest.approx(0.04)
    assert grid[-1]["mean_valid_loss"] is None
    assert grid[-1]["valid_fits"] == 1
    assert grid[-1]["at_upper_bound"] and not grid[-1]["selected"]
    with pytest.raises(ValueError, match="incomplete"):
        choose_recipes(rows[:-1], ("GRU",), rates, 2)


def test_reused_confirmation_fit_and_missing_pair_metrics():
    seed = {"symbols": 4, "seed_index": 0, "seed_string": "ABCD"}
    selected = {"GRU": {"learning_rate": 0.0003, "warmup_epochs": 0}}
    cases = confirmation_cases([seed], ("GRU",), (0, 1), selected, 0.0003)
    assert [case["recipe"] for case in cases] == ["both", "both"]

    record = {**seed, "model": "GRU", "run": 0, "case_id": "reused", "recipe": "both",
              "random_seed": 3407, "learning_rate": 0.0003, "warmup_epochs": 0,
              "status": "ok", "target_string": "ABCD", **dict.fromkeys(METRICS, 1)}
    shared = {**record, "run": 1, "random_seed": 3408, "case_id": "shared", "recipe": "shared"}
    failed = {**seed, "model": "GRU", "run": 1, "case_id": "failed", "recipe": "selected",
              "learning_rate": 0.003, "warmup_epochs": 0, "status": "numerical_failure"}
    pairs = pair_results([record, shared, failed], expected_count=2, base_seed=3407)
    assert pairs[0]["reused_fit"] and pairs[0]["delta_DL"] == 0
    assert pairs[1]["delta_DL"] is None
    summary = summarize_seed(pairs)
    assert summary["complete_pairs"] == 1
    assert summary["selected_failed"] == 1
    assert summary["shared_completed"] == 2
    assert summary["delta_DL"] == 0
    with pytest.raises(ValueError, match="Missing recipe"):
        pair_results([record, shared], expected_count=2, base_seed=3407)
