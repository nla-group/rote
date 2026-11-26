"""Run the period, matched-order, and common-budget reviewer controls."""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path

from ..common import Protocol, fit_one, setup
from ..run_revision import source_fingerprint
from ..seeds import ARCHIVE, archived_seeds
from ..storage import StageStore, atomic_csv, atomic_json
from ..tuning_followup.runner import checksum, csv_rows, rotation_key, runtime, save_once
from .budget import EPOCHS, PARAMETER_TOLERANCE, TARGET_PARAMS, evaluate_budget, matched_widths
from .design import (
    BUDGET_ALPHABETS, BUDGET_MODELS, HORIZON, LZW_LENGTHS, LZW_MODELS,
    SEQUENCE_LENGTH, WINDOW, budget_strings, order_matched_strings,
)
from .period import evaluate_period
from .report import write_intervals


STAGES = ("period", "matched_order", "common_budget")
REPO_ROOT = Path(__file__).resolve().parents[3]
REVISION_RESULTS = REPO_ROOT / "exps/revision/results"
EXISTING_ARCHIVES = (
    REPO_ROOT / "legacy/exps/results_symbolic_matched_size/slurm_104393/results_merged.csv",
    REVISION_RESULTS / "slurm_118058/fixed_length/seeds.csv",
    REVISION_RESULTS / "slurm_118058/capacity/seeds.csv",
    REVISION_RESULTS / "tuning_followup_slurm_118144/confirmation_seeds.csv",
)


def followup_fingerprint():
    digest = hashlib.sha256(source_fingerprint().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_plan(args, selected, protocol, window):
    original = archived_seeds(ARCHIVE).to_dict("records")
    sources = (ARCHIVE,)
    seeds = {"period": original}
    widths = []
    if any(stage in selected for stage in ("matched_order", "common_budget")):
        sources += EXISTING_ARCHIVES
        previous = original + [row for path in EXISTING_ARCHIVES for row in csv_rows(path)]
        blocked = {rotation_key(row["seed_string"]) for row in previous}
        order_alphabets = (8,) if args.smoke else tuple(LZW_LENGTHS)
        budget_alphabets = (2,) if args.smoke else BUDGET_ALPHABETS
        seeds["matched_order"] = order_matched_strings(
            args.pairs_per_alphabet, order_alphabets, blocked
        )
        seeds["common_budget"] = budget_strings(
            args.strings_per_alphabet, budget_alphabets, blocked
        )
        if "common_budget" in selected:
            widths = matched_widths(BUDGET_MODELS, budget_alphabets, protocol.layers, window)
    if args.smoke:
        seeds["period"] = original[:2]
    width_index = {(row["symbols"], row["model"]): row for row in widths}
    models = {"period": ("PeriodDetection",), "matched_order": LZW_MODELS,
              "common_budget": BUDGET_MODELS}
    cases = {
        stage: [(model, row, width_index.get((row["symbols"], model)))
                for row in seeds[stage] for model in models[stage]]
        for stage in selected
    }
    return seeds, cases, widths, sources


def run_stage(root, stage, cases, seeds, config, protocol, window, length, epochs):
    # Use the original split and horizon for the baseline even in the CPU smoke.
    horizon = protocol.horizon
    if stage == "period":
        window, length, horizon = WINDOW, SEQUENCE_LENGTH, HORIZON
    planned = [
        (f"{stage}:n{row['symbols']}:c{row['complexity']}:s{row['seed_index']}:{model}",
         model, row, width)
        for model, row, width in cases
    ]
    stage_config = {**config, "stage": stage, "stage_window": window,
                    "stage_sequence_length": length, "stage_horizon": horizon}
    store = StageStore(root / stage, stage_config, len(planned), seeds)
    try:
        if not store.completed.issubset({case_id for case_id, *_ in planned}):
            raise ValueError(f"The saved case IDs do not match the registered plan for {stage}")
        for index, (case_id, model, row, width) in enumerate(planned, 1):
            if case_id in store.completed:
                continue
            run_index = row["symbols"] * 10_000 + row["seed_index"]
            if stage == "period":
                result = evaluate_period(row, window=window, sequence_length=length, horizon=horizon)
            elif stage == "matched_order":
                result = fit_one(model, row, window=window, sequence_length=length,
                                 run=run_index, protocol=protocol, track=stage)
                result = {**row, **result, "pair_id": row["pair_id"]}
            else:
                result = evaluate_budget(row, width, protocol, epochs=epochs, window=window,
                                         sequence_length=length, horizon=horizon,
                                         initialization=protocol.seed + 1_000_000 + run_index)
            store.append({**result, "case_id": case_id, "stage": stage})
            print(f"{stage}: {index}/{store.total} {model} n={row['symbols']} "
                  f"c={row['complexity']} DL={result['DL']:.4f}", flush=True)
        if stage == "period":
            atomic_json(store.path / "summary.json", {
                "n_strings": len(store.records),
                "exact_rollouts": sum(row["exact"] for row in store.records),
                "exact_periods": sum(row["estimated_period"] == row["seed_length"]
                                     for row in store.records),
            })
        else:
            if stage == "common_budget":
                counts = {(row["optimizer_updates"], row["input_tokens"]) for row in store.records}
                if len(counts) != 1:
                    raise RuntimeError("Fits in the common-budget stage used different budgets")
                updates, tokens = counts.pop()
                atomic_json(store.path / "budget_counts.json",
                            {"optimizer_updates_per_fit": updates, "input_tokens_per_fit": tokens})
            write_intervals(store.path, stage, store.records)
        store.finish()
    except BaseException as error:
        store.fail(error)
        raise


def execute(args, selected, protocol, window, length, epochs, seeds, cases, widths, sources):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "runtime": runtime(args.device), "source_sha256": followup_fingerprint(),
        "input_sha256": {str(path.relative_to(REPO_ROOT)): checksum(path) for path in sources},
        "stages": list(selected), "protocol": vars(protocol), "window": window,
        "sequence_length": length, "smoke": args.smoke, "fixed_budget_epochs": epochs,
        "parameter_target": TARGET_PARAMS, "parameter_tolerance": PARAMETER_TOLERANCE,
        "pairs_per_alphabet": args.pairs_per_alphabet,
        "strings_per_alphabet": args.strings_per_alphabet, "matched_widths": widths,
    }
    totals = {stage: len(items) for stage, items in cases.items()}
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"Another job is using {root}") from error
        save_once(root / "run_config.json", config)
        if widths:
            atomic_csv(root / "matched_widths.csv", widths)
        status = root / "run_status.json"
        completed_stages = []
        atomic_json(status, {"status": "running", "planned": totals, "completed_stages": []})
        try:
            for stage in selected:
                run_stage(root, stage, cases[stage], seeds[stage], config,
                          protocol, window, length, epochs)
                completed_stages.append(stage)
                atomic_json(status, {"status": "running", "planned": totals,
                                     "completed_stages": completed_stages})
        except BaseException as error:
            atomic_json(status, {"status": "failed", "planned": totals,
                                 "completed_stages": completed_stages,
                                 "error": f"{type(error).__name__}: {error}"})
            raise
        atomic_json(status, {"status": "complete", "planned": totals,
                             "completed_stages": completed_stages})
    print(f"Complete: {root}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pairs-per-alphabet", type=int, default=12)
    parser.add_argument("--strings-per-alphabet", type=int, default=12)
    parser.add_argument("--budget-epochs", type=int, default=EPOCHS)
    parser.add_argument("--plan", action="store_true", help="Generate the design and check widths without fitting")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced integration test")
    args = parser.parse_args(argv)
    if not args.plan and args.output_dir is None:
        parser.error("--output-dir is required unless --plan is used")
    if min(args.pairs_per_alphabet, args.strings_per_alphabet, args.budget_epochs) < 1:
        parser.error("String counts and budget epochs must be positive")
    selected = STAGES if args.stage == "all" else (args.stage,)
    protocol = Protocol(device=args.device)
    window, length, epochs = WINDOW, SEQUENCE_LENGTH, args.budget_epochs
    if args.smoke:
        args.pairs_per_alphabet = args.strings_per_alphabet = 2
        protocol = replace(protocol, batch_size=32, horizon=12, max_epochs=1, patience=1,
                           units=16, d_model=32)
        window, length, epochs = 16, 200, 2
    if not args.plan and selected != ("period",):
        setup(args.device)
    seeds, cases, widths, sources = make_plan(args, selected, protocol, window)
    print(f"Planned evaluations: {json.dumps({stage: len(items) for stage, items in cases.items()})}",
          flush=True)
    if widths:
        error = max(abs(row["parameter_rel_error"]) for row in widths)
        print(f"Maximum parameter deviation: {error:.2%} (target {TARGET_PARAMS:,})", flush=True)
    if not args.plan:
        # The period-only path needs no CUDA context.
        if selected == ("period",):
            args.device = "cpu"
            protocol = replace(protocol, device="cpu")
        execute(args, selected, protocol, window, length, epochs, seeds, cases, widths, sources)


if __name__ == "__main__":
    main()
