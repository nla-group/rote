"""Extend the archived learning-rate grid, then evaluate frozen recipes."""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import torch

from rote_benchmark.generation import _shortest_period, lzw_string_generator
from ..common import Protocol, fit_one, setup
from ..run_revision import (ATTENTION_MODELS, MAIN_MODELS, TUNING_RATES,
                            package_versions, source_fingerprint, tasks)
from ..seeds import ARCHIVE, archived_seeds, phase_statistics, predictive_order
from ..storage import StageStore, atomic_csv, atomic_json
from .report import write_confirmation_report


REVISION_DIR = Path(__file__).resolve().parents[1]
MATCHED_ARCHIVE = (REVISION_DIR.parents[1] / "legacy/exps/results_symbolic_matched_size"
                   / "slurm_104393/results_merged.csv")
DEFAULT_RUN = REVISION_DIR / "results/slurm_118058"
EXTRA_RATES = (0.003, 0.01)
ALPHABETS = (4, 8)
RUNS = (0, 1)
CONFIRMATION_SEED = 918431
CONFIRMATION_COUNT = 3


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_fields(case):
    seed = case["row"]
    return {"model": case["model"], "symbols": seed["symbols"],
            "complexity": seed["complexity"], "seed_index": seed["seed_index"],
            "seed_string": seed["seed_string"], "run": case["run"],
            "learning_rate": case["rate"], "warmup_epochs": case["warmup_epochs"]}


def trial_key(row):
    return (row["model"], int(row["symbols"]), int(row["complexity"]),
            int(row["seed_index"]), row["seed_string"], int(row["run"]),
            float(row["learning_rate"]), int(row["warmup_epochs"]))


def load_archived_grid(revision_run, archived):
    directory = revision_run / "sensitivity"
    cfg = json_file(directory / "config.json")
    manifest = json_file(directory / "manifest.json")
    rows = csv_rows(directory / "results.csv")
    original_cases = list(tasks("sensitivity", archived))
    count = len(original_cases)
    expected = {trial_key(fields): fields for fields in map(case_fields, original_cases)}
    if (cfg["stage"] != "sensitivity" or cfg["smoke"]
            or cfg["planned_evaluations"] != count or manifest["status"] != "complete"
            or manifest["completed"] != count or manifest["total"] != count
            or len(rows) != count or {trial_key(row) for row in rows} != set(expected)
            or any(row["stage"] != "sensitivity" for row in rows)):
        raise ValueError("Archived sensitivity results must match the complete original grid")
    if (cfg["archive_sha256"] != checksum(ARCHIVE)
            or cfg["protocol"] != asdict(Protocol())
            or cfg["source_sha256"] != source_fingerprint()):
        raise ValueError("Archived sensitivity data, protocol, or training source differs")
    losses = [float(row["valid_loss"]) for row in rows]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("Archived sensitivity validation losses must be finite")
    # Only validation loss enters recipe selection, including for imported rows.
    return [{**expected[trial_key(row)], "valid_loss": loss, "status": "ok", "origin": "archive",
             "epochs": int(row["epochs"]), "train_time": float(row["train_time"])}
            for row, loss in zip(rows, losses)], cfg


def rotation_key(word):
    period = _shortest_period(word)
    return min(period[i:] + period[:i] for i in range(len(period)))


def confirmation_seeds(previous):
    blocked = {rotation_key(row["seed_string"]) for row in previous}
    seeds = []
    for n in ALPHABETS:
        selected = 0
        for attempt in range(1000):
            rng_seed = CONFIRMATION_SEED + 10000 * n + attempt
            word, score = lzw_string_generator(n, 90, random_state=rng_seed)
            if score != 90 or _shortest_period(word) != word or len(set(word)) != n:
                continue
            key = rotation_key(word)
            if key in blocked or phase_statistics(word, 100)["alpha"] != 0:
                continue
            blocked.add(key)
            seeds.append({"symbols": n, "complexity": score, "seed_index": selected,
                          "seed_string": word, "generator_seed": rng_seed,
                          "seed_length": len(word), "predictive_order": predictive_order(word)})
            selected += 1
            if selected == CONFIRMATION_COUNT:
                break
        if selected != CONFIRMATION_COUNT:
            raise RuntimeError(f"Could not draw {CONFIRMATION_COUNT} fresh strings for n={n}")
    return seeds


def search_cases(seeds, models, runs, rates):
    return [
        {"model": model, "row": seed, "run": run, "rate": rate, "warmup_epochs": warmup}
        for seed in seeds for model in models for run in runs for rate in rates
        for warmup in ((0, 10) if model in ATTENTION_MODELS else (0,))
    ]


def choose_recipes(rows, models, rates, runs_per_recipe):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["learning_rate"], row["warmup_epochs"]].append(row)
    expected = {(model, rate, warmup) for model in models for rate in rates
                for warmup in ((0, 10) if model in ATTENTION_MODELS else (0,))}
    if set(grouped) != expected or any(len(group) != runs_per_recipe for group in grouped.values()):
        raise ValueError("Combined tuning grid is incomplete or off-design")
    if len({trial_key(row) for row in rows}) != len(rows):
        raise ValueError("Combined tuning grid contains duplicate trials")
    selected = {}
    summary = []
    for (model, rate, warmup), group in sorted(grouped.items()):
        valid = [row["valid_loss"] for row in group
                 if row["status"] == "ok" and math.isfinite(row["valid_loss"])]
        score = mean(valid) if len(valid) == runs_per_recipe else None
        entry = {"model": model, "learning_rate": rate, "warmup_epochs": warmup,
                 "valid_fits": len(valid), "planned_fits": runs_per_recipe,
                 "mean_valid_loss": score, "at_upper_bound": rate == max(rates)}
        summary.append(entry)
        if score is None:
            continue
        previous = selected.get(model)
        key = (score, warmup, rate)
        if previous is None or key < (previous["mean_valid_loss"], previous["warmup_epochs"],
                                      previous["learning_rate"]):
            selected[model] = entry
    if set(selected) != set(models):
        raise ValueError("At least one model has no recipe with complete finite validation results")
    for entry in summary:
        entry["selected"] = entry is selected[entry["model"]]
    return selected, summary


def confirmation_cases(seeds, models, runs, selected, shared_rate):
    cases = []
    for seed in seeds:
        for model in models:
            best = selected[model]
            shared = (shared_rate, 0)
            chosen = (best["learning_rate"], best["warmup_epochs"])
            choices = [("both", *shared)] if shared == chosen else [
                ("shared", *shared), ("selected", *chosen)]
            for run in runs:
                for label, rate, warmup in choices:
                    cases.append({"model": model, "row": seed, "run": run, "rate": rate,
                                  "warmup_epochs": warmup, "recipe": label})
    return cases


def evaluate(case, stage, protocol, length, window):
    fields = {"stage": stage, **case_fields(case)}
    searching = stage == "search"
    if searching:
        fields["origin"] = "extension"
    else:
        fields["recipe"] = case["recipe"]
    try:
        result = fit_one(case["model"], case["row"], protocol=protocol,
                         window=window, sequence_length=length, run=case["run"],
                         learning_rate=case["rate"], warmup_epochs=case["warmup_epochs"],
                         track=stage)
    except ValueError as error:
        if not str(error).startswith("Invalid training loss:"):
            raise
        return {**fields, "status": "numerical_failure", "failure": str(error),
                "valid_loss": None}
    if searching:
        loss = result["valid_loss"]
        if math.isfinite(loss):
            return {**fields, "status": "ok", "valid_loss": loss,
                    "epochs": result["epochs"], "train_time": result["train_time"]}
        return {**fields, "status": "numerical_failure", "failure": "Nonfinite validation loss",
                "valid_loss": None}
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        return {**fields, "status": "numerical_failure", "failure": "Nonfinite fit metrics",
                "valid_loss": None}
    return {**result, **fields, "status": "ok"}


def run_stage(root, stage, cases, seeds, config, protocol, length, window):
    store = StageStore(root / stage, {**config, "stage": stage}, len(cases), seeds)
    try:
        for index, case in enumerate(cases, 1):
            case_id = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]
            if case_id in store.completed:
                continue
            record = evaluate(case, stage, protocol, length, window)
            store.append({**record, "case_id": case_id})
            print(f"{stage}: {index}/{len(cases)} {case['model']} "
                  f"n={case['row']['symbols']} seed={case['row']['seed_index']} "
                  f"lr={case['rate']:g} {case.get('recipe', '')} {record['status']}", flush=True)
        store.finish()
    except BaseException as error:
        store.fail(error)
        raise
    return store.records


def runtime(device):
    return {"python": sys.version.split()[0], "packages": package_versions(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None}


def require_matching_runtime(archived_config, actual):
    differences = [key for key in ("packages", "torch", "cuda", "cudnn")
                   if archived_config[key] != actual[key]]
    if archived_config["python"].split(".")[:2] != actual["python"].split(".")[:2]:
        differences.append("python")
    if differences:
        raise ValueError(f"Runtime differs from archived sensitivity in {', '.join(differences)}. "
                         "Use the recorded revision environment before pooling validation losses.")


def save_once(path, data):
    if path.exists():
        if json_file(path) != data:
            raise ValueError(f"Refusing to resume with changed inputs or recipes: {path}")
    else:
        atomic_json(path, data)


def execute(args, old_rows, archive_config, tune_seeds, fresh, input_files):
    length, window = (256, 16) if args.smoke else (3500, 100)
    protocol = Protocol(device=args.device)
    if args.smoke:
        protocol = replace(protocol, layers=1, units=16, max_epochs=1, patience=1, horizon=20)
    setup(args.device)
    environment = runtime(args.device)
    if not args.smoke:
        require_matching_runtime(archive_config, environment)
        print(f"CUDA preflight passed. GPU: {environment['gpu']}", flush=True)
    models = ("GRU",) if args.smoke else MAIN_MODELS
    runs = (0,) if args.smoke else RUNS
    new_rates = (protocol.learning_rate, *EXTRA_RATES) if args.smoke else EXTRA_RATES
    rates = new_rates if args.smoke else (*TUNING_RATES, *EXTRA_RATES)
    if args.smoke:
        old_rows, tune_seeds, fresh = [], tune_seeds[:1], fresh[:1]

    root = args.output_dir
    config = {"protocol": asdict(protocol), "smoke": args.smoke, "window": window,
              "sequence_length": length, "runtime": environment, "rates": list(rates),
              "new_rates": list(new_rates), "source_sha256": source_fingerprint(),
              "followup_source": {path.name: checksum(path)
                                  for path in sorted(Path(__file__).parent.glob("*.py"))},
              "inputs": {str(path): checksum(path) for path in input_files}}
    save_once(root / "followup_config.json", config)
    save_once(root / "confirmation_seeds.json", fresh)
    atomic_csv(root / "confirmation_seeds.csv", fresh)
    status = {"status": "running", "archived_search_fits": len(old_rows)}
    atomic_json(root / "run_status.json", status)
    try:
        cases = search_cases(tune_seeds, models, runs, new_rates)
        new_rows = run_stage(root, "search", cases, tune_seeds, config, protocol, length, window)
        all_trials = old_rows + new_rows
        atomic_csv(root / "tuning_trials.csv", all_trials)
        selected, grid = choose_recipes(all_trials, models, rates, len(tune_seeds) * len(runs))
        atomic_csv(root / "tuning_grid.csv", grid)
        save_once(root / "selected_recipes.json", selected)
        cases = confirmation_cases(fresh, models, runs, selected, protocol.learning_rate)
        confirm = run_stage(root, "confirmation", cases, fresh,
                            {**config, "selected_recipes": selected}, protocol, length, window)
        pair_count = len(fresh) * len(runs) * len(models)
        write_confirmation_report(root, confirm, pair_count, protocol.seed)
        status.update(status="complete", new_search_fits=len(new_rows),
                      confirmation_fits=len(confirm),
                      confirmation_pairs=pair_count,
                      numerical_failures={"search": sum(row["status"] != "ok" for row in new_rows),
                                          "confirmation": sum(row["status"] != "ok" for row in confirm)},
                      selected_at_upper_bound=[model for model, recipe in selected.items()
                                               if recipe["at_upper_bound"]])
        print(f"Tuning follow-up complete: {root}", flush=True)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(root / "run_status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Small local workflow check")
    mode.add_argument("--plan", action="store_true", help="Validate archives and print the plan")
    args = parser.parse_args()
    if args.output_dir is None and not args.plan:
        parser.error("--output-dir is required for training")
    if args.device != "cuda" and not (args.plan or args.smoke):
        parser.error("The archived grid used CUDA. CPU is supported for --smoke or --plan")

    archived = archived_seeds()
    revision_run = args.revision_run.resolve()
    old_rows, archive_config = load_archived_grid(revision_run, archived)
    previous_files = [MATCHED_ARCHIVE, revision_run / "fixed_length/seeds.csv",
                      revision_run / "capacity/seeds.csv"]
    previous = archived.to_dict("records") + [
        row for path in previous_files for row in csv_rows(path)]
    fresh = confirmation_seeds(previous)
    tune_seeds = [row for row in archived.to_dict("records")
                  if row["symbols"] in ALPHABETS and row["complexity"] == 90
                  and row["seed_index"] == 0]
    planned = search_cases(tune_seeds, MAIN_MODELS, RUNS, EXTRA_RATES)
    print(f"Validated {len(old_rows)} archived trials.", flush=True)
    if args.smoke:
        print("Smoke check: 3 tuning fits and up to 2 confirmation fits.", flush=True)
    else:
        print(f"Plan: {len(planned)} new tuning fits, "
              f"up to {2 * len(fresh) * len(RUNS) * len(MAIN_MODELS)} confirmation fits.", flush=True)
    if args.plan:
        for row in fresh:
            print(f"n={row['symbols']} seed={row['seed_index']} "
                  f"generator={row['generator_seed']} length={row['seed_length']}")
        return

    args.output_dir = args.output_dir.resolve()
    if args.output_dir == revision_run or revision_run in args.output_dir.parents:
        parser.error("Use a new output directory outside the previous revision run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_files = [ARCHIVE, *previous_files, *(revision_run / "sensitivity" / name
                   for name in ("config.json", "manifest.json", "results.csv"))]
    with (args.output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another follow-up process is using this output directory")
        execute(args, old_rows, archive_config, tune_seeds, fresh, input_files)


if __name__ == "__main__":
    main()
