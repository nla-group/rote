"""Independent reviewer experiments; run with ``python -m exps.revision.run_revision``."""

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys

import pandas as pd
import torch

from .baselines import evaluate_baseline
from .common import Protocol, fit_one, setup, stage_config
from .seeds import (ARCHIVE, archived_seeds, capacity_seeds, controlled_seeds,
                    phase_statistics, predictive_order)
from .storage import StageStore
from .text_bridge import text_excerpts


STAGES = ("baseline", "fixed_length", "context", "exposure", "convergence",
          "sensitivity", "tuned", "capacity", "text_bridge")
MAIN_MODELS = ("LSTM", "GRU", "minGRU", "minLSTM", "Transformer",
               "LinearAttention", "Performer", "RWKV")
ATTENTION_MODELS = frozenset(("Transformer", "Performer", "LinearAttention"))
TUNING_RATES = (0.0001, 0.0003, 0.001)


def source_fingerprint():
    """Bind a resumable run to the actual benchmark and revision source bytes."""
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "exps/revision").glob("*.py"))
    sources += [root / f"rote_benchmark/{name}.py" for name in
                ("benchmark", "generation", "metrics", "models")]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_versions():
    packages = ("torch", "numpy", "pandas", "minGRU-pytorch",
                "linear-attention-transformer", "performer-pytorch")
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def selected_context_seeds(frame):
    choices = (
        (2, 70, 0), (2, 90, 0),
        (4, 70, 0), (4, 70, 1), (4, 90, 0), (4, 90, 1),
        (8, 70, 0), (8, 90, 0),
    )
    indexed = frame.set_index(["symbols", "complexity", "seed_index"])
    return [dict(indexed.loc[key], symbols=key[0], complexity=key[1], seed_index=key[2])
            for key in choices]


def selected_exposure_seeds(frame):
    indexed = frame.set_index(["symbols", "complexity", "seed_index"])
    choices = ((n, complexity, 0) for n in (4, 8) for complexity in (70, 90))
    return [dict(indexed.loc[key], symbols=key[0], complexity=key[1], seed_index=key[2])
            for key in choices]


def selected_convergence_seeds(frame):
    return [row._asdict() for row in frame.itertuples(index=False)
            if row.symbols in (4, 8) and row.complexity == 90]


def select_tuning(path):
    frame = pd.read_csv(path)
    needed = {"model", "symbols", "complexity", "seed_index", "run",
              "learning_rate", "warmup_epochs", "valid_loss"}
    if not needed <= set(frame):
        raise ValueError(f"Tuning results lack {sorted(needed - set(frame))}")
    if "stage" in frame and set(frame.stage) != {"sensitivity"}:
        raise ValueError("Tuning input must contain sensitivity-stage results only")
    actual = [
        (row.model, int(row.symbols), int(row.complexity), int(row.seed_index),
         int(row.run), float(row.learning_rate), int(row.warmup_epochs))
        for row in frame.itertuples(index=False)
    ]
    expected = {
        (model, symbols, 90, 0, run, rate, warmup)
        for model in MAIN_MODELS
        for symbols in (4, 8)
        for run in (0, 1)
        for rate in TUNING_RATES
        for warmup in ((0, 10) if model in ATTENTION_MODELS else (0,))
    }
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Tuning results are incomplete, duplicated, or off-design")
    if not all(math.isfinite(float(value)) for value in frame.valid_loss):
        raise ValueError("Tuning validation losses must be finite")
    grouped = (frame.groupby(["model", "learning_rate", "warmup_epochs"], as_index=False)
               .agg(valid_loss=("valid_loss", "mean"), trials=("valid_loss", "size")))
    selected = grouped.sort_values(["valid_loss", "warmup_epochs", "learning_rate"])
    selected = selected.drop_duplicates("model")
    if set(selected.model) != set(MAIN_MODELS):
        raise ValueError("Tuning results do not cover every architecture")
    return {row.model: {"learning_rate": float(row.learning_rate),
                        "warmup_epochs": int(row.warmup_epochs)}
            for row in selected.itertuples(index=False)}


def tasks(stage, archived, *, smoke=False, controlled_count=6, tuning=None):
    if stage == "text_bridge":
        rows = [row._asdict() for row in text_excerpts(count=1 if smoke else 3).itertuples(index=False)]
        for row in rows:
            for model in (("GRU",) if smoke else ("GRU", "LSTM", "Transformer", "LinearAttention")):
                for run in range(1 if smoke else 2):
                    yield {"model": model, "row": row, "run": run}
        return
    if stage == "capacity":
        generated = capacity_seeds(count=1 if smoke else 2)
        rows = [row._asdict() for row in generated.itertuples(index=False)]
        if smoke:
            rows = rows[:1]
        for row in rows:
            for model in (("GRU",) if smoke else ("GRU", "Transformer", "LinearAttention")):
                for scale in (("small",) if smoke else ("small", "large")):
                    for run in range(1 if smoke else 2):
                        yield {"model": model, "row": row, "scale": scale,
                               "cycles": 8, "run": run}
        return
    if stage == "baseline":
        rows = [row._asdict() for row in archived.itertuples(index=False)]
        if smoke:
            rows = rows[:1]
        for row in rows:
            order = predictive_order(row["seed_string"])
            for window in dict.fromkeys((max(1, order - 1), order, order + 2, 100)):
                for model in ("Lookup", "NGram"):
                    yield {"model": model, "row": row, "window": window}
        return
    if stage == "fixed_length":
        generated = controlled_seeds(count=1 if smoke else controlled_count)
        rows = [row._asdict() for row in generated.itertuples(index=False)]
        for row in rows:
            for model in (("GRU",) if smoke else ("GRU", "LSTM", "LinearAttention", "Transformer")):
                for run in range(1 if smoke else 2):
                    yield {"model": model, "row": row, "run": run}
        return
    if stage == "context":
        rows = selected_context_seeds(archived)
        if smoke:
            rows = rows[:1]
        for row in rows:
            order = predictive_order(row["seed_string"])
            for window in dict.fromkeys((max(1, order - 1), order, order + 2, 100)):
                for model in (("GRU",) if smoke else ("GRU", "LinearAttention", "Transformer")):
                    for run in range(1 if smoke else 2):
                        yield {"model": model, "row": row, "window": window, "run": run,
                               "window_relation": ("below" if window < order else
                                                   "at" if window == order else "above")}
        return
    if stage == "exposure":
        rows = selected_exposure_seeds(archived)
        if smoke:
            rows = rows[:1]
        for row in rows:
            for cycles in (4, 8, 16):
                for model in (("GRU",) if smoke else ("GRU", "LinearAttention", "Transformer")):
                    for run in range(1 if smoke else 2):
                        yield {"model": model, "row": row, "cycles": cycles, "run": run}
        return
    if stage in ("convergence", "sensitivity", "tuned"):
        rows = selected_convergence_seeds(archived)
        if smoke:
            rows = rows[:1]
        elif stage == "sensitivity":
            rows = [row for row in rows if row["seed_index"] == 0]
        elif stage == "tuned":
            rows = [row for row in rows if row["seed_index"] == 1]
        for row in rows:
            for model in (("GRU",) if smoke else MAIN_MODELS):
                for run in range(1 if smoke else 2):
                    if stage == "tuned":
                        choice = tuning[model]
                        yield {"model": model, "row": row, "run": run,
                               "rate": choice["learning_rate"],
                               "warmup_epochs": choice["warmup_epochs"]}
                    elif stage == "sensitivity":
                        rates = (0.0003,) if smoke else TUNING_RATES
                        warmups = (0, 10) if model in ATTENTION_MODELS else (0,)
                        for rate in rates:
                            for warmup in ((0,) if smoke else warmups):
                                yield {"model": model, "row": row, "run": run,
                                       "rate": rate, "warmup_epochs": warmup}
                    else:
                        yield {"model": model, "row": row, "run": run,
                               "rate": 0.0003, "warmup_epochs": 0}
        return
    raise ValueError(f"Unknown stage: {stage}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="One-epoch CPU-compatible path through every stage")
    parser.add_argument("--controlled-seeds", type=int, default=6)
    parser.add_argument("--tuning-results", type=Path,
                        help="Required for tuned: sensitivity/results.csv from the same run")
    args = parser.parse_args(argv)
    if args.controlled_seeds < 1:
        parser.error("--controlled-seeds must be positive")
    protocol = Protocol(device=args.device)
    if args.smoke:
        protocol = replace(protocol, max_epochs=1, patience=1)
    setup(args.device)
    archived = archived_seeds(args.archive)
    if args.stage == "tuned" and not args.smoke and args.tuning_results is None:
        parser.error("--tuning-results is required for the tuned stage")
    tuning = (select_tuning(args.tuning_results) if args.stage == "tuned" and not args.smoke
              else {"GRU": {"learning_rate": protocol.learning_rate,
                             "warmup_epochs": 0}} if args.stage == "tuned" else None)
    plan = list(tasks(args.stage, archived, smoke=args.smoke,
                      controlled_count=args.controlled_seeds, tuning=tuning))
    if not plan:
        raise RuntimeError("The selected stage has no evaluations")
    config = stage_config(
        args.stage, protocol, archive=str(args.archive.resolve()),
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        planned_evaluations=len(plan), smoke=args.smoke,
        controlled_seed_count=args.controlled_seeds,
        tuning_selection=tuning,
        source_sha256=source_fingerprint(),
        python=sys.version.split()[0], packages=package_versions(),
        torch=str(torch.__version__), cuda=str(torch.version.cuda),
        cudnn=torch.backends.cudnn.version(),
    )
    seeds = [item["row"] for item in plan]
    seeds = list({(row["symbols"], row["complexity"], row["seed_index"]): row
                  for row in seeds}.values())
    store = StageStore(args.output_dir, config, len(plan), seeds)
    try:
        if args.stage in ("fixed_length", "capacity"):
            pd.DataFrame([item["row"] for item in plan]).drop_duplicates(
                ["symbols", "complexity", "seed_index"]
            ).to_csv(args.output_dir / f"{args.stage}_seeds.csv", index=False)
        if args.stage == "baseline":
            diagnostics = []
            for row in archived.itertuples(index=False):
                order = predictive_order(row.seed_string)
                diagnostics.append({**row._asdict(), "seed_length": len(row.seed_string),
                                    "predictive_order": order,
                                    "alpha_100": phase_statistics(row.seed_string, 100)["alpha"]})
            pd.DataFrame(diagnostics).to_csv(args.output_dir / "archived_seed_diagnostics.csv", index=False)
        for index, item in enumerate(plan, start=1):
            case = {key: value for key, value in item.items() if key != "row"}
            case["seed_string"] = item["row"]["seed_string"]
            case_id = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:20]
            if case_id in store.completed:
                continue
            if args.stage == "baseline":
                result = evaluate_baseline(item["row"], window=item["window"],
                                           kind=item["model"])
            else:
                sequence_length = (item["cycles"] * len(item["row"]["seed_string"])
                                   + protocol.horizon) if args.stage in ("exposure", "capacity") else 3500
                case_protocol = protocol
                if args.stage == "capacity":
                    small = item["scale"] == "small"
                    case_protocol = replace(protocol, units=64 if small else 128,
                                            d_model=64 if small else 256)
                result = fit_one(
                    item["model"], item["row"],
                    window=item.get("window", 100), sequence_length=sequence_length,
                    run=item["run"], protocol=case_protocol,
                    learning_rate=item.get("rate"),
                    warmup_epochs=item.get("warmup_epochs", 0),
                    periodic_diagnostics=args.stage != "text_bridge", track=args.stage,
                )
                if args.stage == "context":
                    result["window_relation"] = item["window_relation"]
                if args.stage == "exposure":
                    result["complete_cycles_before_rollout"] = item["cycles"]
                if args.stage == "capacity":
                    result["complete_cycles_before_rollout"] = item["cycles"]
                    result["model_scale"] = item["scale"]
                if args.stage == "text_bridge":
                    result["source_type"] = "nonperiodic_natural_text"
                    result["source_offset"] = item["row"]["source_offset"]
                    result["source_sha256"] = item["row"]["source_sha256"]
            result["case_id"] = case_id
            result["stage"] = args.stage
            store.append(result)
            print(f"{args.stage}: {index}/{len(plan)} model={item['model']} "
                  f"n={item['row']['symbols']} c={item['row']['complexity']} "
                  f"DL={result['DL']:.4f}", flush=True)
        store.finish()
    except BaseException as error:
        store.fail(error)
        raise
    print(f"Complete: {store.path} ({len(store.completed)} rows)")


if __name__ == "__main__":
    main()
