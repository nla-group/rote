"""Aggregate revision runs with the symbolic seed as the sampling unit."""

import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

from .storage import atomic_csv, atomic_json


METRICS = ("test_accuracy", "test_loss", "train_loss", "train_accuracy",
           "valid_loss", "valid_accuracy", "reached_stopping_loss", "DL",
           "exact", "epochs", "train_time", "phase_exposure", "alpha_w",
           "bayes_error_w", "conditional_entropy_w", "unseen_test_context_rate")
GROUPS = ("stage", "model", "model_scale", "symbols", "complexity",
          "window_size", "complete_cycles_before_rollout", "learning_rate",
          "warmup_epochs")


def summarize(root):
    root = Path(root)
    rows = []
    status = {}
    for stage_dir in sorted(root.iterdir()):
        if not stage_dir.is_dir() or not (stage_dir / "manifest.json").exists():
            continue
        with (stage_dir / "manifest.json").open(encoding="utf-8") as stream:
            status[stage_dir.name] = json.load(stream)
        path = stage_dir / "results.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                rows.extend(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No revision results found under {root}")
    grouped = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUPS)
        grouped.setdefault(key, []).append(row)
    summary = []
    seed_summary = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(GROUPS, key))
        item["evaluations"] = len(group)
        by_seed = {}
        for row in group:
            seed_key = (row["symbols"], row["complexity"], row["seed_index"],
                        row["seed_string"])
            by_seed.setdefault(seed_key, []).append(row)
        item["independent_seeds"] = len(by_seed)
        seed_items = []
        for seed_key, seed_rows in sorted(by_seed.items()):
            seed_item = dict(zip(GROUPS, key))
            seed_item["seed_index"] = seed_key[2]
            seed_item["seed_sha256"] = hashlib.sha256(
                seed_key[3].encode("utf-8")
            ).hexdigest()
            seed_item["evaluations"] = len(seed_rows)
            for metric in METRICS:
                values = [float(row[metric]) for row in seed_rows
                          if row.get(metric) not in (None, "")]
                if values:
                    seed_item[f"{metric}_mean"] = statistics.mean(values)
                    seed_item[f"{metric}_run_sd"] = (
                        statistics.stdev(values) if len(values) > 1 else ""
                    )
            seed_summary.append(seed_item)
            seed_items.append(seed_item)
        for metric in METRICS:
            values = [seed_item[f"{metric}_mean"] for seed_item in seed_items
                      if f"{metric}_mean" in seed_item]
            if values:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sd"] = (
                    statistics.stdev(values) if len(values) > 1 else ""
                )
                item[f"{metric}_seed_count"] = len(values)
        summary.append(item)
    atomic_csv(root / "summary_by_configuration.csv", summary)
    atomic_csv(root / "summary_by_seed.csv", seed_summary)
    atomic_json(root / "run_status.json", status)
    print(f"Summarized {len(rows)} evaluations across {len(status)} stages in {root}")
    for stage, manifest in sorted(status.items()):
        print(f"  {stage}: {manifest['status']} ({manifest['completed']}/{manifest['total']})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python summarize.py OUTPUT_ROOT")
    summarize(sys.argv[1])
