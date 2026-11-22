"""Summarize confirmation fits, pairing recipes before aggregating by string."""

from collections import defaultdict
from statistics import mean, stdev

from ..storage import atomic_csv


ARMS = ("shared", "selected")
METRICS = ("exact", "DL", "test_loss", "test_accuracy", "train_time", "memory_mb", "epochs")


def pair_results(records, expected_count, base_seed):
    paired = defaultdict(dict)
    for row in records:
        key = row["model"], row["symbols"], row["seed_index"], row["seed_string"], row["run"]
        arms = ARMS if row["recipe"] == "both" else (row["recipe"],)
        for arm in arms:
            if arm not in ARMS or arm in paired[key]:
                raise ValueError(f"Invalid or duplicate confirmation arm for {key}")
            paired[key][arm] = row

    if len(paired) != expected_count:
        raise ValueError(f"Expected {expected_count} confirmation pairs, got {len(paired)}")
    pairs = []
    for (model, n, seed_index, string, run), arms in sorted(paired.items()):
        if arms.keys() != set(ARMS):
            raise ValueError(f"Missing recipe for {model}, n={n}, seed {seed_index}, run {run}")
        shared, selected = (arms[arm] for arm in ARMS)
        comparable = all(row["status"] == "ok" for row in arms.values())
        if comparable and shared["target_string"] != selected["target_string"]:
            raise ValueError("Paired confirmation fits use different rollout targets")
        pair = {"model": model, "symbols": n, "seed_index": seed_index, "seed_string": string,
                "run": run, "random_seed": base_seed + run,
                "reused_fit": shared["case_id"] == selected["case_id"]}
        for arm, row in arms.items():
            if row["status"] == "ok" and row["random_seed"] != pair["random_seed"]:
                raise ValueError("Confirmation fit used an unexpected initialization seed")
            for field in ("status", "case_id", "learning_rate", "warmup_epochs", *METRICS):
                pair[f"{arm}_{field}"] = row.get(field)
        for metric in METRICS:
            pair[f"delta_{metric}"] = selected[metric] - shared[metric] if comparable else None
        pairs.append(pair)
    return pairs


def observed(rows, field):
    return [row[field] for row in rows if row[field] is not None]


def summarize_seed(pairs):
    row = {"planned_pairs": len(pairs),
           "complete_pairs": len(observed(pairs, "delta_DL")),
           "reused_fits": sum(pair["reused_fit"] for pair in pairs)}
    for arm in ARMS:
        values = observed(pairs, f"{arm}_exact")
        row[f"{arm}_completed"] = len(values)
        row[f"{arm}_failed"] = len(pairs) - len(values)
        row[f"{arm}_exact_count"] = sum(values)
    for metric in METRICS:
        for field in (f"shared_{metric}", f"selected_{metric}", f"delta_{metric}"):
            values = observed(pairs, field)
            row[field] = mean(values) if values else None
    return row


def summarize_model(seeds):
    row = {"strings": len(seeds)}
    for field in ("planned_pairs", "complete_pairs", "reused_fits", *(
        f"{arm}_{suffix}" for arm in ARMS for suffix in ("completed", "failed", "exact_count")
    )):
        row[field] = sum(seed[field] for seed in seeds)
    for metric in METRICS:
        for field in (f"shared_{metric}", f"selected_{metric}", f"delta_{metric}"):
            values = observed(seeds, field)
            row[f"{field}_mean"] = mean(values) if values else None
            row[f"{field}_std"] = stdev(values) if len(values) > 1 else None
            row[f"{field}_strings"] = len(values)
    return row


def write_confirmation_report(root, records, expected_count, base_seed):
    pairs = pair_results(records, expected_count, base_seed)
    seed_groups = defaultdict(list)
    for row in pairs:
        seed_groups[row["model"], row["symbols"], row["seed_index"], row["seed_string"]].append(row)
    seeds = [
        {"model": model, "symbols": n, "seed_index": index, "seed_string": string,
         **summarize_seed(group)}
        for (model, n, index, string), group in seed_groups.items()
    ]
    model_groups = defaultdict(list)
    for row in seeds:
        model_groups[row["model"], "all"].append(row)
        model_groups[row["model"], row["symbols"]].append(row)
    models = [{"model": model, "symbols": n, **summarize_model(group)}
              for (model, n), group in model_groups.items()]
    atomic_csv(root / "confirmation_pairs.csv", pairs)
    atomic_csv(root / "confirmation_by_seed.csv", seeds)
    atomic_csv(root / "summary_by_model.csv", models)
