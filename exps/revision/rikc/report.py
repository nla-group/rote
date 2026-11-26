"""Confidence intervals over strings and matched string pairs."""

from collections import defaultdict
import math

import numpy as np

from ..storage import atomic_csv
from .design import LZW_SCORES


METRICS = ("DL", "exact")
RESAMPLES = 10_000
BOOTSTRAP_SEED = 902680
CONFIDENCE = 0.95


def interval(strata, bounds, *, resamples=RESAMPLES):
    """An equally weighted, stratified interval over the independent units."""
    estimate = float(np.mean([values.mean() for values in strata]))
    if any(len(values) < 2 for values in strata):
        return estimate, None, None, "insufficient_units"
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pooled = np.zeros(resamples)
    for values in strata:
        draws = rng.integers(len(values), size=(resamples, len(values)))
        pooled += values[draws].mean(axis=1) / len(strata)
    tail = (1 - CONFIDENCE) / 2
    low, high = (float(value) for value in np.quantile(pooled, (tail, 1 - tail)))
    if not math.isclose(low, high, abs_tol=1e-12, rel_tol=0):
        return estimate, low, high, "stratified_percentile_bootstrap"
    # All-perfect Exact (or no paired disagreements) makes a bootstrap
    # interval collapse. Report a conservative bounded-mean interval instead.
    lower_bound, upper_bound = bounds
    squared_weights = sum(1 / len(values) for values in strata) / len(strata) ** 2
    radius = (upper_bound - lower_bound) * math.sqrt(
        squared_weights * math.log(2 / (1 - CONFIDENCE)) / 2
    )
    return estimate, max(lower_bound, estimate - radius), min(upper_bound, estimate + radius), "hoeffding"


def contrast_units(stage, records):
    groups = defaultdict(dict)
    for row in records:
        group = groups[row["model"], row["complexity"]]
        key = (row["symbols"], row["seed_index"])
        if key in group:
            raise ValueError(f"Duplicate string outcome for {stage}: {row['model']}, {key}")
        group[key] = row

    def units(label, model, score, rows, reference=None):
        paired = reference is not None
        lzw_pair = label == "higher_minus_lower_score"
        if not rows:
            raise ValueError(f"No independent outcomes for {label} ({model})")
        if paired and rows.keys() != reference.keys():
            raise ValueError(f"Incomplete pairing for {label} ({model})")
        for key, row in sorted(rows.items()):
            other = reference[key] if paired else None
            if lzw_pair:
                attributes = ("pair_id", "predictive_order", "seed_length")
                if any(row[field] != other[field] for field in attributes):
                    raise ValueError(f"LZW pair {key} does not have a common order and ID")
                unit = row["pair_id"]
            else:
                if paired and row["seed_string"] != other["seed_string"]:
                    raise ValueError(f"Architectures did not score the same string at {key}")
                unit = f"n{key[0]}-s{key[1]:03d}"
            values = {metric: row[metric] - (other[metric] if other else 0) for metric in METRICS}
            yield {
                "comparison": label, "model": model, "complexity": score, "symbols": key[0],
                "unit": unit, "unit_type": "matched_string_pair" if lzw_pair else "string",
                "n_rollouts": 2 if paired else 1, **values,
            }

    for (model, score), rows in sorted(groups.items()):
        yield from units("model_mean", model, score, rows)
    if stage == "matched_order":
        low, high = LZW_SCORES
        for model in sorted({model for model, _ in groups}):
            yield from units("higher_minus_lower_score", model, f"{high}-{low}",
                             groups[model, high], groups[model, low])
    if stage == "common_budget":
        yield from units("architecture_difference", "GRU - LinearAttention", 90,
                         groups["GRU", 90], groups["LinearAttention", 90])


def write_intervals(path, stage, records):
    units = list(contrast_units(stage, records))
    if not units:
        raise ValueError("Confidence intervals require recorded string outcomes")
    groups = defaultdict(lambda: defaultdict(list))
    for row in units:
        group = (row["comparison"], row["model"], row["complexity"], row["unit_type"])
        groups[group][row["symbols"]].append(row)
    summaries = []
    for (label, model, score, unit_type), by_n in groups.items():
        strata = list(by_n.values())
        subsets = [(stratum, [values]) for stratum, values in sorted(by_n.items())]
        if len(strata) > 1:
            subsets.append(("pooled_equal_alphabet", strata))
        for scope, subset in subsets:
            for metric in METRICS:
                values = [np.asarray([row[metric] for row in rows], dtype=float) for rows in subset]
                bounds = (0, 1) if label == "model_mean" else (-1, 1)
                estimate, low, high, method = interval(values, bounds)
                summaries.append({
                    "comparison": label, "model": model, "complexity": score, "symbols": scope,
                    "metric": metric, "estimate": estimate, "ci_low": low, "ci_high": high,
                    "confidence": CONFIDENCE, "method": method, "unit_type": unit_type,
                    "n_units": sum(map(len, subset)), "n_alphabets": len(subset),
                    "n_rollouts": sum(row["n_rollouts"] for rows in subset for row in rows),
                })
    atomic_csv(path / "comparison_units.csv", units)
    atomic_csv(path / "confidence_intervals.csv", summaries)
