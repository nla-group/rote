"""Seed diagnostics and fixed-length, fixed-composition seed construction."""

from collections import Counter, defaultdict
import math
import random
from pathlib import Path

import pandas as pd

from rote_benchmark.generation import (ALPHABET, _shortest_period, lzw_complexity,
                                       lzw_string_generator)


ARCHIVE = (Path(__file__).resolve().parents[2] / "legacy/exps/results_symbolic"
           / "slurm_102076/results_merged.csv")


def archived_seeds(path=ARCHIVE):
    frame = pd.read_csv(path)
    required = {"symbols", "complexity", "seed_index", "seed_string"}
    if not required <= set(frame):
        raise ValueError(f"Archive lacks {sorted(required - set(frame))}")
    frame = frame[sorted(required)].drop_duplicates().sort_values(
        ["symbols", "complexity", "seed_index"]
    ).reset_index(drop=True)
    if frame.duplicated(["symbols", "complexity", "seed_index"]).any():
        raise ValueError("An archived seed index maps to multiple strings")
    for row in frame.itertuples(index=False):
        if len(set(row.seed_string)) != row.symbols:
            raise ValueError("Archived seed has the wrong alphabet size")
        if lzw_complexity(_shortest_period(row.seed_string)) != row.complexity:
            raise ValueError("Archived LZW score does not match the seed")
    return frame


def phase_statistics(seed, window):
    """Cyclic ambiguity, empirical Bayes error, and conditional entropy."""
    if window < 1 or not seed:
        raise ValueError("window and seed must be nonempty")
    counts = defaultdict(Counter)
    size = len(seed)
    for phase in range(size):
        context = "".join(seed[(phase - lag) % size] for lag in range(window - 1, -1, -1))
        counts[context][seed[(phase + 1) % size]] += 1
    ambiguous = sum(sum(freq.values()) for freq in counts.values() if len(freq) > 1)
    bayes_errors = sum(sum(freq.values()) - max(freq.values()) for freq in counts.values())
    entropy = -sum(
        count / size * math.log(count / sum(freq.values()))
        for freq in counts.values() for count in freq.values()
    )
    return {"alpha": ambiguous / size, "bayes_error": bayes_errors / size,
            "conditional_entropy": entropy, "cyclic_contexts": len(counts)}


def predictive_order(seed):
    for window in range(1, len(seed) + 1):
        if phase_statistics(seed, window)["alpha"] == 0:
            return window
    raise AssertionError("A finite periodic seed must have finite predictive order")


def fixed_length_seed(n, length, target, seed, used=None, max_attempts=24, steps=5000):
    """Search balanced symbol permutations for an exact LZW score.

    All generated words have the same length and symbol histogram for a
    given (n, length). The search is deterministic for its integer seed.
    A bounded failure is explicit; no unmeasured candidate is accepted.
    """
    if n < 2 or n > 52 or length < n or target < n:
        raise ValueError("Invalid n, length, or target")
    used = used or set()
    rng = random.Random(seed)
    symbols = ALPHABET[:n]
    composition = [symbols[i % n] for i in range(length)]
    best_score = None
    for attempt in range(max_attempts):
        word = sorted(composition) if attempt % 2 == 0 else composition.copy()
        if attempt % 2:
            rng.shuffle(word)
        current = lzw_complexity("".join(word))
        for step in range(steps):
            candidate = "".join(word)
            if current == target and candidate not in used and _shortest_period(candidate) == candidate:
                return candidate, {"attempt": attempt, "steps": step}
            i, j = rng.sample(range(length), 2)
            if word[i] == word[j]:
                continue
            word[i], word[j] = word[j], word[i]
            score = lzw_complexity("".join(word))
            if (abs(score - target) < abs(current - target)
                    or (abs(score - target) == abs(current - target) and rng.random() < 0.75)
                    or rng.random() < 0.015):
                current = score
            else:
                word[i], word[j] = word[j], word[i]
        best_score = current
    raise RuntimeError(
        f"Could not generate n={n}, length={length}, LZW={target}; "
        f"last score={best_score}. Adjust the predeclared design, do not silently drop a cell."
    )


def controlled_seeds(count=6, base_seed=3407):
    """Four balanced cells: two alphabets, two LZW targets, fixed m per n."""
    rows = []
    for n, length in ((4, 196), (8, 130)):
        used = set()
        for target in (70, 90):
            for index in range(count):
                word, search = fixed_length_seed(
                    n, length, target,
                    base_seed + n * 100000 + target * 1000 + index,
                    used=used,
                )
                used.add(word)
                order = predictive_order(word)
                rows.append({
                    "symbols": n, "complexity": target, "seed_index": index,
                    "seed_string": word, "length": length,
                    "predictive_order": order,
                    "alpha_100": phase_statistics(word, 100)["alpha"],
                    "search_attempt": search["attempt"],
                    "search_steps": search["steps"],
                })
    result = pd.DataFrame(rows)
    if len(result) != 4 * count or any(result.groupby("symbols").length.nunique() != 1):
        raise AssertionError("Fixed-length design is incomplete")
    return result


def capacity_seeds(count=2, base_seed=8241):
    """Longer-period stress-test seeds, retaining only exact measured scores."""
    rows = []
    for n in (4, 8):
        for target in (90, 150, 250):
            used = set()
            for index in range(count):
                for attempt in range(100):
                    candidate, measured = lzw_string_generator(
                        n, target, random_state=base_seed + n * 100000
                        + target * 1000 + index * 100 + attempt,
                    )
                    if measured == target and candidate not in used:
                        break
                else:
                    raise RuntimeError(f"No exact capacity seed for n={n}, c={target}")
                used.add(candidate)
                rows.append({"symbols": n, "complexity": target,
                             "seed_index": index, "seed_string": candidate,
                             "length": len(candidate),
                             "predictive_order": predictive_order(candidate),
                             "alpha_100": phase_statistics(candidate, 100)["alpha"]})
    return pd.DataFrame(rows)
