"""LZW-controlled symbolic seeds for the ROTE benchmark."""

from itertools import product
import random

import numpy as np
import pandas as pd


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def lzw_complexity(sequence: str) -> int:
    """Count LZW output codes with the 52 ASCII letters as initial dictionary."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    if any(symbol not in ALPHABET for symbol in sequence):
        raise ValueError("sequence must contain only ASCII letters")
    dictionary = {symbol: index for index, symbol in enumerate(ALPHABET)}
    next_code = len(dictionary)
    current = ""
    count = 0
    for symbol in sequence:
        candidate = current + symbol
        if candidate in dictionary:
            current = candidate
        else:
            count += 1
            dictionary[candidate] = next_code
            next_code += 1
            current = symbol
    return count + bool(current)


def _shortest_period(sequence: str) -> str:
    # Preserve the reduction used to define seed complexity in the original runs.
    for width in range(1, len(sequence) // 2 + 1):
        position = width
        while position < len(sequence):
            if sequence[position : position + width] != sequence[:width]:
                break
            position += width
        if position >= len(sequence):
            return sequence[:width]
    return sequence


def lzw_string_generator(
    nr_symbols: int,
    target_complexity: int,
    priorise_complexity: bool = True,
    random_state: int = 42,
) -> tuple[str, int]:
    """Generate one symbolic seed at a requested LZW code count.

    The random stream matches the historical generator while avoiding changes
    to the process-wide NumPy random state. ``priorise_complexity`` is retained
    for compatibility with the experimental protocol.
    """
    if isinstance(nr_symbols, bool) or not isinstance(nr_symbols, (int, np.integer)):
        raise TypeError("nr_symbols must be an integer")
    if isinstance(target_complexity, bool) or not isinstance(target_complexity, (int, np.integer)):
        raise TypeError("target_complexity must be an integer")
    if not 1 <= nr_symbols <= len(ALPHABET):
        raise ValueError("nr_symbols must be between 1 and 52")
    if target_complexity < nr_symbols:
        raise ValueError("target_complexity must be at least nr_symbols")
    if nr_symbols == 1:
        if target_complexity != 1:
            raise ValueError("complexity above one requires at least two symbols")
        return "A", 1
    if not isinstance(priorise_complexity, bool):
        raise TypeError("priorise_complexity must be a boolean")

    np_rng = np.random.RandomState(random_state)
    py_rng = random.Random(random_state)
    shuffled = list(ALPHABET[:nr_symbols])
    np_rng.shuffle(shuffled)
    sequence = "".join(shuffled)
    measured = nr_symbols
    while measured < target_complexity:
        sequence += ALPHABET[py_rng.randint(0, nr_symbols - 1)]
        measured = lzw_complexity(_shortest_period(sequence))
    return sequence, measured


def _values(value, name: str) -> list[int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        values = [int(value)]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer or a sequence of integers") from exc
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError(f"{name} must contain at least one integer")
    return [int(x) for x in values]


def lzw_string_seeds(
    symbols=(2, 4, 6, 8),
    complexity=(10, 30, 50, 70, 90),
    iterations: int = 1,
    random_state: int = 42,
    priorise_complexity: bool = True,
) -> pd.DataFrame:
    """Return controlled seeds with the columns used by the ROTE experiments.

    Seeds use ``random_state + index`` in the historical iteration order.
    Impossible alphabet/complexity pairs are skipped.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    alphabet_sizes = _values(symbols, "symbols")
    complexities = _values(complexity, "complexity")
    rows = []
    for index, (_, nr_symbols, target) in enumerate(
        product(range(iterations), alphabet_sizes, complexities), start=1
    ):
        if nr_symbols > target:
            continue
        seed, measured = lzw_string_generator(
            nr_symbols, target, priorise_complexity, random_state + index
        )
        rows.append((len(set(seed)), measured, len(seed), seed))
    return pd.DataFrame(rows, columns=["nr_symbols", "LZW_complexity", "length", "string"])
