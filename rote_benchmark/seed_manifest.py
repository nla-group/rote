"""Load exact symbolic seeds from an archived benchmark result or seed table."""

from pathlib import Path

import pandas as pd

from .generation import _shortest_period, lzw_complexity


def load_seed_manifest(path, symbols, complexities, seed_count):
    """Return validated seeds in the table shape expected by both experiments.

    ``path`` may be a results CSV with ``symbols``, ``complexity``,
    ``seed_index``, ``seed_string`` columns, or a seed table with
    ``nr_symbols``, ``LZW_complexity``, ``string`` columns. Results CSVs can
    contain repeated model/run rows; duplicates are collapsed by seed index.
    """
    source = Path(path)
    frame = pd.read_csv(source)
    result_columns = {"symbols", "complexity", "seed_index", "seed_string"}
    seed_columns = {"nr_symbols", "LZW_complexity", "string"}
    if result_columns <= set(frame):
        frame = frame.rename(
            columns={
                "symbols": "nr_symbols",
                "complexity": "LZW_complexity",
                "seed_string": "string",
            }
        )
    elif seed_columns <= set(frame):
        if "seed_index" not in frame:
            frame["seed_index"] = frame.groupby(
                ["nr_symbols", "LZW_complexity"], sort=False
            ).cumcount()
    else:
        raise ValueError(
            f"{source} must contain either benchmark result columns or seed table columns"
        )

    fields = ["nr_symbols", "LZW_complexity", "seed_index", "string"]
    frame = frame[fields].drop_duplicates()
    if frame.empty or frame[fields].isna().any().any():
        raise ValueError("seed manifest is empty or contains missing values")
    for field in fields[:3]:
        values = pd.to_numeric(frame[field], errors="raise")
        if (values % 1 != 0).any():
            raise ValueError(f"{field} must contain integers")
        frame[field] = values.astype(int)
    if (frame.seed_index < 0).any():
        raise ValueError("seed_index must be nonnegative")
    counts = frame.groupby(fields[:3]).size()
    if (counts > 1).any():
        raise ValueError("a seed index maps to more than one string")

    requested_symbols = [symbols] if isinstance(symbols, int) else list(symbols)
    requested_complexities = [complexities] if isinstance(complexities, int) else list(complexities)
    frame = frame[
        frame.nr_symbols.isin(requested_symbols)
        & frame.LZW_complexity.isin(requested_complexities)
    ].sort_values(fields[:3])
    rows = []
    for alphabet_size in requested_symbols:
        for complexity in requested_complexities:
            if alphabet_size > complexity:
                continue
            subset = frame[
                (frame.nr_symbols == alphabet_size)
                & (frame.LZW_complexity == complexity)
            ].head(seed_count)
            if len(subset) != seed_count:
                raise ValueError(
                    f"seed manifest needs {seed_count} seeds for "
                    f"alphabet={alphabet_size}, complexity={complexity}; found {len(subset)}"
                )
            for row in subset.itertuples(index=False):
                if not isinstance(row.string, str) or len(set(row.string)) != alphabet_size:
                    raise ValueError("seed alphabet size does not match manifest")
                if lzw_complexity(_shortest_period(row.string)) != complexity:
                    raise ValueError("seed LZW complexity does not match manifest")
                rows.append((alphabet_size, complexity, row.string))
    return pd.DataFrame(rows, columns=["nr_symbols", "LZW_complexity", "string"])
