"""A train-prefix period detector that forecasts using absolute phase."""

from rote_benchmark.metrics import (
    normalized_damerau_levenshtein_distance,
    normalized_jaro_winkler_distance,
)

from ..baselines import _partition


def shortest_prefix_period(prefix):
    """Return the shortest period compatible with a possibly partial last copy."""
    if not prefix:
        raise ValueError("Period detection requires a nonempty training prefix")
    border_lengths = [0] * len(prefix)
    for index in range(1, len(prefix)):
        length = border_lengths[index - 1]
        while length and prefix[index] != prefix[length]:
            length = border_lengths[length - 1]
        border_lengths[index] = length + (prefix[index] == prefix[length])
    return len(prefix) - border_lengths[-1]


def evaluate_period(row, *, window, sequence_length, horizon):
    seed = row["seed_string"]
    train, valid, test, _, target = _partition(seed, sequence_length, window, horizon)
    prefix = train[0][0] + "".join(symbol for _, symbol in train)
    estimated_period = shortest_prefix_period(prefix)
    first_test_position = len(prefix) + len(valid)
    test_correct = sum(prefix[position % estimated_period] == symbol
                       for position, (_, symbol) in enumerate(test, start=first_test_position))
    forecast = "".join(prefix[position % estimated_period]
                       for position in range(sequence_length - horizon, sequence_length))
    return {
        **row, "model": "PeriodDetection", "train_prefix_length": len(prefix),
        "estimated_period": estimated_period, "seed_length": len(seed),
        "train_windows": len(train), "valid_windows": len(valid),
        "test_windows": len(test), "test_accuracy": test_correct / len(test),
        "window_size": window, "forecast_horizon": horizon,
        "sequence_length": sequence_length, "target_string": target,
        "forecast": forecast, "exact": int(target == forecast),
        "DL": normalized_damerau_levenshtein_distance(target, forecast),
        "JW": normalized_jaro_winkler_distance(target, forecast),
    }
