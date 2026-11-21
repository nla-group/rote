"""Train-only lookup and smoothed variable-order n-gram baselines."""

from collections import Counter, defaultdict
import math

from rote_benchmark.metrics import (
    normalized_damerau_levenshtein_distance,
    normalized_jaro_winkler_distance,
)

from .seeds import phase_statistics, predictive_order


def _partition(seed, sequence_length, window, horizon):
    if sequence_length < window + horizon + 12:
        raise ValueError("Sequence is too short for the declared split")
    sequence = (seed * math.ceil(sequence_length / len(seed)))[:sequence_length]
    observed = sequence[:-horizon]
    windows = [(observed[i:i + window], observed[i + window])
               for i in range(len(observed) - window)]
    train_end = max(1, int(0.8 * len(windows)))
    valid_end = min(max(train_end + 1, int(0.9 * len(windows))), len(windows) - 1)
    return (windows[:train_end], windows[train_end:valid_end],
            windows[valid_end:], observed[-window:], sequence[-horizon:])


class NGram:
    def __init__(self, alphabet, order, smoothing=0.5, backoff=True):
        self.alphabet = sorted(alphabet)
        self.order = order
        self.smoothing = smoothing
        self.backoff = backoff
        self.counts = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, rows):
        for context, target in rows:
            orders = range(self.order + 1) if self.backoff else (self.order, 0)
            for order in orders:
                self.counts[order][context[-order:] if order else ""][target] += 1
        return self

    def probabilities(self, context):
        orders = range(self.order, -1, -1) if self.backoff else (self.order, 0)
        for order in orders:
            counts = self.counts[order].get(context[-order:] if order else "")
            if counts:
                denominator = sum(counts.values()) + self.smoothing * len(self.alphabet)
                return {symbol: (counts[symbol] + self.smoothing) / denominator
                        for symbol in self.alphabet}
        raise RuntimeError("The training set did not contain any symbols")

    def predict(self, context):
        probabilities = self.probabilities(context)
        return max(self.alphabet, key=lambda symbol: probabilities[symbol])


def evaluate_baseline(row, *, window=100, sequence_length=3500, horizon=100,
                      kind="Lookup", smoothing=0.5):
    seed = str(row["seed_string"])
    train, valid, test, initial_context, target = _partition(
        seed, sequence_length, window, horizon
    )
    model = NGram(set(seed), window, smoothing=smoothing, backoff=(kind == "NGram"))
    model.fit(train)
    def metrics(samples):
        accuracy = sum(model.predict(context) == symbol for context, symbol in samples) / len(samples)
        if kind == "Lookup":
            return accuracy, None
        loss = -sum(math.log(model.probabilities(context)[symbol])
                    for context, symbol in samples) / len(samples)
        return accuracy, loss
    train_accuracy, train_loss = metrics(train)
    valid_accuracy, valid_loss = metrics(valid)
    test_accuracy, test_loss = metrics(test)
    unseen = sum(context not in model.counts[window] for context, _ in test) / len(test)
    context = initial_context
    forecast = []
    for _ in target:
        symbol = model.predict(context)
        forecast.append(symbol)
        context = context[1:] + symbol
    forecast = "".join(forecast)
    phase = phase_statistics(seed, window)
    order = predictive_order(seed)
    return {
        "track": "baseline", "model": kind, "symbols": int(row["symbols"]),
        "complexity": int(row["complexity"]), "seed_index": int(row["seed_index"]),
        "seed_string": seed, "seed_length": len(seed), "predictive_order": order,
        "window_size": window, "alpha_w": phase["alpha"],
        "bayes_error_w": phase["bayes_error"],
        "conditional_entropy_w": phase["conditional_entropy"],
        "sequence_length": sequence_length, "forecast_horizon": horizon,
        "train_windows": len(train), "valid_windows": len(valid),
        "test_windows": len(test), "phase_exposure": len(train) / len(seed),
        "unseen_test_context_rate": unseen,
        "stored_contexts": sum(len(table) for table in model.counts),
        "stored_context_characters": sum(
            len(context) for table in model.counts for context in table
        ),
        "smoothing": smoothing if kind == "NGram" else None,
        "train_accuracy": train_accuracy, "train_loss": train_loss,
        "valid_accuracy": valid_accuracy, "valid_loss": valid_loss,
        "test_accuracy": test_accuracy, "test_loss": test_loss,
        "DL": normalized_damerau_levenshtein_distance(target, forecast),
        "JW": normalized_jaro_winkler_distance(target, forecast),
        "target_string": target, "forecast": forecast,
        "exact": int(target == forecast),
    }
