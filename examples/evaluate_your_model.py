"""Evaluate an already constructed PyTorch network with ROTE.

Run: python examples/evaluate_your_model.py --output-dir /tmp/rote-single
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch.nn as nn

from rote_benchmark import lzw_string_generator
from rote_benchmark.artifacts import save_benchmark
from rote_benchmark.evaluation import EvaluationConfig, evaluate_model
from rote_benchmark.visualization import visualize_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("rote-example-single"))
    args = parser.parse_args()

    seed_string, complexity = lzw_string_generator(4, 12, random_state=7)
    config = EvaluationConfig(sequence_length=240, window_size=10,
                              forecast_horizon=24, max_epochs=5)
    n_symbols = len(set(seed_string))
    # Replace this module with any network accepting (batch, window, symbols).
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(config.window_size * n_symbols, 24),
        nn.ReLU(),
        nn.Linear(24, n_symbols),
    )
    result = evaluate_model(model, seed_string, config=config, name="MyMLP")
    table = pd.DataFrame([result])
    saved = save_benchmark(
        table, {"example": "prebuilt MLP", "seed_string": seed_string,
                "evaluation": asdict(config)},
        output_dir=args.output_dir,
    )
    report = visualize_benchmark(saved)
    print(f"Saved run: {saved.path}")
    print(f"Figures: {report.path}")
    print(f"LZW codes: {complexity}; target: {result['target_string']}")
    print(f"forecast: {result['forecast']}")
    print(table[["model", "test_loss", "test_accuracy", "DL", "JW", "model_params"]].to_string(index=False))


if __name__ == "__main__":
    main()
