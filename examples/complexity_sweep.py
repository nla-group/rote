"""A small, self-contained two-complexity ROTE run with two toy models.

Run: python examples/complexity_sweep.py --output-dir /tmp/rote-sweep
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from rote_benchmark import lzw_string_generator
from rote_benchmark.evaluation import EvaluationConfig, run_benchmark
from rote_benchmark.visualization import PlotStyle, visualize_benchmark


class TinyMLP(nn.Module):
    def __init__(self, n_symbols, window_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(n_symbols * window_size, 24),
            nn.ReLU(), nn.Linear(24, n_symbols),
        )

    def forward(self, x):
        return self.network(x)


class TinyGRU(nn.Module):
    def __init__(self, n_symbols):
        super().__init__()
        self.gru = nn.GRU(n_symbols, 16, batch_first=True)
        self.head = nn.Linear(16, n_symbols)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New run directory")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))

    targets = (12, 20)
    seeds = {}
    for index, target in enumerate(targets):
        seed, measured = lzw_string_generator(4, target, random_state=7 + index)
        seeds[f"target-{target}-measured-{measured}"] = seed
    config = EvaluationConfig(
        sequence_length=160, window_size=8, forecast_horizon=12,
        max_epochs=args.epochs, patience=2,
    )
    models = {
        "TinyMLP": lambda n: TinyMLP(n, config.window_size),
        "TinyGRU": TinyGRU,
    }
    saved = run_benchmark(
        models, seeds, config=config, runs=args.runs,
        output_dir=args.output_dir,
        metadata={"example": "toy complexity sweep", "requested_lzw_codes": targets},
    )
    report = visualize_benchmark(saved, style=PlotStyle(formats=("png", "pdf")))
    print(saved.results.groupby(["model", "complexity"])[["test_loss", "DL"]].mean().round(3))
    print(f"Run: {saved.path}")
    print(f"Figures: {report.path}")


if __name__ == "__main__":
    main()
