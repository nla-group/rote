"""Compare three small, self-contained PyTorch models with ROTE.

Run: python examples/compare_toy_models.py --output-dir /tmp/rote-toys
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from rote_benchmark import lzw_string_generator
from rote_benchmark.evaluation import EvaluationConfig, run_benchmark
from rote_benchmark.visualization import PlotStyle, visualize_benchmark


class ToyMLP(nn.Module):
    def __init__(self, n_symbols, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(window_size * n_symbols, 32), nn.ReLU(),
            nn.Linear(32, n_symbols),
        )

    def forward(self, x):
        return self.net(x)


class ToyGRU(nn.Module):
    def __init__(self, n_symbols):
        super().__init__()
        self.gru = nn.GRU(n_symbols, 24, batch_first=True)
        self.head = nn.Linear(24, n_symbols)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class ToyTransformer(nn.Module):
    def __init__(self, n_symbols, window_size):
        super().__init__()
        self.input = nn.Linear(n_symbols, 16)
        self.position = nn.Embedding(window_size, 16)
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=2,
                                           dim_feedforward=32, dropout=0.0,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(16, n_symbols)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        hidden = self.input(x) + self.position(positions)[None, :, :]
        causal_mask = torch.triu(torch.ones(x.size(1), x.size(1),
                                           dtype=torch.bool, device=x.device), diagonal=1)
        return self.head(self.encoder(hidden, mask=causal_mask)[:, -1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New run directory; default: ./rote_runs/run-...")
    parser.add_argument("--font-size", type=float, default=10.5)
    parser.add_argument("--uncertainty", choices=("std", "sem", "none"), default="std")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))

    seed_string, _ = lzw_string_generator(4, 12, random_state=7)
    config = EvaluationConfig(sequence_length=240, window_size=10,
                              forecast_horizon=24, max_epochs=args.epochs,
                              device=args.device)
    models = {
        "ToyMLP": lambda n: ToyMLP(n, config.window_size),
        "ToyGRU": ToyGRU,
        "ToyTransformer": lambda n: ToyTransformer(n, config.window_size),
    }
    saved = run_benchmark(
        models, seed_string, config=config, runs=args.runs,
        output_dir=args.output_dir, metadata={"example": "three toy architectures"},
    )
    report = visualize_benchmark(saved, style=PlotStyle(
        font_size=args.font_size, title_size=args.font_size + 2,
        uncertainty=args.uncertainty,
    ))
    print(saved.results.groupby("model", sort=False)[["test_loss", "DL"]].agg(["mean", "std"]).round(3))
    print(f"Configuration, results, and event log: {saved.path}")
    print(f"Figures and style: {report.path}")


if __name__ == "__main__":
    main()
