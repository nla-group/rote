"""Score a saved PyTorch checkpoint without retraining it.

Run: python examples/score_checkpoint.py --output-dir /tmp/rote-checkpoint
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from rote_benchmark import lzw_string_generator
from rote_benchmark.evaluation import EvaluationConfig, evaluate_model, score_model
from rote_benchmark.visualization import plot_model_comparison


class SmallMLP(nn.Module):
    def __init__(self, n_symbols, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(window_size * n_symbols, 24),
            nn.ReLU(), nn.Linear(24, n_symbols),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("rote-example-checkpoint"))
    args = parser.parse_args()

    seed_string, _ = lzw_string_generator(4, 12, random_state=7)
    n_symbols = len(set(seed_string))
    config = EvaluationConfig(sequence_length=240, window_size=10,
                              forecast_horizon=24, max_epochs=5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = SmallMLP(n_symbols, config.window_size)
    fitted = evaluate_model(model, seed_string, config=config, name="Fitted")
    checkpoint = args.output_dir / "toy_model.pt"
    torch.save(model.state_dict(), checkpoint)

    restored = SmallMLP(n_symbols, config.window_size)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    scored = score_model(restored, seed_string, config=config, name="Reloaded")
    assert scored["forecast"] == fitted["forecast"]
    assert abs(scored["test_loss"] - fitted["test_loss"]) < 1e-6

    results = pd.DataFrame([fitted, scored])
    results.to_csv(args.output_dir / "results.csv", index=False)
    plot_model_comparison(results, args.output_dir / "comparison.png")
    print(results[["model", "test_loss", "test_accuracy", "DL", "JW"]].to_string(index=False))


if __name__ == "__main__":
    main()
