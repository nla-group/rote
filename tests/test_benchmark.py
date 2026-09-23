"""Small CPU checks of the neural benchmark contract."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "install rote-bench[experiment] to test neural models")
class BenchmarkTests(unittest.TestCase):
    def test_model_contract(self):
        import torch
        from rote_benchmark.models import MODEL_REGISTRY, get_model

        self.assertEqual(
            set(MODEL_REGISTRY),
            {"LSTM", "GRU", "minGRU", "minLSTM", "Transformer", "LinearAttention", "Performer", "RWKV"},
        )
        x = torch.zeros(2, 10, 4)
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                optional_module = {
                    "minGRU": "minGRU_pytorch",
                    "LinearAttention": "linear_attention_transformer",
                    "Performer": "performer_pytorch",
                }.get(name)
                if optional_module and importlib.util.find_spec(optional_module) is None:
                    self.skipTest(f"{optional_module} is not installed")
                model = get_model(name, 4, 32, 4, num_layers=1, d_model=16, window_size=10)
                with torch.no_grad():
                    self.assertEqual(tuple(model(x).shape), (2, 4))

    def test_fixed_budget_run_writes_reproducible_schema(self):
        from rote_benchmark import benchmark

        with tempfile.TemporaryDirectory() as output_dir:
            manifest = Path(output_dir) / "seed_manifest.csv"
            manifest.write_text("nr_symbols,LZW_complexity,string\n2,10,ABABABABBBABBBBAA\n", encoding="utf-8")
            argv = [
                "rote-benchmark", "--models", "GRU", "--symbols", "2", "--complexities", "10",
                "--sequence-lengths", "240", "--window-size", "8", "--forecast-horizon", "8",
                "--seed-count", "1", "--seed-manifest", str(manifest), "--runs", "1", "--layers", "1", "--units", "16",
                "--d-models", "16", "--batch-sizes", "16", "--learning-rates", "0.001",
                "--weight-decays", "0.0", "--max-epochs", "1", "--patience", "1",
                "--device", "cpu", "--output-dir", output_dir,
            ]
            with patch.object(sys, "argv", argv):
                args = benchmark.parse_args()
            benchmark.configure(args)
            results = benchmark.run(args)
            self.assertEqual(len(results), 1)
            self.assertEqual(results.iloc[0]["model"], "GRU")
            self.assertEqual(results.iloc[0]["seed_string"], "ABABABABBBABBBBAA")
            self.assertEqual(len(results.iloc[0]["forecast"]), 8)
            self.assertTrue({"test_loss", "test_accuracy", "DL", "JW", "model_params"} <= set(results))
            self.assertTrue((Path(output_dir) / "config.json").is_file())
            self.assertTrue((Path(output_dir) / "results.csv").is_file())

    def test_matched_size_dry_run_writes_configuration(self):
        from rote_benchmark import matched_size

        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "rote-matched-size", "--models", "GRU", "Transformer", "--symbols", "2",
                "--complexities", "10", "--layers", "1", "--recurrent-units", "8", "16",
                "--d-models", "8", "16", "--target-sizes-m", "0.01", "--window-size", "8",
                "--dry-run", "--device", "cpu", "--output-dir", output_dir,
            ]
            with patch.object(sys, "argv", argv):
                args = matched_size.parse_args()
            matched_size.configure(args)
            matched_size.run(args)
            self.assertTrue((Path(output_dir) / "matched_configs.csv").is_file())


if __name__ == "__main__":
    unittest.main()
