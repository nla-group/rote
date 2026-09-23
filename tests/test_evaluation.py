"""Contract tests for user-supplied PyTorch predictors."""

import importlib.util
import unittest


HAS_NEURAL = all(importlib.util.find_spec(name) is not None
                 for name in ("torch", "numpy", "pandas"))
HAS_PLOT = HAS_NEURAL and importlib.util.find_spec("matplotlib") is not None

@unittest.skipUnless(HAS_NEURAL, "install rote-bench[neural] to run adapter tests")
class EvaluationTests(unittest.TestCase):
    def test_factory_runs_and_schema(self):
        import torch
        from rote_benchmark.evaluation import EvaluationConfig, evaluate_models

        config = EvaluationConfig(sequence_length=80, window_size=6,
                                  forecast_horizon=8, max_epochs=1, patience=1,
                                  batch_size=8)
        results = evaluate_models(
            {"Tiny": lambda n: torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(6 * n, n))},
            "ABBAABAB", config=config, runs=2,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(list(results["run"]), [0, 1])
        self.assertEqual(list(results["seed"]), [3407, 3408])
        self.assertTrue({"test_loss", "test_accuracy", "DL", "JW", "forecast",
                         "target_string", "model_params"} <= set(results))
        self.assertTrue(all(results["forecast"].str.len() == 8))

    def test_prebuilt_model_and_rng_state(self):
        import torch
        from rote_benchmark.evaluation import EvaluationConfig, evaluate_model

        config = EvaluationConfig(sequence_length=80, window_size=6,
                                  forecast_horizon=8, max_epochs=1, patience=1)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
        before = torch.random.get_rng_state().clone()
        result = evaluate_model(model, "ABBAABAB", config=config, name="Provided")
        self.assertEqual(result["model"], "Provided")
        self.assertEqual(result["model_params"], 26)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_invalid_output_shape_rejected(self):
        import torch
        from rote_benchmark.evaluation import EvaluationConfig, evaluate_model

        config = EvaluationConfig(sequence_length=80, window_size=6,
                                  forecast_horizon=8, max_epochs=1)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
        with self.assertRaisesRegex(ValueError, "logits of shape"):
            evaluate_model(model, "ABBAABAB", config=config)

    def test_invalid_configuration_rejected(self):
        from rote_benchmark.evaluation import EvaluationConfig

        with self.assertRaisesRegex(ValueError, "sequence_length"):
            EvaluationConfig(sequence_length=20, window_size=12, forecast_horizon=12)

    def test_score_pretrained_does_not_update_weights(self):
        import torch
        from rote_benchmark.evaluation import EvaluationConfig, score_model

        config = EvaluationConfig(sequence_length=80, window_size=6,
                                  forecast_horizon=8, batch_size=8)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
        model.requires_grad_(False)
        weights = [parameter.detach().clone() for parameter in model.parameters()]
        result = score_model(model, "ABBAABAB", config=config, name="Frozen")
        self.assertEqual(result["model"], "Frozen")
        self.assertEqual(len(result["forecast"]), 8)
        self.assertEqual(result["model_params"], 26)
        self.assertEqual(result["trainable_params"], 0)
        self.assertNotIn("train_time", result)
        for before, after in zip(weights, model.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_run_benchmark_writes_incremental_artifacts(self):
        import json
        import tempfile
        from pathlib import Path
        import torch
        from rote_benchmark.evaluation import EvaluationConfig, run_benchmark

        config = EvaluationConfig(sequence_length=80, window_size=6,
                                  forecast_horizon=8, batch_size=8,
                                  max_epochs=1, patience=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial"
            saved = run_benchmark(
                {"Tiny": lambda n: torch.nn.Sequential(
                    torch.nn.Flatten(), torch.nn.Linear(6 * n, n))},
                "ABBAABAB", config=config, runs=2, output_dir=output,
                metadata={"purpose": "smoke"},
            )
            self.assertEqual(saved.manifest["status"], "complete")
            self.assertEqual(saved.manifest["completed"], 2)
            self.assertEqual(len(saved.results), 2)
            self.assertEqual(len(saved.records_jsonl.read_text().splitlines()), 2)
            self.assertEqual(json.loads((saved.path / "config.json").read_text())["metadata"]["purpose"], "smoke")
            with self.assertRaises(FileExistsError):
                run_benchmark({"Tiny": lambda n: torch.nn.Linear(n, n)},
                              "ABBAABAB", config=config, output_dir=output)

    def test_run_benchmark_accepts_named_seeds_for_complexity_sweeps(self):
        import tempfile
        from pathlib import Path
        import torch
        from rote_benchmark.evaluation import EvaluationConfig, run_benchmark

        config = EvaluationConfig(sequence_length=80, window_size=6,
                                  forecast_horizon=8, max_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            saved = run_benchmark(
                {"Tiny": lambda n: torch.nn.Sequential(
                    torch.nn.Flatten(), torch.nn.Linear(6 * n, n))},
                {"first": "ABBAABAB", "second": "AABABB"},
                config=config, output_dir=Path(directory) / "sweep",
            )
            self.assertEqual(saved.manifest["total"], 2)
            self.assertEqual(set(saved.results["seed_id"]), {"first", "second"})
            self.assertEqual(set(saved.config["seed_strings"]), {"first", "second"})

    def test_failed_run_preserves_status_and_events(self):
        import json
        import tempfile
        from pathlib import Path
        from rote_benchmark.evaluation import EvaluationConfig, run_benchmark

        config = EvaluationConfig(sequence_length=80, window_size=6, forecast_horizon=8)
        def broken(_):
            raise RuntimeError("factory failed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with self.assertRaisesRegex(RuntimeError, "factory failed"):
                run_benchmark({"Broken": broken}, "ABBAABAB", config=config, output_dir=output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("run_failed", (output / "events.jsonl").read_text())




@unittest.skipUnless(HAS_PLOT, "install rote-bench[neural,plot] to run figure tests")
class VisualizationTests(unittest.TestCase):
    def test_figures_from_result_rows(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from rote_benchmark.visualization import plot_model_comparison, plot_rollout_error

        rows = pd.DataFrame([
            {"model": "A", "test_loss": 0.1, "DL": 0.0,
             "target_string": "ABBA", "forecast": "ABBA"},
            {"model": "B", "test_loss": 0.2, "DL": 0.25,
             "target_string": "ABBA", "forecast": "ABAA"},
        ])
        comparison = plot_model_comparison(rows)
        horizon = plot_rollout_error(rows)
        self.assertEqual(len(comparison.axes), 2)
        self.assertEqual(len(horizon.axes), 1)
        plt.close(comparison)
        plt.close(horizon)

    def test_visualize_report_and_custom_style(self):
        import json
        import tempfile
        from pathlib import Path
        import pandas as pd
        from rote_benchmark.visualization import PlotStyle, visualize_benchmark

        rows = pd.DataFrame([
            {"model": model, "complexity": complexity,
             "test_loss": loss, "test_accuracy": 0.75, "DL": dl,
             "target_string": "ABBA", "forecast": forecast,
             "train_time": time, "model_size_m": size, "memory_mb": 0.0}
            for model, loss, dl, forecast, time, size in [
                ("A", 0.2, 0.0, "ABBA", 1.0, 0.01),
                ("B", 0.4, 0.25, "ABAA", 2.0, 0.02),
            ]
            for complexity in (10, 20)
        ])
        style = PlotStyle(font_size=12, uncertainty="sem", formats=("png",),
                          model_colors={"A": "#225588"}, horizon_marker_step=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figures"
            report = visualize_benchmark(rows, output_dir=output, style=style)
            self.assertEqual(set(report.figures), {"quality", "rollout", "tradeoffs", "complexity"})
            self.assertTrue(all(paths[0].is_file() for paths in report.figures.values()))
            saved_style = json.loads(report.config_path.read_text())["style"]
            self.assertEqual(saved_style["font_size"], 12)
            self.assertEqual(saved_style["model_colors"]["A"], "#225588")
            with self.assertRaises(FileExistsError):
                visualize_benchmark(rows, output_dir=output, style=style)
            visualize_benchmark(rows, output_dir=output, style=style,
                                figures=("quality",), overwrite=True)



if __name__ == "__main__":
    unittest.main()
