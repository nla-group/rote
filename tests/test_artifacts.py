"""Persistence tests independent of the optional neural stack."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from rote_benchmark.artifacts import load_run, save_benchmark


class ArtifactTests(unittest.TestCase):
    def test_explicit_directory_and_round_trip(self):
        results = pd.DataFrame([
            {"model": "Toy", "run": 0, "DL": 0.25, "forecast": "ABBA"},
            {"model": "Toy", "run": 1, "DL": 0.0, "forecast": "ABAB"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "my-run"
            saved = save_benchmark(results, {"seed_string": "ABBA", "note": "trial"}, output_dir=output)
            self.assertEqual(saved.path, output.resolve())
            self.assertEqual(saved.manifest["status"], "complete")
            self.assertEqual(saved.manifest["completed"], 2)
            self.assertEqual(saved.config["note"], "trial")
            self.assertEqual(len(pd.read_csv(saved.results_csv)), 2)
            self.assertEqual(len(saved.records_jsonl.read_text().splitlines()), 2)
            self.assertEqual(len(saved.events_jsonl.read_text().splitlines()), 4)
            self.assertEqual(load_run(output).results["forecast"].tolist(), ["ABBA", "ABAB"])
            with self.assertRaises(FileExistsError):
                save_benchmark(results, {}, output_dir=output)

    def test_incomplete_last_jsonl_line_does_not_hide_completed_records(self):
        frame = pd.DataFrame([{"model": "Toy", "DL": 0.0}])
        with tempfile.TemporaryDirectory() as directory:
            saved = save_benchmark(frame, {}, output_dir=Path(directory) / "run")
            with saved.records_jsonl.open("a", encoding="utf-8") as stream:
                stream.write('{"model": "unfinished"')
            self.assertEqual(load_run(saved.path).results["model"].tolist(), ["Toy"])

    def test_default_directory_is_unique_and_config_is_json(self):
        results = pd.DataFrame([{"model": "Toy", "DL": 0.0}])
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"ROTE_RUNS_DIR": directory}):
                first = save_benchmark(results, {"settings": {"window": 8}})
                second = save_benchmark(results, {"settings": {"window": 8}})
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(first.path.parent, Path(directory).resolve())
            self.assertEqual(json.loads((first.path / "config.json").read_text())["settings"]["window"], 8)


if __name__ == "__main__":
    unittest.main()
