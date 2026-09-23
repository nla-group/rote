"""The archived paper seeds must remain replayable after the rename."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from rote_benchmark.seed_manifest import load_seed_manifest


class SeedManifestTests(unittest.TestCase):
    def test_repeated_result_rows_are_collapsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            pd.DataFrame(
                [
                    {"symbols": 2, "complexity": 10, "seed_index": 0, "seed_string": "ABABABABBBABBBBAA", "model": model}
                    for model in ["GRU", "LSTM"]
                ]
            ).to_csv(path, index=False)
            loaded = load_seed_manifest(path, 2, [10], 1)
            self.assertEqual(loaded.string.tolist(), ["ABABABABBBABBBBAA"])

    def test_conflicting_seed_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            pd.DataFrame(
                [
                    {"symbols": 2, "complexity": 10, "seed_index": 0, "seed_string": seed}
                    for seed in ["ABABABABBBABBBBAA", "BAABBAABBBBABBBB"]
                ]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "more than one string"):
                load_seed_manifest(path, 2, [10], 1)

    def test_archived_paper_seeds(self):
        repository = Path(__file__).resolve().parents[1]
        locations = [
            repository / "legacy/exps/results_symbolic/slurm_102076",
            repository / "legacy/exps/results_symbolic_matched_size/slurm_104393",
        ]
        if not all((location / "results_merged.csv").is_file() for location in locations):
            self.skipTest("archived paper results are not included")
        for location, expected_count in zip(locations, [40, 4]):
            with self.subTest(archive=location):
                config = json.loads((location / "config.json").read_text(encoding="utf-8"))
                path = location / "results_merged.csv"
                loaded = load_seed_manifest(
                    path, config["symbols"], config["complexities"], config["seed_count"]
                )
                original = (
                    pd.read_csv(path)[["symbols", "complexity", "seed_index", "seed_string"]]
                    .drop_duplicates()
                    .sort_values(["symbols", "complexity", "seed_index"])
                )
                self.assertEqual(len(loaded), expected_count)
                self.assertEqual(loaded.string.tolist(), original.seed_string.tolist())


if __name__ == "__main__":
    unittest.main()
