"""Regression checks for the exact symbolic protocol used in the paper."""

import unittest

import numpy as np

from rote_benchmark import (
    lzw_complexity,
    lzw_string_generator,
    lzw_string_seeds,
    normalized_damerau_levenshtein_distance as dl,
    normalized_jaro_winkler_distance as jw,
)


class GenerationTests(unittest.TestCase):
    def test_golden_seeds_match_manuscript_generator(self):
        self.assertEqual(lzw_string_generator(2, 10, random_state=3407), ("ABABABABBBABBBBAA", 10))
        self.assertEqual(
            lzw_string_generator(4, 30, random_state=3407),
            ("CBADADBCBDCCBDDCCBACBCAABCAACAABBDCBBDDADDCCABD", 30),
        )

    def test_seed_table_order_and_schema(self):
        seeds = lzw_string_seeds(symbols=[2, 4], complexity=[10, 30], iterations=2, random_state=3407)
        self.assertEqual(list(seeds), ["nr_symbols", "LZW_complexity", "length", "string"])
        self.assertEqual(len(seeds), 8)
        self.assertEqual(seeds.loc[0, "string"], "BAABBAABBBBABBBB")
        self.assertEqual(seeds.loc[3, "string"], "CBADACACBCDACDBBADBDBBABADCCDAAABCADCCDAAAAAAAAC")
        for row in seeds.itertuples(index=False):
            self.assertEqual(len(set(row.string)), row.nr_symbols)
            self.assertEqual(len(row.string), row.length)
            self.assertEqual(lzw_complexity(row.string), row.LZW_complexity)

    def test_generation_does_not_modify_numpy_global_rng(self):
        before = np.random.get_state()
        lzw_string_generator(4, 30, random_state=7)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_inputs_fail_explicitly(self):
        for alphabet_size, complexity in [(0, 10), (53, 90), (4, 3), (1, 10)]:
            with self.subTest(alphabet_size=alphabet_size, complexity=complexity):
                with self.assertRaises(ValueError):
                    lzw_string_generator(alphabet_size, complexity)
        with self.assertRaises(ValueError):
            lzw_string_seeds(iterations=0)
        with self.assertRaises(ValueError):
            lzw_complexity("A0")


class DistanceTests(unittest.TestCase):
    def test_reference_values(self):
        self.assertEqual(dl("", ""), 0.0)
        self.assertEqual(jw("", ""), 0.0)
        self.assertEqual(dl("A", ""), 1.0)
        self.assertEqual(jw("A", ""), 1.0)
        self.assertEqual(dl("AB", "BA"), 0.5)
        self.assertAlmostEqual(jw("martha", "marhta"), 0.03888888888888886)

    def test_distances_are_bounded(self):
        strings = ["", "A", "AB", "BA", "ABABBA", "BABABA"]
        for left in strings:
            for right in strings:
                with self.subTest(left=left, right=right):
                    self.assertGreaterEqual(dl(left, right), 0.0)
                    self.assertLessEqual(dl(left, right), 1.0)
                    self.assertGreaterEqual(jw(left, right), 0.0)
                    self.assertLessEqual(jw(left, right), 1.0)
                    self.assertEqual(dl(left, right), dl(right, left))


if __name__ == "__main__":
    unittest.main()
