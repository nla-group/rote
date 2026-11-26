"""Generate independent strings and match LZW scores at equal predictive order."""

from collections import defaultdict
import random

from rote_benchmark.generation import _shortest_period, lzw_string_generator

from ..seeds import fixed_length_seed, predictive_order
from ..tuning_followup.runner import rotation_key


LZW_LENGTHS = {4: 196, 8: 130}
LZW_SCORES = (80, 90)
BUDGET_ALPHABETS = (2, 4, 8)
WINDOW = 100
HORIZON = 100
SEQUENCE_LENGTH = 3500
LZW_MODELS = ("GRU", "LSTM", "LinearAttention", "Transformer")
BUDGET_MODELS = (*LZW_MODELS, "Performer", "RWKV")
LZW_SEED = 782913
BUDGET_SEED = 6290371


def seed_record(word, n, score, index, generator_seed, order):
    return {
        "symbols": n, "complexity": score, "seed_index": index,
        "seed_string": word, "seed_length": len(word),
        "generator_seed": generator_seed, "predictive_order": order,
    }


def order_matched_strings(count, alphabet_sizes, blocked, *, candidates_per_score=192):
    """Match exact scores within predictive order, before any model is fitted."""
    rows = []
    for n in alphabet_sizes:
        pools = {}
        seen = set(blocked)
        for score in LZW_SCORES:
            by_order = defaultdict(list)
            for index in range(candidates_per_score):
                rng_seed = LZW_SEED + n * 100000 + score * 1000 + index
                word, search = fixed_length_seed(n, LZW_LENGTHS[n], score, rng_seed)
                key = rotation_key(word)
                if key in seen:
                    continue
                seen.add(key)
                order = predictive_order(word)
                if order <= WINDOW:
                    row = seed_record(word, n, score, index, rng_seed, order)
                    by_order[order].append({**row, **search})
            pools[score] = by_order
        low, high = (pools[score] for score in LZW_SCORES)
        sampler = random.Random(LZW_SEED + n)
        candidates = []
        for order in sorted(low.keys() & high.keys()):
            sampler.shuffle(low[order])
            sampler.shuffle(high[order])
            candidates.extend(zip(low[order], high[order]))
        sampler.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(
                f"Only {len(candidates)} distinct same-order pairs for n={n}; "
                f"need {count} at scores {LZW_SCORES}."
            )
        for index, pair in enumerate(candidates[:count]):
            for row in pair:
                blocked.add(rotation_key(row["seed_string"]))
                rows.append({**row, "seed_index": index, "pair_id": f"n{n}-p{index:03d}"})
    return rows


def budget_strings(count, alphabet_sizes, blocked):
    rows = []
    for n in alphabet_sizes:
        accepted = 0
        for attempt in range(2000):
            rng_seed = BUDGET_SEED + n * 10000 + attempt
            word, score = lzw_string_generator(n, 90, random_state=rng_seed)
            if score != 90 or len(set(word)) != n or _shortest_period(word) != word:
                continue
            key = rotation_key(word)
            if key in blocked:
                continue
            order = predictive_order(word)
            if order > WINDOW:
                continue
            blocked.add(key)
            rows.append(seed_record(word, n, score, accepted, rng_seed, order))
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise RuntimeError(f"Could only generate {accepted}/{count} new strings for n={n}")
    return rows
