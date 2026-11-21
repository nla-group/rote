"""Deterministic, nonperiodic character-level excerpts from Tiny Shakespeare."""

import hashlib
from pathlib import Path

import pandas as pd

CORPUS = Path(__file__).resolve().parent / "data/tinyshakespeare.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def text_excerpts(path=CORPUS, count=3, length=3500):
    path = Path(path)
    data = path.read_bytes()
    if path.resolve() == CORPUS.resolve() and hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Bundled natural-text corpus checksum changed")
    text = data.decode("utf-8").replace("\r\n", "\n")
    offsets = (0, 150000, 300000)
    if count > len(offsets):
        raise ValueError("Only three preregistered nonoverlapping excerpts are available")
    rows = []
    for index, offset in enumerate(offsets[:count]):
        sample = text[offset:offset + length]
        if len(sample) != length:
            raise ValueError("Text corpus is shorter than the declared excerpt")
        rows.append({"symbols": len(set(sample)), "complexity": None,
                     "seed_index": index, "seed_string": sample, "length": length,
                     "source_offset": offset, "source_sha256": hashlib.sha256(data).hexdigest(),
                     "source_type": "nonperiodic_natural_text"})
    return pd.DataFrame(rows)
