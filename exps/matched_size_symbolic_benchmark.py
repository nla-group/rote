"""Repository entry point for the ROTE matched-size benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rote_benchmark.matched_size import main


if __name__ == "__main__":
    main()
