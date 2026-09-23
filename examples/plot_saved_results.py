"""Regenerate custom-model figures from a saved ROTE run.

Run: python examples/plot_saved_results.py /tmp/rote-toys
"""

import argparse
from pathlib import Path

from rote_benchmark.visualization import PlotStyle, visualize_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved run directory or results.csv")
    parser.add_argument("--output-dir", type=Path, help="Figure directory; defaults to <run>/figures")
    parser.add_argument("--font-family", default="DejaVu Sans")
    parser.add_argument("--font-size", type=float, default=10.5)
    parser.add_argument("--uncertainty", choices=("std", "sem", "none"), default="std")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--figures", nargs="+", choices=("quality", "rollout", "tradeoffs", "complexity"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    style = PlotStyle(
        font_family=args.font_family, font_size=args.font_size,
        title_size=args.font_size + 2, uncertainty=args.uncertainty,
        formats=tuple(args.formats),
    )
    report = visualize_benchmark(
        args.source, output_dir=args.output_dir, style=style,
        figures=tuple(args.figures) if args.figures else None,
        overwrite=args.overwrite,
    )
    print(f"Figures and visualization.json: {report.path}")
    for name, paths in report.figures.items():
        print(f"  {name}: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
