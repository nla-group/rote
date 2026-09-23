"""Console entry points with clear errors for missing optional dependencies."""

from importlib import import_module


def _run(module_name: str, extra: str) -> None:
    try:
        module = import_module(f".{module_name}", package="rote_benchmark")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "matplotlib", "seaborn"}:
            raise SystemExit(
                f"Missing optional dependency '{exc.name}'. Install the "
                f"rote-bench[{extra}] extra and retry."
            ) from exc
        raise
    module.main()


def benchmark() -> None:
    """Run the fixed-budget benchmark."""
    _run("benchmark", "experiment")


def matched_size() -> None:
    """Run the matched-size benchmark."""
    _run("matched_size", "experiment")


def plot() -> None:
    """Generate main-experiment figures."""
    _run("plotting", "plot")


def plot_matched() -> None:
    """Generate the paired matched-size figure."""
    _run("plot_matched", "plot")
