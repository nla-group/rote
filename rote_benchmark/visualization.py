"""Configurable, model-agnostic figures for custom ROTE runs.

The publication figures in :mod:`rote_benchmark.plotting` remain separate;
this module serves user-supplied models and saved run directories.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .artifacts import SavedRun, _write_json, load_run
from .metrics import normalized_damerau_levenshtein_distance


PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#117733", "#882255", "#44AA99",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (6, 2, 1, 2)))
MODEL_ORDER = (
    "LSTM", "GRU", "minGRU", "minLSTM", "Transformer",
    "LinearAttention", "Performer", "RWKV",
)
METRIC_LABELS = {
    "test_loss": "Held-out cross-entropy",
    "test_accuracy": "Next-symbol accuracy",
    "DL": "Normalized DL distance",
    "JW": "Jaro-Winkler distance",
    "train_time": "Training and evaluation time (s)",
    "model_size_m": "Parameters (millions)",
    "memory_mb": "Peak CUDA allocation (MiB)",
}


@dataclass(frozen=True)
class PlotStyle:
    """Shared appearance and metric settings for a custom-model report.

    ``uncertainty`` is descriptive variation across input rows, not a
    confidence interval. ``model_colors`` can pin named models to colors.
    ``formats`` controls files saved by :func:`visualize_benchmark`.
    """

    font_family: str = "DejaVu Sans"
    font_size: float = 10.5
    title_size: float = 12.5
    figure_size: tuple[float, float] = (7.4, 4.4)
    dpi: int = 300
    palette: tuple[str, ...] = PALETTE
    markers: tuple[str, ...] = MARKERS
    linestyles: tuple = LINESTYLES
    model_colors: dict[str, str] = field(default_factory=dict)
    marker_size: float = 7.0
    line_width: float = 2.1
    legend_columns: int = 4
    legend_y: float = 0.025
    uncertainty: str = "std"
    horizon_marker_step: int = 5
    log_cost_axes: bool = False
    quality_metrics: tuple[str, ...] = ("test_loss", "test_accuracy", "DL")
    complexity_metrics: tuple[str, ...] = ("test_loss", "DL")
    cost_metrics: tuple[str, ...] = ("train_time", "model_size_m", "memory_mb")
    formats: tuple[str, ...] = ("png", "pdf")

    def __post_init__(self):
        if self.font_size <= 0 or self.title_size <= 0 or self.marker_size <= 0 or self.line_width <= 0:
            raise ValueError("font, marker, and line sizes must be positive")
        if len(self.figure_size) != 2 or any(value <= 0 for value in self.figure_size):
            raise ValueError("figure_size must contain two positive values")
        if not self.palette or not self.markers or not self.linestyles:
            raise ValueError("palette, markers, and linestyles must be nonempty")
        if self.uncertainty not in {"std", "sem", "none"}:
            raise ValueError("uncertainty must be 'std', 'sem', or 'none'")
        if self.dpi < 72 or self.horizon_marker_step < 1 or self.legend_columns < 1:
            raise ValueError("dpi, horizon_marker_step, and legend_columns must be positive")
        if not self.formats or set(self.formats) - {"png", "pdf", "svg"}:
            raise ValueError("formats must be a nonempty subset of png, pdf, and svg")
        if not self.quality_metrics or set(self.quality_metrics) - {"test_loss", "test_accuracy", "DL", "JW"}:
            raise ValueError("quality_metrics contains an unknown metric")
        if not self.complexity_metrics or set(self.complexity_metrics) - {"test_loss", "test_accuracy", "DL", "JW"}:
            raise ValueError("complexity_metrics contains an unknown metric")
        if not self.cost_metrics or set(self.cost_metrics) - {"train_time", "model_size_m", "memory_mb"}:
            raise ValueError("cost_metrics contains an unknown cost metric")


@dataclass(frozen=True)
class VisualizationReport:
    """Output paths from :func:`visualize_benchmark`."""

    path: Path
    figures: dict[str, tuple[Path, ...]]
    config_path: Path


def _frame(results, required):
    frame = pd.DataFrame(results).copy()
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("results must have at least one row")
    if frame["model"].isna().any() or any(not str(name).strip() for name in frame["model"]):
        raise ValueError("every row must have a nonempty model name")
    frame["model"] = frame["model"].astype(str)
    return frame


def _numeric(frame, columns):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
        if (frame[column] < 0).any():
            raise ValueError(f"{column} must be nonnegative")
        if column in {"DL", "JW", "test_accuracy"} and (frame[column] > 1).any():
            raise ValueError(f"{column} must not exceed one")


def _models(frame):
    present = set(frame["model"])
    return [name for name in MODEL_ORDER if name in present] + sorted(present.difference(MODEL_ORDER))


def _appearance(models, style):
    return {
        name: {
            "color": style.model_colors.get(name, style.palette[index % len(style.palette)]),
            "marker": style.markers[index % len(style.markers)],
            "linestyle": style.linestyles[index % len(style.linestyles)],
            "filled": index % 2 == 0,
        }
        for index, name in enumerate(models)
    }


def _rc(style):
    return plt.rc_context({
        "font.family": style.font_family,
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.title_size,
        "xtick.labelsize": style.font_size,
        "ytick.labelsize": style.font_size,
        "legend.fontsize": style.font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _polish(ax, style):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#777777", alpha=0.16, linewidth=0.7)
    ax.tick_params(axis="both", width=0.8, length=4)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def _error(values, style):
    if style.uncertainty == "none" or len(values) < 2:
        return 0.0
    deviation = float(np.std(values, ddof=1))
    return deviation / np.sqrt(len(values)) if style.uncertainty == "sem" else deviation


def _legend(fig, models, appearance, style):
    handles = []
    for name in models:
        item = appearance[name]
        handles.append(Line2D(
            [0], [0], label=name, color=item["color"], linestyle=item["linestyle"],
            linewidth=style.line_width, marker=item["marker"], markersize=style.marker_size,
            markerfacecolor=item["color"] if item["filled"] else "white",
            markeredgecolor=item["color"], markeredgewidth=1.3,
        ))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, style.legend_y),
               ncol=min(style.legend_columns, len(models)), frameon=False)


def _finish(fig, style, *, with_legend=False):
    if with_legend:
        handles = len(fig.legends[0].texts) if fig.legends else 0
        legend_rows = max(1, int(np.ceil(handles / style.legend_columns)))
        fig.subplots_adjust(bottom=0.24 + 0.06 * (legend_rows - 1),
                            top=0.88, wspace=0.32)
    else:
        fig.subplots_adjust(bottom=0.17, top=0.86, wspace=0.28)
    return fig


def plot_model_comparison(results, output=None, *, style: PlotStyle | None = None):
    """Compare quality metrics with horizontal mean/error point plots."""
    style = style or PlotStyle()
    frame = _frame(results, ("model",))
    metrics = [metric for metric in style.quality_metrics if metric in frame]
    if not metrics:
        raise ValueError("No requested quality metrics are available")
    _numeric(frame, metrics)
    models = _models(frame)
    appearance = _appearance(models, style)
    height = max(style.figure_size[1], 2.0 + 0.48 * len(models))
    with _rc(style):
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(style.figure_size[0], 3.4 * len(metrics)),
                                                         height), squeeze=False, sharey=True)
        for ax, metric in zip(axes[0], metrics):
            for index, name in enumerate(models):
                values = frame.loc[frame["model"] == name, metric].to_numpy(dtype=float)
                mean = float(values.mean())
                error = _error(values, style)
                lower = min(error, mean)
                upper = min(error, 1 - mean) if metric in {"DL", "JW", "test_accuracy"} else error
                item = appearance[name]
                ax.errorbar(mean, index, xerr=np.array([[lower], [upper]]),
                            fmt=item["marker"], color=item["color"],
                            markerfacecolor=item["color"] if item["filled"] else "white",
                            markeredgecolor=item["color"], markeredgewidth=1.3,
                            markersize=style.marker_size + 1, capsize=3,
                            zorder=4, clip_on=False)
            ax.set_yticks(range(len(models)), models)
            ax.set_ylim(len(models) - 0.5, -0.5)
            ax.set_xlabel(METRIC_LABELS[metric])
            ax.set_title(METRIC_LABELS[metric], pad=12)
            ax.margins(x=0.15)
            _polish(ax, style)
            ax.grid(axis="x", color="#777777", alpha=0.13, linewidth=0.7)
        for ax in axes[0][1:]:
            ax.tick_params(labelleft=False)
        fig.suptitle("Prediction and rollout quality", fontsize=style.title_size,
                     fontweight="bold")
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.85, wspace=0.32)
    if output is not None:
        _save_direct(fig, output, style.dpi)
    return fig

def plot_rollout_error(results, output=None, *, style: PlotStyle | None = None):
    """Plot prefix DL over horizon, using consistent model styles."""
    style = style or PlotStyle()
    frame = _frame(results, ("model", "target_string", "forecast"))
    if any(
        not isinstance(row.target_string, str) or not isinstance(row.forecast, str)
        or not row.target_string or len(row.target_string) != len(row.forecast)
        for row in frame.itertuples()
    ):
        raise ValueError("target_string and forecast must be nonempty and equal length")
    models = _models(frame)
    appearance = _appearance(models, style)
    with _rc(style):
        fig, ax = plt.subplots(figsize=style.figure_size)
        for name in models:
            group = frame.loc[frame["model"] == name]
            horizons = {len(value) for value in group["target_string"]}
            if len(horizons) != 1:
                raise ValueError(f"rollout horizons differ within model {name!r}")
            horizon = horizons.pop()
            curves = np.asarray([
                [normalized_damerau_levenshtein_distance(target[:step], forecast[:step])
                 for step in range(1, horizon + 1)]
                for target, forecast in zip(group["target_string"], group["forecast"])
            ])
            mean = curves.mean(axis=0)
            x = np.arange(1, horizon + 1)
            item = appearance[name]
            ax.plot(x, mean, color=item["color"], linestyle=item["linestyle"],
                    linewidth=style.line_width, marker=item["marker"],
                    markevery=max(1, style.horizon_marker_step),
                    markersize=style.marker_size, markeredgewidth=1.2,
                    markerfacecolor=item["color"] if item["filled"] else "white", zorder=3)
            if len(curves) > 1 and style.uncertainty != "none":
                deviations = curves.std(axis=0, ddof=1)
                if style.uncertainty == "sem":
                    deviations /= np.sqrt(len(curves))
                ax.fill_between(x, np.maximum(0, mean - deviations), np.minimum(1, mean + deviations),
                                color=item["color"], alpha=0.1, linewidth=0)
        ax.set(xlabel="Rollout horizon", ylabel=METRIC_LABELS["DL"], xlim=(1, None), ylim=(0, 1))
        ax.set_title("Closed-loop rollout stability", pad=12)
        _polish(ax, style)
        _legend(fig, models, appearance, style)
        _finish(fig, style, with_legend=True)
    if output is not None:
        _save_direct(fig, output, style.dpi)
    return fig


def plot_tradeoffs(results, output=None, *, style: PlotStyle | None = None):
    """Plot DL against time, parameter count, and available CUDA memory."""
    style = style or PlotStyle()
    frame = _frame(results, ("model", "DL"))
    costs = [metric for metric in style.cost_metrics if metric in frame]
    if "memory_mb" in costs and not (pd.to_numeric(frame["memory_mb"], errors="coerce") > 0).any():
        costs.remove("memory_mb")
    if not costs:
        raise ValueError("No usable cost metrics are available")
    _numeric(frame, ["DL", *costs])
    models = _models(frame)
    appearance = _appearance(models, style)
    with _rc(style):
        fig, axes = plt.subplots(1, len(costs), figsize=(max(style.figure_size[0], 3.5 * len(costs)),
                                                       style.figure_size[1]), squeeze=False)
        for ax, metric in zip(axes[0], costs):
            for name in models:
                group = frame.loc[frame["model"] == name]
                xvalues = group[metric].to_numpy(dtype=float)
                yvalues = group["DL"].to_numpy(dtype=float)
                item = appearance[name]
                ax.errorbar(
                    float(xvalues.mean()), float(yvalues.mean()),
                    xerr=_error(xvalues, style), yerr=_error(yvalues, style),
                    fmt=item["marker"], color=item["color"],
                    markerfacecolor=item["color"] if item["filled"] else "white",
                    markeredgecolor=item["color"], markeredgewidth=1.3,
                    markersize=style.marker_size + 1, capsize=3, zorder=4, clip_on=False,
                )
            ax.set_xlabel(METRIC_LABELS[metric])
            ax.set_ylabel(METRIC_LABELS["DL"])
            ax.margins(x=0.15, y=0.15)
            if style.log_cost_axes and (frame[metric] > 0).all():
                ax.set_xscale("log")
            _polish(ax, style)
        fig.suptitle("Rollout quality versus resource cost", fontsize=style.title_size,
                     fontweight="bold")
        _legend(fig, models, appearance, style)
        _finish(fig, style, with_legend=True)
    if output is not None:
        _save_direct(fig, output, style.dpi)
    return fig


def plot_complexity_sweep(results, output=None, *, style: PlotStyle | None = None):
    """Plot predictive and rollout metrics against measured LZW code count."""
    style = style or PlotStyle()
    frame = _frame(results, ("model", "complexity"))
    metrics = [metric for metric in style.complexity_metrics if metric in frame]
    if not metrics:
        raise ValueError("No requested complexity metrics are available")
    _numeric(frame, ["complexity", *metrics])
    if frame["complexity"].nunique() < 2:
        raise ValueError("A complexity sweep requires at least two measured LZW values")
    models = _models(frame)
    appearance = _appearance(models, style)
    with _rc(style):
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(style.figure_size[0], 3.6 * len(metrics)),
                                                         style.figure_size[1]), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            for name in models:
                group = frame.loc[frame["model"] == name]
                summary = group.groupby("complexity", sort=True)[metric].agg(["mean", "std", "count"])
                x = summary.index.to_numpy(dtype=float)
                y = summary["mean"].to_numpy(dtype=float)
                item = appearance[name]
                ax.plot(x, y, color=item["color"], linestyle=item["linestyle"],
                        linewidth=style.line_width, marker=item["marker"],
                        markersize=style.marker_size, markeredgewidth=1.2,
                        markerfacecolor=item["color"] if item["filled"] else "white", zorder=3)
                if style.uncertainty != "none":
                    error = summary["std"].fillna(0).to_numpy(dtype=float)
                    if style.uncertainty == "sem":
                        error = error / np.sqrt(summary["count"].to_numpy(dtype=float))
                    ax.fill_between(x, np.maximum(0, y - error), y + error,
                                    color=item["color"], alpha=0.1, linewidth=0)
            ax.set_xlabel("Measured LZW code count")
            ax.set_ylabel(METRIC_LABELS[metric])
            _polish(ax, style)
        fig.suptitle("Performance across symbolic complexity", fontsize=style.title_size,
                     fontweight="bold")
        _legend(fig, models, appearance, style)
        _finish(fig, style, with_legend=True)
    if output is not None:
        _save_direct(fig, output, style.dpi)
    return fig


def _save_direct(fig, output, dpi):
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _source_frame(source):
    if isinstance(source, SavedRun):
        return source.results.copy(), source.path
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    path = Path(source).expanduser()
    if path.is_dir():
        saved = load_run(path)
        return saved.results, saved.path
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), None
    raise ValueError("source must be a SavedRun, DataFrame, run directory, or CSV file")


def _new_figures_dir(source_run, output_dir):
    if output_dir is not None:
        path = Path(output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    if source_run is not None:
        path = source_run / "figures"
        path.mkdir(exist_ok=True)
        return path
    base = Path(os.environ.get("ROTE_FIGURES_DIR", Path.cwd() / "rote_figures")).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / f"figures-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    return path.resolve()


def visualize_benchmark(
    source,
    *,
    output_dir: str | Path | None = None,
    style: PlotStyle | None = None,
    figures: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> VisualizationReport:
    """Save a coordinated figure set from a run, DataFrame, or CSV.

    With a saved run, figures default to ``<run>/figures``. Otherwise a new
    directory is created under ``./rote_figures`` (or ``ROTE_FIGURES_DIR``).
    ``figures`` can select ``quality``, ``rollout``, ``tradeoffs``, and
    ``complexity``. The default includes all that the data support. Existing
    figure files are protected unless ``overwrite=True``.
    """
    style = style or PlotStyle()
    frame, source_run = _source_frame(source)
    frame = _frame(frame, ("model",))
    available = ["quality"] if any(metric in frame for metric in style.quality_metrics) else []
    if {"target_string", "forecast"} <= set(frame):
        available.append("rollout")
    if "DL" in frame and any(
        metric in frame and (metric != "memory_mb" or
                             (pd.to_numeric(frame[metric], errors="coerce") > 0).any())
        for metric in style.cost_metrics
    ):
        available.append("tradeoffs")
    if "complexity" in frame and pd.to_numeric(frame["complexity"], errors="coerce").nunique() >= 2:
        available.append("complexity")
    requested = tuple(available) if figures is None else tuple(figures)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("figures must contain unique figure names")
    unknown = set(requested).difference({"quality", "rollout", "tradeoffs", "complexity"})
    if unknown:
        raise ValueError(f"Unknown figures: {sorted(unknown)}")
    missing = set(requested).difference(available)
    if missing:
        raise ValueError(f"Requested figures are not supported by these results: {sorted(missing)}")
    path = _new_figures_dir(source_run, output_dir)
    targets = {name: tuple(path / f"{name}.{fmt}" for fmt in style.formats) for name in requested}
    config_path = path / "visualization.json"
    existing = [file for paths in targets.values() for file in paths if file.exists()]
    if config_path.exists():
        existing.append(config_path)
    if existing and not overwrite:
        raise FileExistsError(f"Visualization files already exist in {path}; use overwrite=True")
    plotters = {
        "quality": plot_model_comparison,
        "rollout": plot_rollout_error,
        "tradeoffs": plot_tradeoffs,
        "complexity": plot_complexity_sweep,
    }
    for name in requested:
        fig = plotters[name](frame, style=style)
        try:
            for destination in targets[name]:
                temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.{destination.suffix[1:]}")
                try:
                    fig.savefig(temporary, dpi=style.dpi, bbox_inches="tight")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            plt.close(fig)
    _write_json(config_path, {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": str(source_run) if source_run is not None else None,
        "figures": list(requested),
        "style": asdict(style),
        "rows": len(frame),
    })
    return VisualizationReport(path=path, figures=targets, config_path=config_path)
