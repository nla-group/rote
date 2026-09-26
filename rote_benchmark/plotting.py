"""Generate publication-style figures for symbolic sequence experiments.

Every figure is drawn at the size it is printed at in the manuscript (6.5 in text width), so
font sizes are true point sizes. With LaTeX on the PATH the text is typeset in Times with
Computer Modern math, matching the manuscript; otherwise a Times-like fallback is used.
"""

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rote-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rote")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODEL_ORDER = [
    "LSTM",
    "GRU",
    "minGRU",
    "minLSTM",
    "Transformer",
    "LinearAttention",
    "Performer",
    "RWKV",
]

TEXT_WIDTH = 6.5  # inches, manuscript \textwidth
FONT_SIZE = 8
LINE_WIDTH = 1.1
MARKER_SIZE = 3.6
MARKER_EDGE_WIDTH = 0.8

MODEL_STYLES = {
    "LSTM": {"color": "#0072B2", "marker": "o", "linestyle": "-", "filled": True},
    "GRU": {"color": "#D55E00", "marker": "s", "linestyle": "--", "filled": False},
    "minGRU": {"color": "#009E73", "marker": "^", "linestyle": "-.", "filled": True},
    "minLSTM": {"color": "#CC79A7", "marker": "v", "linestyle": ":", "filled": False},
    "Transformer": {"color": "#E69F00", "marker": "D", "linestyle": (0, (5, 2)), "filled": True},
    "LinearAttention": {"color": "#117733", "marker": "<", "linestyle": (0, (6, 2, 1, 2)), "filled": False},
    "Performer": {"color": "#882255", "marker": ">", "linestyle": (0, (2, 2)), "filled": True},
    "RWKV": {"color": "#44AA99", "marker": "h", "linestyle": (0, (4, 2, 1, 2, 1, 2)), "filled": False},
}
MODEL_PALETTE = {model: MODEL_STYLES[model]["color"] for model in MODEL_ORDER}
METRIC_LABELS = {
    "test_accuracy": r"Next-token accuracy",
    "test_loss": r"Next-token cross-entropy",
    "DL": r"Normalized $\mathrm{DL}$ distance",
    "JW": r"$\mathrm{JW}$ similarity",
    "train_time": r"Training time (s)",
    "time_per_epoch": r"Time per epoch (s)",
    "memory_mb": r"Memory usage (MB)",
    "model_size_m": r"Trainable parameters, $|\theta|$ (M)",
}
# Two-line variants for the half-width panels, whose axes are too short for one line.
SHORT_AXIS_LABELS = {
    "test_loss": "Next-token\ncross-entropy",
    "DL": "Normalized\n" r"$\mathrm{DL}$ distance",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="exps/results_symbolic/results.csv",
        help="CSV file, merged CSV file, or directory containing results*.csv shards.",
    )
    parser.add_argument("--output-dir", default="exps/figures_symbolic")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_style():
    use_latex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    params = {
        "text.usetex": use_latex,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}" if use_latex else "",
        "font.family": "serif",
        "axes.titlesize": FONT_SIZE * 1.2,
        "axes.labelsize": FONT_SIZE,
        "font.size": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "lines.linewidth": 0.4,
        "figure.facecolor": "white",
    }
    if not use_latex:
        params["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
        params["mathtext.fontset"] = "stix"
    plt.rcParams.update(params)


def new_figure(width_fraction, height):
    """A print-sized figure: `width_fraction` of the text width by `height` inches."""
    return plt.figure(figsize=(width_fraction * TEXT_WIDTH, height), layout="constrained")


def load_results(path):
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("results_merged.csv"))
        files = candidates or sorted(path.glob("results*.csv"))
        if not files:
            raise FileNotFoundError(f"No result CSV files found in {path}")
        df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    else:
        df = pd.read_csv(path)
    return normalize_columns(df)


def normalize_columns(df):
    df = df.copy()
    aliases = {"model_size": "model_size_m", "memory": "memory_mb", "lr": "learning_rate", "wd": "weight_decay"}
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "model_params" not in df.columns and "model_size_m" in df.columns:
        df["model_params"] = df["model_size_m"] * 1e6
    for column in ["model", "symbols", "complexity"]:
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")
    seen_models = list(dict.fromkeys(df["model"].astype(str)))
    present_models = [model for model in MODEL_ORDER if model in seen_models]
    present_models.extend(model for model in seen_models if model not in present_models)
    df["model"] = pd.Categorical(df["model"].astype(str), categories=present_models, ordered=True)
    return df.sort_values(["model", "symbols", "complexity"]).reset_index(drop=True)


def aggregate(df, metrics):
    keys = [key for key in ["model", "symbols", "complexity", "sequence_length", "window_size"] if key in df.columns]
    available = [metric for metric in metrics if metric in df.columns]
    grouped = df.groupby(keys, observed=True)[available]
    mean = grouped.mean().reset_index()
    sem = grouped.sem().reset_index()
    for metric in available:
        mean[f"{metric}_sem"] = sem[metric].fillna(0.0)
    return mean


def save_figure(fig, output_dir, stem, formats, dpi):
    # Constrained layout cannot shrink a legend wider than the figure; it gets clipped instead.
    renderer = fig.canvas.get_renderer()
    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        if box.x0 < fig.bbox.x0 or box.x1 > fig.bbox.x1:
            raise RuntimeError(f"{stem}: legend is wider than the figure")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Each save re-runs constrained layout starting from the previous save's positions, so the
    # PDF goes first to come out the same whichever other formats are requested.
    for fmt in sorted(formats, key=lambda fmt: fmt != "pdf"):
        # Figures are print-sized, so no tight bounding box; no timestamp keeps reruns identical.
        metadata = {"CreationDate": None} if fmt == "pdf" else None
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, metadata=metadata)
    plt.close(fig)


def model_style(model):
    default = {"color": "#4C4C4C", "marker": "o", "linestyle": "-", "filled": True}
    return MODEL_STYLES.get(str(model), default)


def marker_facecolor(style):
    return style["color"] if style.get("filled", True) else "white"


def ordered_models(values):
    present = {str(value) for value in values}
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def legend_handle(model, markers=True):
    style = model_style(model)
    return Line2D(
        [0],
        [0],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=LINE_WIDTH,
        marker=style["marker"] if markers else None,
        markersize=MARKER_SIZE,
        markerfacecolor=marker_facecolor(style),
        markeredgecolor=style["color"],
        markeredgewidth=MARKER_EDGE_WIDTH,
        label=str(model),
    )


def place_bottom_legend(fig, models, ncol, markers=True, compact=False):
    fig.legend(
        handles=[legend_handle(model, markers) for model in models],
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        handlelength=1.9 if compact else 2.6,
        handletextpad=0.3 if compact else 0.4,
        columnspacing=0.55 if compact else 1.0,
        borderaxespad=0.0,
    )


def style_axes(ax, xlabel, ylabel=None, x_from_zero=False):
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if x_from_zero:
        ax.set_xlim(left=0)
    ax.grid(True, which="major", color="grey", alpha=0.2, linewidth=0.3)
    ax.tick_params(which="major", axis="y", direction="in", width=0.5, color="grey")
    ax.tick_params(which="major", axis="x", direction="in", width=0.5, color="grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("grey")
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("grey")
    ax.spines["left"].set_linewidth(0.5)
    if not x_from_zero:
        ax.margins(x=0.035)


def plot_model_series(ax, data, x, y, yerr=None, show_markers=True, markevery=None):
    for model in ordered_models(data["model"]):
        model_data = data[data["model"].astype(str) == model].sort_values(x)
        if model_data.empty:
            continue
        style = model_style(model)
        ax.plot(
            model_data[x],
            model_data[y],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            marker=style["marker"] if show_markers else None,
            markersize=MARKER_SIZE,
            markerfacecolor=marker_facecolor(style),
            markeredgecolor=style["color"],
            markeredgewidth=MARKER_EDGE_WIDTH,
            markevery=markevery,
            alpha=0.98,
            zorder=3,
        )
        if yerr and yerr in model_data:
            ax.errorbar(
                model_data[x],
                model_data[y],
                yerr=model_data[yerr],
                fmt="none",
                color=style["color"],
                alpha=0.42,
                capsize=1.6,
                capthick=0.5,
                elinewidth=0.6,
                zorder=1,
            )


def plot_metric_vs_complexity(df, metric, ylabel, output_dir, formats, dpi):
    if metric not in df.columns:
        return
    summary = aggregate(df, [metric])
    symbols = sorted(summary["symbols"].dropna().unique())
    fig = new_figure(1.0, 1.87)
    axes = np.atleast_1d(fig.subplots(1, len(symbols), sharey=True))
    for index, (ax, symbol_count) in enumerate(zip(axes, symbols)):
        sub = summary[summary["symbols"] == symbol_count]
        plot_model_series(ax, sub, "complexity", metric, yerr=f"{metric}_sem")
        ax.set_title(rf"$|\Sigma| = {int(symbol_count)}$")
        ax.set_xticks(sorted(summary["complexity"].unique()))
        style_axes(ax, r"$\mathrm{LZW}$ complexity, $c$", ylabel if index == 0 else None)
    place_bottom_legend(fig, ordered_models(summary["model"]), ncol=8, compact=True)
    save_figure(fig, output_dir, f"{metric}_vs_lzw_complexity", formats, dpi)


def plot_rollout_error_by_horizon(df, output_dir, formats, dpi):
    required = {"target_string", "forecast", "model"}
    if not required.issubset(df.columns):
        return
    rows = []
    for _, row in df.dropna(subset=["target_string", "forecast"]).iterrows():
        target = str(row["target_string"])
        forecast = str(row["forecast"])
        horizon = min(len(target), len(forecast))
        if horizon == 0:
            continue
        misses = np.fromiter((target[i] != forecast[i] for i in range(horizon)), dtype=float)
        cumulative = np.cumsum(misses) / np.arange(1, horizon + 1)
        for step, value in enumerate(cumulative, start=1):
            rows.append({"model": row["model"], "step": step, "cumulative_error": value})
    if not rows:
        return
    horizon_df = pd.DataFrame(rows)
    summary = horizon_df.groupby(["model", "step"], observed=True)["cumulative_error"].mean().reset_index()
    fig = new_figure(0.62, 2.46)
    ax = fig.subplots()
    plot_model_series(ax, summary, "step", "cumulative_error", markevery=10)
    style_axes(ax, r"Forecast horizon, $k$", r"Cumulative rollout error", x_from_zero=True)
    ax.set_ylim(0, min(1.0, max(0.05, summary["cumulative_error"].max() * 1.15)))
    place_bottom_legend(fig, ordered_models(summary["model"]), ncol=4)
    save_figure(fig, output_dir, "rollout_error_vs_horizon", formats, dpi)


def plot_pareto(df, output_dir, formats, dpi):
    if not {"train_time", "DL", "model_size_m"}.issubset(df.columns):
        return
    summary = df.groupby("model", observed=True).agg(
        train_time=("train_time", "median"), dl=("DL", "median"), model_size_m=("model_size_m", "median")
    ).reset_index()
    fig = new_figure(0.495, 2.05)
    ax = fig.subplots()
    sizes = 12 + 60 * summary["model_size_m"] / max(summary["model_size_m"].max(), 1e-9)
    for idx, row in summary.iterrows():
        style = model_style(row["model"])
        ax.scatter(
            row["train_time"],
            row["dl"],
            s=sizes.iloc[idx],
            marker=style["marker"],
            facecolor=marker_facecolor(style),
            edgecolor=style["color"],
            linewidth=0.8,
            alpha=0.88,
        )
    style_axes(ax, r"Median training time (s)", "Median normalized\n" r"$\mathrm{DL}$ distance")
    if summary["train_time"].min() > 0:
        ax.set_xscale("log")
    place_bottom_legend(fig, ordered_models(summary["model"]), ncol=4, compact=True)
    save_figure(fig, output_dir, "compute_performance_pareto", formats, dpi)


def plot_scaling(df, output_dir, formats, dpi):
    candidates = [
        ("model_params", "test_loss", r"Model parameters, $|\theta|$", "test_loss_vs_model_params"),
        ("sequence_length", "test_loss", r"Training sequence length, $N$", "test_loss_vs_sequence_length"),
        ("model_params", "DL", r"Model parameters, $|\theta|$", "dl_vs_model_params"),
    ]
    for x_col, y_col, xlabel, stem in candidates:
        if not {x_col, y_col}.issubset(df.columns):
            continue
        summary = df.groupby(["model", x_col], observed=True)[y_col].mean().reset_index()
        if summary[x_col].nunique() < 2:
            continue
        fig = new_figure(0.495, 1.93)
        ax = fig.subplots()
        plot_model_series(ax, summary, x_col, y_col)
        style_axes(ax, xlabel, SHORT_AXIS_LABELS[y_col])
        if summary[x_col].min() > 0:
            ax.set_xscale("log")
        if y_col == "test_loss" and summary[y_col].min() > 0:
            ax.set_yscale("log")
        place_bottom_legend(fig, ordered_models(summary["model"]), ncol=4, compact=True)
        save_figure(fig, output_dir, stem, formats, dpi)


def plot_context_sensitivity(df, output_dir, formats, dpi):
    if not {"window_size", "DL"}.issubset(df.columns) or df["window_size"].nunique() < 2:
        return
    summary = df.groupby(["model", "window_size"], observed=True)["DL"].mean().reset_index()
    fig = new_figure(0.62, 2.46)
    ax = fig.subplots()
    plot_model_series(ax, summary, "window_size", "DL")
    style_axes(ax, r"Context window size, $w$", METRIC_LABELS["DL"])
    place_bottom_legend(fig, ordered_models(summary["model"]), ncol=4)
    save_figure(fig, output_dir, "context_sensitivity", formats, dpi)


def plot_metric_heatmap(df, metric, output_dir, formats, dpi):
    if metric not in df.columns:
        return
    import seaborn as sns

    summary = df.groupby(["model", "complexity"], observed=True)[metric].mean().reset_index()
    pivot = summary.pivot(index="model", columns="complexity", values=metric)
    pivot = pivot.reindex([model for model in MODEL_ORDER if model in pivot.index])
    fig = new_figure(0.62, 2.6)
    ax = fig.subplots()
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlGnBu_r" if metric in {"DL", "test_loss"} else "YlGnBu",
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": FONT_SIZE - 1},
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
    )
    ax.set_xlabel(r"$\mathrm{LZW}$ complexity, $c$")
    ax.set_ylabel("Model")
    ax.tick_params(axis="both", length=0)
    save_figure(fig, output_dir, f"{metric}_complexity_heatmap", formats, dpi)


def write_summary(df, output_dir):
    metrics = [
        metric
        for metric in ["test_loss", "test_accuracy", "DL", "JW", "train_time", "time_per_epoch", "memory_mb", "model_size_m"]
        if metric in df.columns
    ]
    summary = df.groupby("model", observed=True)[metrics].agg(["median", "mean", "std"]).round(5)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary_by_model.csv")


def main():
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    df = load_results(args.results)
    write_summary(df, output_dir)
    for metric, ylabel in [
        ("test_accuracy", METRIC_LABELS["test_accuracy"]),
        ("test_loss", METRIC_LABELS["test_loss"]),
        ("DL", METRIC_LABELS["DL"]),
        ("JW", METRIC_LABELS["JW"]),
    ]:
        plot_metric_vs_complexity(df, metric, ylabel, output_dir, args.formats, args.dpi)
    plot_rollout_error_by_horizon(df, output_dir, args.formats, args.dpi)
    plot_pareto(df, output_dir, args.formats, args.dpi)
    plot_scaling(df, output_dir, args.formats, args.dpi)
    plot_context_sensitivity(df, output_dir, args.formats, args.dpi)
    for metric in ["DL", "JW", "test_accuracy", "test_loss"]:
        plot_metric_heatmap(df, metric, output_dir, args.formats, args.dpi)
    print(f"Wrote figures and summary to {output_dir}")


if __name__ == "__main__":
    main()
