"""Generate publication-style figures for symbolic sequence experiments."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/slearn-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-slearn")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_ORDER = [
    "LSTM",
    "GRU",
    "minGRU",
    "minLSTM",
    "Transformer",
    "BERT",
    "GPT",
    "LinearAttention",
    "Performer",
    "RWKV",
]

FONT = {"title": 16, "label": 14, "tick": 12, "legend": 12, "annotation": 10}
FONT_WEIGHT = "semibold"
LINE_WIDTH = 2.25
MARKER_SIZE = 8.6
MARKER_EDGE_WIDTH = 1.45

MODEL_STYLES = {
    "LSTM": {"color": "#0072B2", "marker": "o", "linestyle": "-", "filled": True},
    "GRU": {"color": "#D55E00", "marker": "s", "linestyle": "--", "filled": False},
    "minGRU": {"color": "#009E73", "marker": "^", "linestyle": "-.", "filled": True},
    "minLSTM": {"color": "#CC79A7", "marker": "v", "linestyle": ":", "filled": False},
    "Transformer": {"color": "#E69F00", "marker": "D", "linestyle": (0, (5, 2)), "filled": True},
    "BERT": {"color": "#56B4E9", "marker": "P", "linestyle": (0, (1, 1)), "filled": False},
    "GPT": {"color": "#332288", "marker": "X", "linestyle": (0, (3, 1, 1, 1)), "filled": True},
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

DEFAULT_LAYOUT = {"figsize": (6.8, 4.2), "legend_y": -0.05, "bottom": 0.28, "legend_ncol": 5}
FIGURE_LAYOUTS = {
    "rollout_error_vs_horizon": {"figsize": (7.4, 4.7), "legend_y": 0.12, "bottom": 0.24, "legend_ncol": 5},
    "compute_performance_pareto": {"figsize": (7.0, 4.9), "legend_y": 0.12, "bottom": 0.24, "legend_ncol": 5},
    "test_loss_vs_model_params": {"figsize": (7.0, 4.6), "legend_y": 0.02, "bottom": 0.24, "legend_ncol": 5},
    "dl_vs_model_params": {"figsize": (7.0, 4.6), "legend_y": -.02, "bottom": 0.24, "legend_ncol": 5},
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
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "axes.titlesize": FONT["title"],
            "axes.labelsize": FONT["label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
            "legend.title_fontsize": FONT["legend"],
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": FONT["tick"],
            "font.weight": FONT_WEIGHT,
            "axes.titleweight": FONT_WEIGHT,
            "axes.labelweight": FONT_WEIGHT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "figure.constrained_layout.use": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "mathtext.fontset": "stix",
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


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
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_layout(stem):
    layout = DEFAULT_LAYOUT.copy()
    layout.update(FIGURE_LAYOUTS.get(stem, {}))
    return layout


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


def legend_handle(model):
    from matplotlib.lines import Line2D

    style = model_style(model)
    return Line2D(
        [0],
        [0],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=LINE_WIDTH,
        marker=style["marker"],
        markersize=MARKER_SIZE,
        markerfacecolor=marker_facecolor(style),
        markeredgecolor=style["color"],
        markeredgewidth=MARKER_EDGE_WIDTH,
        label=str(model),
    )


def plot_model_series(
    ax,
    data,
    x,
    y,
    yerr=None,
    show_markers=True,
    marker_size=MARKER_SIZE,
    marker_edge_width=MARKER_EDGE_WIDTH,
    markevery=None,
):
    for model in ordered_models(data["model"]):
        model_data = data[data["model"].astype(str) == model].sort_values(x)
        if model_data.empty:
            continue
        style = model_style(model)
        ax.plot(
            model_data[x],
            model_data[y],
            label=model,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            marker=style["marker"] if show_markers else None,
            markersize=marker_size if show_markers else 0,
            markerfacecolor=marker_facecolor(style) if show_markers else "none",
            markeredgecolor=style["color"] if show_markers else "none",
            markeredgewidth=marker_edge_width if show_markers else 0,
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
                capsize=3.2,
                capthick=1.0,
                elinewidth=1.05,
                zorder=1,
            )


def place_bottom_legend(fig, axes, title="Model", layout=None):
    layout = layout or DEFAULT_LAYOUT
    labels = []
    for ax in np.ravel(axes):
        _, axis_labels = ax.get_legend_handles_labels()
        for label in axis_labels:
            if label and not label.startswith("_") and label not in labels:
                labels.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    labels = ordered_models(labels)
    if labels:
        fig.legend(
            [legend_handle(label) for label in labels],
            labels,
            title=title,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=min(5, len(labels)),
            frameon=False,
            prop={"size": FONT["legend"], "weight": FONT_WEIGHT},
            handlelength=3.0,
            handletextpad=0.55,
            columnspacing=1.25,
            borderaxespad=0.0,
            markerscale=1.08,
        )
        fig.subplots_adjust(bottom=layout["bottom"])


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=FONT["label"], fontweight=FONT_WEIGHT, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=FONT["label"], fontweight=FONT_WEIGHT, labelpad=9)
    ax.tick_params(axis="both", labelsize=FONT["tick"], width=0.9, length=4.5)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight(FONT_WEIGHT)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.22, linestyle="--", linewidth=0.75)
    ax.grid(True, axis="x", alpha=0.10, linestyle="--", linewidth=0.65)
    ax.margins(x=0.035)


def plot_metric_vs_complexity(df, metric, ylabel, output_dir, formats, dpi):
    if metric not in df.columns:
        return
    summary = aggregate(df, [metric])
    symbols = sorted(summary["symbols"].dropna().unique())
    fig, axes = plt.subplots(1, len(symbols), figsize=(4.2 * len(symbols), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, symbol_count in zip(axes, symbols):
        sub = summary[summary["symbols"] == symbol_count]
        plot_model_series(ax, sub, "complexity", metric, yerr=f"{metric}_sem")
        ax.set_title(rf"$|\Sigma| = {int(symbol_count)}$", fontsize=FONT["title"], fontweight=FONT_WEIGHT, pad=10)
        style_axes(ax, r"$\mathrm{LZW}$ complexity, $c$", ylabel)
    place_bottom_legend(fig, axes)
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
    stem = "rollout_error_vs_horizon"
    layout = figure_layout(stem)
    fig, ax = plt.subplots(figsize=layout["figsize"])
    plot_model_series(ax, summary, "step", "cumulative_error", markevery=10)
    style_axes(ax, r"Forecast horizon, $k$", r"Cumulative rollout error")
    ax.set_ylim(0, min(1.0, max(0.05, summary["cumulative_error"].max() * 1.15)))
    place_bottom_legend(fig, [ax], layout=layout)
    save_figure(fig, output_dir, stem, formats, dpi)


def plot_pareto(df, output_dir, formats, dpi):
    if not {"train_time", "DL", "model_size_m"}.issubset(df.columns):
        return
    summary = df.groupby("model", observed=True).agg(
        train_time=("train_time", "median"), dl=("DL", "median"), model_size_m=("model_size_m", "median")
    ).reset_index()
    stem = "compute_performance_pareto"
    layout = figure_layout(stem)
    fig, ax = plt.subplots(figsize=layout["figsize"])
    sizes = 90 + 440 * summary["model_size_m"] / max(summary["model_size_m"].max(), 1e-9)
    for idx, row in summary.iterrows():
        model = str(row["model"])
        style = model_style(model)
        ax.scatter(
            row["train_time"],
            row["dl"],
            s=sizes.iloc[idx],
            marker=style["marker"],
            facecolor=marker_facecolor(style),
            edgecolor=style["color"],
            linewidth=1.35,
            alpha=0.88,
            label=model,
        )
    style_axes(ax, r"Median training time (s)", r"Median normalized $\mathrm{DL}$ distance")
    if summary["train_time"].min() > 0:
        ax.set_xscale("log")
    place_bottom_legend(fig, [ax], layout=layout)
    save_figure(fig, output_dir, stem, formats, dpi)


def plot_scaling(df, output_dir, formats, dpi):
    candidates = [
        ("model_params", "test_loss", r"Model parameters, $|\theta|$", METRIC_LABELS["test_loss"], "test_loss_vs_model_params"),
        ("sequence_length", "test_loss", r"Training sequence length, $N$", METRIC_LABELS["test_loss"], "test_loss_vs_sequence_length"),
        ("model_params", "DL", r"Model parameters, $|\theta|$", METRIC_LABELS["DL"], "dl_vs_model_params"),
    ]
    for x_col, y_col, xlabel, ylabel, stem in candidates:
        if not {x_col, y_col}.issubset(df.columns):
            continue
        summary = df.groupby(["model", x_col], observed=True)[y_col].mean().reset_index()
        if summary[x_col].nunique() < 2:
            continue
        layout = figure_layout(stem)
        fig, ax = plt.subplots(figsize=layout["figsize"])
        plot_model_series(ax, summary, x_col, y_col)
        style_axes(ax, xlabel, ylabel)
        if summary[x_col].min() > 0:
            ax.set_xscale("log")
        if y_col == "test_loss" and summary[y_col].min() > 0:
            ax.set_yscale("log")
        place_bottom_legend(fig, [ax], layout=layout)
        save_figure(fig, output_dir, stem, formats, dpi)


def plot_context_sensitivity(df, output_dir, formats, dpi):
    if not {"window_size", "DL"}.issubset(df.columns) or df["window_size"].nunique() < 2:
        return
    summary = df.groupby(["model", "window_size"], observed=True)["DL"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    plot_model_series(ax, summary, "window_size", "DL")
    style_axes(ax, r"Context window size, $w$", r"Normalized $\mathrm{DL}$ distance")
    place_bottom_legend(fig, [ax])
    save_figure(fig, output_dir, "context_sensitivity", formats, dpi)


def plot_metric_heatmap(df, metric, output_dir, formats, dpi):
    if metric not in df.columns:
        return
    summary = df.groupby(["model", "complexity"], observed=True)[metric].mean().reset_index()
    pivot = summary.pivot(index="model", columns="complexity", values=metric)
    pivot = pivot.reindex([model for model in MODEL_ORDER if model in pivot.index])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlGnBu_r" if metric in {"DL", "test_loss"} else "YlGnBu",
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": FONT["annotation"], "fontweight": FONT_WEIGHT},
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
    )
    style_axes(ax, r"$\mathrm{LZW}$ complexity, $c$", "Model")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT["tick"])
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight(FONT_WEIGHT)
    cbar.set_label(METRIC_LABELS.get(metric, metric), fontsize=FONT["label"], fontweight=FONT_WEIGHT)
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
