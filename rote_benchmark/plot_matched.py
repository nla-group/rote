"""Generate the matched-size robustness comparison figure."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rote-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rote")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .plotting import FONT_SIZE, MODEL_ORDER, configure_style, load_results, new_figure, save_figure, style_axes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-budget-results", required=True)
    parser.add_argument("--matched-size-results", required=True)
    parser.add_argument("--output-dir", default="exps/figures_symbolic_matched_size")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--comparison-complexity", type=int, default=90)
    parser.add_argument("--comparison-symbols", nargs="+", type=int, default=[4, 8])
    return parser.parse_args()


def comparison_model_label(model):
    labels = {"LinearAttn": "LinearAttention", "LinearAttention": "LinearAttention"}
    return labels.get(str(model), str(model))


def summarize_matched_comparison(fixed_df, matched_df, complexity, symbols):
    symbol_set = {int(symbol) for symbol in symbols}
    fixed = fixed_df[
        (fixed_df["complexity"].astype(int) == int(complexity))
        & fixed_df["symbols"].astype(int).isin(symbol_set)
    ].copy()
    matched = matched_df[
        (matched_df["complexity"].astype(int) == int(complexity))
        & matched_df["symbols"].astype(int).isin(symbol_set)
    ].copy()
    models = [
        model
        for model in MODEL_ORDER
        if model in set(fixed["model"].astype(str)) and model in set(matched["model"].astype(str))
    ]
    fixed = fixed[fixed["model"].astype(str).isin(models)]
    matched = matched[matched["model"].astype(str).isin(models)]
    rows = []
    alphabet_label = ";".join(str(symbol) for symbol in sorted(symbol_set))
    for track, data in [("Fixed budget", fixed), ("Matched size", matched)]:
        for model in models:
            subset = data[data["model"].astype(str) == model]
            if subset.empty:
                continue
            rows.append(
                {
                    "track": track,
                    "model": model,
                    "alphabets": alphabet_label,
                    "DL_mean": subset["DL"].mean(),
                    "DL_std": subset["DL"].std(ddof=1),
                    "test_loss_mean": subset["test_loss"].mean(),
                    "test_loss_std": subset["test_loss"].std(ddof=1),
                }
            )
    columns = ["track", "model", "alphabets", "DL_mean", "DL_std", "test_loss_mean", "test_loss_std"]
    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        return summary
    summary["model"] = pd.Categorical(summary["model"], categories=models, ordered=True)
    summary["track"] = pd.Categorical(summary["track"], categories=["Fixed budget", "Matched size"], ordered=True)
    return summary.sort_values(["model", "track"]).reset_index(drop=True)


def plot_matched_size_comparison(fixed_df, matched_df, output_dir, formats, dpi, complexity=90, symbols=(4, 8)):
    required = {"model", "symbols", "complexity", "DL", "test_loss"}
    if not required.issubset(fixed_df.columns) or not required.issubset(matched_df.columns):
        missing = required.difference(fixed_df.columns).union(required.difference(matched_df.columns))
        raise ValueError(f"Cannot build matched-size comparison; missing columns: {sorted(missing)}")
    summary = summarize_matched_comparison(fixed_df, matched_df, complexity, symbols)
    if summary.empty:
        raise ValueError("No overlapping rows found for the requested matched-size comparison.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "matched_size_fixed_budget_comparison.csv", index=False)

    models = [model for model in MODEL_ORDER if model in set(summary["model"].astype(str))]
    x = np.arange(len(models))
    offsets = {"Fixed budget": -0.12, "Matched size": 0.12}
    track_styles = {
        "Fixed budget": {"color": "#1B4F9C", "marker": "o", "facecolor": "white", "markeredgewidth": 0.9},
        "Matched size": {"color": "#D95F02", "marker": "D", "facecolor": "#D95F02", "markeredgewidth": 0.7},
    }
    panels = [
        ("DL_mean", "DL_std", r"$\mathrm{DL}$ distance"),
        ("test_loss_mean", "test_loss_std", r"Test loss"),
    ]

    fig = new_figure(1.0, 3.57)
    axes = fig.subplots(1, 2)
    fig.suptitle(r"Fixed-budget versus matched-size results at high $\mathrm{LZW}$ complexity")

    for ax, (mean_col, std_col, title) in zip(axes, panels):
        track_subsets = {
            track: summary[summary["track"].astype(str) == track].set_index("model").reindex(models)
            for track in track_styles
        }
        means = {track: subset[mean_col].to_numpy(dtype=float) for track, subset in track_subsets.items()}
        stds = {track: subset[std_col].fillna(0.0).to_numpy(dtype=float) for track, subset in track_subsets.items()}
        for model_index in range(len(models)):
            ax.plot(
                [x[model_index] + offsets["Fixed budget"], x[model_index] + offsets["Matched size"]],
                [means["Fixed budget"][model_index], means["Matched size"][model_index]],
                color="#B8B8B8",
                linewidth=0.5,
                zorder=1,
                clip_on=False,
            )
        for track, style in track_styles.items():
            lower_err = np.minimum(stds[track], np.maximum(means[track], 0.0))
            ax.errorbar(
                x + offsets[track],
                means[track],
                yerr=np.vstack([lower_err, stds[track]]),
                fmt=style["marker"],
                markersize=4.2,
                color=style["color"],
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
                elinewidth=0.7,
                capsize=2.0,
                linestyle="none",
                zorder=5,
                clip_on=False,
            )
        upper = max((means[track] + stds[track]).max() for track in track_styles)
        ax.set_ylim(0.0, max(upper * 1.12, 1e-3))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [comparison_model_label(model) for model in models], rotation=18, ha="right", rotation_mode="anchor"
        )
        ax.set_title(title)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axes(ax, "Model")
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="none",
            color=style["color"],
            markerfacecolor=style["facecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=style["markeredgewidth"],
            markersize=4.2,
            label=track,
        )
        for track, style in track_styles.items()
    ]
    note = (
        r"Points show means; vertical bars show one standard deviation over runs at "
        rf"$c={complexity}$ and $n\in\{{{','.join(str(symbol) for symbol in symbols)}\}}$."
    )
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        title=note,
        title_fontsize=FONT_SIZE,
        columnspacing=1.9,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(wspace=0.08)
    save_figure(fig, output_dir, "matched_size_comparison", formats, dpi)


def main():
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir)
    fixed_df = load_results(args.fixed_budget_results)
    matched_df = load_results(args.matched_size_results)
    plot_matched_size_comparison(
        fixed_df,
        matched_df,
        output_dir,
        args.formats,
        args.dpi,
        complexity=args.comparison_complexity,
        symbols=args.comparison_symbols,
    )
    print(f"Wrote matched-size comparison to {output_dir}")


if __name__ == "__main__":
    main()
