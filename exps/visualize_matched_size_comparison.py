"""Generate the matched-size robustness comparison figure."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from visualize_symbolic_results import MODEL_ORDER, configure_style, load_results, save_figure


MATCHED_SIZE_COMPARISON_FONT_SIZE = 7.0
MATCHED_SIZE_COMPARISON_FONT_WEIGHT = "semibold"


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


def style_matched_comparison_axes(ax, title):
    font_size = MATCHED_SIZE_COMPARISON_FONT_SIZE
    font_weight = MATCHED_SIZE_COMPARISON_FONT_WEIGHT
    ax.set_title(title, fontsize=font_size, fontweight=font_weight, pad=8)
    ax.set_xlabel("Model", fontsize=font_size, fontweight=font_weight, labelpad=7)
    ax.tick_params(axis="both", labelsize=font_size, width=0.65, length=3.5)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.28, linestyle="-", linewidth=0.45)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_zorder(0)
    ax.spines["bottom"].set_zorder(0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight(font_weight)


def comparison_model_label(model):
    labels = {"LinearAttention": "LinearAttn"}
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
        "Fixed budget": {"color": "#1B4F9C", "marker": "o", "facecolor": "white", "markeredgewidth": 1.1},
        "Matched size": {"color": "#D95F02", "marker": "D", "facecolor": "#D95F02", "markeredgewidth": 0.9},
    }
    panels = [
        ("DL_mean", "DL_std", r"$\mathrm{DL}$ distance"),
        ("test_loss_mean", "test_loss_std", r"Test loss"),
    ]

    font_size = MATCHED_SIZE_COMPARISON_FONT_SIZE
    font_weight = MATCHED_SIZE_COMPARISON_FONT_WEIGHT
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.45))
    fig.suptitle(
        r"Fixed-budget versus matched-size results at high $\mathrm{LZW}$ complexity",
        fontsize=font_size + 2,
        fontweight=font_weight,
        x=0.53,
        y=0.928,
    )

    for ax, (mean_col, std_col, title) in zip(axes, panels):
        track_subsets = {
            track: summary[summary["track"].astype(str) == track].set_index("model").reindex(models)
            for track in track_styles
        }
        for model_index in range(len(models)):
            y_pair = [
                track_subsets["Fixed budget"].iloc[model_index][mean_col],
                track_subsets["Matched size"].iloc[model_index][mean_col],
            ]
            ax.plot(
                [x[model_index] + offsets["Fixed budget"], x[model_index] + offsets["Matched size"]],
                y_pair,
                color="#B8B8B8",
                linewidth=0.55,
                zorder=1,
                clip_on=False,
            )
        for track, style in track_styles.items():
            subset = track_subsets[track]
            positions = x + offsets[track]
            values = subset[mean_col].to_numpy(dtype=float)
            std = subset[std_col].fillna(0.0).to_numpy(dtype=float)
            lower_err = np.minimum(std, np.maximum(values, 0.0))
            yerr = np.vstack([lower_err, std])
            ax.errorbar(
                positions,
                values,
                yerr=yerr,
                fmt=style["marker"],
                markersize=5.5,
                color=style["color"],
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
                elinewidth=0.7,
                capsize=2.4,
                linestyle="none",
                label=track,
                zorder=5,
                clip_on=False,
            )
        upper = 0.0
        for subset in track_subsets.values():
            upper = max(upper, (subset[mean_col] + subset[std_col].fillna(0.0)).max())
        y_top = max(upper * 1.12, 1e-3)
        ax.set_ylim(bottom=0.0, top=y_top)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [comparison_model_label(model) for model in models],
            fontsize=font_size,
            fontweight=font_weight,
        )
        style_matched_comparison_axes(ax, title)
        ax.margins(x=0.03)

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
            markersize=5.5,
            label=track,
        )
        for track, style in track_styles.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.108),
        ncol=2,
        frameon=False,
        prop={"size": font_size, "weight": font_weight},
        handletextpad=0.55,
        columnspacing=1.9,
        borderaxespad=0.0,
    )
    fig.text(
        0.5,
        0.065,
        rf"Points show means; vertical bars show one standard deviation over runs at $c={complexity}$ and $n\in\{{{','.join(str(symbol) for symbol in symbols)}\}}$.",
        ha="center",
        va="center",
        fontsize=font_size + 2,
        fontweight=font_weight,
        color="#404040",
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.265, top=0.835, wspace=0.26)
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
