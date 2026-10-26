"""Step 3: Heatmap of score_std.csv: macro concept x difficulty level.

Rows are macro concepts (long labels read best on the y-axis), columns are
levels in easy -> medium -> hard order, and each cell is colored by a value
column (std_norm by default) on a pale -> ACM blue ramp. Each cell shows the
value and n (the number of question ids behind it); a missing combination is
drawn as a gray "-" cell. The figure is sized for one ACM column. The ids of
each block are in score_std.csv and block_ids_in_metrics.csv; --show-ids also
prints them in the cells (full ACM text width, for a figure* or a check).

Row order: ROW_ORDER below (or --order); macro concepts not listed follow,
sorted by mean value, with Other last unless placed explicitly. An empty
ROW_ORDER sorts every row by mean value.

    python plot_score_std.py --file run_scores/score_std.csv --outdir run_scores
    python plot_score_std.py --value mean        # any numeric column
    python plot_score_std.py --order "Data Understanding" "Data Summary" PCA
    python plot_score_std.py --show-ids          # wide version with ids in cells
    python plot_score_std.py --file run_scores/score_std_macro2.csv   # one version only

By default both versions are drawn: score_std.csv (macro_concept rows) ->
score_<value>_heatmap.*, and score_std_macro2.csv (macro2_concept rows, order
ROW_ORDER_MACRO2) -> score_<value>_heatmap_macro2.*.

Writes <outdir>/score_<value>_heatmap.png / .pdf (--show-ids:
score_<value>_heatmap_ids.png / .pdf) and <outdir>/block_ids_in_metrics.csv, the
ids of each block that already appear in the jsonl files of --metrics-dir
(default .metrics/).
"""

import argparse
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from plot_concept_levels import (
    ACM_BLUE,
    ACM_DARK_BLUE,
    LEVEL_ORDER,
    UNMAPPED_MACRO,
    metrics_ids,
    sort_ids,
)
from round_plot_style import MUTED_TEXT, PRIMARY_TEXT

ACM_RAMP = LinearSegmentedColormap.from_list("acm_blue", ["#eaf8fc", ACM_BLUE, ACM_DARK_BLUE])
MISSING_COLOR = "#ececea"
VALUE_LABELS = {
    "std_norm": "Normalized std of score",
    "std": "Std of score",
    "mean": "Mean score",
}
ACM_COLUMN_WIDTH = 3.33  # inches, one column
ACM_TEXT_WIDTH = 7.0      # inches, full width (figure*)
IDS_PER_LINE = 6

# Top-to-bottom order of macro concepts; edit here or pass --order.
ROW_ORDER = [
    "Data Understanding",
    "Data Summary",
    "Data Pre-processing",
    "Data Visualization",
    "Linear Regression",
    "Logistic Regression",
    "PCA",
    "Clustering",
    "Hypothesis Testing",
]

# Same for the macro2_concept version (score_std_macro2.csv); empty = sort by mean.
ROW_ORDER_MACRO2 = [
    "Data Pre-processing",
    "Data Visualization",
    "Supervised Learning",
    "Unsupervised Learning",
    "Hypothesis Testing",
    "Model Evaluation"
]

# The two heatmaps drawn by default: (score_std file, row order, file-name suffix).
VERSIONS = [
    (os.path.join("run_scores", "score_std.csv"), ROW_ORDER, ""),
    (os.path.join("run_scores", "score_std_macro2.csv"), ROW_ORDER_MACRO2, "_macro2"),
]


def split_ids(cell):
    """'1, 2.0, 14.1' -> ['1', '2.0', '14.1']; NaN/empty -> []."""
    if not isinstance(cell, str):
        return [] if pd.isna(cell) else [str(cell)]
    return [i for i in cell.split(", ") if i]


def row_order(grid, order=()):
    """Listed macro concepts first (in the given order, unknown names ignored
    with a warning), then the rest by mean value, Other last."""
    present = grid.index.tolist()
    unknown = [r for r in order if r not in present]
    if unknown:
        print(f"Ignoring --order names not in the data: {unknown}")
    first = [r for r in dict.fromkeys(order) if r in present]
    rest = [r for r in grid.mean(axis=1).sort_values(ascending=False).index if r not in first]
    if UNMAPPED_MACRO in rest:
        rest = [r for r in rest if r != UNMAPPED_MACRO] + [UNMAPPED_MACRO]
    return first + rest


def load_grid(path, value, order=()):
    """score_std csv -> (value, n_ids, ids) grids, groups x levels. The group
    column (macro_concept or macro2_concept) is the csv's first column."""
    df = pd.read_csv(path)
    group = df.columns[0]
    df["ids"] = df["ids"].map(lambda c: ", ".join(sort_ids(split_ids(c))))
    grid = df.pivot_table(index=group, columns="level", values=value, aggfunc="first")
    counts = df.pivot_table(index=group, columns="level", values="n_ids", aggfunc="first")
    ids = df.pivot(index=group, columns="level", values="ids")

    levels = [l for l in LEVEL_ORDER if l in grid.columns]
    levels += sorted(c for c in grid.columns if c not in LEVEL_ORDER)
    rows = row_order(grid, order)
    return tuple(t.reindex(index=rows, columns=levels) for t in (grid, counts, ids))


def ids_in_metrics(path, metrics_dirs):
    """Per block of score_std.csv: its ids, and which of them already appear
    in the metrics jsonl files."""
    df = pd.read_csv(path)
    group = df.columns[0]
    seen = metrics_ids(metrics_dirs)
    rows = []
    for _, r in df.iterrows():
        ids = sort_ids(split_ids(r["ids"]))
        hit = [i for i in ids if i in seen]
        rows.append({
            group: r[group],
            "level": r["level"],
            "n_ids": len(ids),
            "n_in_metrics": len(hit),
            "ids_in_metrics": ", ".join(hit),
            "ids_not_in_metrics": ", ".join(i for i in ids if i not in seen),
        })
    return pd.DataFrame(rows), seen


def plot_heatmap(grid, counts, value, out_base, ids=None):
    n_rows, n_cols = grid.shape
    if ids is not None:
        wrapped = ids.map(lambda c: "\n".join(textwrap.wrap(
            c, width=IDS_PER_LINE * 6, break_long_words=False)) if isinstance(c, str) else "")
        lines = wrapped.map(lambda c: c.count("\n") + 1 if c else 0).max(axis=1)
        heights = (0.22 + 0.085 * lines).tolist()
        width = ACM_TEXT_WIDTH
    else:
        heights = [0.28] * n_rows
        width = ACM_COLUMN_WIDTH
    fig, ax = plt.subplots(figsize=(width, sum(heights) + 1.0))

    vals = grid.to_numpy(dtype=float)
    vmin, vmax = (0.0, 1.0) if value == "std_norm" else (np.nanmin(vals), np.nanmax(vals))
    ax.set_facecolor(MISSING_COLOR)  # shows through NaN (missing) cells
    # Row edges top-down, so rows can be as tall as their id lists need.
    edges = np.concatenate([[0], np.cumsum(heights)])
    centers = (edges[:-1] + edges[1:]) / 2
    im = ax.pcolormesh(np.arange(n_cols + 1) - 0.5, edges, np.ma.masked_invalid(vals),
                       cmap=ACM_RAMP, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=2)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(edges[-1], 0)

    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, centers[i], "–", ha="center", va="center", fontsize=7, color=MUTED_TEXT)
                continue
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
            color = "white" if frac > 0.6 else PRIMARY_TEXT
            n = counts.iat[i, j]
            label = f"{v:.2f}" + (f" (n={int(n)})" if pd.notna(n) else "")
            if ids is not None and wrapped.iat[i, j]:
                ax.text(j, edges[i] + 0.05, label, ha="center", va="top", fontsize=6,
                        color=color, fontweight="bold")
                ax.text(j, edges[i] + 0.17, wrapped.iat[i, j], ha="center", va="top",
                        fontsize=5, color=color, linespacing=1.05)
            else:
                ax.text(j, centers[i], label.replace(" (", "\n("), ha="center", va="center",
                        fontsize=6, color=color, linespacing=0.9)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([l.capitalize() for l in grid.columns], fontsize=7, color=PRIMARY_TEXT)
    ax.set_yticks(centers)
    ax.set_yticklabels(grid.index, fontsize=7, color=PRIMARY_TEXT)
    ax.xaxis.tick_top()
    ax.tick_params(which="major", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Difficulty level", fontsize=7, color=MUTED_TEXT, labelpad=4)
    ax.xaxis.set_label_position("top")

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6, colors=MUTED_TEXT, length=2)
    cbar.set_label(VALUE_LABELS.get(value, value), fontsize=7, color=MUTED_TEXT)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default=None,
                        help="one score_std csv to plot (default: both versions in VERSIONS)")
    parser.add_argument("--outdir", default="run_scores")
    parser.add_argument("--value", default="std_norm", help="column to color by (std_norm, std, mean)")
    parser.add_argument("--order", nargs="*", default=None,
                        help="rows top to bottom (default: ROW_ORDER / ROW_ORDER_MACRO2)")
    parser.add_argument("--show-ids", action="store_true",
                        help="also list the ids in each cell (full text width)")
    parser.add_argument("--metrics-dir", nargs="*", default=[".metrics"],
                        help="folders of jsonl files to check the block ids against")
    args = parser.parse_args()

    if args.file:
        group = pd.read_csv(args.file, nrows=0).columns[0]
        suffix = "_macro2" if group == "macro2_concept" else ""
        versions = [(args.file, ROW_ORDER_MACRO2 if suffix else ROW_ORDER, suffix)]
    else:
        versions = VERSIONS
    os.makedirs(args.outdir, exist_ok=True)
    for path, default_order, suffix in versions:
        if not os.path.exists(path):
            print(f"{path} not found, skipped (run score_runs.py)")
            continue
        order = args.order if args.order is not None else default_order
        grid, counts, ids = load_grid(path, args.value, order)
        print(grid.round(3))
        out_base = os.path.join(args.outdir, f"score_{args.value}_heatmap{suffix}"
                                + ("_ids" if args.show_ids else ""))
        plot_heatmap(grid, counts, args.value, out_base, ids if args.show_ids else None)
        print(f"Wrote {out_base}.png / .pdf")

        in_metrics, seen = ids_in_metrics(path, args.metrics_dir)
        group = in_metrics.columns[0]
        row_pos = {m: k for k, m in enumerate(grid.index)}
        in_metrics = in_metrics.sort_values(
            [group, "level"],
            key=lambda s: s.map(row_pos) if s.name == group
            else s.map({l: k for k, l in enumerate(grid.columns)}))
        out_csv = os.path.join(args.outdir, f"block_ids_in_metrics{suffix}.csv")
        in_metrics.to_csv(out_csv, index=False)
        print(f"{len(seen)} id(s) found in {args.metrics_dir}; "
              f"{in_metrics['n_in_metrics'].sum()} of {in_metrics['n_ids'].sum()} block ids present. "
              f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
