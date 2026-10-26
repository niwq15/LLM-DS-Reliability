"""Step 1.2: Proportion of difficulty levels per concept, from a questions jsonl file.

Each line of the input has (at least) a "concepts" list and a "level" string.
A question tagged with several concepts counts once towards each of them.
Concepts are also grouped into macro concepts via concept_map; concepts
missing from the map go to UNMAPPED_MACRO. A question counts once per macro
concept even if several of its concepts map to the same one.

    python plot_concept_levels.py --file questions.jsonl --outdir concept_plots

A second grouping, macro2_concept, maps each macro concept through macro_map
(macro concepts not in macro_map keep their name).

Writes, per grouping (concept, macro_concept, macro2_concept), <outdir>/<grouping>_levels.csv
(the group x level proportions) and <outdir>/<grouping>_levels.png / .pdf
(100% stacked bar chart). Questions that land in UNMAPPED_MACRO are listed in
<outdir>/other_questions.csv (id, level, all concepts, the unmapped ones) so
concept_map can be checked and extended.

<outdir>/question_ids_by_block.csv (and .tex, a booktabs table for the paper)
lists the question ids in each block of task type (macro concept) x
difficulty level, with per-level counts and a row total. Ids that already
appear in a metrics jsonl file (--metrics-dir, default metrics/) are
highlighted in ACM dark blue in the .tex and counted in the csv's m_<level> / in_metrics columns.

    python plot_concept_levels.py --metrics-dir metrics .metrics
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from round_plot_style import MUTED_TEXT, PRIMARY_TEXT, style_axis

# Colors from the ACM acmart class (used by The Web Conference template),
# converted from its CMYK definitions. Difficulty is ordinal, so levels get
# a light -> dark blue ramp; extra/unknown levels fall back to other ACM hues.
ACM_LIGHT_BLUE = "#82fcff"  # ACMLightBlue cmyk(0.49, 0.01, 0, 0)
ACM_BLUE = "#00cfe6"        # ACMBlue      cmyk(1, 0.1, 0, 0.1)
ACM_DARK_BLUE = "#0055c9"   # ACMDarkBlue  cmyk(1, 0.58, 0, 0.21)
ACM_FALLBACK = ["#fc9200", "#6200d9", "#a5cf00", "#ff1924", "#ffd600"]  # Orange, Purple, Green, Red, Yellow

LEVEL_ORDER = ["easy", "medium", "hard"]
LEVEL_COLORS = {"easy": ACM_LIGHT_BLUE, "medium": ACM_BLUE, "hard": ACM_DARK_BLUE}

concept_map_overview = {
    "Regression Modeling": "Statistical Models",
    "Bayesian Regression Model":"Statistical Models",
    "Logistic Regression Model":"Statistical Models",
    "Regression Model Diagnostics":"Statistical Models",
    "Linear Regression":"Statistical Models",
    "Regression Modeling Interpretation":"Statistical Models",
    "Correlation Analysis":"Statistical Models",
    "Linear Discriminant Analysis":"Statistical Models",
    "Clustering":"Statistical Models",
    "Confidence Interval":"Statistical Models",
    "PCA":"Statistical Models",
    "Model Improvement":"Statistical Models",
    "Regression Model":"Statistical Models",
    "Logistic Regression Model Performance":"Statistical Models",
    "Ridge-Lasso Regression":"Statistical Models",
    "Modeling Comparison":"Statistical Models",
    "Action Suggestion":"Statistical Models",
    "Confusion Matrix Details":"Statistical Models",
    "Classification Model":"Statistical Models",
    "CI Interpretation":"Statistical Models",
    "Bootstrap CI":"Statistical Models",
    "Bayesian Regression Model Checks":"Statistical Models",

    "Data Summary":"Data Summary",
    "Data Transformation-Summary":"Data Summary",
    "Function Writing-Data Summary":"Data Summary",
    "Data Summary-Interpretation":"Data Summary",
    "Summary Statistics":"Data Summary",

    "Data Visualization":"Data Visualization",
    "Data Viz And EDA": "Data Visualization",
    "Model Diagnostic Visualization":"Data Visualization",
    "Model Performance-Visualization":"Data Visualization",
    "Data Visualization-Summary":"Data Visualization",
    "General EDA":"Data Visualization",
    "Descriptive Statistics": "Data Visualization",
    "Data Visualization-Interpretation":"Data Visualization",
    "Data Visualization-Customization":"Data Visualization",

    "Data Understanding":"Data Understanding",
    "Data Description": "Data Understanding",
    "Interpretation":"Data Understanding",    
    "Specific Descriptions":"Data Understanding",


    "Data Transformation-Data Summary":"Data Pre-processing",
    "Data Transformation":"Data Pre-processing",
    "Data Preparation":"Data Pre-processing",
    "Data Cleaning-Preparation":"Data Pre-processing",
    "Data Cleaning":"Data Pre-processing",
    
    "Hypothesis Testing":"Hypothesis Testing"
}
concept_map = {
    "Regression Modeling": "Linear Regression",
    "Linear Regression":"Linear Regression",
    "Regression Modeling Interpretation":"Linear Regression",

    "Bayesian Regression Model":"Logistic Regression",
    "Logistic Regression Model":"Logistic Regression",
    "Regression Model Diagnostics":"Statistical Models",
    
    "Correlation Analysis":"Data Understanding",
    "Linear Discriminant Analysis":"Linear Regression",
    "Clustering":"Clustering",
    "Confidence Interval":"Model Evaluation",
    "PCA":"PCA",
    "Model Improvement":"Model Evaluation",
    "Regression Model":"Statistical Models",
    "Logistic Regression Model Performance":"Model Evaluation",
    "Ridge-Lasso Regression":"Statistical Models",
    "Modeling Comparison":"Model Evaluation",
    "Action Suggestion":"Model Evaluation",
    "Confusion Matrix Details":"Model Evaluation",
    "Classification Model":"Statistical Models",
    "CI Interpretation":"Model Evaluation",
    "Bootstrap CI":"Model Evaluation",
    "Bayesian Regression Model Checks":"Model Evaluation",

    "Data Summary":"Data Summary",
    "Data Transformation-Summary":"Data Summary",
    "Function Writing-Data Summary":"Data Summary",
    "Data Summary-Interpretation":"Data Summary",
    "Summary Statistics":"Data Summary",

    "Data Visualization":"Data Visualization",
    "Data Viz And EDA": "Data Visualization",
    "Model Diagnostic Visualization":"Data Visualization",
    "Model Performance-Visualization":"Data Visualization",
    "Data Visualization-Summary":"Data Visualization",
    "General EDA":"Data Visualization",
    "Descriptive Statistics": "Data Visualization",
    "Data Visualization-Interpretation":"Data Visualization",
    "Data Visualization-Customization":"Data Visualization",

    "Data Understanding":"Data Understanding",
    "Data Description": "Data Understanding",
    "Interpretation":"Data Understanding",    
    "Specific Descriptions":"Data Understanding",


    "Data Transformation-Data Summary":"Data Pre-processing",
    "Data Transformation":"Data Pre-processing",
    "Data Preparation":"Data Pre-processing",
    "Data Cleaning-Preparation":"Data Pre-processing",
    "Data Cleaning":"Data Pre-processing",
    
    "Hypothesis Testing":"Hypothesis Testing"
}
UNMAPPED_MACRO = "Other"

macro_map = {
    "Data Understanding": "Data Pre-processing",
    "Data Summary": "Data Pre-processing",
    "Data Pre-processing": "Data Pre-processing",
    "Data Visualization": "Data Visualization",
    "Statistical Models": "Supervised Learning",
    "Linear Regression": "Supervised Learning",
    "Logistic Regression": "Supervised Learning",
    "Clustering": "Unsupervised Learning",
    "PCA": "Unsupervised Learning",
    "Hypothesis Testing": "Hypothesis Testing",
    "Statistical Models": "Model Evaluation",
    "Model Evaluation": "Model Evaluation"
}

MACRO2_COLUMN_ORDER = [
    "Data Pre-processing",
    "Data Visualization",
    "Supervised Learning",
    "Unsupervised Learning",
    "Hypothesis Testing",
    "Model Evaluation"
]


def macro2_of(macro):
    """macro_concept -> macro2_concept via macro_map; a macro concept missing
    from macro_map keeps its own name."""
    return macro_map.get(macro, macro)

def load_concept_levels(path):
    """Return a long dataframe with one row per (question, concept):
    question, concepts, macro_concept, level. "question" is the jsonl "id"
    (or the line number if there is no id)."""
    df = pd.read_json(path, lines=True)
    df["question"] = df["id"] if "id" in df.columns else df.index
    df = df[["question", "concepts", "level"]]
    df["concepts"] = df["concepts"].apply(lambda c: c if isinstance(c, list) else [c])
    df = df.explode("concepts").dropna(subset=["concepts"])
    df["concepts"] = df["concepts"].astype(str).str.strip()
    df["macro_concept"] = df["concepts"].map(concept_map).fillna(UNMAPPED_MACRO)
    df["macro2_concept"] = df["macro_concept"].map(macro2_of)
    df["level"] = df["level"].astype(str).str.strip().str.lower()
    return df[["question", "concepts", "macro_concept", "macro2_concept", "level"]].reset_index(drop=True)


def other_questions(df):
    """One row per question with at least one concept outside concept_map:
    id, level, all its concepts, and the unmapped ones."""
    grouped = df.groupby("question", sort=True)
    unmapped = (df[df["macro_concept"] == UNMAPPED_MACRO]
                .groupby("question", sort=True)["concepts"].agg("; ".join))
    table = pd.DataFrame({
        "level": grouped["level"].first(),
        "concepts": grouped["concepts"].agg("; ".join),
    }).loc[unmapped.index]
    table["unmapped_concepts"] = unmapped
    return table.rename_axis("id").reset_index()


def id_key(x):
    """Canonical text form of a question id, so 14, 14.0 and "14" all match:
    whole numbers lose the .0, other numbers keep their decimals (14.1)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x).strip()
    return str(int(f)) if f.is_integer() else format(f, "g")


def sort_ids(ids):
    """Distinct ids as id_key strings, in numeric order where possible."""
    keys = {id_key(i) for i in ids}
    def order(k):
        try:
            return (0, float(k), k)
        except ValueError:
            return (1, 0.0, k)
    return sorted(keys, key=order)


def question_id_table(df):
    """Macro concept x level table of question ids, plus counts per level and
    a total of distinct questions per macro concept. Rows follow the bar
    chart order (most questions first, UNMAPPED_MACRO last)."""
    df = df.drop_duplicates(subset=["question", "macro_concept"])
    _, totals = level_proportions(df, "macro_concept")
    ids = df.pivot_table(index="macro_concept", columns="level", values="question",
                         aggfunc=lambda s: ", ".join(sort_ids(s)))
    counts = pd.crosstab(df["macro_concept"], df["level"])
    levels = [l for l in LEVEL_ORDER if l in counts.columns]
    levels += sorted(c for c in counts.columns if c not in LEVEL_ORDER)

    table = ids.reindex(index=totals.index, columns=levels).fillna("")
    for level in levels:
        table[f"n_{level}"] = counts.reindex(index=totals.index)[level]
    table["total"] = df.groupby("macro_concept")["question"].nunique().reindex(totals.index)
    return table.rename_axis("macro_concept").rename_axis(None, axis=1).reset_index(), levels


def metrics_ids(dirs):
    """Distinct ids found in the *.jsonl files of the given folders."""
    ids = set()
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rid = json.loads(line).get("id")
                        if rid is not None:
                            ids.add(id_key(rid))
    return ids


def add_metrics_counts(table, levels, seen):
    """Per level, how many of the block's ids are in seen (m_<level>), plus a
    row total of distinct ids already in metrics (in_metrics)."""
    def split(cell):
        return [i for i in str(cell).split(", ") if i]
    table = table.copy()
    for level in levels:
        table[f"m_{level}"] = table[level].map(lambda c: sum(i in seen for i in split(c)))
    table["in_metrics"] = table[levels].apply(
        lambda row: len({i for c in row for i in split(c)} & seen), axis=1)
    return table


def _highlight(cell, seen, mark):
    return ", ".join(mark(i) if i in seen else i for i in str(cell).split(", ") if i)


def write_question_id_tex(table, levels, path, seen=frozenset()):
    """booktabs LaTeX version of question_id_table: ids with (n) per cell,
    ids in seen colored ACMDarkBlue (defined by acmart; provided otherwise)."""
    def esc(s):
        return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    cols = "l" + "p{0.22\\linewidth}" * len(levels) + "r"
    lines = [
        r"\providecolor{ACMDarkBlue}{cmyk}{1,0.58,0,0.21}",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        " & ".join(["Task type"] + [l.capitalize() for l in levels] + ["Total"]) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [esc(row["macro_concept"])]
        for level in levels:
            n = int(row[f"n_{level}"])
            ids = _highlight(row[level], seen, lambda i: r"\textcolor{ACMDarkBlue}{" + i + "}")
            cells.append(f"{ids} ({n})" if n else "--")
        cells.append(str(int(row["total"])))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def level_proportions(df, group_col="concepts"):
    """Group x level table of proportions (each row sums to 1), plus row totals."""
    df = df.drop_duplicates(subset=["question", group_col])
    counts = pd.crosstab(df[group_col], df["level"])
    levels = [l for l in LEVEL_ORDER if l in counts.columns]
    levels += sorted(c for c in counts.columns if c not in LEVEL_ORDER)
    counts = counts.reindex(columns=levels)
    # Most-asked groups first, with the unmapped bucket last.
    order = counts.sum(axis=1).sort_values(ascending=False).index.tolist()
    if group_col == "macro2_concept":
        first = [g for g in MACRO2_COLUMN_ORDER if g in order]
        order = first + [g for g in order if g not in first]
    if group_col in ("macro_concept", "macro2_concept") and UNMAPPED_MACRO in order:
        order = [g for g in order if g != UNMAPPED_MACRO] + [UNMAPPED_MACRO]
    counts = counts.loc[order]
    return counts.div(counts.sum(axis=1), axis=0), counts.sum(axis=1)


def plot_proportions(props, totals, out_base, group_label="Concept"):
    colors = dict(LEVEL_COLORS)
    extra = iter(ACM_FALLBACK)
    for level in props.columns:
        colors.setdefault(level, next(extra, "#898781"))

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(props) + 2), 4))
    style_axis(ax)
    ax.grid(False, axis="x")

    x = range(len(props))
    bottom = pd.Series(0.0, index=props.index)
    for level in props.columns:
        vals = props[level]
        ax.bar(x, vals, bottom=bottom, width=0.6, color=colors[level],
               edgecolor="white", linewidth=1.5, label=level.capitalize())
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 0.06:
                light = colors[level] in (ACM_LIGHT_BLUE, ACM_BLUE, "#ffd600", "#a5cf00")
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color=PRIMARY_TEXT if light else "white")
        bottom += vals

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\n(n={totals[c]})" for c in props.index],
                       rotation=30 if len(props) > 5 else 0,
                       ha="right" if len(props) > 5 else "center")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Question Category", color="black")
    ax.set_ylabel("Proportion of questions by difficulty level", color=MUTED_TEXT)
    # ax.set_title(f"Difficulty level distribution per {group_label.lower()}",
    #              color=PRIMARY_TEXT, fontsize=10)
    ax.legend(title="Level", frameon=False, fontsize=8, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default="local_input/questions-all.jsonl")
    parser.add_argument("--outdir", default="concept_plots")
    parser.add_argument("--metrics-dir", nargs="*", default=["metrics"],
                        help="folders of metrics jsonl files whose ids get highlighted")
    args = parser.parse_args()

    df = load_concept_levels(args.file)
    print(df)
    unmapped = sorted(df.loc[df["macro_concept"] == UNMAPPED_MACRO, "concepts"].unique())
    if unmapped:
        print(f"{len(unmapped)} concept(s) not in concept_map, grouped as {UNMAPPED_MACRO!r}: {unmapped}")
    kept = sorted(set(df["macro_concept"]) - set(macro_map))
    if kept:
        print(f"{len(kept)} macro concept(s) not in macro_map, kept as their own macro2_concept: {kept}")

    os.makedirs(args.outdir, exist_ok=True)
    others = other_questions(df)
    others.to_csv(os.path.join(args.outdir, "other_questions.csv"), index=False)
    print(f"{len(others)} question(s) in {UNMAPPED_MACRO!r}, ids: {others['id'].tolist()}")
    id_table, levels = question_id_table(df)
    seen = metrics_ids(args.metrics_dir)
    id_table = add_metrics_counts(id_table, levels, seen)
    print(f"{len(seen)} id(s) found in {args.metrics_dir}")
    print(id_table.to_string(index=False))
    id_table.to_csv(os.path.join(args.outdir, "question_ids_by_block.csv"), index=False)
    write_question_id_tex(id_table, levels, os.path.join(args.outdir, "question_ids_by_block.tex"), seen)
    for group_col, name, label in (("concepts", "concept", "Concept"),
                                   ("macro_concept", "macro_concept", "Macro concept"),
                                   ("macro2_concept", "macro2_concept", "Macro2 concept")):
        props, totals = level_proportions(df, group_col)
        print(props.round(3))
        props.to_csv(os.path.join(args.outdir, f"{name}_levels.csv"))
        plot_proportions(props, totals, os.path.join(args.outdir, f"{name}_levels"), label)
    print(f"Wrote chart and table to {args.outdir}/")


if __name__ == "__main__":
    main()
