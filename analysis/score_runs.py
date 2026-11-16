"""Step 2: score each run, then measure how much the score varies
across rounds of the same question.
"""

import argparse
import os

import pandas as pd

from extract_run_features import (CONCEPT_SEP, complete_ratio_to_numeric, macro2_concepts, macro_concepts,
                                  parse_model_temperature)
from plot_concept_levels import sort_ids

SCORE_FORMULA = "complete_ratio + accuracy + code_execute"
CODE_EXECUTE_MAX = 2
STD_GROUP_KEYS = ["macro_concept", "level"]
STD_GROUP_KEYS_MACRO2 = ["macro2_concept", "level"]
# Model whose runs make up score_std.csv / score_std_macro2.csv (the heatmaps).
HEATMAP_MODEL = "gpt-4o"


def select_model(runs, model):
    """Runs of one model (as in the model column, e.g. gpt-4o); 'all' keeps every run.
    Older run_features.csv files without a model column get it from source_file."""
    if model == "all":
        return runs
    if "model" not in runs.columns:
        runs = runs.assign(model=runs["source_file"].map(lambda f: parse_model_temperature(f)[0]))
    found = runs["model"].value_counts(dropna=False)
    print("runs per model: " + ", ".join(f"{m}: {n}" for m, n in found.items()))
    picked = runs[runs["model"] == model]
    if picked.empty:
        raise SystemExit(f"No runs with model {model!r}; models in the data: {sorted(map(str, found.index))}")
    print(f"score_std uses model {model!r} only: {len(picked)} of {len(runs)} runs, "
          f"{picked['source_file'].nunique()} file(s)")
    return picked

def transform_features(runs):
    """Apply the pre-scoring transformations (returns a copy)."""
    runs = runs.copy()
    runs["accuracy"] = pd.to_numeric(runs["accuracy"], errors="coerce").fillna(0)
    runs["code_execute"] = pd.to_numeric(runs["code_execute"], errors="coerce") / CODE_EXECUTE_MAX
    # Also done at extraction; repeated here for csv files made before that.
    runs["complete_ratio"] = complete_ratio_to_numeric(runs["complete_ratio"])
    # Older run_features.csv files have no macro_concept column.
    if "macro_concept" not in runs.columns:
        runs.insert(runs.columns.get_loc("concepts") + 1, "macro_concept",
                    runs["concepts"].map(macro_concepts))
    # Older run_features.csv files have no macro2_concept column.
    if "macro2_concept" not in runs.columns:
        runs.insert(runs.columns.get_loc("macro_concept") + 1, "macro2_concept",
                    runs["macro_concept"].map(macro2_concepts))
    return runs


def compute_score(runs, formula=SCORE_FORMULA):
    """Transform features and add a 'score' column computed from formula."""
    runs = transform_features(runs)
    runs["score"] = runs.eval(formula)
    return runs


def score_std(runs, keys=STD_GROUP_KEYS):
    """Mean and population std of score per group, a min-max normalized
    std_norm, and the question ids each group covers."""
    runs = runs.assign(
        concept=runs["concepts"].astype(str).str.split(CONCEPT_SEP),
        macro_concept=runs["macro_concept"].astype(str).str.split(CONCEPT_SEP),
        macro2_concept=runs["macro2_concept"].astype(str).str.split(CONCEPT_SEP),
    )
    for col in ("concept", "macro_concept", "macro2_concept"):
        if col in keys:
            runs = runs.explode(col)
    grouped = runs.groupby(keys)
    std = pd.DataFrame({
        "mean": grouped["score"].mean(),
        "std": grouped["score"].std(ddof=0),
        "n_ids": grouped["id"].nunique(),
        "n_runs": grouped["score"].size(),
        "ids": grouped["id"].agg(lambda s: ", ".join(sort_ids(s))),
    }).reset_index()
    lo, hi = std["std"].min(), std["std"].max()
    std.insert(std.columns.get_loc("std") + 1, "std_norm",
               (std["std"] - lo) / (hi - lo) if hi > lo else 0.0)
    return std


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=os.path.join("run_scores", "run_features.csv"))
    parser.add_argument("--outdir", default="run_scores")
    parser.add_argument("--formula", default=SCORE_FORMULA)
    parser.add_argument("--model", default=HEATMAP_MODEL,
                        help="model used for score_std*.csv, e.g. gpt-4o, gpt-4.1-mini, or 'all'")
    args = parser.parse_args()

    runs = compute_score(pd.read_csv(args.features), args.formula)
    print(f"score = {args.formula}")
    print(runs["score"].describe().round(3))

    heat_runs = select_model(runs, args.model)
    std = score_std(heat_runs)
    print(std.round(3).to_string(index=False))

    os.makedirs(args.outdir, exist_ok=True)
    runs.to_csv(os.path.join(args.outdir, "run_scores.csv"), index=False)
    std.to_csv(os.path.join(args.outdir, "score_std.csv"), index=False)
    std2 = score_std(heat_runs, STD_GROUP_KEYS_MACRO2)
    print(std2.round(3).to_string(index=False))
    std2.to_csv(os.path.join(args.outdir, "score_std_macro2.csv"), index=False)
    print(f"Wrote run_scores.csv, score_std.csv and score_std_macro2.csv to {args.outdir}/")


if __name__ == "__main__":
    main()
