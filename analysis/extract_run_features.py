"""Step 1: extract per-run features from a set of metrics jsonl files.
"""

import argparse
import json
import os

import pandas as pd

import re

from plot_concept_levels import UNMAPPED_MACRO, concept_map, id_key, macro2_of, macro_map
from round_metrics_utils import resolve_target_files

RUN_FEATURES = [
    "id",
    "round",
    "concepts",
    "level",
    "status",
    "runtime",
    "verbosity_ratio_words",
    "verbosity_ratio_tokens",
    "Similarity_Jaccard",
    "code_execute",
    "complete_ratio",
    "accuracy",
    "tokens",
]
CONCEPT_SEP = "; "
# complete_ratio may be logged as text; anything else (not numeric, not in the
# map) becomes NaN so it shows up instead of being scored silently.
COMPLETE_RATIO_MAP = {"completed": 1, "failed": 0}
COLUMN_ALIASES = {"CR": "complete_ratio"}

# Per source file: old id -> new id, applied first (before QUESTION_FIXES).
ID_REMAP = {
    "laptop_data_cleaned_gpt_4o_1.jsonl": {"0": "46", "1": "47", "2": "48"},
}

# id (after ID_REMAP) -> {"level": ..., "macro_concept": ..., optional
# "source_file": limit the fix to that file}. A blank value leaves the field
# unchanged. Applied to every run of the id before run_features.csv is written.
QUESTION_FIXES = {
    "34": {"level": "medium", "macro_concept": "Data Visualization"},
    "49": {"level": "hard", "macro_concept": "Model Evaluation"},
    "50": {"level": "hard", "macro_concept": "Model Evaluation"},
    "51": {"level": "medium", "macro_concept": "Logistic Regression"},
    "53": {"level": "medium", "macro_concept": "Data Visualization"},
    "54": {"level": "easy", "macro_concept": "Data Summary"},
    "46": {"level": "easy", "macro_concept": "Data Summary", "source_file": "laptop_data_cleaned_gpt_4o_1.jsonl"},
    "47": {"level": "medium", "macro_concept": "Data Visualization", "source_file": "laptop_data_cleaned_gpt_4o_1.jsonl"},
    "48": {"level": "medium", "macro_concept": "Data Visualization", "source_file": "laptop_data_cleaned_gpt_4o_1.jsonl"},
    "72": {"level": "medium", "macro_concept": "Data Understanding", "source_file": "Edinburgh_rainfall_gpt_4o_1.jsonl"}
}
# Optional hard-coded list of files to process (bare names resolve to metrics/).
file_path = "../simulations/metrics"
TARGET_FILES = [ 
    f'{file_path}/aeroplane_gpt_4o_1.jsonl', 
    f'{file_path}/duke-forest_gpt_4o_1.jsonl', 
    f'{file_path}/evals_gpt_4o_1.jsonl',
    f'{file_path}/UK-visitor-numbers_gpt_4o_1.jsonl', 
    f'{file_path}/weatherAUS_gpt_4o_1.jsonl',
    f'{file_path}/mouse_gpt_4o_1.jsonl',
    f'{file_path}/laptop_data_cleaned_gpt_4o_1.jsonl',
    f'{file_path}/cherryblossom_gpt_4o_1.jsonl',
    f'{file_path}/instructional-staff_gpt_4o_1.jsonl',
    f'{file_path}/Edinburgh_rainfall_gpt_4o_1.jsonl',

    f'{file_path}/aeroplane_gpt_4o_mini_1.jsonl',
    f'{file_path}/cherryblossom0_gpt_4.1_mini_1.jsonl',
    f'{file_path}/evals_gpt_4.1_mini_0.2.jsonl',
    f'{file_path}/evals_gpt_4.1_mini_0.7.jsonl',
    f'{file_path}/evals_gpt_4.1_mini_1.0.jsonl',
    f'{file_path}/evals_gpt_4.1_mini_1.2.jsonl',
] 

def complete_ratio_to_numeric(s):
    """completed/failed -> 1/0; numeric values are kept as they are."""
    return pd.to_numeric(s, errors="coerce").fillna(
        s.astype(str).str.strip().str.lower().map(COMPLETE_RATIO_MAP))


# 2. A function that renames the keys in each record
def macro_concepts(concepts):
    """'A; B' -> '; '-joined distinct macro concepts, in order of first appearance."""
    if not isinstance(concepts, str) or not concepts:
        return UNMAPPED_MACRO
    macros = [concept_map.get(c.strip(), UNMAPPED_MACRO) for c in concepts.split(CONCEPT_SEP)]
    return CONCEPT_SEP.join(dict.fromkeys(macros))


# File name -> dataname, model, temperature. The model starts with one of
# MODEL_PREFIXES so data names containing "_" (laptop_data_cleaned) still split.
MODEL_PREFIXES = ("gpt", "o1", "o3", "o4", "claude", "gemini", "llama", "mistral", "qwen", "deepseek")
FILE_NAME_PATTERN = re.compile(
    r"^(?P<data>.+?)_(?P<model>(?:" + "|".join(MODEL_PREFIXES) + r")(?:_[^_]+)*?)"
    r"_(?P<temperature>\d+(?:\.\d+)?)(?:_no_CodeInterp)?\.jsonl$", re.IGNORECASE)


def parse_model_temperature(path):
    """'evals_gpt_4.1_mini_0.2.jsonl' -> ('gpt-4.1-mini', 0.2); (None, None) if
    the name does not follow {dataname}_{model}_{temperature}.jsonl."""
    m = FILE_NAME_PATTERN.match(_file_name(path))
    if not m:
        return None, None
    return m.group("model").replace("_", "-"), float(m.group("temperature"))


def macro2_concepts(macros):
    """'A; B' macro concepts -> '; '-joined distinct macro2 concepts."""
    if not isinstance(macros, str) or not macros:
        return UNMAPPED_MACRO
    return CONCEPT_SEP.join(dict.fromkeys(macro2_of(m.strip()) for m in macros.split(CONCEPT_SEP)))


def normalize_columns(rec):
    """Rename aliased keys (e.g. CR -> complete_ratio) unless the real name is present."""
    for alias, name in COLUMN_ALIASES.items():
        if alias in rec and name not in rec:
            rec[name] = rec.pop(alias)
    return rec


def _file_name(name):
    """Base name, tolerating the '.jsonal' typo for '.jsonl'."""
    return re.sub(r"\.jsonal$", ".jsonl", os.path.basename(str(name)))


def _input_tokens(rec, path):
    """input_tokens_<model> for the model named in the file (e.g. gpt_4o in
    aeroplane_gpt_4o_1.jsonl); None if there is no matching field."""
    name = os.path.basename(path)
    matches = [k for k in rec if k.startswith("input_tokens_")
               and f"_{k[len('input_tokens_'):]}_" in f"_{name}"]
    return rec[max(matches, key=len)] if matches else None


def remap_ids(df, source_file, remap=ID_REMAP):
    """Apply ID_REMAP[source_file] to df["id"] (returns a copy and a log)."""
    table = remap.get(_file_name(source_file), {})
    if not table:
        return df, []
    df = df.copy()
    keys = df["id"].map(id_key)
    log = []
    for old, new in table.items():
        hit = keys == id_key(old)
        if hit.any():
            df.loc[hit, "id"] = float(new) if re.fullmatch(r"-?\d+(\.\d+)?", str(new)) else new
            log.append(f"{_file_name(source_file)}: id {old} -> {new} ({hit.sum()} run(s))")
        else:
            log.append(f"{_file_name(source_file)}: id {old} not found, not remapped")
    return df, log


def apply_question_fixes(df, fixes=QUESTION_FIXES):
    """Apply QUESTION_FIXES to df (all files stacked); returns a copy and a log."""
    df = df.copy()
    keys = df["id"].map(id_key)
    files = df["source_file"].map(_file_name)
    log = []
    for rid, fields in fixes.items():
        hit = keys == id_key(rid)
        src = _file_name(fields["source_file"]) if fields.get("source_file") else ""
        if src:
            hit &= files == src
        if not hit.any():
            log.append(f"id {rid}{' in ' + src if src else ''}: not found, fix not applied")
            continue
        for col in ("level", "macro_concept"):
            value = str(fields.get(col) or "").strip()
            if value:
                value = value.lower() if col == "level" else value
                df.loc[hit, col] = value
        log.append(f"id {rid}{' in ' + src if src else ''}: level={fields.get('level')!r}, "
                   f"macro_concept={fields.get('macro_concept')!r} ({hit.sum()} run(s))")
    return df, log


def extract_file_features(path):
    """One file -> dataframe with source_file + RUN_FEATURES, one row per run."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = normalize_columns(json.loads(line))
                row = {k: rec.get(k) for k in RUN_FEATURES}
                row["input_tokens"] = _input_tokens(rec, path)
                records.append(row)
    df = pd.DataFrame.from_records(records, columns=RUN_FEATURES + ["input_tokens"])

    with open(path, "r", encoding="utf-8") as fh:
        first = normalize_columns(next((json.loads(l) for l in fh if l.strip()), {}))
    missing = [c for c in RUN_FEATURES if c not in first]
    if missing:
        print(f"  {os.path.basename(path)}: missing {missing}, left empty")

    df["concepts"] = df["concepts"].apply(
        lambda c: CONCEPT_SEP.join(map(str, c)) if isinstance(c, list) else c)
    df["complete_ratio"] = complete_ratio_to_numeric(df["complete_ratio"])
    df.insert(df.columns.get_loc("concepts") + 1, "macro_concept", df["concepts"].map(macro_concepts))
    df.insert(0, "source_file", os.path.basename(path))
    model, temperature = parse_model_temperature(path)
    if model is None:
        print(f"  WARNING: {os.path.basename(path)} does not follow "
              f"{{dataname}}_{{model}}_{{temperature}}.jsonl; model/temperature left empty")
    df.insert(1, "model", model)
    df.insert(2, "temperature", temperature)
    df, log = remap_ids(df, path)
    for line in log:
        print(f"  remap: {line}")
    return df


def check_remap_collisions(runs, remap=ID_REMAP):
    """Warn if a remapped id also exists in another file."""
    for src, table in remap.items():
        for new in table.values():
            other = runs[(runs["id"].map(id_key) == id_key(new)) & (runs["source_file"].map(_file_name) != src)]
            if len(other):
                print(f"WARNING: remapped id {new} ({src}) also exists in "
                      f"{sorted(other['source_file'].unique())}")


def extract_features(paths):
    """Extract every file, combine into one dataframe, then apply QUESTION_FIXES."""
    frames = []
    for path in paths:
        print(f"Reading {path}")
        frames.append(extract_file_features(path))
    runs = pd.concat(frames, ignore_index=True)
    check_remap_collisions(runs)
    runs, log = apply_question_fixes(runs)
    for line in log:
        print(f"  fix: {line}")
    runs.insert(runs.columns.get_loc("macro_concept") + 1, "macro2_concept",
                runs["macro_concept"].map(macro2_concepts))
    kept = sorted({m.strip() for v in runs["macro_concept"].dropna() for m in str(v).split(CONCEPT_SEP)}
                  - set(macro_map))
    if kept:
        print(f"{len(kept)} macro concept(s) not in macro_map, kept as their own macro2_concept: {kept}")
    return runs.sort_values(["source_file", "id", "round"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="*", help="metrics jsonl files (default: metrics/*.jsonl)")
    parser.add_argument("--out", default=os.path.join("run_scores", "run_features.csv"))
    args = parser.parse_args()

    paths = resolve_target_files(args.files, TARGET_FILES)
    if not paths:
        raise SystemExit("No metrics jsonl files found.")
    runs = extract_features(paths)
    concepts = runs["concepts"].dropna().astype(str).str.split(CONCEPT_SEP).explode().str.strip()
    unmapped = sorted(set(concepts) - set(concept_map))
    if unmapped:
        print(f"{len(unmapped)} concept(s) not in concept_map, macro_concept {UNMAPPED_MACRO!r}: {unmapped}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs.to_csv(args.out, index=False)
    print(f"Wrote {len(runs)} runs from {len(paths)} file(s) to {args.out}")


if __name__ == "__main__":
    main()
