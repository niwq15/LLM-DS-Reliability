"""Shared helpers for computing/plotting per-id, across-round statistics
from a metrics .jsonl file (one record per id x round trial).
"""

import glob
import json
import os

import pandas as pd

METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")

# Numeric columns to summarize across rounds for each id.
NUMERIC_FEATURES = [
    "verbosity_ratio_words",
    "verbosity_ratio_tokens",
    "complete_ratio",
    "Similarity_Jaccard",
    "code_execute",
]

# Categorical column summarized as a completion rate across rounds for each id.
STATUS_FEATURE = "status"
STATUS_COMPLETE_VALUE = "completed"

ALL_FEATURES = [STATUS_FEATURE] + NUMERIC_FEATURES

# Metadata columns carried through from the records (constant per id).
ID_META_COLUMNS = ["file_name", "concepts", "level"]


def default_metrics_files():
    """All metrics .jsonl files under metrics/."""
    return sorted(glob.glob(os.path.join(METRICS_DIR, "*.jsonl")))


def resolve_target_files(cli_files, target_files):
    """Decide which metrics files to process, in priority order:

    1. cli_files - files passed explicitly on the command line.
    2. target_files - a script's own hard-coded TARGET_FILES list (bare
       filenames are resolved against METRICS_DIR; a path containing a
       separator is used as-is).
    3. Every .jsonl file under metrics/ (default_metrics_files()).
    """
    if cli_files:
        return list(cli_files)
    if target_files:
        return [
            name if os.path.sep in name or os.path.isabs(name) else os.path.join(METRICS_DIR, name)
            for name in target_files
        ]
    return default_metrics_files()


def load_metrics_jsonl(path):
    """Load a metrics .jsonl file into a DataFrame, one row per id x round."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    missing = [c for c in ["id", "round"] + ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {missing}")
    return df.sort_values(["id", "round"]).reset_index(drop=True)


FEATURE_COLUMNS = ["id", "round"] + ID_META_COLUMNS + ALL_FEATURES


def extract_features(df):
    """Slim a loaded metrics DataFrame down to id, round, metadata, and the tracked features."""
    columns = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[columns].copy()


def save_features_csv(df, path, outdir):
    """Write the per-round, per-id feature table (extract_features) to <outdir>/<basename>_features.csv."""
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(path))[0]
    outpath = os.path.join(outdir, f"{basename}_features.csv")
    extract_features(df).to_csv(outpath, index=False)
    return outpath
