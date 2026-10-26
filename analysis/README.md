# Overview

This folder contains the code for result analysis. 

## Implementation
The following step outputs the main result files, run_features.csv, run_scores.csv. For easy use, we already share the local version on the 'run_scores/' folder. 
```
# Extract data to obtain run_features.csv, run_scores.csv
python extract_run_features.py
python plot_concept_levels.py
```

```
# Compute scores and plot score distribution
python score_runs.py
python plot_score_std.py
```

```
# Conduct verbosity analysis
python verbosity_relation.py
```

