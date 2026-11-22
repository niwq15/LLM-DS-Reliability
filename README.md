# Are LLMs Ready for Data Science?

<!-- ===================== [REQUIRED] 3. TL;DR + TEASER ===================== -->
## 💡 Overview

Existing data-science benchmarks score an LLM **a few runs per question**. But whether the same prompt can produce the same answer, code and charts remains unexplored. To study the reliability of LLM-generated responses, we ask each question **100 times** and measure how much the answers vary. The toolkit shows how to implement multi-round simulations on individual or sequential questions under different settings of model choices, temperatures, prompts, and input datasets. 

### ⭐ If you like this project, please support it by giving it a star!
---

<!-- ===================== [RECOMMENDED] 2. NEWS ===================== -->
## 🔥 News

<!-- Newest first. Only list confirmed events; update the venue line after notification. -->
<!-- Add after notification (4 Jan 2027): - **[2027-01]** 🎉 Paper accepted at **TheWebConf 2027**. -->
<!-- During review, use the arXiv BibTeX entry in the Citation section and do not name the venue. -->

- **[2026-11-25]** 🎤 Preliminary results will be presented at **UKAIRS 2026**, Edinburgh (Emerging Research Track).
- **[2026-11-06]** 📄 Preprint released on [arXiv](https://arxiv.org/abs/2611.09729).
- **[2026-11]** 🗂️ Benchmark and evaluation code released.

<!-- [📄 Paper](TODO) ·  -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<!-- ===================== [REQUIRED] 4. HIGHLIGHTS ===================== -->
## ✨ Highlights

- **🎲 Monte Carlo consistency framework** — each prompt is issued 100× under identical settings, giving an empirical response *distribution* instead of a point estimate.
<!-- - **📏 Normalized instability metrics** for mixed outputs:
  - binary correctness → Bernoulli variance
  - categorical answers → normalized Shannon entropy
  - numeric answers → variance normalized by Popoviciu's bound -->
- **📚 Curriculum-grounded benchmark** — 100 questions from university-level statistics courses, 15 datasets, 6 task categories, 3 difficulty levels, organized by the De Bie et al. data-science taxonomy.
- **🔗 Sequential-QA analysis** — follow-up questions (e.g. `14 → 14.1 → 14.2`) reveal how early errors propagate.
- **🛠️ Actionable prompt guidance** derived from task-specific failure patterns.

<!-- TODO(highlights): add 1 bullet with your headline NUMBER, e.g.
     "**Key finding:** TODO% of questions that are 'solved' on a single run fail in ≥ TODO% of repeated runs." -->

---

<!-- ===================== [REQUIRED] 6. INSTALLATION ===================== -->
## ⚙️ Installation

```bash
git clone https://github.com/niwq15/LLM-DS-Reliability.git
cd LLM-DS-Reliability
pip install -r requirements.txt
# Set your API key:
export OPENAI_API_KEY="sk-..."
```

### 📖 Simulation: How to conduct experiments for new dataset or questions

To run multi-round experiments on a selected set of questions specific to one dataset, we provide a single file 
(*runner.py*) that can accommodate both independent questions and sequential questions. The adjustable setting is 
- model choice (default: 'gpt-4o')
- temperature (1.0)
- instruction for the AI assistant
- data-file-related features: dataname, filename, file_id, outfolder
- question-related features: selected questions (Qs), the number of rounds (Ns), and the number of already-done rounds (ks).

If all the questions are independent, try the function **multi_round_assistant**.
If the questions are sequentially related, try the alternative **sequential_question_assistant**.

After obtaining the experimental outcomes, you can use (*evaluation.py*) to collect results and make evaluation. The adjustable setting is the same as above, except the instruction being no longer used. 

```bash
python runner.py --datanames aeroplane --models gpt-4o-mini --temperatures 1 --Qs 56 --sequential False --with_code_interpreter False --vector_store_id ... --ks 0 # one example
```

### 📖 Analysis

The code in *analysis/* is used to analyse the simulation outcomes (several examples can be found in the folder *simulations/metrics/*). The following step outputs the main result files, run_features.csv, run_scores.csv. For easy use, we already share the local version in *analysis/run_scores/*. 


```bash
# Preliminary result processing
python plot_concept_levels.py --metrics-dir .metrics   # Section 1，含 macro2_concept_levels.png
python extract_run_features.py                         # add the parameters of model、temperature、macro2_concept 
# Final score calculation
python score_runs.py                        # generate score_std.csv for gpt-4o (default)
python score_runs.py --model gpt-4.1-mini   # another model
python score_runs.py --model all            # every model, as before
# Plotting score distribution and heatmaps
python plot_score_std.py                             
# Comparison across models
python plot_config_compare.py model --source aeroplane --id 9 --temperature 1 --features score accuracy
# Comparison across temperatures
python plot_config_compare.py temperature --source evals --id 21 --model gpt-4.1-mini --features tokens --stats median   # only ask medium values and IQR
# Explore token verbosity vs. word token verbositys
python verbosity_relation.py      # default configuration: model gpt-4o, temperature 1
python verbosity_relation.py --model gpt-4.1-mini --temperature 1   # another configuration
```
---

<!-- ===================== [RECOMMENDED] 11. REPO STRUCTURE ===================== -->
## 🗂️ Repository Structure

```text
LLM-DS-Reliability/
├── data/
│   ├── DA-tasks-questions.jsonl        # benchmark questions
│   ├── DA-tasks-responses.jsonl        # benchmark reference answers
│   └── datasets/                       # 15 source datasets (add later)
├── requirements.txt                    # required dependencies
├── runner.py                           # run experiments over 100 times
├── evaluation.py                       # collect results and make initial evaluation
├── llmds
|   ├── scripts/             
|       ├── automation.py              # Monte Carlo querying
│       ├── collect_result.py          # extract metrics per iteration
│       └── evaluate.py                # evaluate metrics 
|   └── READMD.md            
├── simulations/metrics                   # metric tables
├── analysis/                          # reproduction scripts
|   ├── run_score/                     # instability scores
|   ├── plot_concept_levels.py         # code for analysis (same for below)
|   ├── extract_run_features.py       
|   ├── score_runs.py        
|   ├── plot_score_std.py 
|   ├── plot_config_compare.py 
|   └── verbosity_relation.py  
```
<!-- TODO(structure): update to match the real tree (`tree -L 2`). -->

---

<!-- ===================== [REQUIRED] 13. CITATION ===================== -->
## 📝 Citation

If you find this work useful, please cite:
@misc{ni2026llmds,
  title         = {One Run Is Not Enough: A Monte Carlo Protocol for Evaluating LLM Reliability on Data Science Tasks},
  author        = {Ni, W., Evkaya, O. and de Carvalho, M.},
  year          = {2026},
  eprint        = {2611.09729},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.09729 }
}

---

<!-- ===================== [RECOMMENDED] 15. ACKNOWLEDGEMENTS ===================== -->
## 🙏 Acknowledgements

This work was funded by the **Edinburgh Generative AI Lab (GAIL)**, University of Edinburgh.
<!-- TODO(funding source / grant number). TODO(thank course instructors who provided questions, if applicable). -->

---

<!-- ===================== [REQUIRED] 16. CONTACT ===================== -->
## 📬 Contact

Questions and issues: open a [GitHub Issue](https://github.com/niwq15/GAIL-DS-project-Analysis/issues).

<!-- [OPTIONAL] Star history — only worth adding once the repo has some stars -->
<!--
## ⭐ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=TODO-user/TODO-repo&type=Date)](https://star-history.com/#TODO-user/TODO-repo&Date)
-->
