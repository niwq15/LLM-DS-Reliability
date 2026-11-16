"""Relationship between verbosity_ratio_tokens (y) and verbosity_ratio_words (x).
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from extract_run_features import FILE_NAME_PATTERN, _file_name, parse_model_temperature  # noqa: E402
from plot_concept_levels import ACM_DARK_BLUE, ACM_FALLBACK, id_key  # noqa: E402
from round_plot_style import LINE_COLOR, MUTED_TEXT, PRIMARY_TEXT, style_axis  # noqa: E402

# ---- Human input: edit these, or override on the command line --------------
RUN_SCORES = os.path.join("run_scores", "run_scores.csv")
MODEL = "gpt-4o"
TEMPERATURE = 1.0
X_COL = "verbosity_ratio_words"
Y_COL = "verbosity_ratio_tokens"
# -----------------------------------------------------------------------------

DEFAULT_OUTDIR = "verbosity_analysis"
CONFIG_COLORS = [LINE_COLOR, *ACM_FALLBACK]


def load(path):
    runs = pd.read_csv(path)
    if "model" not in runs or "temperature" not in runs:
        parsed = runs["source_file"].map(parse_model_temperature)
        runs["model"], runs["temperature"] = parsed.str[0], parsed.str[1]
    data = runs["source_file"].map(lambda f: (m.group("data") if (m := FILE_NAME_PATTERN.match(_file_name(f)))
                                              else str(f)))
    runs["question"] = data + ":" + runs["id"].map(id_key)
    runs["config"] = runs["model"].astype(str) + " T=" + runs["temperature"].map(lambda t: f"{t:g}")
    x = pd.to_numeric(runs[X_COL], errors="coerce")
    y = pd.to_numeric(runs[Y_COL], errors="coerce")
    ok = (x > 0) & (y > 0)
    dropped = int((~ok).sum())
    if dropped:
        print(f"dropped {dropped} of {len(runs)} run(s) with a missing or non-positive ratio")
    runs = runs[ok].copy()
    runs["x"], runs["y"] = x[ok], y[ok]
    runs["log_x"], runs["log_y"] = np.log(runs["x"]), np.log(runs["y"])
    return runs


def ols(formula, d):
    """OLS with cluster-robust (by question) standard errors when there are >1 questions."""
    m = smf.ols(formula, d)
    if d["question"].nunique() > 1:
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["question"])[0]})
    return m.fit()


def fit_subset(name, d):
    out = {"subset": name, "n_runs": len(d), "n_questions": d["question"].nunique(),
           "n_configs": d["config"].nunique(), "configs": "; ".join(sorted(d["config"].unique()))}
    out["pearson_raw"] = d["x"].corr(d["y"])
    out["pearson_log"] = d["log_x"].corr(d["log_y"])
    out["spearman"] = d["x"].corr(d["y"], method="spearman")
    lin = ols("y ~ x", d)
    out.update(lin_intercept=lin.params["Intercept"], lin_slope=lin.params["x"],
               lin_slope_lo=lin.conf_int().loc["x", 0], lin_slope_hi=lin.conf_int().loc["x", 1],
               lin_r2=lin.rsquared)
    log = ols("log_y ~ log_x", d)
    ci = log.conf_int().loc["log_x"]
    out.update(log_intercept=log.params["Intercept"], log_slope=log.params["log_x"],
               log_slope_lo=ci[0], log_slope_hi=ci[1], log_r2=log.rsquared,
               proportionality_factor=float(np.exp(log.params["Intercept"])),
               slope_eq_1_p=float(log.t_test("log_x = 1").pvalue))
    resid = log.resid
    out["log_resid_sd"] = float(resid.std(ddof=0))
    out["share_abs_resid_gt_2sd"] = float((resid.abs() > 2 * resid.std(ddof=0)).mean())
    if d["question"].nunique() > 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # boundary warnings when few questions
                mix = smf.mixedlm("log_y ~ log_x", d, groups=d["question"]).fit(reml=True)
            mci = mix.conf_int().loc["log_x"]
            out.update(mixed_slope=mix.params["log_x"], mixed_slope_lo=mci[0], mixed_slope_hi=mci[1],
                       mixed_question_var=float(mix.cov_re.iloc[0, 0]), mixed_resid_var=float(mix.scale))
        except Exception as e:  # singular fits etc.
            print(f"  {name}: mixed model failed ({e}); skipped")
    return out, log


def interaction_test(d):
    """Does the log-log slope differ between configs? Wald test on the
    log_x:config interaction terms (cluster-robust)."""
    if d["config"].nunique() < 2:
        return None
    m = ols("log_y ~ log_x * C(config)", d)
    terms = [t for t in m.params.index if t.startswith("log_x:")]
    if not terms:
        return None
    w = m.wald_test(" , ".join(f"{t} = 0" for t in terms), scalar=True)
    slopes = {}
    for c in sorted(d["config"].unique()):
        sub = d[d["config"] == c]
        if sub["question"].nunique() > 1 and len(sub) > 10:
            slopes[c] = (ols("log_y ~ log_x", sub).params["log_x"], len(sub), sub["question"].nunique())
    return {"wald_stat": float(np.squeeze(w.statistic)), "p_value": float(w.pvalue), "df": len(terms),
            "slopes": slopes}


def plot_subset(name, d, log, out_base):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for k, (c, g) in enumerate(d.groupby("config")):
        ax.scatter(g["x"], g["y"], s=7, alpha=0.45, linewidths=0, color=CONFIG_COLORS[k % len(CONFIG_COLORS)],
                   label=c if d["config"].nunique() > 1 else "runs")
    xs = np.exp(np.linspace(d["log_x"].min(), d["log_x"].max(), 100))
    a, b = log.params["Intercept"], log.params["log_x"]
    ax.plot(xs, np.exp(a) * xs ** b, color=ACM_DARK_BLUE, lw=1.6,
            label=f"log-log fit: y = {np.exp(a):.3f} x^{b:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_COL, color=MUTED_TEXT)
    ax.set_ylabel(Y_COL, color=MUTED_TEXT)
    # ax.set_title(f"{name}: n={len(d)} runs, {d['question'].nunique()} questions", fontsize=8.5,
    #              color=PRIMARY_TEXT)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    style_axis(ax)
    ax2 = axes[1]
    ax2.scatter(log.fittedvalues, log.resid, s=7, alpha=0.45, linewidths=0, color=LINE_COLOR)
    ax2.axhline(0, color=ACM_DARK_BLUE, lw=1)
    sd = log.resid.std(ddof=0)
    for s in (-2, 2):
        ax2.axhline(s * sd, color=MUTED_TEXT, lw=0.8, ls="--")
    ax2.set_xlabel("fitted log y", color=MUTED_TEXT)
    ax2.set_ylabel("residual (log scale)", color=MUTED_TEXT)
    # ax2.set_title("residuals (dashed: ±2 sd)", fontsize=8.5, color=PRIMARY_TEXT)
    style_axis(ax2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_lines(rows, inter, model, temperature):
    L = [f"y = {Y_COL}, x = {X_COL}; filtered = model {model}, temperature {temperature:g}", ""]
    for r in rows:
        L += [f"[{r['subset']}] {r['n_runs']} runs, {r['n_questions']} questions, configs: {r['configs']}",
              f"  correlation: Pearson raw {r['pearson_raw']:.3f}, Pearson log {r['pearson_log']:.3f}, "
              f"Spearman {r['spearman']:.3f}",
              f"  linear:  y = {r['lin_intercept']:.4f} + {r['lin_slope']:.4f} x  "
              f"(slope 95% CI {r['lin_slope_lo']:.4f} to {r['lin_slope_hi']:.4f}), R2 {r['lin_r2']:.3f}",
              f"  log-log: log y = {r['log_intercept']:.3f} + {r['log_slope']:.3f} log x  "
              f"(slope 95% CI {r['log_slope_lo']:.3f} to {r['log_slope_hi']:.3f}), R2 {r['log_r2']:.3f}",
              f"           i.e. y = {r['proportionality_factor']:.4f} * x^{r['log_slope']:.3f}; "
              f"test slope = 1: p = {r['slope_eq_1_p']:.3g}",
              f"           residual sd (log) {r['log_resid_sd']:.3f}; share |resid| > 2 sd: "
              f"{r['share_abs_resid_gt_2sd']:.3f}"]
        if "mixed_slope" in r and pd.notna(r.get("mixed_slope")):
            L.append(f"  mixed (random intercept per question): slope {r['mixed_slope']:.3f} "
                     f"(95% CI {r['mixed_slope_lo']:.3f} to {r['mixed_slope_hi']:.3f}); "
                     f"between-question var {r['mixed_question_var']:.4f}, residual var {r['mixed_resid_var']:.4f}")
        if r["n_questions"] < 20:
            L.append(f"  note: only {r['n_questions']} questions; cluster-robust CIs and the mixed model "
                     f"are unreliable with this few clusters")
        L.append("")
    if inter:
        L.append(f"[all] slope differs between configs? Wald chi2 = {inter['wald_stat']:.2f}, "
                 f"df = {inter['df']}, p = {inter['p_value']:.3g}")
        for c, (s, n, q) in inter["slopes"].items():
            L.append(f"  {c}: log-log slope {s:.3f} ({n} runs, {q} questions)")
    else:
        L.append("[all] only one config in the data: no between-config test")
    return L


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=RUN_SCORES)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    runs = load(args.runs)
    model = args.model.replace("_", "-")
    filt = runs[(runs["model"] == model) & np.isclose(runs["temperature"].astype(float), args.temperature)]
    if filt.empty:
        raise SystemExit(f"No runs for model {model}, temperature {args.temperature:g}; configs: "
                         f"{sorted(runs['config'].unique())}")
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for name, d in (("filtered", filt), ("all", runs)):
        row, log = fit_subset(name, d)
        rows.append(row)
        plot_subset(name, d, log, os.path.join(args.outdir, f"verbosity_relation_{name}"))
    inter = interaction_test(runs)
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "verbosity_relation_summary.csv"), index=False)
    lines = report_lines(rows, inter, model, args.temperature)
    with open(os.path.join(args.outdir, "verbosity_relation_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote summary csv, report and figures to {args.outdir}/")


if __name__ == "__main__":
    main()
