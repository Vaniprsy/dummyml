"""Step 8 - collect the report tables from the saved experiment outputs.

Writes reports/results_tables.md. Numbers only; the analysis in the report must be
written by the team from these results.
"""
from __future__ import annotations

import json

import pandas as pd

from pathsnap import config

R = config.REPORTS_DIR


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.4f}")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()) + "\n"


def main() -> None:
    out = ["# Result tables (generated from reports/*.csv; do not edit by hand)\n"]
    mc = pd.read_csv(R / "model_comparison.csv")
    out += ["## 3.1 Model comparison (CV = 5-fold GroupKFold on 80 train trips; test = 10 student-test trips)\n",
            md(mc[["model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean",
                   "test_top1_all", "test_macro_f1", "test_log_loss", "test_route_continuity",
                   "train_fit_top1", "train_time_s", "inference_s_per_1k_candidates"]])]
    bt = pd.read_csv(R / "bayes_untuned_vs_tuned.csv").drop(columns=["params"])
    bs = json.loads((R / "bayes_summary.json").read_text())
    out += ["## 4. Bayesian optimisation\n",
            f"Trials: {bs['n_trials']} (budget {bs['time_budget_s']} s), sampler {bs['sampler']}, "
            f"objective: {bs['objective']}, CV: {bs['cv']}\n\nBest params: `{json.dumps(bs['best_params'])}`\n\n",
            md(bt)]
    trials = pd.read_csv(R / "bayes_trials.csv")
    out += ["### All trials\n", md(trials)]
    out += ["## 5. HMM vs classifier-only\n", md(pd.read_csv(R / "hmm_comparison.csv"))]
    out += ["### Decoder grid (top 8 by top1_all, tuning trips only)\n",
            md(pd.read_csv(R / "hmm_tuning.csv").head(8))]
    for part in ["train", "student_test"]:
        cs = pd.read_csv(R / f"candidate_stats_{part}.csv")
        out.append(f"## Candidate miss rate, {part}\n\nmedian {cs.miss_rate.median():.4f}, "
                   f"max {cs.miss_rate.max():.4f}, trips above 5%: {(cs.miss_rate > .05).sum()}\n")
    (R / "results_tables.md").write_text("\n".join(out))
    print("wrote reports/results_tables.md")


if __name__ == "__main__":
    main()
