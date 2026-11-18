"""Step 3b - Bayesian hyperparameter optimisation (Optuna TPE) of HistGradientBoosting.

Objective : maximise mean candidate-level macro-F1 over 5-fold GroupKFold CV on the
            80 training trips (folds never split a trip).
Search    : learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization.
Student-test trips are NOT used to choose the search space, objective or parameters;
they are scored once, after the best configuration is refit on all training trips.

Outputs: reports/bayes_trials.csv, reports/bayes_summary.json,
         reports/bayes_untuned_vs_tuned.csv,
         data/processed/{oof,student}_hist_gb_tuned.npy, configs/hgb_best_params.json
Run:  python scripts/04b_bayes_opt.py [max_trials] [time_budget_seconds]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.models.evaluate import evaluate
from pathsnap.models.zoo import FEATURES

SEED = config.RANDOM_SEED
N_SPLITS = 5


def make(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def main(n_trials: int, timeout_s: int) -> None:
    P = config.PROCESSED_DIR
    tr_c = pd.read_parquet(P / "train_candidates.parquet")
    tr_o = pd.read_parquet(P / "train_obs.parquet")
    te_c = pd.read_parquet(P / "student_test_candidates.parquet")
    te_o = pd.read_parquet(P / "student_test_obs.parquet")
    part_of = {t: "train" for t in tr_o.trip_id.unique()}
    part_of.update({t: "student_test" for t in te_o.trip_id.unique()})
    X, y = tr_c[FEATURES].to_numpy(np.float32), tr_c["label"].to_numpy()
    groups = tr_c["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 400, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 500, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        )
        scores = []
        for tri, vai in folds:
            est = make(params).fit(X[tri], y[tri])
            scores.append(f1_score(y[vai], est.predict(X[vai]), average="macro"))
        return float(np.mean(scores))

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    # start from the default configuration so the search has a reference point
    study.enqueue_trial({"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31,
                         "min_samples_leaf": 20, "l2_regularization": 1e-4})
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s,
                   callbacks=[lambda s, t: print(
                       f"trial {t.number:>2} cv_macroF1={t.value:.4f} {t.params}", flush=True)])
    search_s = time.time() - t0

    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    trials.to_csv(config.REPORTS_DIR / "bayes_trials.csv", index=False)
    best = study.best_params
    print("best:", best, "cv macro-F1 =", study.best_value, flush=True)

    # --- compare untuned (library defaults used in step 3) vs tuned ---
    untuned = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)
    rows = []
    for tag, params in [("untuned", untuned), ("tuned_bayes", best)]:
        oof = np.zeros(len(y))
        for tri, vai in folds:
            oof[vai] = make(params).fit(X[tri], y[tri]).predict_proba(X[vai])[:, 1]
        t1 = time.time(); est = make(params).fit(X, y); fit_s = time.time() - t1
        t1 = time.time(); pte = est.predict_proba(te_c[FEATURES].to_numpy(np.float32))[:, 1]
        inf_s = time.time() - t1
        cv = evaluate(tr_c, tr_o, oof, part_of)
        te = evaluate(te_c, te_o, pte, part_of)
        fold_f1 = [f1_score(y[v], (oof[v] >= .5).astype(int), average="macro") for _, v in folds]
        rows.append({"model": tag, "params": json.dumps(params),
                     "cv_macro_f1": np.mean(fold_f1), "cv_top1_all": cv["top1_all"],
                     "cv_log_loss": cv["log_loss"],
                     "test_top1_all": te["top1_all"], "test_macro_f1": te["cand_macro_f1"],
                     "test_log_loss": te["log_loss"], "test_route_continuity": te["route_continuity"],
                     "train_time_s": fit_s, "inference_s_per_1k_candidates": inf_s / len(pte) * 1000})
        if tag == "tuned_bayes":
            np.save(P / "oof_hist_gb_tuned.npy", oof)
            np.save(P / "student_hist_gb_tuned.npy", pte)
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "bayes_untuned_vs_tuned.csv", index=False)
    (config.REPORTS_DIR / "bayes_summary.json").write_text(json.dumps({
        "n_trials": len(study.trials), "max_trials": n_trials, "time_budget_s": timeout_s, "sampler": "Optuna TPE", "seed": SEED, "cv": f"GroupKFold({N_SPLITS}) by trip, 80 train trips",
        "objective": "mean candidate-level macro-F1", "best_value": study.best_value,
        "best_params": best, "search_seconds": search_s}, indent=2))
    (config.REPO_ROOT / "configs").mkdir(exist_ok=True)
    (config.REPO_ROOT / "configs" / "hgb_best_params.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25,
         int(sys.argv[2]) if len(sys.argv) > 2 else 600)
