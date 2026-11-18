"""Step 3 - train and compare candidate classifiers.

For every model:
  * 5-fold GroupKFold CV on the 80 training trips (folds never split a trip)
  * out-of-fold probabilities saved for decoder tuning
  * refit on all 80 trips, then scored once on the 10 student-test trips

Outputs: reports/cv_folds.csv, reports/model_comparison.csv,
         reports/mlp_loss_curve.csv, data/processed/{oof,student}_<model>.npy
Run:  python scripts/03_train_compare.py [model_name ...]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.models.evaluate import evaluate
from pathsnap.models.zoo import FEATURES, get_models


def main(selected: list[str]) -> None:
    P = config.PROCESSED_DIR
    tr_c = pd.read_parquet(P / "train_candidates.parquet")
    tr_o = pd.read_parquet(P / "train_obs.parquet")
    te_c = pd.read_parquet(P / "student_test_candidates.parquet")
    te_o = pd.read_parquet(P / "student_test_obs.parquet")
    part_of = {t: "train" for t in tr_o.trip_id.unique()}
    part_of.update({t: "student_test" for t in te_o.trip_id.unique()})

    X, y = tr_c[FEATURES].to_numpy(np.float32), tr_c["label"].to_numpy()
    groups = tr_c["trip_id"].to_numpy()
    Xte = te_c[FEATURES].to_numpy(np.float32)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))
    models = get_models()
    names = selected or list(models)
    fold_rows, summary = [], []

    for name in names:
        family, cfg, _ = models[name]
        print(f"\n=== {name} ({family}) ===", flush=True)
        oof = np.zeros(len(y))
        t_cv = time.time()
        for k, (tri, vai) in enumerate(folds):
            est = get_models()[name][2]
            t0 = time.time()
            est.fit(X[tri], y[tri])
            fit_s = time.time() - t0
            p = est.predict_proba(X[vai])[:, 1]
            oof[vai] = p
            vc = tr_c.iloc[vai]
            vo = tr_o[tr_o.trip_id.isin(vc.trip_id.unique())]
            m = evaluate(vc, vo, p, part_of)
            m.update(model=name, fold=k, fit_s=fit_s)
            fold_rows.append(m)
            print(f"  fold {k}: macroF1={m['cand_macro_f1']:.4f} top1={m['top1_all']:.4f} "
                  f"logloss={m['log_loss']:.4f} fit={fit_s:.0f}s", flush=True)
        cv_time = time.time() - t_cv
        np.save(P / f"oof_{name}.npy", oof)
        cv_all = evaluate(tr_c, tr_o, oof, part_of)

        # refit on all training trips, score once on student_test
        est = get_models()[name][2]
        t0 = time.time(); est.fit(X, y); train_s = time.time() - t0
        t0 = time.time(); pte = est.predict_proba(Xte)[:, 1]; inf_s = time.time() - t0
        np.save(P / f"student_{name}.npy", pte)
        te = evaluate(te_c, te_o, pte, part_of)
        tr_fit = evaluate(tr_c, tr_o, est.predict_proba(X)[:, 1], part_of)
        if name == "mlp":
            pd.DataFrame({"epoch": range(1, len(est[-1].loss_curve_) + 1),
                          "train_loss": est[-1].loss_curve_}).to_csv(
                config.REPORTS_DIR / "mlp_loss_curve.csv", index=False)

        fold_df = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": family, "config": cfg,
            "cv_macro_f1_mean": fold_df.cand_macro_f1.mean(),
            "cv_macro_f1_std": fold_df.cand_macro_f1.std(),
            "cv_top1_mean": fold_df.top1_all.mean(), "cv_top1_std": fold_df.top1_all.std(),
            "cv_log_loss": fold_df.log_loss.mean(),
            "cv_route_continuity": fold_df.route_continuity.mean(),
            "train_fit_top1": tr_fit["top1_all"], "train_fit_macro_f1": tr_fit["cand_macro_f1"],
            "test_macro_f1": te["cand_macro_f1"], "test_top1_all": te["top1_all"],
            "test_top1_covered": te["top1_covered"], "test_log_loss": te["log_loss"],
            "test_route_continuity": te["route_continuity"],
            "train_time_s": train_s, "cv_time_s": cv_time,
            "inference_s_per_1k_candidates": inf_s / len(Xte) * 1000,
        })
        print(f"  CV  macroF1={fold_df.cand_macro_f1.mean():.4f}  top1={fold_df.top1_all.mean():.4f}\n"
              f"  TEST macroF1={te['cand_macro_f1']:.4f}  top1={te['top1_all']:.4f}  "
              f"logloss={te['log_loss']:.4f}  continuity={te['route_continuity']:.4f}", flush=True)

        # write incrementally so partial results survive interruption
        pd.DataFrame(fold_rows).to_csv(config.REPORTS_DIR / "cv_folds.csv", index=False)
        pd.DataFrame(summary).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
