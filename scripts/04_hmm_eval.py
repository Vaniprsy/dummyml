"""Step 4 - HMM/Viterbi decoding: tune decoder parameters and compare with
classifier-only predictions.

Decoder parameters are tuned on OUT-OF-FOLD probabilities of a subset of the
80 training trips only. The student-test trips are scored once with the chosen
parameters. Evaluator-test data is never touched.

Run:  python scripts/04_hmm_eval.py <model_name>
Outputs: reports/hmm_tuning.csv, reports/hmm_comparison.csv, configs/hmm_config.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.hmm.viterbi import prepare_trip, viterbi
from pathsnap.models.evaluate import pick_top1, score_sequence
from pathsnap.preprocess.io import load_trip

# first pass (logreg smoke test) peaked at beta=10, w=0.5, the edge of the grid, so the
# grid was extended towards smaller values
GRID = {"beta": [3.0, 6.0, 10.0, 20.0], "lam": [4.0, 8.0, 16.0], "w": [0.15, 0.3, 0.5, 1.0]}
N_TUNE_TRIPS = 30


def _prep(tid, part, cand_t, obs_t, rows):
    trip = load_trip(tid, part, with_route=False)   # decoder never sees .route
    return prepare_trip(trip, cand_t, obs_t, rows)


def build_structs(part: str, cand, obs, trips):
    idx = cand.groupby("trip_id").indices
    return dict(zip(trips, Parallel(n_jobs=-2)(
        delayed(_prep)(t, part, cand, obs[obs.trip_id == t], idx[t]) for t in trips)))


def decode_all(structs: dict, probs: np.ndarray, **params) -> pd.DataFrame:
    parts = []
    for tid, st in structs.items():
        pred = viterbi(st, probs[st.rows], **params)
        parts.append(pd.DataFrame({"trip_id": tid, "obs_idx": np.arange(len(pred)), "pred_arc": pred}))
    return pd.concat(parts, ignore_index=True)


def main(model: str) -> None:
    P = config.PROCESSED_DIR
    tr_c = pd.read_parquet(P / "train_candidates.parquet")
    tr_o = pd.read_parquet(P / "train_obs.parquet")
    te_c = pd.read_parquet(P / "student_test_candidates.parquet")
    te_o = pd.read_parquet(P / "student_test_obs.parquet")
    oof = np.load(P / f"oof_{model}.npy")
    pte = np.load(P / f"student_{model}.npy")
    part_of = {t: "train" for t in tr_o.trip_id.unique()}
    part_of.update({t: "student_test" for t in te_o.trip_id.unique()})

    t0 = time.time()
    tr_trips = sorted(tr_o.trip_id.unique())
    te_trips = sorted(te_o.trip_id.unique())
    tr_struct = build_structs("train", tr_c, tr_o, tr_trips)
    te_struct = build_structs("student_test", te_c, te_o, te_trips)
    print(f"prepared network distances for {len(tr_struct)}+{len(te_struct)} trips "
          f"in {time.time() - t0:.0f}s", flush=True)

    # --- tune on a seeded subset of training trips (OOF probabilities) ---
    rng = np.random.default_rng(config.RANDOM_SEED)
    tune_trips = sorted(rng.choice(tr_trips, N_TUNE_TRIPS, replace=False).tolist())
    sub = {t: tr_struct[t] for t in tune_trips}
    sub_obs = tr_o[tr_o.trip_id.isin(tune_trips)]
    base = score_sequence(sub_obs, pick_top1(tr_c[tr_c.trip_id.isin(tune_trips)],
                                             oof[tr_c.trip_id.isin(tune_trips).to_numpy()])
                          .rename(columns={}), part_of)
    print(f"classifier-only on tuning trips: top1_all={base['top1_all']:.4f}", flush=True)

    rows = []
    for beta, lam, w in itertools.product(GRID["beta"], GRID["lam"], GRID["w"]):
        pred = decode_all(sub, oof, beta=beta, lam=lam, w=w)
        sc = score_sequence(sub_obs, pred, part_of)
        rows.append({"beta": beta, "lam": lam, "w": w, **sc})
        print(f"  beta={beta:>5} lam={lam:>5} w={w:>4}  top1_all={sc['top1_all']:.4f} "
              f"continuity={sc['route_continuity']:.4f}", flush=True)
    tune = pd.DataFrame(rows).sort_values("top1_all", ascending=False)
    tune.to_csv(config.REPORTS_DIR / "hmm_tuning.csv", index=False)
    best = tune.iloc[0]
    params = {"beta": float(best.beta), "lam": float(best.lam), "w": float(best.w)}
    print("best decoder params:", params, flush=True)

    # --- compare classifier-only vs HMM on all train (OOF) and student_test ---
    out = []
    for name, c, o, st, pr in [("train_oof_80_trips", tr_c, tr_o, tr_struct, oof),
                               ("student_test_10_trips", te_c, te_o, te_struct, pte)]:
        co = score_sequence(o, pick_top1(c, pr), part_of)
        t1 = time.time()
        hm = score_sequence(o, decode_all(st, pr, **params), part_of)
        out.append({"partition": name, "method": "classifier_only", **co})
        out.append({"partition": name, "method": "classifier+HMM", **hm,
                    "decode_s": time.time() - t1})
    cmp = pd.DataFrame(out)
    cmp.to_csv(config.REPORTS_DIR / "hmm_comparison.csv", index=False)
    print(cmp.to_string(index=False))

    config_dir = config.REPO_ROOT / "configs"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "hmm_config.json").write_text(json.dumps(
        {"model": model, **params, "dmax_m": 400.0, "gap_s": 10.0,
         "candidate_radius_m": config.CANDIDATE_RADIUS_M,
         "max_candidates_per_obs": config.MAX_CANDIDATES_PER_OBS,
         "tuned_on": f"{N_TUNE_TRIPS} train trips, out-of-fold probabilities"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
