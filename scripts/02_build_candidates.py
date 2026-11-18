"""Step 2 - candidate generation for train and student_test.

Writes data/processed/{partition}_candidates.parquet, {partition}_obs.parquet
and reports/candidate_stats_{partition}.csv (candidate miss rate per trip).
"""
from __future__ import annotations

import time

import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.preprocess.candidates import build_candidates
from pathsnap.preprocess.io import list_trips, load_trip


def one(trip_id: str, partition: str):
    trip = load_trip(trip_id, partition)
    return build_candidates(trip)


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    for part in ["train", "student_test"]:
        t0 = time.time()
        trips = list_trips(part)
        res = Parallel(n_jobs=-2)(delayed(one)(tid, part) for tid in trips)
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        cand.to_parquet(config.PROCESSED_DIR / f"{part}_candidates.parquet")
        obs.to_parquet(config.PROCESSED_DIR / f"{part}_obs.parquet")

        stats = obs.groupby("trip_id").agg(
            n_obs=("obs_idx", "size"),
            mean_candidates=("n_cand", "mean"),
            obs_no_candidate=("n_cand", lambda s: int((s == 0).sum())),
            miss_rate=("gt_in_candidates", lambda s: 1 - s.mean()),
        ).reset_index()
        stats.to_csv(config.REPORTS_DIR / f"candidate_stats_{part}.csv", index=False)
        print(f"{part}: {len(trips)} trips, {len(obs)} obs, {len(cand)} candidate rows, "
              f"positive rate {cand.label.mean():.3f}, "
              f"overall candidate miss rate {1 - obs.gt_in_candidates.mean():.4f}, "
              f"obs with zero candidates {(obs.n_cand == 0).sum()} "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
