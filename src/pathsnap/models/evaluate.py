"""Metrics shared by model comparison, tuning and decoder evaluation.

Candidate level : binary macro-F1 at threshold 0.5, log loss.
Observation level: top-1 accuracy (highest-scoring candidate == reference arc),
                   reported over ALL observations (observations whose reference
                   arc is not in the candidate set count as wrong) and over
                   covered observations only.
Route level     : continuity = share of consecutive observations whose arcs are
                   identical or head-to-tail adjacent in the road graph.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from pathsnap import config
from pathsnap.preprocess.io import load_trip


def pick_top1(cand: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """One predicted arc per observation: the highest-scoring candidate."""
    df = cand[["trip_id", "obs_idx", "arc_id"]].copy()
    df["score"] = score
    df = df.sort_values(["trip_id", "obs_idx", "score"], ascending=[True, True, False])
    return df.drop_duplicates(["trip_id", "obs_idx"])[["trip_id", "obs_idx", "arc_id"]] \
             .rename(columns={"arc_id": "pred_arc"})


@lru_cache(maxsize=None)
def _arcs(trip_id: str, partition: str) -> np.ndarray:
    return load_trip(trip_id, partition, with_route=False).arcs


def route_continuity(seq: pd.DataFrame, arc_col: str, partition_of: dict[str, str]) -> float:
    """seq: columns trip_id, obs_idx, <arc_col>. Fraction of consecutive pairs
    (within a trip, both arcs present) that are the same arc or adjacent."""
    ok, total = 0, 0
    for tid, g in seq.sort_values(["trip_id", "obs_idx"]).groupby("trip_id"):
        arcs = _arcs(tid, partition_of[tid])
        a = g[arc_col].to_numpy()
        valid = a[:-1] >= 0
        valid &= a[1:] >= 0
        a0, a1 = a[:-1][valid], a[1:][valid]
        if len(a0) == 0:
            continue
        adj = (a0 == a1) | (arcs[a0, 1] == arcs[a1, 0])
        ok += int(adj.sum())
        total += len(a0)
    return ok / total if total else float("nan")


def score_sequence(obs: pd.DataFrame, pred: pd.DataFrame, partition_of: dict[str, str]) -> dict:
    """Score per-observation predictions (columns trip_id, obs_idx, pred_arc;
    pred_arc = -1 means no prediction) against the reference arcs."""
    m = obs[["trip_id", "obs_idx", "gt_arc", "gt_in_candidates"]].merge(
        pred, on=["trip_id", "obs_idx"], how="left")
    m["pred_arc"] = m["pred_arc"].fillna(-1).astype(np.int64)
    correct = (m["pred_arc"] == m["gt_arc"]).to_numpy()
    covered = m["gt_in_candidates"].to_numpy() == 1
    return {
        "top1_all": correct.mean(),
        "top1_covered": correct[covered].mean() if covered.any() else float("nan"),
        "route_continuity": route_continuity(m.assign(arc=m["pred_arc"]), "arc", partition_of),
    }


def evaluate(cand: pd.DataFrame, obs: pd.DataFrame, proba: np.ndarray,
             partition_of: dict[str, str]) -> dict:
    y = cand["label"].to_numpy()
    pred = pick_top1(cand, proba)
    m = obs[["trip_id", "obs_idx", "gt_arc", "gt_in_candidates"]].merge(
        pred, on=["trip_id", "obs_idx"], how="left")
    m["pred_arc"] = m["pred_arc"].fillna(-1).astype(np.int64)
    correct = (m["pred_arc"] == m["gt_arc"]).to_numpy()
    covered = m["gt_in_candidates"].to_numpy() == 1
    seq = m.assign(arc=m["pred_arc"])
    return {
        "cand_macro_f1": f1_score(y, (proba >= 0.5).astype(int), average="macro"),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "top1_all": correct.mean(),
        "top1_covered": correct[covered].mean() if covered.any() else float("nan"),
        "route_continuity": route_continuity(seq, "arc", partition_of),
    }
