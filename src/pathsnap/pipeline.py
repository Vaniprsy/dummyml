"""End-to-end inference pipeline used by the API.

predict()  : raw GPS + road network  ->  per-observation arcs. It never reads
             the .route file (the Trip is loaded with with_route=False).
score()    : server-side only. Loads the reference route separately and
             compares it with the prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from pathsnap import config
from pathsnap.hmm.viterbi import prepare_trip, viterbi
from pathsnap.models.zoo import FEATURES
from pathsnap.preprocess.candidates import build_candidates
from pathsnap.preprocess.io import Trip, load_trip


def _continuity(arcs: np.ndarray, seq: np.ndarray) -> float:
    a0, a1 = seq[:-1], seq[1:]
    ok = (a0 >= 0) & (a1 >= 0)
    a0, a1 = a0[ok], a1[ok]
    if len(a0) == 0:
        return float("nan")
    return float(((a0 == a1) | (arcs[a0, 1] == arcs[a1, 0])).mean())


class PathSnapPipeline:
    def __init__(self, model_path: Path | None = None, hmm_cfg_path: Path | None = None):
        model_path = model_path or config.ARTIFACTS_DIR / "model.joblib"
        hmm_cfg_path = hmm_cfg_path or config.REPO_ROOT / "configs" / "hmm_config.json"
        self.model = joblib.load(model_path)
        self.hmm = json.loads(Path(hmm_cfg_path).read_text())

    # ---------------------------------------------------------------- predict
    def predict(self, trip: Trip) -> dict:
        assert len(trip.route) == 0, "prediction pipeline must not receive reference labels"
        cand, obs = build_candidates(trip, with_labels=False)
        n_obs = len(trip.track)
        if len(cand) == 0:
            return {"pred_arc": np.full(n_obs, -1), "clf_arc": np.full(n_obs, -1), "n_cand": 0}
        proba = self.model.predict_proba(cand[FEATURES].to_numpy(np.float32))[:, 1]

        # classifier-only: highest-probability candidate per observation
        clf = np.full(n_obs, -1, dtype=np.int64)
        order = np.lexsort((-proba, cand["obs_idx"].to_numpy()))
        first = np.unique(cand["obs_idx"].to_numpy()[order], return_index=True)[1]
        sel = order[first]
        clf[cand["obs_idx"].to_numpy()[sel]] = cand["arc_id"].to_numpy()[sel]

        st = prepare_trip(trip, cand, obs, np.arange(len(cand)),
                          dmax=self.hmm["dmax_m"], gap_s=self.hmm["gap_s"])
        pred = viterbi(st, proba, beta=self.hmm["beta"], lam=self.hmm["lam"], w=self.hmm["w"])
        return {"pred_arc": pred, "clf_arc": clf, "n_cand": len(cand)}

    # ------------------------------------------------------------------ score
    @staticmethod
    def score(trip_id: str, partition: str, result: dict, arcs: np.ndarray) -> dict:
        """Server-side evaluation against the reference route."""
        ref_trip = load_trip(trip_id, partition, with_route=True)
        _, obs = build_candidates(ref_trip, with_labels=True)
        gt = obs["gt_arc"].to_numpy()
        covered = obs["gt_in_candidates"].to_numpy() == 1
        out = {}
        for key, name in [("pred_arc", "hmm"), ("clf_arc", "classifier_only")]:
            ok = result[key] == gt
            out[f"{name}_top1_all"] = float(ok.mean())
            out[f"{name}_top1_covered"] = float(ok[covered].mean()) if covered.any() else None
            out[f"{name}_route_continuity"] = _continuity(arcs, result[key])
        out["reference_route_continuity"] = _continuity(arcs, gt)
        out["n_obs"] = int(len(gt))
        out["candidate_miss_rate"] = float(1 - covered.mean())
        return out

    # --------------------------------------------------------------- geometry
    @staticmethod
    def route_polyline(trip: Trip, pred_arc: np.ndarray) -> list:
        """Predicted route as a list of [lat, lon] segments (consecutive duplicate arcs merged)."""
        seq = pred_arc[pred_arc >= 0]
        if len(seq) == 0:
            return []
        keep = np.concatenate([[True], seq[1:] != seq[:-1]])
        seq = seq[keep]
        segs = []
        for a in seq:
            u, v = trip.arcs[a]
            segs.append([[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])],
                         [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]])
        return segs
