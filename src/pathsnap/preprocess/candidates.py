"""Candidate generation: for every GPS observation, find nearby road arcs.

A "candidate" is (observation, arc) within CANDIDATE_RADIUS_M metres. The
classifier then scores each candidate (is this the arc the vehicle is on?).

Labels: the reference route (.route) is a sequence of arc_ids. For each
observation we take the *nearest route arc* as its reference arc. This is an
assumption (the route has no per-observation timestamps) and is documented in
the report. Labels are only read when `with_labels=True`; the inference path
never touches the .route file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from pathsnap import config
from pathsnap.preprocess.geo import (angle_diff, bearing_xy, make_projector,
                                     point_to_segments, project)
from pathsnap.preprocess.io import Trip

MIN_SPEED_FOR_HEADING = 0.5  # m/s; below this the heading is unreliable


def _obs_kinematics(xy: np.ndarray, t: np.ndarray):
    """Heading (rad), speed (m/s), dt to previous / next obs."""
    n = len(xy)
    prev_i = np.maximum(np.arange(n) - 1, 0)
    next_i = np.minimum(np.arange(n) + 1, n - 1)
    d = xy[next_i] - xy[prev_i]
    dt = np.maximum(t[next_i] - t[prev_i], 1e-6)
    speed = np.linalg.norm(d, axis=1) / dt
    heading = bearing_xy(d)
    dt_prev = np.where(np.arange(n) > 0, t - t[prev_i], 0.0)
    dt_next = np.where(np.arange(n) < n - 1, t[next_i] - t, 0.0)
    return heading, speed, dt_prev, dt_next


def build_candidates(trip: Trip, radius: float = config.CANDIDATE_RADIUS_M,
                     k: int = config.MAX_CANDIDATES_PER_OBS,
                     with_labels: bool = True):
    """Return (cand_df, obs_df) for one trip."""
    lon, lat, t = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]
    tr = make_projector(lon, lat)
    obs_xy = project(tr, lon, lat)
    node_xy = project(tr, trip.nodes[:, 0], trip.nodes[:, 1])
    a = node_xy[trip.arcs[:, 0]]
    b = node_xy[trip.arcs[:, 1]]
    n_obs, n_arcs = len(obs_xy), len(trip.arcs)

    heading, speed, dt_prev, dt_next = _obs_kinematics(obs_xy, t)

    lines = shapely.linestrings(np.stack([a, b], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(obs_xy)
    oi, ai = tree.query(pts, predicate="dwithin", distance=radius)

    dist, frac = point_to_segments(obs_xy[oi], a[ai], b[ai]) if len(oi) else (np.empty(0), np.empty(0))

    # keep the k nearest arcs per observation
    order = np.lexsort((dist, oi))
    oi, ai, dist, frac = oi[order], ai[order], dist[order], frac[order]
    counts = np.bincount(oi, minlength=n_obs)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(oi)) - starts[oi]
    keep = rank < k
    n_within_radius = counts
    oi, ai, dist, frac, rank = oi[keep], ai[keep], dist[keep], frac[keep], rank[keep]

    # arc geometry / topology features
    arc_vec = b[ai] - a[ai]
    arc_len = np.linalg.norm(arc_vec, axis=1)
    arc_head = bearing_xy(arc_vec)
    head_diff = angle_diff(heading[oi], arc_head)
    heading_valid = (speed[oi] >= MIN_SPEED_FOR_HEADING).astype(np.int8)
    head_diff = np.where(heading_valid == 1, head_diff, 0.0)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    in_deg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))

    cand = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": oi.astype(np.int32),
        "arc_id": ai.astype(np.int64),
        "dist_m": dist.astype(np.float32),
        "frac": frac.astype(np.float32),
        "arc_len_m": arc_len.astype(np.float32),
        "head_diff": head_diff.astype(np.float32),
        "heading_valid": heading_valid,
        "to_out_deg": out_deg[trip.arcs[ai, 1]].astype(np.int16),
        "from_in_deg": in_deg[trip.arcs[ai, 0]].astype(np.int16),
        "speed_mps": speed[oi].astype(np.float32),
        "dt_prev_s": dt_prev[oi].astype(np.float32),
        "dt_next_s": dt_next[oi].astype(np.float32),
        "rank": rank.astype(np.int16),
        "n_within_radius": n_within_radius[oi].astype(np.int16),
    })
    min_d = cand.groupby("obs_idx")["dist_m"].transform("min")
    cand["rel_dist_m"] = (cand["dist_m"] - min_d).astype(np.float32)

    obs = pd.DataFrame({
        "trip_id": trip.trip_id, "obs_idx": np.arange(n_obs, dtype=np.int32),
        "t": t, "lon": lon, "lat": lat, "x": obs_xy[:, 0], "y": obs_xy[:, 1],
        "n_cand": np.bincount(oi, minlength=n_obs).astype(np.int16),
    })

    if with_labels and len(trip.route):
        route_arcs = np.unique(trip.route)
        rtree = STRtree(lines[route_arcs])
        qi, ri = rtree.query_nearest(pts, all_matches=False)
        gt = np.full(n_obs, -1, dtype=np.int64)
        gt[qi] = route_arcs[ri]
        obs["gt_arc"] = gt
        cand["label"] = (cand["arc_id"].to_numpy() == gt[cand["obs_idx"].to_numpy()]).astype(np.int8)
        hit = cand.groupby("obs_idx")["label"].max().reindex(range(n_obs), fill_value=0)
        obs["gt_in_candidates"] = hit.to_numpy().astype(np.int8)
    return cand, obs
