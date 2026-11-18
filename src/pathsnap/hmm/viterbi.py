"""HMM + Viterbi route reconstruction on top of classifier probabilities.

Hidden states   : at each GPS observation, the candidate road arcs found within
                  CANDIDATE_RADIUS_M. State identity = arc_id, so the
                  candidate -> road-segment mapping is preserved.
Emission        : classifier probability, renormalised over the candidates of
                  the observation:  log e(c) = w * log( p_c / sum_k p_k ).
Transition      : Newson-Krumm style. For arcs a (obs i-1) and b (obs i),
                  d_net = network distance between the projected positions on
                  the two arcs, d_obs = straight-line distance between the two
                  observations:
                      log T(a,b) = -|d_net - d_obs| / beta
                  If b is not reachable from a within DMAX metres (including
                  driving backwards on the same arc), the transition is
                  "disconnected" and gets a flat penalty -lam (not -inf, so
                  decoding can always continue).
Gaps            : if the time gap to the previous observation exceeds GAP_S, or
                  observations in between had no candidates, connectivity says
                  nothing reliable, so the chain is reset (zero transition
                  cost: the best previous state is carried over unconstrained).
Missing candidates: observations with zero candidates are skipped and get
                  pred_arc = -1.
Decoding        : Viterbi recurrence in log space,
                  delta_i(b) = w*log e_i(b) + max_a [ delta_{i-1}(a) + log T(a,b) ].
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pathsnap.preprocess.geo import make_projector, project
from pathsnap.preprocess.io import Trip

DMAX_M = 400.0   # farthest network distance explored between consecutive obs
GAP_S = 10.0     # larger time gaps reset the chain


@dataclass
class TripStructure:
    trip_id: str
    n_obs: int
    rows: np.ndarray                 # candidate row positions (into the cand frame), trip-ordered
    obs_of_step: np.ndarray          # observation index for each step
    starts: np.ndarray               # start offset of each step in `rows`
    sizes: np.ndarray                # number of candidates per step
    arc_ids: np.ndarray              # arc id per candidate row (aligned with rows)
    dnet: list = field(default_factory=list)   # per step>=1: (Kp,Kc) matrix or None (reset)
    dobs: list = field(default_factory=list)


def _dijkstra(src: int, out_ptr, out_to, out_len, dmax: float) -> dict:
    dist = {src: 0.0}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for e in range(out_ptr[u], out_ptr[u + 1]):
            v = out_to[e]
            nd = d + out_len[e]
            if nd <= dmax and nd < dist.get(v, 1e18):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def prepare_trip(trip: Trip, cand: pd.DataFrame, obs: pd.DataFrame, rows: np.ndarray,
                 dmax: float = DMAX_M, gap_s: float = GAP_S) -> TripStructure:
    """Precompute network distances between candidate pairs of consecutive
    observations. `rows` are positions of this trip's candidates in `cand`,
    sorted by observation. Decoding parameters are applied later, so this
    (expensive) step is done once and reused while tuning."""
    c = cand.iloc[rows]
    obs_idx = c["obs_idx"].to_numpy()
    arc = c["arc_id"].to_numpy()
    frac = c["frac"].to_numpy().astype(np.float64)
    alen = c["arc_len_m"].to_numpy().astype(np.float64)
    uniq, starts, sizes = np.unique(obs_idx, return_index=True, return_counts=True)

    tr = make_projector(trip.track[:, 0], trip.track[:, 1])
    node_xy = project(tr, trip.nodes[:, 0], trip.nodes[:, 1])
    arcs = trip.arcs
    arc_len_all = np.linalg.norm(node_xy[arcs[:, 1]] - node_xy[arcs[:, 0]], axis=1)
    order = np.argsort(arcs[:, 0], kind="stable")
    out_to = arcs[order, 1].tolist()
    out_len = arc_len_all[order].tolist()
    out_ptr = np.concatenate([[0], np.cumsum(np.bincount(arcs[:, 0], minlength=len(node_xy)))]).tolist()

    t = trip.track[:, 2]
    xy = project(tr, trip.track[:, 0], trip.track[:, 1])
    reach: dict[int, dict] = {}

    st = TripStructure(trip.trip_id, len(trip.track), rows, uniq, starts, sizes, arc)
    st.dnet.append(None)
    st.dobs.append(0.0)
    for s in range(1, len(uniq)):
        i, j = uniq[s - 1], uniq[s]
        if j - i != 1 or (t[j] - t[i]) > gap_s:
            st.dnet.append(None)
            st.dobs.append(0.0)
            continue
        pa = slice(starts[s - 1], starts[s - 1] + sizes[s - 1])
        pb = slice(starts[s], starts[s] + sizes[s])
        D = np.full((sizes[s - 1], sizes[s]), np.inf, dtype=np.float32)
        for ia, a in enumerate(arc[pa]):
            fa, la = frac[pa][ia], alen[pa][ia]
            n_end = int(arcs[a, 1])
            r = reach.get(n_end)
            if r is None:
                r = reach[n_end] = _dijkstra(n_end, out_ptr, out_to, out_len, dmax)
            for ib, b in enumerate(arc[pb]):
                fb, lb = frac[pb][ib], alen[pb][ib]
                if a == b:
                    d = (fb - fa) * la
                    D[ia, ib] = d if d >= -10.0 else np.inf
                else:
                    g = r.get(int(arcs[b, 0]))
                    if g is not None:
                        D[ia, ib] = (1 - fa) * la + g + fb * lb
        st.dnet.append(D)
        st.dobs.append(float(np.linalg.norm(xy[j] - xy[i])))
    return st


def viterbi(st: TripStructure, p: np.ndarray, beta: float = 30.0, lam: float = 8.0,
            w: float = 1.0) -> np.ndarray:
    """Decode one trip. `p` = classifier probability per candidate row of the
    trip (aligned with st.rows). Returns predicted arc per observation (-1 if
    the observation had no candidates)."""
    S = len(st.obs_of_step)
    pred = np.full(st.n_obs, -1, dtype=np.int64)
    if S == 0:
        return pred
    # emission: renormalise per observation
    logp = np.empty(len(p))
    for s in range(S):
        sl = slice(st.starts[s], st.starts[s] + st.sizes[s])
        q = np.clip(p[sl], 1e-6, None)
        logp[sl] = np.log(q / q.sum())
    logp *= w

    deltas, psis = [], []
    delta = logp[st.starts[0]:st.starts[0] + st.sizes[0]]
    deltas.append(delta)
    psis.append(None)
    for s in range(1, S):
        sl = slice(st.starts[s], st.starts[s] + st.sizes[s])
        D = st.dnet[s]
        if D is None:                                   # reset: unconstrained
            psi = np.full(st.sizes[s], int(np.argmax(delta)))
            new = delta.max() + logp[sl]
        else:
            T = np.where(np.isfinite(D), -np.abs(D - st.dobs[s]) / beta, -lam)
            T = np.maximum(T, -lam)                     # cap: never worse than "disconnected"
            sc = delta[:, None] + T
            psi = sc.argmax(axis=0)
            new = sc[psi, np.arange(sc.shape[1])] + logp[sl]
        deltas.append(new)
        psis.append(psi)
        delta = new
    k = int(np.argmax(delta))
    for s in range(S - 1, -1, -1):
        sl = slice(st.starts[s], st.starts[s] + st.sizes[s])
        pred[st.obs_of_step[s]] = st.arc_ids[sl][k]
        if s > 0:
            k = int(psis[s][k])
    return pred
