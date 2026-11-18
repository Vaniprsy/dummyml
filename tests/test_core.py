import numpy as np

from pathsnap.hmm.viterbi import TripStructure, viterbi
from pathsnap.preprocess.geo import angle_diff, point_to_segments


def test_point_to_segment_distance_and_fraction():
    a, b = np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]])
    d, f = point_to_segments(np.array([4.0, 3.0]), a, b)
    assert np.isclose(d[0], 3.0) and np.isclose(f[0], 0.4)
    d, f = point_to_segments(np.array([-5.0, 0.0]), a, b)      # beyond the start -> clipped
    assert np.isclose(d[0], 5.0) and f[0] == 0.0


def test_angle_diff_wraps():
    assert np.isclose(angle_diff(np.array([0.1]), np.array([2 * np.pi - 0.1]))[0], 0.2)


def _toy_structure(dnet):
    # two observations, two candidate arcs each: arcs [10, 11] then [20, 21]
    return TripStructure(
        trip_id="toy", n_obs=2, rows=np.arange(4), obs_of_step=np.array([0, 1]),
        starts=np.array([0, 2]), sizes=np.array([2, 2]), arc_ids=np.array([10, 11, 20, 21]),
        dnet=[None, dnet], dobs=[0.0, 20.0])


def test_viterbi_prefers_connected_route_over_noisy_classifier():
    # classifier slightly prefers arc 21 at obs 1, but only 10 -> 20 is network-consistent
    # (network distance 20 m matches the 20 m travelled); 11 -> 21 is disconnected.
    dnet = np.array([[20.0, np.inf], [np.inf, np.inf]], dtype=np.float32)
    st = _toy_structure(dnet)
    p = np.array([0.6, 0.4, 0.45, 0.55])
    clf_only = [10, 21]
    assert [int(st.arc_ids[2 + np.argmax(p[2:])])] == [21]      # classifier alone picks 21
    pred = viterbi(st, p, beta=10.0, lam=8.0, w=1.0)
    assert list(pred) == [10, 20] and clf_only[1] == 21


def test_viterbi_reset_on_gap_is_unconstrained():
    st = _toy_structure(None)
    p = np.array([0.9, 0.1, 0.2, 0.8])
    assert list(viterbi(st, p)) == [10, 21]


def test_viterbi_handles_observation_without_candidates():
    st = TripStructure("toy", 3, np.arange(2), np.array([0]), np.array([0]), np.array([2]),
                       np.array([5, 6]), dnet=[None], dobs=[0.0])
    pred = viterbi(st, np.array([0.3, 0.7]))
    assert list(pred) == [6, -1, -1]
