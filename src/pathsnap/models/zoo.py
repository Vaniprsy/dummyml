"""The candidate classifiers compared in scripts/03_train_compare.py.

Each classifier scores a (observation, arc) candidate: P(arc is the reference arc).
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathsnap import config

FEATURES = [
    "dist_m", "rel_dist_m", "rank", "frac", "arc_len_m", "head_diff",
    "heading_valid", "to_out_deg", "from_in_deg", "speed_mps",
    "dt_prev_s", "dt_next_s", "n_within_radius",
]

SEED = config.RANDOM_SEED


def get_models() -> dict:
    """name -> (algorithm family, key configuration string, estimator)"""
    return {
        "logreg": ("Linear", "C=1.0, L2, scaled",
                   make_pipeline(StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=300, random_state=SEED))),
        "gaussian_nb": ("Probabilistic", "var_smoothing=1e-9",
                        make_pipeline(StandardScaler(), GaussianNB())),
        "decision_tree": ("Tree", "max_depth=12, min_samples_leaf=50",
                          DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                                 random_state=SEED)),
        "random_forest": ("Ensemble (bagging)", "n_estimators=60, max_depth=14, min_samples_leaf=30",
                          RandomForestClassifier(n_estimators=60, max_depth=14,
                                                 min_samples_leaf=30, n_jobs=-1,
                                                 random_state=SEED)),
        "hist_gb": ("Boosting", "max_iter=200, lr=0.1, max_leaf_nodes=31",
                    HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                   max_leaf_nodes=31, random_state=SEED)),
        "mlp": ("Neural network", "hidden=(64,32), adam, max_iter=25, scaled",
                make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024,
                                            max_iter=25, random_state=SEED))),
    }
