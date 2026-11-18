"""Step 5 - export the final model for the API.

Fits the final classifier on the 80 training trips only and writes
artifacts/model.joblib and artifacts/model_meta.json. The configuration is read
from configs/hgb_best_params.json (written by 04b_bayes_opt.py) unless
--untuned is passed.
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap import config
from pathsnap.models.zoo import FEATURES


def main(untuned: bool) -> None:
    if untuned:
        params = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)
    else:
        params = json.loads((config.REPO_ROOT / "configs" / "hgb_best_params.json").read_text())
    tr = pd.read_parquet(config.PROCESSED_DIR / "train_candidates.parquet")
    est = HistGradientBoostingClassifier(random_state=config.RANDOM_SEED, **params)
    est.fit(tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy())
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(est, config.ARTIFACTS_DIR / "model.joblib", compress=3)
    (config.ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps({
        "algorithm": "HistGradientBoostingClassifier", "params": params, "features": FEATURES,
        "trained_on": "80 training trips", "seed": config.RANDOM_SEED}, indent=2))
    print("saved artifacts/model.joblib", params)


if __name__ == "__main__":
    main("--untuned" in sys.argv)
