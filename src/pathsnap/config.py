"""Central configuration. Override paths with environment variables.

PATHSNAP_DATA_DIR : folder that contains `train/` and `student_test/`
                    (default: ../Student folder, next to this repo)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.environ.get("PATHSNAP_DATA_DIR", REPO_ROOT.parent / "Student folder")
)

REPORTS_DIR = REPO_ROOT / "reports"
MAPS_DIR = REPO_ROOT / "maps"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Trips come from many places worldwide (see reports/inventory_*.csv), so there
# is no single projected CRS. Each trip is projected to metres with a local
# azimuthal-equidistant projection centred on its own GPS track
# (see pathsnap.preprocess.geo.make_projector).
GEO_CRS = "EPSG:4326"

# Candidate generation
CANDIDATE_RADIUS_M = 50.0
MAX_CANDIDATES_PER_OBS = 15

RANDOM_SEED = 42
