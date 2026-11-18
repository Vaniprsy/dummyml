"""PathSnap API (runs on Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment:
  PATHSNAP_DATA_DIR        folder containing the trip partitions (default ../Student folder)
  PATHSNAP_EVAL_PARTITION  partition served to the UI (default: student_test; for the
                           final evaluation point this at the evaluator partition folder)

The reference route (.route) is read ONLY inside the server-side scoring step; it is
never passed to the prediction pipeline and never returned to the UI.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config
from pathsnap.pipeline import PathSnapPipeline
from pathsnap.preprocess.io import list_trips, load_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "student_test")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = PathSnapPipeline()
log.info("model + HMM config loaded | serving partition '%s' from %s", PARTITION, config.DATA_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "partition": PARTITION, "hmm": pipeline.hmm}


@app.get("/trips")
def trips():
    ids = list_trips(PARTITION)
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    if req.trip_id not in list_trips(PARTITION):
        log.warning("POST /predict trip_id=%s -> 404", req.trip_id)
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    log.info("POST /predict trip_id=%s received", req.trip_id)

    trip = load_trip(req.trip_id, PARTITION, with_route=False)   # no labels in the pipeline
    result = pipeline.predict(trip)
    metrics = pipeline.score(req.trip_id, PARTITION, result, trip.arcs)  # server-side only
    elapsed = time.time() - t0

    pred = result["pred_arc"]
    log.info("POST /predict trip_id=%s done | n_obs=%d | HMM top1=%.4f | classifier-only top1=%.4f "
             "| continuity=%.4f | %.2fs", req.trip_id, metrics["n_obs"], metrics["hmm_top1_all"],
             metrics["classifier_only_top1_all"], metrics["hmm_route_continuity"], elapsed)
    return {
        "trip_id": req.trip_id,
        "n_obs": metrics["n_obs"],
        "gps_points": [[float(a), float(b)] for b, a in zip(trip.track[:, 0], trip.track[:, 1])],
        "predicted_route": pipeline.route_polyline(trip, pred),
        "predicted_arcs": [int(a) for a in pred],
        "metrics": metrics,
        "elapsed_s": round(elapsed, 3),
    }
