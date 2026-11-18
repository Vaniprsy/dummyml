# PathSnap: ML map-matching with HMM/Viterbi route reconstruction

Noisy 1 Hz GPS points are matched to a road network. A classifier scores every
(GPS observation, nearby road arc) candidate; an HMM decoded with Viterbi then picks a
connected route. A FastAPI backend (Laptop 1) serves the pipeline and a browser UI
(Laptop 2) lets the evaluator pick a trip and see the predicted route and accuracy.

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/preprocess/` | raw readers, geometry helpers, candidate generation + labels |
| `src/pathsnap/features/` | feature definitions (see `models/zoo.py::FEATURES`; features are produced in `preprocess/candidates.py`) |
| `src/pathsnap/models/` | model zoo, metrics (`evaluate.py`) |
| `src/pathsnap/hmm/` | HMM + Viterbi decoder (`viterbi.py`) |
| `src/pathsnap/pipeline.py` | inference pipeline used by the API |
| `scripts/` | numbered scripts, run in order (see below) |
| `api/` | FastAPI backend |
| `ui/` | evaluator UI (single static page) |
| `configs/` | tuned hyperparameters and HMM configuration |
| `artifacts/` | exported model (`model.joblib`) and metadata |
| `reports/` | inventory, experiment logs, result CSVs, figures |
| `maps/` | interactive Folium maps |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Raw data is **not** committed. Point the code at the folder that contains `train/` and
`student_test/` (each trip folder holds `.track .nodes .arcs .route`):

```bash
export PATHSNAP_DATA_DIR="/path/to/Student folder"   # default: ../Student folder
```

## Reproduce the experiments

```bash
python scripts/01_inventory.py          # dataset inventory + data-quality checks -> reports/inventory_*.csv
python scripts/02_build_candidates.py   # candidates, features, labels, candidate miss rate
python scripts/03_train_compare.py      # >=5 models, 5-fold GroupKFold CV, student-test scores
python scripts/04b_bayes_opt.py 25 600  # Bayesian optimisation (Optuna TPE) of the chosen model
python scripts/05_export_model.py       # fit final model on the 80 training trips -> artifacts/
python scripts/04_hmm_eval.py <model>   # tune HMM parameters (train OOF only) + classifier vs HMM
python scripts/06_make_maps.py student_test 00000000   # Folium maps -> maps/
python scripts/07_plots.py              # figures -> reports/figures/
python scripts/08_results_summary.py    # tables -> reports/results_tables.md
```

Seeds are fixed (`RANDOM_SEED = 42` in `src/pathsnap/config.py`). Every result quoted in
the report comes from the CSV/JSON files in `reports/`.

## Run the system (two laptops)

**Laptop 1 - API + model** (set the partition that holds the trips to be served):

```bash
export PATHSNAP_DATA_DIR="/path/to/data"
export PATHSNAP_EVAL_PARTITION=student_test     # or the evaluator partition folder name
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Every request is logged to this console (`POST /predict trip_id=...`).

**Laptop 2 - UI:**

```bash
cd ui && python3 -m http.server 8080
# open http://localhost:8080, enter  http://<laptop-1-ip>:8000  as the API address,
# click "Load trips", choose a trip, click "Run prediction".
```

Ground truth (`.route`) is read only inside the server-side scoring step, never by the
prediction pipeline and never sent to the UI.

## Notes on the data

- Trips come from many places worldwide, so each trip is projected to metres with its own
  local azimuthal-equidistant projection (not one global UTM zone).
- Reference route labels: each GPS point's reference arc is the nearest arc of that trip's
  `.route` (the route has no per-point timestamps). This is an assumption, see the report.
- A few training trips have road networks that do not cover the whole track
  (e.g. trip 00000054); these show up as a high candidate miss rate in
  `reports/candidate_stats_train.csv`.
