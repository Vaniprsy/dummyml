"""Raw file readers for PathSnap trips.

Per-trip files (no headers, whitespace separated):
  <id>.track  : lon, lat, timestamp_s
  <id>.nodes  : lon, lat            (row index = node_id)
  <id>.arcs   : from_node, to_node  (row index = arc_id)
  <id>.route  : arc_id              (chronological reference route)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pathsnap import config


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray   # (n_obs, 3) lon, lat, t
    nodes: np.ndarray   # (n_nodes, 2) lon, lat
    arcs: np.ndarray    # (n_arcs, 2) from_node, to_node
    route: np.ndarray   # (n_route,) arc_id, empty if unavailable


def list_trips(partition: str, data_dir: Path | None = None) -> list[str]:
    root = (data_dir or config.DATA_DIR) / partition
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _load(path: Path, dtype, ndmin_cols: int) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ndmin_cols), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr


def load_trip(trip_id: str, partition: str, data_dir: Path | None = None,
              with_route: bool = True) -> Trip:
    """Load one trip. Set with_route=False for the inference path so that
    reference labels are never read by the prediction pipeline."""
    d = (data_dir or config.DATA_DIR) / partition / trip_id
    track = _load(d / f"{trip_id}.track", float, 3)
    nodes = _load(d / f"{trip_id}.nodes", float, 2)
    arcs = _load(d / f"{trip_id}.arcs", np.int64, 2)
    if with_route:
        route = _load(d / f"{trip_id}.route", np.int64, 1).reshape(-1)
    else:
        route = np.empty(0, dtype=np.int64)
    return Trip(trip_id, partition, track, nodes, arcs, route)
