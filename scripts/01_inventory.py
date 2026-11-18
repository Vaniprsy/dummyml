"""Step 1 - dataset inventory and data-quality checks.

Writes reports/inventory_<partition>.csv and reports/inventory_summary.txt.
Run:  python scripts/01_inventory.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from pathsnap import config
from pathsnap.preprocess.io import list_trips, load_trip

PARTITIONS = ["train", "student_test"]


def inspect_trip(trip_id: str, partition: str) -> dict:
    t = load_trip(trip_id, partition)
    d = config.DATA_DIR / partition / trip_id
    sizes = {ext: (d / f"{trip_id}.{ext}").stat().st_size
             for ext in ("track", "nodes", "arcs", "route")}

    lon, lat, ts = t.track[:, 0], t.track[:, 1], t.track[:, 2]
    dt = np.diff(ts)
    n_nodes, n_arcs = len(t.nodes), len(t.arcs)

    # Graph connectivity (undirected weak components, directed strong ones)
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(map(tuple, t.arcs))
    weak = list(nx.weakly_connected_components(g))
    largest_weak = max(len(c) for c in weak) if weak else 0
    touched = set(t.arcs.ravel().tolist())
    isolated = n_nodes - len(touched)

    # Route validity
    route_ok = bool(len(t.route)) and t.route.min() >= 0 and t.route.max() < n_arcs
    # Consecutive route arcs should connect (to_node of a == from_node of b)
    if len(t.route) > 1 and route_ok:
        a, b = t.route[:-1], t.route[1:]
        connected = (t.arcs[a, 1] == t.arcs[b, 0]) | (a == b)
        route_break_frac = float(1.0 - connected.mean())
    else:
        route_break_frac = float("nan")

    return {
        "trip_id": trip_id, "partition": partition,
        "n_obs": len(t.track),
        "n_nodes": n_nodes, "n_arcs": n_arcs, "n_route_arcs": len(t.route),
        "track_bytes": sizes["track"], "nodes_bytes": sizes["nodes"],
        "arcs_bytes": sizes["arcs"], "route_bytes": sizes["route"],
        "nan_obs": int(np.isnan(t.track).any(axis=1).sum()),
        "lon_min": lon.min(), "lon_max": lon.max(),
        "lat_min": lat.min(), "lat_max": lat.max(),
        # lon in [-180,180], lat in [-90,90]; a swapped pair would break this
        "coord_order_ok": bool(np.abs(lon).max() <= 180 and np.abs(lat).max() <= 90),
        "lon_span_deg": float(lon.max() - lon.min()),
        "lat_span_deg": float(lat.max() - lat.min()),
        "mean_lon": float(lon.mean()), "mean_lat": float(lat.mean()),
        "t_sorted": bool((dt >= 0).all()),
        "dup_timestamps": int((dt == 0).sum()),
        "pct_dt_1s": float((dt == 1).mean() * 100) if len(dt) else float("nan"),
        "dt_max_s": float(dt.max()) if len(dt) else float("nan"),
        "n_gaps_gt_5s": int((dt > 5).sum()),
        "duration_s": float(ts[-1] - ts[0]),
        "nodes_in_arcs_ok": bool(len(t.arcs) == 0 or
                                 (t.arcs.min() >= 0 and t.arcs.max() < n_nodes)),
        "isolated_nodes": isolated,
        "weak_components": len(weak),
        "largest_component_frac": largest_weak / n_nodes if n_nodes else float("nan"),
        "route_ids_valid": route_ok,
        "route_break_frac": route_break_frac,
    }


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    lines = []
    for part in PARTITIONS:
        trips = list_trips(part)
        rows = [inspect_trip(tid, part) for tid in trips]
        df = pd.DataFrame(rows)
        df.to_csv(config.REPORTS_DIR / f"inventory_{part}.csv", index=False)

        lines.append(f"=== {part}: {len(df)} trips ===")
        lines.append(f"files per trip: 4 expected; missing-value obs total: {df.nan_obs.sum()}")
        lines.append(f"obs/trip   min={df.n_obs.min()} median={int(df.n_obs.median())} max={df.n_obs.max()} total={df.n_obs.sum()}")
        lines.append(f"nodes/trip min={df.n_nodes.min()} median={int(df.n_nodes.median())} max={df.n_nodes.max()}")
        lines.append(f"arcs/trip  min={df.n_arcs.min()} median={int(df.n_arcs.median())} max={df.n_arcs.max()}")
        lines.append(f"route arcs min={df.n_route_arcs.min()} median={int(df.n_route_arcs.median())} max={df.n_route_arcs.max()}")
        lines.append(f"coord order ok: {df.coord_order_ok.all()} | timestamps sorted: {df.t_sorted.all()} | dup timestamps: {df.dup_timestamps.sum()}")
        lines.append(f"% steps exactly 1s: mean={df.pct_dt_1s.mean():.2f} min={df.pct_dt_1s.min():.2f}")
        lines.append(f"largest time gap (s): {df.dt_max_s.max():.0f}; trips with a gap >5s: {(df.n_gaps_gt_5s > 0).sum()}")
        lines.append(f"graph: isolated nodes total={df.isolated_nodes.sum()}, "
                     f"trips with >1 weak component={(df.weak_components > 1).sum()}, "
                     f"min largest-component frac={df.largest_component_frac.min():.3f}")
        lines.append(f"arc/node ids valid in all trips: {df.nodes_in_arcs_ok.all()}")
        lines.append(f"route ids valid in all trips: {df.route_ids_valid.all()}")
        lines.append(f"route consecutive-arc break fraction: mean={df.route_break_frac.mean():.4f} max={df.route_break_frac.max():.4f}")
        lines.append("")

    text = "\n".join(lines)
    (config.REPORTS_DIR / "inventory_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
