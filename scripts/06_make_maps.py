"""Step 6 - interactive Folium maps.

Layers: raw GPS observations, road network (bounding box of the track),
reference route, classifier-only route, classifier+HMM route. Layer control and
legend included.

Run:  python scripts/06_make_maps.py <partition> <trip_id> [<trip_id> ...]
      e.g. python scripts/06_make_maps.py student_test 00000000
Prediction layers are added only if artifacts/model.joblib exists.
"""
from __future__ import annotations

import sys

import folium
import numpy as np

from pathsnap import config
from pathsnap.pipeline import PathSnapPipeline
from pathsnap.preprocess.io import load_trip

MAX_NETWORK_SEGMENTS = 6000


def _segments(trip, arc_ids):
    u, v = trip.arcs[arc_ids, 0], trip.arcs[arc_ids, 1]
    return [[[trip.nodes[a, 1], trip.nodes[a, 0]], [trip.nodes[b, 1], trip.nodes[b, 0]]]
            for a, b in zip(u, v)]


def _route_layer(trip, arcs, name, color, weight, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    seq = np.asarray(arcs)
    seq = seq[seq >= 0]
    if len(seq):
        seq = seq[np.concatenate([[True], seq[1:] != seq[:-1]])]
        # one multi-segment polyline keeps the HTML small
        folium.PolyLine(_segments(trip, seq), color=color, weight=weight, opacity=0.85,
                        tooltip=name).add_to(fg)
    return fg


def make_map(partition: str, trip_id: str) -> str:
    trip = load_trip(trip_id, partition, with_route=True)       # offline analysis: labels allowed
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=15,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])

    # road network around the track
    pad = 0.002
    nx_, ny_ = trip.nodes[:, 0], trip.nodes[:, 1]
    in_box = (nx_ > lon.min() - pad) & (nx_ < lon.max() + pad) & (ny_ > lat.min() - pad) & (ny_ < lat.max() + pad)
    sel = np.where(in_box[trip.arcs[:, 0]] & in_box[trip.arcs[:, 1]])[0]
    if len(sel) > MAX_NETWORK_SEGMENTS:
        sel = sel[np.linspace(0, len(sel) - 1, MAX_NETWORK_SEGMENTS).astype(int)]
    net = folium.FeatureGroup(name=f"Road network ({len(sel)} arcs)", show=False)
    folium.PolyLine(_segments(trip, sel), color="#888888", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for la, lo, t in trip.track[:, [1, 0, 2]]:
        folium.CircleMarker([la, lo], radius=2, color="#1f6feb", fill=True, fill_opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)

    _route_layer(trip, trip.route, "Reference route", "#2ca02c", 6).add_to(m)

    if (config.ARTIFACTS_DIR / "model.joblib").exists():
        pipe = PathSnapPipeline()
        res = pipe.predict(load_trip(trip_id, partition, with_route=False))
        _route_layer(trip, res["clf_arc"], "Classifier-only route", "#ff7f0e", 3, show=False).add_to(m)
        _route_layer(trip, res["pred_arc"], "Classifier + HMM route", "#d62728", 4).add_to(m)

    legend = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;color:#222;
      padding:8px 12px;border:1px solid #999;border-radius:6px;font:13px sans-serif">
      <b>Legend</b><br><span style="color:#1f6feb">&#9679;</span> GPS observations<br>
      <span style="color:#2ca02c">&#9644;</span> Reference route<br>
      <span style="color:#ff7f0e">&#9644;</span> Classifier-only route<br>
      <span style="color:#d62728">&#9644;</span> Classifier + HMM route<br>
      <span style="color:#888">&#9644;</span> Road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    config.MAPS_DIR.mkdir(exist_ok=True)
    out = config.MAPS_DIR / f"map_{partition}_{trip_id}.html"
    m.save(out)
    return str(out)


if __name__ == "__main__":
    part, ids = sys.argv[1], sys.argv[2:]
    for tid in ids:
        print("wrote", make_map(part, tid))
