"""Geometry helpers.

Degrees are not metres, so every distance is computed after projecting the
trip to a local metric plane. Trips are spread worldwide, so the projection
is centred on each trip's own track (azimuthal equidistant, WGS84).
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def make_projector(lon: np.ndarray, lat: np.ndarray) -> Transformer:
    """lon/lat (deg) -> local x/y (m), centred on the mean of the given points."""
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", local, always_xy=True)


def project(tr: Transformer, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([x, y])


def point_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from point p (2,) to segments a->b (n,2).

    Returns (dist, frac) where frac in [0,1] is the projection position along
    the segment.
    """
    ab = b - a
    ap = p - a
    denom = np.einsum("ij,ij->i", ab, ab)
    frac = np.where(denom > 0, np.einsum("ij,ij->i", ap, ab) / np.where(denom > 0, denom, 1), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    closest = a + ab * frac[:, None]
    dist = np.linalg.norm(p - closest, axis=1)
    return dist, frac


def bearing_xy(d: np.ndarray) -> np.ndarray:
    """Direction angle (rad, math convention) of displacement vectors (n,2)."""
    return np.arctan2(d[:, 1], d[:, 0])


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between angles, in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)
