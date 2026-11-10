"""Driving circuity: network distance from planar / geodesic centroid distance.

Fitted on off-diagonal England & Wales MSOA pairs in
``3-msoa_snetwork_geodesic_dist.ipynb``:

    circuity(d) = alpha + beta * d_km ** (-gamma)

with geodesic distance in kilometres.  NTS ``TripDisIncSW`` is a reported
(network-like) length, so destination sampling and mode choice should compare
POI Euclidean lengths after this conversion.

Uncertainty is a lognormal residual around the mean circuity,
``c = c_mean(d) * exp(eps)``, ``eps ~ N(0, sigma^2)``.  ``SIGMA_LOG_CIRCUITY``
is the scale of log(observed / fitted) circuity implied by the MSOA fit
(R^2 ~ 0.89 on network km; residual CV ~ 0.18).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# circuity  d_net/d_geo = alpha + beta * d_geo^(-gamma)   (d_geo in km)
CIRCUITY_ALPHA = 1.1621965301302377
CIRCUITY_BETA = 1.935804956795695
CIRCUITY_GAMMA = 0.33680921795536917
CIRCUITY_MIN_KM = 0.05
CIRCUITY_CLIP = (1.02, 3.5)
SIGMA_LOG_CIRCUITY = 0.18
SIGMA_LOG_POI = 0.08
_MIN_GEO_M = 1.0


def mean_circuity(geodesic_km: ArrayLike) -> np.ndarray:
    """Mean driving circuity at geodesic distance(s) in kilometres."""
    d = np.clip(np.asarray(geodesic_km, dtype=float), CIRCUITY_MIN_KM, None)
    c = CIRCUITY_ALPHA + CIRCUITY_BETA * np.power(d, -CIRCUITY_GAMMA)
    return np.clip(c, CIRCUITY_CLIP[0], CIRCUITY_CLIP[1])


def geodesic_m_to_network_m(
    geodesic_m: ArrayLike,
    rng: Optional[np.random.Generator] = None,
    sigma: float = 0.0,
    poi_sigma: float = 0.0,
) -> ArrayLike:
    """Convert Euclidean / geodesic metres to implied driving-network metres.

    ``sigma`` is a *shared* trip-level log residual (one draw).  ``poi_sigma``
    is extra independent noise per candidate, used when ranking POIs against
    an NTS target length.
    """
    scalar = np.ndim(geodesic_m) == 0
    geo = np.asarray(geodesic_m, dtype=float)
    out = np.full(geo.shape, np.nan, dtype=float)
    ok = np.isfinite(geo) & (geo >= 0)
    if not ok.any():
        return float(out) if scalar else out

    g = np.clip(geo[ok], _MIN_GEO_M, None)
    c = mean_circuity(g / 1000.0)
    scale = 1.0
    if rng is not None and sigma and sigma > 0:
        scale = float(np.exp(rng.normal(0.0, float(sigma))))
    if rng is not None and poi_sigma and poi_sigma > 0:
        extra = np.exp(rng.normal(0.0, float(poi_sigma), size=g.shape))
        net = g * c * scale * extra
    else:
        net = g * c * scale
    out[ok] = net
    if scalar:
        return float(out.reshape(-1)[0])
    return out


def network_m_to_geodesic_m(network_m: ArrayLike, n_iter: int = 10) -> ArrayLike:
    """Invert the mean circuity curve: reported network metres → Euclidean metres."""
    scalar = np.ndim(network_m) == 0
    net = np.asarray(network_m, dtype=float)
    out = np.full(net.shape, np.nan, dtype=float)
    ok = np.isfinite(net) & (net > 0)
    if not ok.any():
        return float(out) if scalar else out

    n = np.clip(net[ok], _MIN_GEO_M, None)
    geo = n / 1.48
    a, b, g = CIRCUITY_ALPHA, CIRCUITY_BETA, CIRCUITY_GAMMA
    lo, hi = CIRCUITY_CLIP
    for _ in range(int(n_iter)):
        km = np.clip(geo / 1000.0, CIRCUITY_MIN_KM, None)
        c_raw = a + b * np.power(km, -g)
        clipped = (c_raw <= lo) | (c_raw >= hi)
        c = np.clip(c_raw, lo, hi)
        dc_dgeo = b * (-g) * np.power(km, -g - 1.0) * 0.001
        dc_dgeo = np.where(clipped, 0.0, dc_dgeo)
        deriv = c + geo * dc_dgeo
        deriv = np.where(np.abs(deriv) < 1e-8, 1.0, deriv)
        geo = np.clip(geo - (geo * c - n) / deriv, _MIN_GEO_M, None)
    out[ok] = geo
    if scalar:
        return float(out.reshape(-1)[0])
    return out
