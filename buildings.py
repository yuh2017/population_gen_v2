"""OSM / activity-POI classification and destination sampling.

Education, shop, personal, leisure and other destinations are drawn from a
purpose-labelled point pool (OSM footprints, or a prepared activity POI
layer).  A trip length ``d*`` is sampled from the NTS
``TripDisIncSW`` distribution for that purpose and the person's LAD
(``HHoldOSLAUA_B01ID``).  A POI in the same activity class is then chosen so
its driving-network distance (MSOA circuity
``c(d_km) = 1.162 + 1.936 d_km^(-0.34)``, plus a lognormal residual) is close
to ``d*``.  Footprint area is a secondary weight among polygon POIs at similar
length; prepared point layers use equal weights unless a weight column is
passed.  If the NTS pack is missing, sampling falls back to ``exp(-β d)``.

Home *zones* come from commuting flows.  A ``home`` POI is drawn inside the
origin OA when that OA has enough POIs; otherwise a point is sampled
uniformly in the OA polygon (new-development zones with no buildings).
The workplace *zone* is the destination MSOA of the OD pair.  A ``work``
POI inside that MSOA is matched to NTS length ``d*`` when the MSOA has
enough job POIs; otherwise a point is sampled in the destination MSOA
(or OA) polygon.  Gravity ``exp(-β d)`` is only the fallback if the NTS
pack is missing.

Prepare classified points outside this package (map activity labels and zone
codes onto generic columns), then call :func:`build_home_work_poi_pools` and
:func:`build_purpose_poi_pools`.  The OSM building loaders remain available
as a fallback.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .circuity import (
    SIGMA_LOG_CIRCUITY,
    SIGMA_LOG_POI,
    geodesic_m_to_network_m,
    network_m_to_geodesic_m,
)

DECAY_PURPOSES = ("education", "shop", "personal", "leisure", "other")

METERS_PER_MILE = 1609.34

# Fallback β (per mile) when the survey fit is missing. Values from the NTS CPCA extract.
DEFAULT_DECAY_BETA = {
    "education": 0.20,
    "shop": 0.15,
    "personal": 0.15,
    "leisure": 0.11,
    "other": 0.12,
}

# OSM `type` (building=*) → exclusive purpose, checked in this order.
_TYPE_TO_PURPOSE: Dict[str, str] = {}

for _t in (
    "school", "university", "college", "kindergarten", "education",
    "dormitory",
):
    _TYPE_TO_PURPOSE[_t] = "education"

for _t in (
    "retail", "supermarket", "kiosk", "shop", "retali", "mall", "commercial",
):
    _TYPE_TO_PURPOSE[_t] = "shop"

for _t in (
    "hospital", "healthcare", "medical_centre", "veterinary", "clinic",
    "civic", "government", "townhall", "public", "fire_station",
    "police_station", "social_facility", "care_home", "prison",
):
    _TYPE_TO_PURPOSE[_t] = "personal"

for _t in (
    "church", "chapel", "mosque", "cathedral", "pub", "hotel", "motel",
    "inn", "cafe", "restaurant", "sports_centre", "sports_hall", "stadium",
    "grandstand", "pavilion", "clubhouse", "club", "museum",
    "community_centre", "village_hall", "hall", "multiplex", "cinema",
    "bandstand", "events", "conference_centre", "social_club", "hostel",
    "arts centre", "sports", "theatre", "church_hall",
):
    _TYPE_TO_PURPOSE[_t] = "leisure"

for _t in (
    "office", "industrial", "warehouse", "train_station",
    "service", "works", "factory", "industrial_unit", "manufacture",
    "plant", "utility", "transportation", "railway",
    "laboratory", "amenity",
):
    _TYPE_TO_PURPOSE[_t] = "other"

_RESIDENTIAL = {
    "house", "residential", "detached", "semidetached_house", "terrace",
    "apartments", "bungalow", "static_caravan", "houseboat", "flats",
    "terraced_house", "houses", "cottage", "farmhouse", "park_home",
    "cabin", "country_house", "manor_house", "mansion", "stately_home",
    "pair of semi detache", "three delling block", "two semis",
    "four flats",
}

_ANCILLARY = {
    "garage", "garages", "shed", "roof", "carport", "hut", "outbuilding",
    "greenhouse", "glasshouse", "barn", "farm_auxiliary", "stable",
    "stables", "silo", "container", "ruins", "construction", "collapsed",
    "demolished", "abandoned", "no", "porch", "canopy", "gazebo",
    "conservatory", "summerhouse", "tent", "bridge", "tower", "clock_tower",
    "water_tower", "storage_tank", "pumping_station", "pumphouse",
    "pump_house", "transformer_kiosk", "signal_box", "bunker", "hangar",
    "toilets", "shelter", "seat", "lych_gate", "elevator", "lift",
    "parking", "disused", "fixme", "demolished 2024",
}

_NAME_KEYWORDS: Dict[str, Sequence[str]] = {
    "education": (
        "school", "college", "university", "academy", "nursery",
        "kindergarten", "primary", "infant", "junior", "campus",
    ),
    "shop": (
        "tesco", "sainsbury", "asda", "aldi", "lidl", "waitrose",
        "morrison", "co-op", "cooperative", "costcutter", "londis",
        "nisa", "one stop", "iceland", "marks", "m&s", "whsmith",
        "boots", "superstore", "supermarket", "retail park",
    ),
    "personal": (
        "hospital", "surgery", "clinic", "medical", "dental", "dentist",
        "pharmacy", "gp ", "health centre", "council", "library",
        "police", "fire station", "jobcentre",
    ),
    "leisure": (
        "leisure", "sports", "gym", "fitness", "theatre", "cinema",
        "pub", "hotel", "church", "chapel", "mosque", "pavilion",
        "swimming", "play centre", "golf", "stadium", "museum",
        "community centre", "village hall", "scout",
        "mcdonald", "kfc", "burger king", "costa", "starbucks",
        "subway", "greggs", "pizza", "nando",
    ),
}

# Fast-food / cafe names are leisure (eat-out) rather than grocery shop.
_LEISURE_NAME_OVERRIDE = {
    "mcdonald", "kfc", "burger king", "costa", "starbucks", "subway",
    "greggs", "pizza", "nando", "pub", "hotel", "travelodge", "premier inn",
}


def _norm_text(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower()


def classify_building_purpose(btype: str, name: str = "") -> Optional[str]:
    """Map one OSM building to a trip purpose, or None if it is not a destination."""
    t = _norm_text(btype)
    n = _norm_text(name)
    if t in _ANCILLARY or t in _RESIDENTIAL:
        return None
    if t in _TYPE_TO_PURPOSE:
        return _TYPE_TO_PURPOSE[t]
    for purpose, keys in _NAME_KEYWORDS.items():
        if any(k in n for k in keys):
            if purpose == "shop" and any(k in n for k in _LEISURE_NAME_OVERRIDE):
                return "leisure"
            return purpose
    if t in {"commercial", "office", "industrial"}:
        return "other"
    if n:
        # Named, untyped, non-residential building: treat as a generic destination.
        return "other"
    return None


def classify_osm_buildings(gdf: "pd.DataFrame") -> "pd.DataFrame":
    """Add ``purpose`` from OSM ``type`` and ``name``. Drops non-destinations."""
    out = gdf.copy()
    types = out["type"] if "type" in out.columns else ""
    names = out["name"] if "name" in out.columns else ""
    if not isinstance(types, pd.Series):
        types = pd.Series([""] * len(out), index=out.index)
    if not isinstance(names, pd.Series):
        names = pd.Series([""] * len(out), index=out.index)
    out["purpose"] = [
        classify_building_purpose(t, n) for t, n in zip(types, names)
    ]
    return out.loc[out["purpose"].notna()].copy()


def _footprint_weights(gdf, min_m2: float = 25.0) -> np.ndarray:
    geom = gdf.geometry
    try:
        area = geom.area.to_numpy(dtype=float)
    except Exception:
        area = np.ones(len(gdf), dtype=float)
    area = np.where(np.isfinite(area) & (area > 0), area, min_m2)
    area = np.clip(area, min_m2, np.nanpercentile(area, 99.5) if len(area) else min_m2)
    return area


def load_purpose_building_pools(
    buildings_path: str,
    oa_gdf=None,
    geoid_col: str = "OA11CD",
    target_crs: str = "EPSG:27700",
) -> Dict[str, Dict]:
    """Load OSM footprints, classify by purpose, reproject, attach OA codes.

    Each pool is a dict of numpy arrays ``x``, ``y``, ``weight``, ``geoid``
    in *target_crs* (British National Grid when persons use EPSG:27700).
    """
    import geopandas as gpd

    gdf = gpd.read_file(buildings_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    classified = classify_osm_buildings(gdf)
    if classified.empty:
        return {}

    classified = classified.to_crs(target_crs)
    classified["geometry"] = classified.geometry.centroid
    classified["_w"] = _footprint_weights(classified)

    if oa_gdf is not None and geoid_col in oa_gdf.columns:
        oa = oa_gdf[[geoid_col, "geometry"]].copy()
        if oa.crs != classified.crs:
            oa = oa.to_crs(classified.crs)
        joined = gpd.sjoin_nearest(
            classified, oa, how="left", max_distance=500.0, distance_col="_oa_dist",
        )
        if len(joined) > len(classified):
            joined = joined.loc[~joined.index.duplicated(keep="first")]
        if geoid_col in joined.columns:
            classified = classified.copy()
            classified["_geoid"] = (
                joined[geoid_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
            )
        else:
            classified["_geoid"] = ""
    else:
        classified["_geoid"] = ""

    pools: Dict[str, Dict] = {}
    for purpose in DECAY_PURPOSES:
        sub = classified.loc[classified["purpose"] == purpose]
        if len(sub) == 0:
            continue
        xy = np.column_stack([sub.geometry.x.to_numpy(dtype=float),
                              sub.geometry.y.to_numpy(dtype=float)])
        ok = np.isfinite(xy).all(axis=1)
        xy = xy[ok]
        w = sub["_w"].to_numpy(dtype=float)[ok]
        geoids = np.asarray(sub["_geoid"].to_numpy())[ok]
        pools[purpose] = {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "weight": w,
            "geoid": geoids,
            "n": int(len(xy)),
        }
    return pools


def sample_decay_destination(
    origin_x: float,
    origin_y: float,
    purpose: str,
    pools: Dict[str, Dict],
    beta: Optional[float] = None,
    coords_are_projected: bool = True,
    rng: Optional[np.random.Generator] = None,
    min_sep_m: float = 0.0,
    nts_by_purpose: Optional[Dict] = None,
    lad: Optional[str] = None,
    nts_purpose: Optional[str] = None,
) -> Optional[tuple]:
    """Draw one building. Returns ``(x, y, geoid)`` or None.

    When ``nts_by_purpose`` is given, a reported NTS length ``d*`` is sampled
    for *purpose* (or ``nts_purpose``) in *lad*, then a POI in that activity
    class is chosen whose driving-network distance is close to ``d*``.
    Otherwise footprint area × ``exp(-β d)``.  ``min_sep_m`` excludes
    candidates closer than that Euclidean distance when at least one farther
    building exists.
    """
    key = str(purpose).strip().lower() if purpose else ""
    pool = pools.get(key) or pools.get("other")
    if not pool or pool["n"] < 1:
        return None
    ox, oy = float(origin_x), float(origin_y)
    if not np.isfinite(ox) or not np.isfinite(oy):
        return None

    rng = np.random.default_rng() if rng is None else rng
    pack = {
        "x": np.asarray(pool["x"], dtype=float),
        "y": np.asarray(pool["y"], dtype=float),
        "weight": np.asarray(pool["weight"], dtype=float),
        "geoid": np.asarray(pool["geoid"]) if "geoid" in pool else np.array([""] * int(pool["n"]), dtype=object),
        "n": int(pool["n"]),
    }
    sep = float(min_sep_m) if min_sep_m else 0.0
    nts_key = str(nts_purpose or key).strip().lower()
    fallback = None if nts_by_purpose and nts_key in nts_by_purpose else "other"
    dist_pack = _dist_pack_for_lad(
        nts_by_purpose, nts_key, str(lad or ""), fallback_purpose=fallback,
    )
    d_star = None
    if coords_are_projected:
        d_star = sample_nts_distance_m(dist_pack, rng, dmin_m=max(sep, 40.0))
    if d_star is not None and np.isfinite(d_star) and d_star > 0:
        drawn = _match_poi_to_dstar(
            ox, oy, pack, d_star, rng,
            min_sep_m=sep,
            relax_last=True,
        )
        if drawn is not None:
            return drawn

    dx = pack["x"] - ox
    dy = pack["y"] - oy
    dist = np.hypot(dx, dy)
    if not coords_are_projected:
        dist = dist * 111_320.0
    net_m = geodesic_m_to_network_m(dist, rng=rng, sigma=SIGMA_LOG_CIRCUITY)
    dist_mi = np.asarray(net_m, dtype=float) / METERS_PER_MILE
    b = DEFAULT_DECAY_BETA.get(key, 0.12) if beta is None else float(beta)
    b = float(np.clip(b, 0.02, 1.5))
    logp = np.log(np.clip(pack["weight"], 1e-12, None)) - b * np.clip(dist_mi, 0.05, 80.0)
    if sep and sep > 0:
        far = dist >= sep
        if bool(np.any(far)):
            logp = np.where(far, logp, -np.inf)
    finite = np.isfinite(logp)
    if not bool(np.any(finite)):
        return None
    logp = np.where(finite, logp, -np.inf)
    logp = logp - np.nanmax(logp)
    p = np.exp(logp)
    p = np.where(np.isfinite(p), p, 0.0)
    i = _choice_index(p, rng)
    geoid = pack["geoid"][i]
    geoid = "" if geoid is None or str(geoid) in {"nan", "None"} else str(geoid)
    return float(pack["x"][i]), float(pack["y"][i]), geoid


def summarise_pools(pools: Dict[str, Dict]) -> str:
    parts = [f"{p}={pools[p]['n']:,}" for p in DECAY_PURPOSES if p in pools]
    return ", ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Home / workplace POIs + LAD-specific NTS distance matching
# ---------------------------------------------------------------------------

_MIN_COMMUTE_M = 80.0
_POI_JITTER_M = 12.0
_NEAR_OA_M = 250.0
_DEFAULT_WORK_BETA = 0.10
_DEFAULT_HOME_BETA = 0.12
_MIN_POIS_PER_ZONE = 3
_MSOA_GEOM_KEY = "__msoa_geom__"


def classify_building_role(btype: str, name: str = "") -> Optional[str]:
    """``home`` for residential footprints, ``work`` for job-hosting buildings."""
    t = _norm_text(btype)
    if t in _ANCILLARY:
        return None
    if t in _RESIDENTIAL:
        return "home"
    if classify_building_purpose(btype, name) is not None:
        return "work"
    return None


def _finite_xy(x: np.ndarray, y: np.ndarray, *arrays):
    ok = np.isfinite(x) & np.isfinite(y)
    out = [np.asarray(x)[ok], np.asarray(y)[ok]]
    for a in arrays:
        out.append(np.asarray(a)[ok])
    return tuple(out)


def _index_pool_by_key(keys: np.ndarray, x, y, weight, geoid=None, lad=None, msoa=None) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    keys = np.asarray(keys).astype(str)
    for i, key in enumerate(keys):
        if not key or key in {"nan", "None", "NoneType"}:
            continue
        rec = out.setdefault(key, {"_i": []})
        rec["_i"].append(i)
    for key, rec in out.items():
        idx = np.asarray(rec["_i"], dtype=int)
        pack = {
            "x": x[idx],
            "y": y[idx],
            "weight": weight[idx],
            "n": int(len(idx)),
        }
        if geoid is not None:
            pack["geoid"] = np.asarray(geoid)[idx]
        if lad is not None:
            pack["lad"] = np.asarray(lad)[idx]
        if msoa is not None:
            pack["msoa"] = np.asarray(msoa)[idx]
        out[key] = pack
    return out


def _empty_role_pack() -> Dict:
    return {
        "n": 0, "by_oa": {}, "by_lad": {}, "by_msoa": {},
        "x": np.array([]), "y": np.array([]),
        "weight": np.array([]), "geoid": np.array([]),
        "lad": np.array([]), "msoa": np.array([]),
    }


def _empty_purpose_pack() -> Dict:
    return {
        "n": 0,
        "x": np.array([]),
        "y": np.array([]),
        "weight": np.array([]),
        "geoid": np.array([]),
    }


def _codes_from_column(gdf, col: Optional[str], n: int) -> np.ndarray:
    if col and col in gdf.columns:
        s = gdf[col].astype(str).replace(
            {"nan": "", "None": "", "<NA>": "", "NoneType": ""},
        )
        return s.fillna("").to_numpy()
    return np.full(n, "", dtype=object)


def _point_or_area_weights(gdf, weight_col: Optional[str] = None, min_w: float = 1.0) -> np.ndarray:
    if weight_col and weight_col in gdf.columns:
        w = pd.to_numeric(gdf[weight_col], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(w) & (w > 0), w, min_w)
    try:
        area = gdf.geometry.area.to_numpy(dtype=float)
        if np.isfinite(area).any() and float(np.nanmax(area)) > 1.0:
            return _footprint_weights(gdf)
    except Exception:
        pass
    return np.full(len(gdf), float(min_w), dtype=float)


def _to_target_points(gdf, target_crs: str):
    out = gdf.copy()
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    if target_crs and str(out.crs) != str(target_crs):
        out = out.to_crs(target_crs)
    non_point = ~out.geom_type.astype(str).str.contains("Point", case=False, na=False)
    if bool(non_point.any()):
        out.loc[non_point, "geometry"] = out.loc[non_point].geometry.centroid
    return out


def build_labeled_poi_pools(
    gdf,
    label_col: str,
    labels: Sequence[str],
    *,
    geoid_col: Optional[str] = None,
    lad_col: Optional[str] = None,
    msoa_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    target_crs: str = "EPSG:27700",
    with_zone_index: bool = False,
) -> Dict[str, Dict]:
    """Sampling pools from already-classified point (or polygon) features.

    Column names are supplied by the caller. The package does not read OSM
    ``type`` / ``name`` or a specific activity-GPKG schema here: map those
    fields onto *label_col* / zone columns first.
    """
    labels = [str(x).strip().lower() for x in labels]
    empty = _empty_role_pack() if with_zone_index else _empty_purpose_pack()
    if gdf is None or len(gdf) == 0 or not label_col or label_col not in gdf.columns:
        return {lab: dict(empty) for lab in labels}

    pts = _to_target_points(gdf, target_crs)
    lab = pts[label_col].astype(str).str.strip().str.lower().to_numpy()
    w_all = _point_or_area_weights(pts, weight_col)
    geoids = _codes_from_column(pts, geoid_col, len(pts))
    lads = _codes_from_column(pts, lad_col, len(pts))
    msoas = _codes_from_column(pts, msoa_col, len(pts))
    x_all = pts.geometry.x.to_numpy(dtype=float)
    y_all = pts.geometry.y.to_numpy(dtype=float)

    pools: Dict[str, Dict] = {}
    for role in labels:
        mask = lab == role
        if not bool(mask.any()):
            pools[role] = dict(empty)
            continue
        x, y, w, gids, lids, mids = _finite_xy(
            x_all[mask], y_all[mask], w_all[mask], geoids[mask], lads[mask], msoas[mask],
        )
        pack: Dict = {
            "x": x, "y": y, "weight": w, "geoid": gids, "n": int(len(x)),
        }
        if with_zone_index:
            pack["lad"] = lids
            pack["msoa"] = mids
            pack["by_oa"] = _index_pool_by_key(gids, x, y, w, geoid=gids, lad=lids, msoa=mids)
            pack["by_lad"] = _index_pool_by_key(lids, x, y, w, geoid=gids, lad=lids, msoa=mids)
            pack["by_msoa"] = _index_pool_by_key(mids, x, y, w, geoid=gids, lad=lids, msoa=mids)
        pools[role] = pack
    return pools


def build_home_work_poi_pools(
    gdf,
    role_col: str = "purpose",
    *,
    geoid_col: Optional[str] = "OA11CD",
    lad_col: Optional[str] = "LAD11CD",
    msoa_col: Optional[str] = "MSOA11CD",
    weight_col: Optional[str] = None,
    target_crs: str = "EPSG:27700",
) -> Dict[str, Dict]:
    """Home/work sampling pools from points already labelled ``home`` / ``work``."""
    return build_labeled_poi_pools(
        gdf, role_col, ("home", "work"),
        geoid_col=geoid_col, lad_col=lad_col, msoa_col=msoa_col,
        weight_col=weight_col, target_crs=target_crs, with_zone_index=True,
    )


def build_purpose_poi_pools(
    gdf,
    purpose_col: str = "purpose",
    purposes: Optional[Sequence[str]] = None,
    *,
    geoid_col: Optional[str] = "OA11CD",
    weight_col: Optional[str] = None,
    target_crs: str = "EPSG:27700",
) -> Dict[str, Dict]:
    """Education/shop/personal/leisure/other pools from labelled activity points."""
    labels = tuple(purposes) if purposes is not None else DECAY_PURPOSES
    return build_labeled_poi_pools(
        gdf, purpose_col, labels,
        geoid_col=geoid_col, weight_col=weight_col,
        target_crs=target_crs, with_zone_index=False,
    )


def _choice_index(weights: np.ndarray, rng: np.random.Generator) -> int:
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    n = int(len(w))
    if n == 0:
        raise ValueError("empty weights")
    if s <= 0:
        return int(rng.integers(n))
    return int(rng.choice(n, p=w / s))


def _jitter_xy(x: float, y: float, rng: np.random.Generator, jitter_m: float = _POI_JITTER_M):
    if jitter_m and jitter_m > 0:
        x = float(x) + float(rng.uniform(-jitter_m, jitter_m))
        y = float(y) + float(rng.uniform(-jitter_m, jitter_m))
    return float(x), float(y)


def load_home_work_building_pools(
    buildings_path: str,
    oa_gdf=None,
    geoid_col: str = "OA11CD",
    lad_col: str = "LAD11CD",
    msoa_col: str = "MSOA11CD",
    target_crs: str = "EPSG:27700",
) -> Dict[str, Dict]:
    """Residential (home) and job-hosting (work) OSM footprints, indexed by OA, MSOA and LAD."""
    import geopandas as gpd

    empty_role = {
        "n": 0, "by_oa": {}, "by_lad": {}, "by_msoa": {},
        "x": np.array([]), "y": np.array([]),
        "weight": np.array([]), "geoid": np.array([]),
        "lad": np.array([]), "msoa": np.array([]),
    }

    gdf = gpd.read_file(buildings_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    types = gdf["type"] if "type" in gdf.columns else ""
    names = gdf["name"] if "name" in gdf.columns else ""
    if not isinstance(types, pd.Series):
        types = pd.Series([""] * len(gdf), index=gdf.index)
    if not isinstance(names, pd.Series):
        names = pd.Series([""] * len(gdf), index=gdf.index)
    gdf = gdf.copy()
    gdf["_role"] = [classify_building_role(t, n) for t, n in zip(types, names)]
    gdf = gdf.loc[gdf["_role"].notna()].copy()
    if gdf.empty:
        return {"home": dict(empty_role), "work": dict(empty_role)}

    gdf = gdf.to_crs(target_crs)
    gdf["geometry"] = gdf.geometry.centroid
    gdf["_w"] = _footprint_weights(gdf)

    geoids = np.full(len(gdf), "", dtype=object)
    lads = np.full(len(gdf), "", dtype=object)
    msoas = np.full(len(gdf), "", dtype=object)
    if oa_gdf is not None and geoid_col in oa_gdf.columns:
        keep = [geoid_col, "geometry"]
        if lad_col in oa_gdf.columns:
            keep.append(lad_col)
        if msoa_col in oa_gdf.columns:
            keep.append(msoa_col)
        oa = oa_gdf[keep].copy()
        if oa.crs != gdf.crs:
            oa = oa.to_crs(gdf.crs)
        joined = gpd.sjoin_nearest(
            gdf, oa, how="left", max_distance=500.0, distance_col="_oa_dist",
        )
        if len(joined) > len(gdf):
            joined = joined.loc[~joined.index.duplicated(keep="first")]
        if geoid_col in joined.columns:
            geoids = (
                joined[geoid_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
            )
        if lad_col in joined.columns:
            lads = (
                joined[lad_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
            )
        elif lad_col in oa_gdf.columns:
            oa_to_lad = {
                str(o): ("" if pd.isna(v) else str(v))
                for o, v in zip(oa_gdf[geoid_col], oa_gdf[lad_col])
            }
            lads = np.array([oa_to_lad.get(str(g), "") for g in geoids], dtype=object)
        if msoa_col in joined.columns:
            msoas = (
                joined[msoa_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
            )
        elif msoa_col in oa_gdf.columns:
            oa_to_msoa = {
                str(o): ("" if pd.isna(v) else str(v))
                for o, v in zip(oa_gdf[geoid_col], oa_gdf[msoa_col])
            }
            msoas = np.array([oa_to_msoa.get(str(g), "") for g in geoids], dtype=object)

    pools: Dict[str, Dict] = {}
    for role in ("home", "work"):
        mask = (gdf["_role"].to_numpy() == role)
        if not mask.any():
            pools[role] = dict(empty_role)
            continue
        xy = np.column_stack([
            gdf.geometry.x.to_numpy(dtype=float)[mask],
            gdf.geometry.y.to_numpy(dtype=float)[mask],
        ])
        w = gdf["_w"].to_numpy(dtype=float)[mask]
        gids = geoids[mask]
        lids = lads[mask]
        mids = msoas[mask]
        xy0, xy1, w, gids, lids, mids = _finite_xy(xy[:, 0], xy[:, 1], w, gids, lids, mids)
        pools[role] = {
            "x": xy0,
            "y": xy1,
            "weight": w,
            "geoid": gids,
            "lad": lids,
            "msoa": mids,
            "n": int(len(xy0)),
            "by_oa": _index_pool_by_key(gids, xy0, xy1, w, geoid=gids, lad=lids, msoa=mids),
            "by_lad": _index_pool_by_key(lids, xy0, xy1, w, geoid=gids, lad=lids, msoa=mids),
            "by_msoa": _index_pool_by_key(mids, xy0, xy1, w, geoid=gids, lad=lids, msoa=mids),
        }
    return pools


def build_oa_location_index(
    oa_gdf,
    geoid_col: str = "OA11CD",
    lad_col: str = "LAD11CD",
    msoa_col: str = "MSOA11CD",
) -> Dict[str, Dict]:
    """Per-OA centroid, envelope diagonal, LAD/MSOA codes, and polygon."""
    out: Dict[str, Dict] = {}
    if oa_gdf is None or geoid_col not in oa_gdf.columns:
        return out
    has_lad = lad_col in oa_gdf.columns
    has_msoa = msoa_col in oa_gdf.columns
    for row in oa_gdf.itertuples(index=False):
        oa = str(getattr(row, geoid_col, "") or "")
        if not oa or oa in {"nan", "None"}:
            continue
        geom = getattr(row, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        c = geom.centroid
        lad = ""
        if has_lad:
            raw = getattr(row, lad_col, "")
            lad = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw)
        msoa = ""
        if has_msoa:
            raw = getattr(row, msoa_col, "")
            msoa = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw)
        out[oa] = {
            "lad": lad,
            "msoa": msoa,
            "cx": float(c.x),
            "cy": float(c.y),
            "diag_m": float(np.hypot(maxx - minx, maxy - miny)),
            "geom": geom,
        }
    buckets: Dict[str, list] = {}
    for rec in out.values():
        m = rec.get("msoa") or ""
        geom = rec.get("geom")
        if m and m not in {"nan", "None"} and geom is not None:
            buckets.setdefault(m, []).append(geom)
    msoa_geoms: Dict[str, object] = {}
    if buckets:
        from shapely.ops import unary_union
        for m, geoms in buckets.items():
            try:
                msoa_geoms[m] = geoms[0] if len(geoms) == 1 else unary_union(geoms)
            except Exception:
                msoa_geoms[m] = geoms[0]
    if msoa_geoms:
        out[_MSOA_GEOM_KEY] = msoa_geoms
    return out


def add_msoa_to_oa(
    oa_gdf,
    msoa_gdf,
    oa_col: str = "OA11CD",
    msoa_col: str = "MSOA11CD",
):
    """Attach ``MSOA11CD`` to OA polygons by nearest MSOA centroid join."""
    import geopandas as gpd

    oa = oa_gdf.copy()
    if msoa_col in oa.columns and oa[msoa_col].notna().any():
        return oa
    if msoa_gdf is None or msoa_col not in msoa_gdf.columns:
        oa[msoa_col] = ""
        return oa
    msoa = msoa_gdf[[msoa_col, "geometry"]].copy()
    if oa.crs != msoa.crs:
        msoa = msoa.to_crs(oa.crs)
    pts = gpd.GeoDataFrame(
        {oa_col: oa[oa_col].to_numpy()},
        geometry=oa.geometry.centroid,
        crs=oa.crs,
        index=oa.index,
    )
    joined = gpd.sjoin_nearest(pts, msoa, how="left", max_distance=2500.0)
    if len(joined) > len(pts):
        joined = joined.loc[~joined.index.duplicated(keep="first")]
    oa[msoa_col] = (
        joined[msoa_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
    )
    return oa


def summarise_home_work_pools(pools: Dict[str, Dict]) -> str:
    parts = []
    for role in ("home", "work"):
        pack = pools.get(role) or {}
        parts.append(
            f"{role}={int(pack.get('n') or 0):,} "
            f"({len(pack.get('by_oa') or {})} OAs, "
            f"{len(pack.get('by_msoa') or {})} MSOAs, "
            f"{len(pack.get('by_lad') or {})} LADs)"
        )
    return "; ".join(parts)


def sample_nts_distance_m(
    dist_pack: Optional[Dict],
    rng: np.random.Generator,
    dmin_m: float = _MIN_COMMUTE_M,
    dmax_m: Optional[float] = None,
) -> Optional[float]:
    """Draw one trip length (metres) from an empirical NTS pack."""
    if not dist_pack:
        return None
    miles = np.asarray(dist_pack.get("miles"), dtype=float)
    weight = np.asarray(dist_pack.get("weight"), dtype=float)
    if miles.size == 0:
        return None
    if weight.size != miles.size:
        weight = np.ones(miles.size, dtype=float)
    meters = miles * METERS_PER_MILE
    lo = float(dmin_m)
    hi = float(dmax_m) if dmax_m is not None and np.isfinite(dmax_m) else float(meters.max())
    if hi <= lo:
        hi = lo + 50.0
    mask = np.isfinite(meters) & (meters >= lo) & (meters <= hi) & np.isfinite(weight) & (weight > 0)
    if int(mask.sum()) < 8:
        mask = np.isfinite(meters) & (meters >= lo) & np.isfinite(weight) & (weight > 0)
    if not mask.any():
        return float(np.clip(0.5 * (lo + hi), lo, hi))
    sub_m = meters[mask]
    sub_w = weight[mask]
    return float(rng.choice(sub_m, p=sub_w / sub_w.sum()))


def _dist_pack_for_lad(
    nts_by_purpose: Optional[Dict],
    purpose: str,
    lad: str,
    fallback_purpose: Optional[str] = None,
) -> Optional[Dict]:
    if not nts_by_purpose:
        return None
    key = str(purpose or "").strip().lower()
    block = nts_by_purpose.get(key)
    if not block and fallback_purpose:
        block = nts_by_purpose.get(str(fallback_purpose).strip().lower())
    if not block:
        return None
    by_lad = block.get("by_lad") or {}
    lad_key = str(lad or "").strip()
    if lad_key and lad_key in by_lad and (by_lad[lad_key].get("n") or 0) >= 15:
        return by_lad[lad_key]
    return block.get("all")


def _beta_for_lad(nts_by_purpose: Optional[Dict], purpose: str, lad: str, default: float) -> float:
    pack = _dist_pack_for_lad(nts_by_purpose, purpose, lad)
    beta = None if pack is None else pack.get("beta")
    if beta is None or not np.isfinite(beta) or float(beta) <= 0:
        return float(default)
    return float(np.clip(beta, 0.02, 1.5))


def _sample_poi(
    ox: float,
    oy: float,
    pack: Dict,
    rng: np.random.Generator,
    *,
    target_m: Optional[float] = None,
    beta_per_mile: Optional[float] = None,
    min_sep_m: float = _MIN_COMMUTE_M,
    max_radius_m: Optional[float] = None,
    sigma_m: Optional[float] = None,
    target_is_network_m: bool = True,
    relax_radius: bool = True,
) -> Optional[Tuple[float, float, str]]:
    """Sample a POI. ``target_m`` / ``beta_per_mile`` are NTS reported lengths."""
    xs = np.asarray(pack["x"], dtype=float)
    ys = np.asarray(pack["y"], dtype=float)
    ww = np.asarray(pack["weight"], dtype=float)
    if len(xs) == 0:
        return None
    dist = np.hypot(xs - ox, ys - oy)
    geoid = np.asarray(pack["geoid"]) if "geoid" in pack else np.array([""] * len(xs), dtype=object)
    net = geodesic_m_to_network_m(dist)
    keep = np.ones(len(xs), dtype=bool)
    if max_radius_m is not None and np.isfinite(max_radius_m) and max_radius_m > 0:
        radius_metric = net if target_is_network_m else dist
        keep &= radius_metric <= float(max_radius_m)
        n_keep = int(keep.sum())
        if n_keep < 5:
            if relax_radius:
                keep = np.ones(len(xs), dtype=bool)
            elif n_keep == 0:
                return None
    xs, ys, ww, dist, net, geoid = xs[keep], ys[keep], ww[keep], dist[keep], np.asarray(net)[keep], geoid[keep]

    poi_sigma = SIGMA_LOG_POI if (target_m is not None and np.isfinite(target_m)) else 0.0
    net_draw = geodesic_m_to_network_m(
        dist, rng=rng, sigma=SIGMA_LOG_CIRCUITY, poi_sigma=poi_sigma,
    )
    net_draw = np.asarray(net_draw, dtype=float)
    match_d = net_draw if target_is_network_m else dist
    decay_d = net_draw if target_is_network_m else dist

    logp = np.log(np.clip(ww, 1e-12, None))
    far_enough = dist >= float(min_sep_m)
    if target_m is not None and np.isfinite(target_m):
        sig = float(sigma_m) if sigma_m else max(150.0, 0.25 * float(target_m))
        logp = logp - 0.5 * ((match_d - float(target_m)) / sig) ** 2
    if beta_per_mile is not None and np.isfinite(beta_per_mile):
        dist_mi = np.clip(decay_d / METERS_PER_MILE, 0.05, 80.0)
        logp = logp - float(beta_per_mile) * dist_mi
    if far_enough.any():
        logp = np.where(far_enough, logp, -1e12)
    logp = logp - np.nanmax(logp)
    p = np.exp(logp)
    p = np.where(np.isfinite(p), p, 0.0)
    i = _choice_index(p, rng)
    gid = geoid[i] if len(geoid) else ""
    gid = "" if gid is None or str(gid) in {"nan", "None"} else str(gid)
    x, y = _jitter_xy(xs[i], ys[i], rng)
    return x, y, gid


def _dstar_caps(d_star: float, *, wide: bool = True) -> list:
    caps = [
        max(d_star * 1.5, 800.0),
        max(d_star * 2.5, 1600.0),
    ]
    if wide:
        caps.append(max(d_star * 4.0, 3200.0))
        caps.append(40.0 * METERS_PER_MILE)
    return caps


def _match_poi_to_dstar(
    ox: float,
    oy: float,
    pack: Dict,
    d_star: float,
    rng: np.random.Generator,
    *,
    min_sep_m: float = _MIN_COMMUTE_M,
    caps: Optional[Sequence[float]] = None,
    relax_last: bool = False,
) -> Optional[Tuple[float, float, str]]:
    """Pick a POI whose network length is close to reported NTS metres ``d_star``."""
    if not pack or not pack.get("n"):
        return None
    if d_star is None or not np.isfinite(d_star) or d_star <= 0:
        return None
    d_star = float(d_star)
    if caps is None:
        caps = _dstar_caps(d_star, wide=True)
    sigma = max(150.0, 0.25 * d_star)
    for cap in caps:
        drawn = _sample_poi(
            ox, oy, pack, rng,
            target_m=d_star,
            beta_per_mile=None,
            min_sep_m=min_sep_m,
            max_radius_m=float(cap),
            sigma_m=sigma,
            target_is_network_m=True,
            relax_radius=False,
        )
        if drawn is not None:
            return drawn
    if not relax_last:
        return None
    return _sample_poi(
        ox, oy, pack, rng,
        target_m=d_star,
        min_sep_m=min_sep_m,
        max_radius_m=None,
        sigma_m=max(150.0, 0.35 * d_star),
        target_is_network_m=True,
        relax_radius=True,
    )


def _role_flat_pack(pools: Dict, role: str) -> Optional[Dict]:
    rec = pools.get(role) or {}
    if not rec.get("n"):
        return None
    return {
        "x": rec["x"],
        "y": rec["y"],
        "weight": rec["weight"],
        "geoid": rec.get("geoid"),
        "n": rec["n"],
    }


def _boost_geoid_weight(pack: Optional[Dict], geoid: str, factor: float = 4.0) -> Optional[Dict]:
    if not pack or not geoid or factor == 1.0:
        return pack
    g = pack.get("geoid")
    if g is None:
        return pack
    g = np.asarray(g).astype(str)
    w = np.asarray(pack["weight"], dtype=float).copy()
    if w.size != g.size:
        return pack
    w[g == str(geoid)] *= float(factor)
    out = dict(pack)
    out["weight"] = w
    return out


def _offset_from_home(
    hx: float,
    hy: float,
    dist_m: float,
    rng: np.random.Generator,
    polygon=None,
) -> Tuple[float, float]:
    dist_m = float(max(dist_m, _MIN_COMMUTE_M))
    from shapely.geometry import Point

    for _ in range(12):
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        x = hx + dist_m * np.cos(ang)
        y = hy + dist_m * np.sin(ang)
        if polygon is None:
            return float(x), float(y)
        try:
            if polygon.contains(Point(x, y)):
                return float(x), float(y)
        except Exception:
            return float(x), float(y)
    if polygon is not None:
        try:
            rp = polygon.representative_point()
            dx, dy = float(rp.x) - hx, float(rp.y) - hy
            n = float(np.hypot(dx, dy))
            if n > 1.0:
                scale = min(dist_m, max(n, _MIN_COMMUTE_M)) / n
                x = hx + dx * scale
                y = hy + dy * scale
                if polygon.contains(Point(x, y)):
                    return float(x), float(y)
            return float(rp.x), float(rp.y)
        except Exception:
            pass
    return float(hx + dist_m), float(hy)


def _pack_from_oa(pools: Dict[str, Dict], role: str, oa: str) -> Optional[Dict]:
    by_oa = (pools.get(role) or {}).get("by_oa") or {}
    pack = by_oa.get(str(oa or ""))
    if pack and pack.get("n"):
        return pack
    return None


def _pack_from_msoa(pools: Dict[str, Dict], role: str, msoa: str) -> Optional[Dict]:
    by_msoa = (pools.get(role) or {}).get("by_msoa") or {}
    pack = by_msoa.get(str(msoa or ""))
    if pack and pack.get("n"):
        return pack
    return None


def _pack_if_enough(pack: Optional[Dict], min_n: int) -> Optional[Dict]:
    if pack is None:
        return None
    try:
        n = int(pack.get("n") or 0)
    except (TypeError, ValueError):
        n = 0
    return pack if n >= max(int(min_n), 1) else None


def _random_point_in_polygon(geom, rng: np.random.Generator, n_try: int = 40) -> Optional[Tuple[float, float]]:
    """Uniform point in *geom* (rejection sampling in the envelope)."""
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        return None
    from shapely.geometry import Point

    try:
        minx, miny, maxx, maxy = geom.bounds
    except Exception:
        return None
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return None
    if maxx <= minx or maxy <= miny:
        try:
            c = geom.centroid
            return float(c.x), float(c.y)
        except Exception:
            return None
    for _ in range(n_try):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        try:
            pt = Point(x, y)
            if geom.contains(pt) or geom.covers(pt):
                return x, y
        except Exception:
            continue
    try:
        rp = geom.representative_point()
        return float(rp.x), float(rp.y)
    except Exception:
        try:
            c = geom.centroid
            return float(c.x), float(c.y)
        except Exception:
            return None


def _msoa_polygon(oa_index: Dict[str, Dict], msoa: str, oa_meta: Optional[Dict] = None):
    if msoa:
        geoms = oa_index.get(_MSOA_GEOM_KEY) or {}
        if isinstance(geoms, dict) and msoa in geoms:
            return geoms[msoa]
    if oa_meta:
        return oa_meta.get("geom")
    return None


def sample_home_work_xy(
    home_oa: str,
    work_oa: str,
    pools: Dict[str, Dict],
    oa_index: Dict[str, Dict],
    rng: np.random.Generator,
    nts_by_purpose: Optional[Dict] = None,
    home_fallback_xy: Optional[Tuple[float, float]] = None,
    work_fallback_xy: Optional[Tuple[float, float]] = None,
    min_pois_per_zone: int = _MIN_POIS_PER_ZONE,
) -> Dict:
    """Draw home and workplace coordinates from flow zones + NTS lengths.

    Home stays in the flow origin OA. If that OA has fewer than
    *min_pois_per_zone* home POIs, a point is sampled inside the OA polygon
    (new development, or a zone with no POIs). The point stays in that
    polygon. Workplace *zone* is the destination MSOA. Job POIs inside
    that MSOA are used when there are enough of them. Otherwise a point is
    sampled in the destination MSOA (or OA) polygon.
    """
    home_oa = str(home_oa or "")
    work_oa = str(work_oa or "")
    min_n = int(min_pois_per_zone) if min_pois_per_zone is not None else _MIN_POIS_PER_ZONE
    hmeta = oa_index.get(home_oa) or {}
    wmeta = oa_index.get(work_oa) or {}
    lad = str(hmeta.get("lad") or wmeta.get("lad") or "")
    hx = hy = wx = wy = np.nan
    home_src = work_src = "missing"

    home_pack = _pack_if_enough(_pack_from_oa(pools, "home", home_oa), min_n)
    if home_pack is not None:
        i = _choice_index(home_pack["weight"], rng)
        hx, hy = _jitter_xy(home_pack["x"][i], home_pack["y"][i], rng)
        home_src = "residential_poi"
    else:
        drawn = _random_point_in_polygon(hmeta.get("geom"), rng)
        if drawn is not None:
            hx, hy = drawn
            home_src = "zone_random"
        elif home_fallback_xy is not None and np.isfinite(home_fallback_xy[0]):
            hx, hy = float(home_fallback_xy[0]), float(home_fallback_xy[1])
            home_src = "raster"
        elif hmeta:
            hx, hy = float(hmeta["cx"]), float(hmeta["cy"])
            home_src = "centroid"

    same_zone = bool(home_oa and work_oa and home_oa == work_oa)
    centroid_gap = np.nan
    if hmeta and wmeta:
        centroid_gap = float(np.hypot(hmeta["cx"] - wmeta["cx"], hmeta["cy"] - wmeta["cy"]))
        if np.isfinite(centroid_gap) and centroid_gap < _NEAR_OA_M:
            same_zone = True

    work_beta = _beta_for_lad(nts_by_purpose, "work", lad, _DEFAULT_WORK_BETA)
    home_beta = _beta_for_lad(nts_by_purpose, "home", lad, _DEFAULT_HOME_BETA)
    mix_beta = 0.5 * (work_beta + home_beta)
    work_geoid = work_oa
    work_msoa = str(wmeta.get("msoa") or "").strip()
    if work_msoa in {"nan", "None"}:
        work_msoa = ""

    def _work_zone_point() -> Optional[Tuple[float, float]]:
        geom = _msoa_polygon(oa_index, work_msoa, wmeta)
        return _random_point_in_polygon(geom, rng)

    def _finish(wx_, wy_, geoid_, src_):
        if not np.isfinite(wx_) or not np.isfinite(wy_):
            zone_xy = _work_zone_point()
            if zone_xy is not None:
                wx_, wy_ = zone_xy
                src_ = "zone_random"
                geoid_ = work_oa
            elif work_fallback_xy is not None and np.isfinite(work_fallback_xy[0]):
                wx_, wy_ = float(work_fallback_xy[0]), float(work_fallback_xy[1])
                src_ = "centroid"
                geoid_ = work_oa
            elif wmeta:
                wx_, wy_ = float(wmeta["cx"]), float(wmeta["cy"])
                src_ = "centroid"
                geoid_ = work_oa
        dist = float(np.hypot(wx_ - hx, wy_ - hy)) if np.isfinite(hx) and np.isfinite(wx_) else np.nan
        return {
            "home_x": float(hx) if np.isfinite(hx) else np.nan,
            "home_y": float(hy) if np.isfinite(hy) else np.nan,
            "work_x": float(wx_) if np.isfinite(wx_) else np.nan,
            "work_y": float(wy_) if np.isfinite(wy_) else np.nan,
            "work_geoid": str(geoid_ or work_oa),
            "home_source": home_src,
            "work_source": src_,
            "same_zone": same_zone,
            "distance_m": dist,
            "lad": lad,
        }

    if not np.isfinite(hx):
        return _finish(np.nan, np.nan, work_oa, "missing")

    work_pack_oa = _pack_if_enough(_pack_from_oa(pools, "work", work_oa), min_n)
    work_pack_msoa = _pack_if_enough(
        _pack_from_msoa(pools, "work", work_msoa) if work_msoa else None, min_n,
    )
    dest_pack = work_pack_msoa or work_pack_oa

    if dest_pack is None:
        zone_xy = _work_zone_point()
        if zone_xy is not None:
            return _finish(zone_xy[0], zone_xy[1], work_oa, "zone_random")

    d_star = sample_nts_distance_m(
        _dist_pack_for_lad(nts_by_purpose, "work", lad),
        rng,
        dmin_m=_MIN_COMMUTE_M,
        dmax_m=40.0 * METERS_PER_MILE,
    )
    if d_star is None:
        d_star = sample_nts_distance_m(
            _dist_pack_for_lad(nts_by_purpose, "home", lad),
            rng,
            dmin_m=_MIN_COMMUTE_M,
            dmax_m=40.0 * METERS_PER_MILE,
        )

    if d_star is not None and np.isfinite(d_star) and d_star > 0 and dest_pack is not None:
        cand = _boost_geoid_weight(dest_pack, work_oa, 1.5)
        picked = _sample_poi(
            hx, hy, cand, rng,
            target_m=d_star,
            beta_per_mile=None,
            min_sep_m=_MIN_COMMUTE_M,
            max_radius_m=None,
            sigma_m=max(150.0, 0.25 * float(d_star)),
            target_is_network_m=True,
            relax_radius=True,
        )
        if picked is not None:
            wx, wy, gid = picked
            if np.hypot(wx - hx, wy - hy) >= _MIN_COMMUTE_M:
                if gid:
                    gmeta = oa_index.get(gid) or {}
                    gid_msoa = str(gmeta.get("msoa") or "")
                    if (not work_msoa) or (not gid_msoa) or gid_msoa == work_msoa:
                        work_geoid = gid
                src = "work_poi_msoa" if work_pack_msoa is not None else "work_poi_oa"
                return _finish(wx, wy, work_geoid, src)

        origin = work_fallback_xy
        if origin is None and wmeta:
            origin = (wmeta["cx"], wmeta["cy"])
        if origin is None:
            origin = (hx, hy)
        wx, wy = _offset_from_home(
            float(origin[0]), float(origin[1]),
            min(float(d_star), 400.0), rng, polygon=wmeta.get("geom"),
        )
        return _finish(wx, wy, work_oa, "polar_nts")

    if dest_pack is not None:
        picked = _sample_poi(
            hx, hy, dest_pack, rng,
            target_m=None,
            beta_per_mile=mix_beta,
            min_sep_m=_MIN_COMMUTE_M,
        )
        if picked is not None:
            wx, wy, gid = picked
            if np.hypot(wx - hx, wy - hy) >= _MIN_COMMUTE_M:
                return _finish(wx, wy, gid or work_oa, "work_poi")

    zone_xy = _work_zone_point()
    if zone_xy is not None:
        return _finish(zone_xy[0], zone_xy[1], work_oa, "zone_random")
    origin = work_fallback_xy
    if origin is None and wmeta:
        origin = (wmeta["cx"], wmeta["cy"])
    if origin is None:
        origin = (hx, hy)
    wx, wy = _offset_from_home(
        float(origin[0]), float(origin[1]), 150.0, rng, polygon=wmeta.get("geom"),
    )
    return _finish(wx, wy, work_oa, "polar")


def assign_home_work_poi_coordinates(
    persons: Sequence[Dict],
    pools: Dict[str, Dict],
    oa_index: Dict[str, Dict],
    rng: np.random.Generator,
    nts_by_purpose: Optional[Dict] = None,
    home_fallback: Optional[Callable] = None,
    work_centroid: Optional[Dict] = None,
    min_pois_per_zone: int = _MIN_POIS_PER_ZONE,
) -> Dict[str, float]:
    """Overwrite home/work xy on *persons* in place. Returns placement stats.

    Zones with fewer than *min_pois_per_zone* POIs (including none) get a
    uniform random point inside the origin OA (home) or destination MSOA
    (work), so SSP growth areas are not snapped onto distant 2011 buildings.
    """
    n = len(persons)
    n_same = n_home_poi = n_work_poi = n_work_updated = n_short = 0
    n_home_zone = n_work_zone = 0
    dists = []
    for person in persons:
        home_oa = str(person.get("home_geoid") or "")
        work_oa = str(person.get("work_geoid") or "")
        hfb = home_fallback(home_oa) if home_fallback is not None else None
        wfb = None
        if work_centroid is not None:
            wfb = work_centroid.get(work_oa)
        rec = sample_home_work_xy(
            home_oa, work_oa, pools, oa_index, rng,
            nts_by_purpose=nts_by_purpose,
            home_fallback_xy=hfb,
            work_fallback_xy=wfb,
            min_pois_per_zone=min_pois_per_zone,
        )
        person["home_lon"] = rec["home_x"]
        person["home_lat"] = rec["home_y"]
        person["work_lon"] = rec["work_x"]
        person["work_lat"] = rec["work_y"]
        new_work = rec["work_geoid"]
        if new_work and new_work != work_oa:
            person["work_geoid"] = new_work
            n_work_updated += 1
        if rec["same_zone"]:
            n_same += 1
        if rec["home_source"] == "residential_poi":
            n_home_poi += 1
        elif rec["home_source"] == "zone_random":
            n_home_zone += 1
        if str(rec["work_source"]).startswith("work_poi"):
            n_work_poi += 1
        elif rec["work_source"] == "zone_random":
            n_work_zone += 1
        d = rec["distance_m"]
        if np.isfinite(d):
            dists.append(d)
            if d < 100.0:
                n_short += 1
    arr = np.asarray(dists, dtype=float) if dists else np.array([np.nan])
    stats = {
        "n": n,
        "same_zone": n_same,
        "home_poi": n_home_poi,
        "work_poi": n_work_poi,
        "home_zone_random": n_home_zone,
        "work_zone_random": n_work_zone,
        "work_geoid_updated": n_work_updated,
        "share_lt_100m": float(n_short / max(n, 1)),
        "median_m": float(np.nanmedian(arr)),
        "p10_m": float(np.nanpercentile(arr, 10)),
        "p90_m": float(np.nanpercentile(arr, 90)),
    }
    return stats
