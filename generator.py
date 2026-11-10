"""
Revised generate_population_profiles and generate_travel_activities
for the population_gen package.

Survey column conventions used throughout:
  - HHID, HHPERSONID         : household / person identifiers
  - AGE2, RACE, INCOME_RANGE : demographic attributes
  - ODATE                    : activity date
  - OTimin                   : trip start time in minutes from midnight (numeric)
  - OTripPur / DTripPur      : origin / destination trip purpose (string labels)
  - O_GEOID20 / D_GEOID20    : origin / destination census-block GEOID (str)

LODES / flows column conventions:
  - h_geocode  : home census-block GEOID
  - w_geocode  : work census-block GEOID
  - S000       : total jobs (flow weight)

RAC column conventions:
  - h_geocode  : home census-block GEOID
  - C000       : total workers (residential density weight)
"""

from typing import Optional, List, Dict, Tuple, Sequence
import time
import random
import pandas as pd
import numpy as np
from math import cos, radians
import shapely


def _normalise_geoid(geoid: str, target_len: int) -> str:
    """Zero-pad a numeric GEOID; leave alphanumeric ONS codes unchanged.

    LODES CSVs frequently store US GEOIDs as plain integers (dropping leading
    zeros). UK codes such as OA11CD (``E00078837``) must not be left-padded.
    """
    s = str(geoid).strip()
    if not s or s[:1].isalpha():
        return s
    if len(s) < target_len:
        s = s.zfill(target_len)
    return s


def _pad_geoid_series(s: pd.Series, width: int = 15) -> pd.Series:
    """Left-pad numeric GEOIDs to *width*; keep UK ONS codes as-is."""
    s = s.astype(str).str.strip()
    is_numeric = s.str.fullmatch(r"\d+", na=False)
    out = s.copy()
    if is_numeric.any():
        out.loc[is_numeric] = s.loc[is_numeric].str.zfill(width)
    return out


def _expand_code_map(mapping: Optional[Dict]) -> Dict:
    """Accept int, float, and string forms of the same NTS code."""
    if not mapping:
        return {}
    out = dict(mapping)
    for k, v in list(mapping.items()):
        try:
            f = float(k)
            out[f] = v
            if float(f).is_integer():
                out[int(f)] = v
        except (TypeError, ValueError, OverflowError):
            pass
        out[str(k).strip()] = v
        out[str(k).strip().lower()] = v
    return out


def _lookup_mapped(val, mapping: Dict, default=None):
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    if val in mapping:
        return mapping[val]
    try:
        f = float(val)
        if f in mapping:
            return mapping[f]
        if int(f) in mapping:
            return mapping[int(f)]
    except (ValueError, TypeError, OverflowError):
        pass
    s = str(val).strip()
    if s in mapping:
        return mapping[s]
    sl = s.lower()
    if sl in mapping:
        return mapping[sl]
    return default


def _label_mode(val, mode_mapping: Optional[Dict] = None) -> Optional[str]:
    """Map a survey mode code/string to a mode_category label."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    mapped = _lookup_mapped(val, _expand_code_map(mode_mapping), default=None)
    if mapped is not None:
        return mapped
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    return s


def _infer_geoid_len(geoids) -> int:
    """Return the modal GEOID length in a collection (ignoring NaN / empty)."""
    lengths = [len(str(g).strip()) for g in geoids
               if g is not None and str(g).strip() not in ('', 'nan')]
    if not lengths:
        return 15  # default census-block length
    return max(set(lengths), key=lengths.count)


def _build_coverage_report(
    source_geoids,
    centroid_lookup: Dict[str, Tuple[float, float]],
    label: str,
) -> Tuple[np.ndarray, float]:
    """Normalise GEOIDs and report match rate against centroid_lookup.

    Returns (normalised_geoid_array, match_fraction).
    """
    target_len   = _infer_geoid_len(centroid_lookup.keys())
    normalised   = np.array([_normalise_geoid(g, target_len) for g in source_geoids])
    matched      = np.array([g in centroid_lookup for g in normalised])
    match_frac   = matched.mean()

    print(f"  [{label}] GEOID match rate: {match_frac*100:.1f}%  "
          f"({matched.sum():,} / {len(normalised):,} zones matched shapefile)")
    if match_frac < 0.5:
        sample_miss = normalised[~matched][:3].tolist()
        sample_hit  = list(centroid_lookup.keys())[:3]
        print(f"    ⚠  Low match rate — check GEOID format.")
        print(f"    Example unmatched : {sample_miss}")
        print(f"    Example shapefile : {sample_hit}")

    return normalised, match_frac


def _random_point_in_block(geoid: str, gdf_index: Optional[Dict], fallback_lon: float, fallback_lat: float):
    """Sample a random point inside the census block polygon if available."""
    if gdf_index and geoid in gdf_index:
        polygon = gdf_index[geoid]
        minx, miny, maxx, maxy = polygon.bounds
        for _ in range(200):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            from shapely.geometry import Point
            if polygon.contains(Point(x, y)):
                return x, y
    return fallback_lon, fallback_lat


# ---------------------------------------------------------------------------
# Internal helper: rich activity-chain template extraction
# ---------------------------------------------------------------------------

def _norm_zone_key(val) -> str:
    """Normalise OA / ward codes for matching (drop NA sentinels and .0 suffixes)."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return ""
    if s.endswith(".0"):
        core = s[:-2]
        if core.lstrip("-").isdigit():
            s = core
    if s in ("-10", "-8", "-9"):
        return ""
    return s


def _index_templates_by_key(templates: List[Dict], field: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for tmpl in templates:
        k = _norm_zone_key(tmpl.get(field))
        if k:
            out.setdefault(k, []).append(tmpl)
    return out


def _choose_spatial_template_pool(
    home_geoid: str,
    home_ward: str,
    by_oa: Dict[str, List[Dict]],
    by_ward: Dict[str, List[Dict]],
    all_templates: List[Dict],
    min_zone_templates: int,
) -> Tuple[List[Dict], str]:
    """Prefer same-OA survey diaries, then same ward, then all CPCA."""
    oa = _norm_zone_key(home_geoid)
    ward = _norm_zone_key(home_ward)
    oa_pool = by_oa.get(oa, []) if oa else []
    ward_pool = by_ward.get(ward, []) if ward else []
    if oa_pool and len(oa_pool) >= min_zone_templates:
        return oa_pool, "oa"
    if ward_pool and len(ward_pool) >= min_zone_templates:
        return ward_pool, "ward"
    if oa_pool and ward_pool:
        if len(oa_pool) >= len(ward_pool):
            return oa_pool, "oa"
        return ward_pool, "ward"
    if oa_pool:
        return oa_pool, "oa"
    if ward_pool:
        return ward_pool, "ward"
    return all_templates, "global"


def _choose_persona_pool(
    home_geoid: str,
    home_ward: str,
    pt: Dict,
    min_zone_templates: int,
) -> Tuple[List[Dict], str]:
    """Same-persona diaries: OA, then ward, then the persona-wide pool (CPCA + regional)."""
    oa = _norm_zone_key(home_geoid)
    ward = _norm_zone_key(home_ward)
    oa_pool = pt["by_oa"].get(oa, []) if oa else []
    if len(oa_pool) >= min_zone_templates:
        return oa_pool, "oa"
    ward_pool = pt["by_ward"].get(ward, []) if ward else []
    if len(ward_pool) >= min_zone_templates:
        return ward_pool, "ward"
    return pt["all"], "persona"


def _extract_rich_activity_chain_templates(
    survey_df: pd.DataFrame,
    person_id_col: str = 'HHPERSONID',
    date_col: str = 'ODATE',
    trip_no_col: str = 'TRIPNO',
    trip_time_col: str = 'OTimin',
    origin_activity_col: str = 'OTripPur',
    dest_activity_col: str = 'DTripPur',
    origin_geoid_col: Optional[str] = 'O_GEOID20',
    dest_geoid_col: Optional[str] = 'D_GEOID20',
    home_geoid_col: Optional[str] = None,
    home_ward_col: Optional[str] = None,
    min_trips: int = 1,
    persona_col: Optional[str] = None,
) -> List[Dict]:
    """Extract complete person-day activity chains from survey with purposes and GEOIDs.

    Unlike :func:`extract_activity_chain_templates` (which stores only times),
    each template here carries per-trip purpose labels and zone IDs so the
    activity-chain sampler can propagate them to synthetic persons.

    Parameters
    ----------
    survey_df : pd.DataFrame
        Trip-level survey data (one row per trip).
    person_id_col, date_col, trip_no_col : str
        Grouping / sorting columns.
    trip_time_col : str
        Numeric minutes-from-midnight column (``OTimin``).
    origin_activity_col, dest_activity_col : str
        Trip-purpose columns (e.g. ``OTripPur``, ``DTripPur``).
    origin_geoid_col, dest_geoid_col : str, optional
        Census-block GEOID columns for origin and destination. Can be None.
    min_trips : int
        Minimum trips per person-day to include.

    Returns
    -------
    List[Dict]
        Each element::

            {
                'n_trips': int,
                'times':   [int, ...],          # trip start times in minutes
                'trips': [                       # per-trip detail
                    {
                        'time':      int | None,
                        'OTripPur':  str | None,
                        'DTripPur':  str | None,
                        'O_GEOID20': str | None,
                        'D_GEOID20': str | None,
                    },
                    ...
                ],
            }
    """
    templates = []

    # Ensure numeric time column
    df = survey_df.copy()
    if trip_time_col in df.columns:
        if df[trip_time_col].dtype not in [int, float, 'int64', 'float64']:
            def _to_min(v):
                try:
                    s = str(v).strip()
                    if ':' in s:
                        h, m = s.split(':')[:2]
                        return int(h) * 60 + int(m)
                    return int(float(s))
                except Exception:
                    return np.nan
            df[trip_time_col] = df[trip_time_col].apply(_to_min)

    group_cols = [c for c in [person_id_col, date_col] if c in df.columns]
    if not group_cols:
        return templates

    for _, group in df.groupby(group_cols):
        if trip_no_col in group.columns:
            group = group.sort_values(trip_no_col)
        
        trips = []
        for _, row in group.iterrows():
            t = row.get(trip_time_col)
            t = int(t) if pd.notna(t) else None

            # Handle None for geoid columns
            o_geoid = row.get(origin_geoid_col) if origin_geoid_col else None
            d_geoid = row.get(dest_geoid_col) if dest_geoid_col else None

            trips.append({
                'time':      t,
                'OTripPur':  row.get(origin_activity_col),
                'DTripPur':  row.get(dest_activity_col),
                'O_GEOID20': str(o_geoid) if pd.notna(o_geoid) else None,
                'D_GEOID20': str(d_geoid) if pd.notna(d_geoid) else None,
                'mode_category': row.get('mode_category'),
            })

        n_trips = len(trips)
        if n_trips < min_trips:
            continue

        first = group.iloc[0]
        home_geoid = first.get(home_geoid_col) if home_geoid_col else None
        home_ward = first.get(home_ward_col) if home_ward_col else None
        templates.append({
            'n_trips': n_trips,
            'times':   [tr['time'] for tr in trips],
            'trips':   trips,
            'home_geoid': _norm_zone_key(home_geoid),
            'home_ward': _norm_zone_key(home_ward),
            'persona': int(first.get(persona_col)) if persona_col and pd.notna(first.get(persona_col)) else 0,
        })

    return templates


# ---------------------------------------------------------------------------
# Internal helper: assign O/D GEOID + coordinates for one trip record
# ---------------------------------------------------------------------------

# Euclidean metres below this are treated as the same point (home/work snap).
_MIN_OD_M = 80.0
_WORK_PURPOSES = frozenset({"work", "commuting"})


def _purpose_key(val) -> str:
    return str(val or "").strip().lower()


def _euclid_m(x0, y0, x1, y1) -> float:
    try:
        d = float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
    except (TypeError, ValueError):
        return float("nan")
    return d if np.isfinite(d) else float("nan")


def _nearest_geoid(lon: float, lat: float, centroid_lookup: Optional[Dict]) -> Optional[str]:
    if not centroid_lookup:
        return None
    keys = list(centroid_lookup.keys())
    xy = np.asarray(list(centroid_lookup.values()), dtype=float)
    d2 = (xy[:, 0] - float(lon)) ** 2 + (xy[:, 1] - float(lat)) ** 2
    return keys[int(np.argmin(d2))]


def _polar_offset(ox: float, oy: float, rng: np.random.Generator,
                  min_m: float = 400.0, max_m: float = 6000.0) -> Tuple[float, float]:
    r = float(rng.uniform(min_m, max_m))
    a = float(rng.uniform(0.0, 2.0 * np.pi))
    return ox + r * np.cos(a), oy + r * np.sin(a)


def _force_home_tour_ends(trip_details: List[Dict], home_geoid: str) -> List[Dict]:
    """Start at home and end at home, and keep a positive-length return trip.

    If the last diary trip is already ``home→X``, append ``X→home``.
    Rewriting that destination to home would collapse the trip to 0 m.
    Drop leading arrivals at home that would become ``home→home`` once
    the origin is forced to home.
    """
    if not trip_details:
        return trip_details
    details = [dict(td) for td in trip_details]
    details[0]["OTripPur"] = "home"
    details[0]["O_GEOID20"] = home_geoid
    while (
        len(details) > 1
        and _purpose_key(details[0].get("DTripPur")) == "home"
    ):
        details.pop(0)
        details[0]["OTripPur"] = "home"
        details[0]["O_GEOID20"] = home_geoid

    last = details[-1]
    last_o = _purpose_key(last.get("OTripPur"))
    last_d = _purpose_key(last.get("DTripPur"))
    if last_d != "home":
        if last_o == "home":
            details.append({
                "time": None,
                "OTripPur": last.get("DTripPur"),
                "DTripPur": "home",
                "O_GEOID20": last.get("D_GEOID20"),
                "D_GEOID20": home_geoid,
                "mode_category": last.get("mode_category"),
            })
        else:
            last["DTripPur"] = "home"
            last["D_GEOID20"] = home_geoid
    return details


def _assign_secondary_work_dest(
    rec: Dict,
    survey_dest_col: str,
    purpose_building_pools: Optional[Dict],
    decay_beta: Optional[Dict],
    centroid_lookup: Optional[Dict],
    coords_are_projected: bool,
    rng: Optional[np.random.Generator],
    nts_by_purpose: Optional[Dict] = None,
    lad: Optional[str] = None,
) -> None:
    """Place an in-course-of-work destination away from the current origin."""
    from .buildings import sample_decay_destination

    rng = np.random.default_rng() if rng is None else rng
    ox, oy = rec.get("OLON_RANDOM"), rec.get("OLAT_RANDOM")
    try:
        ox, oy = float(ox), float(oy)
    except (TypeError, ValueError):
        return
    if not (np.isfinite(ox) and np.isfinite(oy)):
        return

    beta = None
    if decay_beta:
        beta = decay_beta.get("work", decay_beta.get("other"))
    drawn = None
    if purpose_building_pools:
        drawn = sample_decay_destination(
            ox, oy, "other", purpose_building_pools, beta=beta,
            coords_are_projected=coords_are_projected, rng=rng,
            min_sep_m=_MIN_OD_M,
            nts_by_purpose=nts_by_purpose,
            lad=lad,
            nts_purpose="work",
        )
    lon = lat = None
    geoid = None
    if drawn:
        lon, lat, geoid = drawn
    if lon is None or _euclid_m(ox, oy, lon, lat) < _MIN_OD_M:
        lon, lat = _polar_offset(ox, oy, rng)
        geoid = None
    rec["DLON_RANDOM"] = float(lon)
    rec["DLAT_RANDOM"] = float(lat)
    rec[survey_dest_col] = geoid or _nearest_geoid(lon, lat, centroid_lookup) or rec.get(survey_dest_col)


def _assign_trip_locations(
    rec: Dict,
    o_purpose: Optional[str],
    d_purpose: Optional[str],
    person_home_geoid: str,
    person_home_lon: float,
    person_home_lat: float,
    person_work_geoid: str,
    person_work_lon: float,
    person_work_lat: float,
    centroid_lookup: Optional[Dict],
    od_destination_lookup: Optional[Dict],
    flows_od_lookup: Optional[Dict],
    activity_transition_matrix: Optional[Dict],
    survey_origin_col: str = 'O_GEOID20',
    survey_dest_col: str = 'D_GEOID20',
    purpose_building_pools: Optional[Dict] = None,
    decay_beta: Optional[Dict] = None,
    decay_purposes: Optional[Sequence] = None,
    coords_are_projected: bool = True,
    rng: Optional[np.random.Generator] = None,
    origin_locked: bool = False,
    nts_by_purpose: Optional[Dict] = None,
    lad: Optional[str] = None,
) -> Optional[Dict]:
    """Populate OLON/OLAT/DLON/DLAT and O/D GEOID fields in *rec* in-place.

    Priority for each endpoint:
      1. If purpose == 'home'  → use person home location
      2. If purpose == 'work'  → use person work location
         (work→work uses a secondary workplace POI, not the usual commute site)
      3. Education / shop / personal / leisure / other: draw ``d*`` from the
         NTS length distribution for that purpose and the person's LAD, then
         pick an OSM building whose network length is close to ``d*``
      4. Activity-transition matrix (purpose-conditioned OD)
      5. Survey OD destination lookup (empirical hotspot distribution)
      6. LODES flows OD lookup (employment-weighted fallback)
      7. Uniform random draw from centroid_lookup

    Returns ``None`` when a home destination would sit on the person's home
    origin (a 0 m home→home stub).  When *origin_locked* is True the origin
    coordinates already come from the previous destination and are not
    overwritten by purpose.
    """
    def _get_centroid(geoid):
        if centroid_lookup and geoid and geoid in centroid_lookup:
            return centroid_lookup[geoid]
        return None

    def _sample_dest(current_geoid, o_pur, d_pur):
        """Try lookups in priority order; return (geoid, lon, lat) or (None,None,None)."""
        # 1. Activity-transition matrix
        if activity_transition_matrix:
            key = (o_pur, d_pur)
            if key in activity_transition_matrix and current_geoid:
                origin_dict = activity_transition_matrix[key]
                if current_geoid in origin_dict:
                    choices = list(origin_dict[current_geoid].keys())
                    probs   = np.array(list(origin_dict[current_geoid].values()))
                    geoid   = np.random.choice(choices, p=probs / probs.sum())
                    coords  = _get_centroid(geoid)
                    if coords:
                        return geoid, coords[0], coords[1]

        # 2. Survey OD lookup
        if od_destination_lookup and current_geoid:
            key2 = str(current_geoid)
            if key2 in od_destination_lookup:
                choices = list(od_destination_lookup[key2].keys())
                probs   = np.array(list(od_destination_lookup[key2].values()))
                geoid   = np.random.choice(choices, p=probs / probs.sum())
                coords  = _get_centroid(geoid)
                if coords:
                    return geoid, coords[0], coords[1]

        # 3. LODES OD lookup
        if flows_od_lookup and current_geoid:
            key3 = str(current_geoid)
            if key3 in flows_od_lookup:
                choices = list(flows_od_lookup[key3].keys())
                probs   = np.array(list(flows_od_lookup[key3].values()))
                geoid   = np.random.choice(choices, p=probs / probs.sum())
                coords  = _get_centroid(geoid)
                if coords:
                    return geoid, coords[0], coords[1]

        # 4. Uniform random from centroid_lookup
        if centroid_lookup:
            geoid  = np.random.choice(list(centroid_lookup.keys()))
            coords = centroid_lookup[geoid]
            return geoid, coords[0], coords[1]

        return None, None, None

    o_pur_l = _purpose_key(o_purpose)
    d_pur_l = _purpose_key(d_purpose)

    # --- Origin ---
    if not origin_locked:
        if o_pur_l == 'home':
            rec['OLON_RANDOM']     = person_home_lon
            rec['OLAT_RANDOM']     = person_home_lat
            rec[survey_origin_col] = person_home_geoid
        elif o_pur_l in _WORK_PURPOSES:
            rec['OLON_RANDOM']     = person_work_lon
            rec['OLAT_RANDOM']     = person_work_lat
            rec[survey_origin_col] = person_work_geoid
        else:
            existing = rec.get(survey_origin_col)
            if existing and centroid_lookup and str(existing) in centroid_lookup:
                lon, lat = centroid_lookup[str(existing)]
                rec['OLON_RANDOM'] = lon
                rec['OLAT_RANDOM'] = lat
            else:
                # Sample from centroid_lookup uniformly (origin from survey is unanchored)
                if centroid_lookup:
                    geoid = np.random.choice(list(centroid_lookup.keys()))
                    lon, lat = centroid_lookup[geoid]
                    rec['OLON_RANDOM']     = lon
                    rec['OLAT_RANDOM']     = lat
                    rec[survey_origin_col] = geoid

    # --- Destination ---
    current_o_geoid = rec.get(survey_origin_col)
    at_usual_work = _euclid_m(
        rec.get("OLON_RANDOM"), rec.get("OLAT_RANDOM"),
        person_work_lon, person_work_lat,
    ) < _MIN_OD_M
    at_home = _euclid_m(
        rec.get("OLON_RANDOM"), rec.get("OLAT_RANDOM"),
        person_home_lon, person_home_lat,
    ) < _MIN_OD_M

    if d_pur_l == 'home':
        rec['DLON_RANDOM']   = person_home_lon
        rec['DLAT_RANDOM']   = person_home_lat
        rec[survey_dest_col] = person_home_geoid
        if at_home:
            return None
    elif d_pur_l in _WORK_PURPOSES:
        if o_pur_l in _WORK_PURPOSES or at_usual_work:
            _assign_secondary_work_dest(
                rec, survey_dest_col, purpose_building_pools, decay_beta,
                centroid_lookup, coords_are_projected, rng,
                nts_by_purpose=nts_by_purpose, lad=lad,
            )
        else:
            rec['DLON_RANDOM']   = person_work_lon
            rec['DLAT_RANDOM']   = person_work_lat
            rec[survey_dest_col] = person_work_geoid
        if _euclid_m(
            rec.get("OLON_RANDOM"), rec.get("OLAT_RANDOM"),
            rec.get("DLON_RANDOM"), rec.get("DLAT_RANDOM"),
        ) < _MIN_OD_M:
            _assign_secondary_work_dest(
                rec, survey_dest_col, purpose_building_pools, decay_beta,
                centroid_lookup, coords_are_projected, rng,
                nts_by_purpose=nts_by_purpose, lad=lad,
            )
    elif (
        purpose_building_pools
        and d_pur_l in (set(decay_purposes) if decay_purposes is not None else set(purpose_building_pools))
        and np.isfinite(float(rec['OLON_RANDOM']) if rec.get('OLON_RANDOM') is not None else np.nan)
        and np.isfinite(float(rec['OLAT_RANDOM']) if rec.get('OLAT_RANDOM') is not None else np.nan)
    ):
        from .buildings import sample_decay_destination
        beta = None if not decay_beta else decay_beta.get(d_pur_l)
        drawn = sample_decay_destination(
            rec['OLON_RANDOM'], rec['OLAT_RANDOM'], d_pur_l,
            purpose_building_pools, beta=beta,
            coords_are_projected=coords_are_projected,
            rng=rng,
            nts_by_purpose=nts_by_purpose,
            lad=lad,
        )
        if drawn:
            lon, lat, geoid = drawn
            rec['DLON_RANDOM'] = lon
            rec['DLAT_RANDOM'] = lat
            if geoid:
                rec[survey_dest_col] = geoid
            elif centroid_lookup:
                keys = list(centroid_lookup.keys())
                xy = np.asarray(list(centroid_lookup.values()), dtype=float)
                d2 = (xy[:, 0] - lon) ** 2 + (xy[:, 1] - lat) ** 2
                rec[survey_dest_col] = keys[int(np.argmin(d2))]
        else:
            geoid, lon, lat = _sample_dest(current_o_geoid, o_pur_l, d_pur_l)
            if geoid:
                rec['DLON_RANDOM']   = lon
                rec['DLAT_RANDOM']   = lat
                rec[survey_dest_col] = geoid
    else:
        existing_d = rec.get(survey_dest_col)
        if existing_d and centroid_lookup and str(existing_d) in centroid_lookup:
            lon, lat = centroid_lookup[str(existing_d)]
            rec['DLON_RANDOM'] = lon
            rec['DLAT_RANDOM'] = lat
        else:
            geoid, lon, lat = _sample_dest(current_o_geoid, o_pur_l, d_pur_l)
            if geoid:
                rec['DLON_RANDOM']   = lon
                rec['DLAT_RANDOM']   = lat
                rec[survey_dest_col] = geoid

    return rec


# ===========================================================================
# PUBLIC FUNCTION 1: generate_population_profiles
# ===========================================================================

def generate_population_profiles(
    survey_path: str,
    flows_path: str,
    target_population: int = 22200,
    census_block_shp_path: Optional[str] = None,
    rac_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    
    flows_origin_col: str = 'h_geocode',
    flows_work_geocode_col: str = 'w_geocode',
    flows_weight_col: str = 'S000',

    census_geoid_col: str = 'GEOID20',
    census_crs_target: str = 'EPSG:4326',
    survey_person_cols: Optional[Dict[str, str]] = None,
    rac_geoid_col: str = 'h_geocode',
    rac_weight_col: str = 'C000',
    # Generation options
    batch_size: int = 5000,
    home_jitter_m: float = 100.0,
    work_jitter_m: float = 100.0,
    use_area_weight: bool = False,
    use_polygon_sampling: bool = False,
) -> List[Dict]:
    """Generate synthetic population profiles with demographic attributes and spatial locations.

    Each returned person dict contains::

        {
            'synthetic_person_id': int,
            'household_id':        int,
            'person_id':           int,   # within household (0-indexed)
            'home_geoid':          str,   # census-block GEOID
            'home_lon':            float,
            'home_lat':            float,
            'work_geoid':          str,
            'work_lon':            float,
            'work_lat':            float,
            # + demographic columns from survey_person_cols mapping
        }

    Parameters
    ----------
    survey_path : str
        Path to household travel survey CSV.
    flows_path : str
        Path to LODES OD flow CSV.
        Columns: ``h_geocode``, ``w_geocode``, ``S000`` (or overridden below).
    target_population : int
        Number of synthetic persons to generate.
    census_block_shp_path : str, optional
        Path to census-block shapefile (GEOID20 + geometry).  Required to
        assign geographic coordinates.
    rac_path : str, optional
        Path to LODES RAC (Residence Area Characteristics) CSV.
        Columns: ``h_geocode`` (zone GEOID), ``C000`` (total worker count).
        When provided, home zones are sampled from the RAC residential
        density rather than from the marginal of the flow file.
    random_seed : int, optional
        RNG seed for reproducibility.
    flows_origin_col : str
        Column for home zone in flows file (default ``'h_geocode'``).
    flows_work_geocode_col : str
        Column for work zone in flows file (default ``'w_geocode'``).
    flows_weight_col : str
        Flow weight column (default ``'S000'`` = total jobs).
    census_geoid_col : str
        GEOID column in the shapefile (default ``'GEOID20'``).
    census_crs_target : str
        Target CRS for the shapefile (default ``'EPSG:4326'``).
    survey_person_cols : dict, optional
        Mapping ``{output_col: survey_col}`` of demographic attributes to
        copy from a randomly sampled survey row.  Defaults to::

            {'HHID': 'HHID', 'HHPERSONID': 'HHPERSONID',
             'AGE2': 'AGE2', 'RACE': 'RACE', 'INCOME_RANGE': 'INCOME_RANGE'}

    batch_size : int
        Number of persons to generate per batch (controls memory footprint).
    home_jitter_m : float
        Maximum random spatial jitter applied to home coordinates (metres).
        Set to 0 to use raw block centroids.
    work_jitter_m : float
        Maximum random spatial jitter applied to work coordinates (metres).

    Returns
    -------
    List[Dict]
        One dict per synthetic person.
    """
    from .data_io import (
        load_survey_csv, load_flow_csv,
        load_census_block_shapefile,
        build_hw_lookup, build_centroid_lookup,
    )

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if survey_person_cols is None:
        survey_person_cols = {
            'HHID':         'HHID',
            'HHPERSONID':   'HHPERSONID',
            'AGE2':         'AGE2',
            'RACE':         'RACE',
            'INCOME_RANGE': 'INCOME_RANGE',
        }

    METERS_PER_DEG_LAT = 111_320.0

    # ------------------------------------------------------------------
    # 1. Load survey (for demographics)
    # ------------------------------------------------------------------
    print("[1/6] Loading survey from:", survey_path)
    df_survey = load_survey_csv(survey_path)
    valid_cols = {k: v for k, v in survey_person_cols.items() if v in df_survey.columns}
    missing = set(survey_person_cols.values()) - set(df_survey.columns)
    if missing:
        print(f"  ⚠  Survey columns not found (will be skipped): {sorted(missing)}")
    if not valid_cols:
        print("  ⚠  No demographic columns matched — demographics will be empty.")
    # ------------------------------------------------------------------
    # 2. Load LODES OD flows (for work-zone sampling)
    # ------------------------------------------------------------------
    print("[2/6] Loading LODES flows from:", flows_path)
    flows = load_flow_csv(flows_path)
    flows[flows_origin_col]       = _pad_geoid_series(flows[flows_origin_col])
    flows[flows_work_geocode_col] = _pad_geoid_series(flows[flows_work_geocode_col])

    hw_lookup = build_hw_lookup(
        flows,
        h_geocode_col=flows_origin_col,
        w_geocode_col=flows_work_geocode_col,
        weight_col=flows_weight_col,
    )
    print(f"  Home→work lookup: {len(hw_lookup):,} home zones.")

    # ------------------------------------------------------------------
    # 3. Load census-block shapefile → centroid lookup
    # ------------------------------------------------------------------
    centroid_lookup:  Dict[str, Tuple[float, float]] = {}
    polygon_index:    Optional[Dict] = None          # geoid -> shapely Polygon
    area_lookup:      Dict[str, float] = {}          # geoid -> area (sq m, EPSG:3857)
    if census_block_shp_path is not None:
        print("[3/6] Loading census blocks from:", census_block_shp_path)
        gdf_blocks = load_census_block_shapefile(census_block_shp_path, to_crs=census_crs_target)
        gdf_blocks[census_geoid_col] = _pad_geoid_series(gdf_blocks[census_geoid_col])
        centroid_lookup = build_centroid_lookup(gdf_blocks, geoid_col=census_geoid_col)
        print(f"  Centroid lookup: {len(centroid_lookup):,} census blocks.")

        if use_polygon_sampling or use_area_weight:
            # Project to metric CRS for area calculation
            gdf_metric = gdf_blocks.to_crs('EPSG:3857')
            polygon_index = {}
            for _, row in gdf_blocks.iterrows():
                geoid = str(row[census_geoid_col])
                polygon_index[geoid] = row['geometry']
            if use_area_weight:
                for idx, row in gdf_metric.iterrows():
                    geoid = str(gdf_blocks.loc[idx, census_geoid_col])
                    area_lookup[geoid] = row['geometry'].area  # sq metres
            print(f"  Polygon index built ({len(polygon_index):,} blocks).")
    else:
        print("[3/6] census_block_shp_path not provided — coordinates will be NaN.")

    # ------------------------------------------------------------------
    # 4. Build home-zone sampling distribution
    # ------------------------------------------------------------------
    if rac_path is not None:
        print("[4/6] Building home-zone distribution from RAC:", rac_path)
        rac_df = pd.read_csv(rac_path, dtype={rac_geoid_col: str})
        rac_df[rac_geoid_col] = _pad_geoid_series(rac_df[rac_geoid_col])
        rac_df = rac_df[rac_df[rac_weight_col] > 0].copy()

        # --- GEOID normalisation & coverage check ---
        if centroid_lookup:
            rac_normalised, rac_match = _build_coverage_report(
                rac_df[rac_geoid_col].values, centroid_lookup, label='RAC'
            )
            rac_df[rac_geoid_col] = rac_normalised
            # Keep only zones that matched the shapefile
            pre = len(rac_df)
            rac_df = rac_df[rac_df[rac_geoid_col].isin(centroid_lookup)].copy()
            print(f"  RAC zones retained after GEOID match: {len(rac_df):,} / {pre:,}")

        # Optional: multiply weight by block area (larger blocks → more persons)
        weights = rac_df[rac_weight_col].values.astype(float)
        if use_area_weight and area_lookup:
            area_arr = np.array([area_lookup.get(g, 1.0) for g in rac_df[rac_geoid_col]])
            weights  = weights * area_arr
            print("  Applied area weighting to RAC probabilities.")

        rac_df['_prob'] = weights / weights.sum()
        home_choices    = rac_df[rac_geoid_col].values
        home_probs      = rac_df['_prob'].values
        print(f"  Home distribution: {len(home_choices):,} zones  "
              f"(weight col = '{rac_weight_col}').")
        if rac_weight_col == 'C000':
            print("  ℹ  C000 counts employed workers only.  For full residential")
            print("     distribution pass a total-population column via rac_weight_col.")
    else:
        print("[4/6] No rac_path — using flows origin marginal for home zones.")
        home_dist = flows.groupby(flows_origin_col)[flows_weight_col].sum().reset_index()
        if centroid_lookup:
            flows_normalised, _ = _build_coverage_report(
                home_dist[flows_origin_col].values, centroid_lookup, label='Flows'
            )
            home_dist[flows_origin_col] = flows_normalised
            home_dist = home_dist[home_dist[flows_origin_col].isin(centroid_lookup)].copy()
        home_dist['_prob'] = home_dist[flows_weight_col] / home_dist[flows_weight_col].sum()
        home_choices = home_dist[flows_origin_col].values
        home_probs   = home_dist['_prob'].values
        print(f"  Flows home distribution: {len(home_choices):,} zones.")

    # Also normalise GEOID keys in hw_lookup to match shapefile length
    if centroid_lookup:
        target_len = _infer_geoid_len(centroid_lookup.keys())
        hw_lookup_normalised = {}
        for h_key, w_dict in hw_lookup.items():
            h_norm = _normalise_geoid(h_key, target_len)
            w_norm = {_normalise_geoid(w, target_len): p for w, p in w_dict.items()}
            hw_lookup_normalised[h_norm] = w_norm
        hw_lookup = hw_lookup_normalised

        # Coverage check for work zones
        all_work_geoids = [w for wd in hw_lookup.values() for w in wd.keys()]
        work_matched = sum(1 for w in all_work_geoids if w in centroid_lookup)
        print(f"  Work GEOID match rate: "
              f"{100*work_matched/max(len(all_work_geoids),1):.1f}%  "
              f"({work_matched:,} / {len(all_work_geoids):,})")

    # Fallback pool: centroid GEOIDs that matched the shapefile.
    # A sampled zone missing from this lookup draws from the pool,
    # so misses do not all land on one arbitrary point.
    fallback_geoids = np.array(list(centroid_lookup.keys())) if centroid_lookup else None

    # ------------------------------------------------------------------
    # 5. Generate persons in batches
    # ------------------------------------------------------------------
    print(f"[5/6] Generating {target_population:,} persons in batches of {batch_size:,}...")
    # Detect whether target CRS is geographic (degrees) or projected (metres)
    _is_geographic_crs = census_crs_target.upper() in (
        'EPSG:4326', 'WGS84', 'WGS 84', 'CRS84',
    )
    METERS_PER_DEG_LAT = 111_320.0
    persons: List[Dict] = []
    synthetic_pid = 0
    n_batches = (target_population + batch_size - 1) // batch_size
    avg_hh_size = 2

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, target_population)
        n_this      = batch_end - batch_start
        pct         = 100.0 * batch_end / target_population

        # --- Home zones ---
        home_geoids = np.random.choice(home_choices, size=n_this, p=home_probs).astype(str)

        # --- Home coordinates + jitter ---
        home_lons = np.full(n_this, np.nan)
        home_lats = np.full(n_this, np.nan)
        for i, geoid in enumerate(home_geoids):
            if geoid in centroid_lookup:
                home_lons[i], home_lats[i] = centroid_lookup[geoid]
            elif fallback_geoids is not None:
                # Draw one matched zone at random. Index 0 would pin every miss to one zone.
                fb = fallback_geoids[np.random.randint(len(fallback_geoids))]
                home_lons[i], home_lats[i] = centroid_lookup[fb]
                home_geoids[i] = fb        # update geoid too

        # Jitter or polygon sampling
        if use_polygon_sampling and polygon_index:
            for i in range(n_this):
                lon, lat = _random_point_in_block(
                    home_geoids[i], polygon_index, home_lons[i], home_lats[i]
                )
                home_lons[i], home_lats[i] = lon, lat
        elif home_jitter_m > 0:
            for i in range(n_this):
                if not np.isnan(home_lats[i]):
                    if _is_geographic_crs:
                        lat_deg = home_jitter_m / METERS_PER_DEG_LAT
                        lon_deg = home_jitter_m / (
                            METERS_PER_DEG_LAT * max(cos(radians(home_lats[i])), 1e-6)
                        )
                        home_lats[i] += np.random.uniform(-lat_deg, lat_deg)
                        home_lons[i] += np.random.uniform(-lon_deg, lon_deg)
                    else:
                        # Projected CRS (e.g. BNG): jitter directly in metres
                        home_lats[i] += np.random.uniform(-home_jitter_m, home_jitter_m)
                        home_lons[i] += np.random.uniform(-home_jitter_m, home_jitter_m)


        # --- Work zones (conditional on home) ---
        work_geoids = []
        for hg in home_geoids:
            if hg in hw_lookup and hw_lookup[hg]:
                w_choices = list(hw_lookup[hg].keys())
                w_probs   = np.array(list(hw_lookup[hg].values()), dtype=float)
                work_geoids.append(
                    np.random.choice(w_choices, p=w_probs / w_probs.sum())
                )
            elif fallback_geoids is not None:
                # Draw one centroid zone at random when this home has no work lookup.
                work_geoids.append(
                    fallback_geoids[np.random.randint(len(fallback_geoids))]
                )
            else:
                work_geoids.append(hg)


        # --- Work coordinates + jitter ---
        work_lons = np.full(n_this, np.nan)
        work_lats = np.full(n_this, np.nan)

        for i, geoid in enumerate(work_geoids):
            if geoid in centroid_lookup:
                work_lons[i], work_lats[i] = centroid_lookup[geoid]
            elif fallback_geoids is not None:
                fb = fallback_geoids[np.random.randint(len(fallback_geoids))]
                work_lons[i], work_lats[i] = centroid_lookup[fb]
                work_geoids[i] = fb

        if use_polygon_sampling and polygon_index:
            for i in range(n_this):
                lon, lat = _random_point_in_block(
                    work_geoids[i], polygon_index, work_lons[i], work_lats[i]
                )
                work_lons[i], work_lats[i] = lon, lat
        elif work_jitter_m > 0:
            for i in range(n_this):
                if not np.isnan(work_lats[i]):
                    if _is_geographic_crs:
                        lat_deg = work_jitter_m / METERS_PER_DEG_LAT
                        lon_deg = work_jitter_m / (
                            METERS_PER_DEG_LAT * max(cos(radians(work_lats[i])), 1e-6)
                        )
                        work_lats[i] += np.random.uniform(-lat_deg, lat_deg)
                        work_lons[i] += np.random.uniform(-lon_deg, lon_deg)
                    else:
                        work_lats[i] += np.random.uniform(-work_jitter_m, work_jitter_m)
                        work_lons[i] += np.random.uniform(-work_jitter_m, work_jitter_m)


        # --- Demographics: sample from survey with replacement ---
        demo_rows = df_survey.sample(n=n_this, replace=True).reset_index(drop=True)

        # --- Assemble person dicts ---
        for i in range(n_this):
            p = {
                'synthetic_person_id': synthetic_pid,
                'household_id':        synthetic_pid // avg_hh_size,
                'person_id':           synthetic_pid %  avg_hh_size,
                'home_geoid':          str(home_geoids[i]),
                'home_lon':            float(home_lons[i]),
                'home_lat':            float(home_lats[i]),
                'work_geoid':          str(work_geoids[i]),
                'work_lon':            float(work_lons[i]),
                'work_lat':            float(work_lats[i]),
            }
            for out_col, surv_col in valid_cols.items():
                p[out_col] = demo_rows.loc[i, surv_col]
            persons.append(p)
            synthetic_pid += 1

        print(f"  Batch {batch_idx+1}/{n_batches}: "
              f"{batch_end:,}/{target_population:,} ({pct:.1f}%)")
    # ------------------------------------------------------------------
    # 6. Post-generation diagnostics
    # ------------------------------------------------------------------
    print(f"[6/6] Done. {len(persons):,} persons generated.")
    nan_home = sum(1 for p in persons if np.isnan(p['home_lon']))
    nan_work = sum(1 for p in persons if np.isnan(p['work_lon']))
    if nan_home:
        print(f"  ⚠  {nan_home:,} persons have NaN home coordinates.")
    if nan_work:
        print(f"  ⚠  {nan_work:,} persons have NaN work coordinates.")
    if nan_home == 0 and nan_work == 0:
        print("  ✓  All persons have valid home and work coordinates.")

    # Zone-count diagnostics
    import pandas as _pd
    home_zone_counts = _pd.Series([p['home_geoid'] for p in persons]).value_counts()
    work_zone_counts = _pd.Series([p['work_geoid'] for p in persons]).value_counts()
    print(f"  Home zones used : {home_zone_counts.shape[0]:,}  "
          f"(top zone = {home_zone_counts.index[0]}, "
          f"n={home_zone_counts.iloc[0]:,}, "
          f"{100*home_zone_counts.iloc[0]/len(persons):.1f}%)")
    print(f"  Work zones used : {work_zone_counts.shape[0]:,}  "
          f"(top zone = {work_zone_counts.index[0]}, "
          f"n={work_zone_counts.iloc[0]:,}, "
          f"{100*work_zone_counts.iloc[0]/len(persons):.1f}%)")
    if home_zone_counts.iloc[0] / len(persons) > 0.10:
        print("  ⚠  Top home zone holds >10% of population — likely a GEOID "
              "mismatch is collapsing persons into one zone.  Check _build_coverage_report "
              "output above and verify GEOID lengths.")

    return persons


# ===========================================================================
# PUBLIC FUNCTION 2: generate_travel_activities
# ===========================================================================

def generate_travel_activities(
    persons: List[Dict],
    survey_path: str,
    census_block_shp_path: Optional[str] = None,
    flows_path: Optional[str] = None,
    activity_mapping: Optional[Dict] = None,
    random_seed: Optional[int] = None,
    use_time_stratification: bool = True,
    time_bin_size: int = 10,
    # Survey column names
    survey_time_col: str = 'OTimin',        # numeric minutes from midnight
    survey_origin_col: str = 'O_GEOID20',
    survey_dest_col: str = 'D_GEOID20',
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur',
    # Flows / geography column names
    flows_origin_col: str = 'h_geocode',
    flows_work_geocode_col: str = 'w_geocode',
    flows_weight_col: str = 'S000',
    census_geoid_col: str = 'GEOID20',
    census_crs_target: str = 'EPSG:4326',
    force_home_start_end: bool = False,
    survey_mode_col: Optional[str] = None,
    mode_mapping: Optional[Dict] = None,
    mode_nests: Optional[Dict] = None,
    mode_reference: str = "car",
    default_speed_ms: float = 13.89,
    survey_home_geoid_lookup: Optional[Dict] = None,
    survey_home_lookup_key: str = 'HouseholdID',
    survey_ward_col: Optional[str] = 'HHoldOSWard_B01ID',
    person_home_ward_key: str = 'home_ward',
    min_zone_templates: int = 5,
    continuous_start_times: bool = True,
    min_trip_gap_minutes: float = 5.0,
    osm_buildings_path: Optional[str] = None,
    purpose_poi_pools: Optional[Dict] = None,
    decay_purposes: Optional[Sequence[str]] = None,
    use_mode_choice_model: bool = True,
    mode_choice_cache_path: Optional[str] = None,
    persona_map: Optional[Dict] = None,
    persona_fallback_survey_path: Optional[str] = None,
    persona_min_persons: int = 150,
    persona_min_zone_templates: int = 20,
    persona_survey_id_col: str = 'IndividualID',
    person_persona_key: str = 'persona',
) -> List[Dict]:
    """Generate daily travel activity chains for pre-generated synthetic persons.

    This function is the second step of the two-step pipeline::

        persons     = generate_population_profiles(...)
        all_records = generate_travel_activities(persons, ...)

    Activity generation follows ``generate_population_stream``.
    Home and work locations are read from *persons*, which already stores them.

    Parameters
    ----------
    persons : List[Dict]
        Output of :func:`generate_population_profiles`. Each dict must contain
        at minimum: ``synthetic_person_id``, ``home_geoid``, ``home_lon``,
        ``home_lat``, ``work_geoid``, ``work_lon``, ``work_lat``.
    survey_path : str
        Path to the household travel survey CSV.
    census_block_shp_path : str, optional
        Path to the census-block shapefile.  Needed to look up coordinates for
        non-home / non-work activity destinations.
    flows_path : str, optional
        Path to LODES OD flows CSV.  Used as a fallback destination-sampling
        distribution when the survey OD lookup is sparse.
    activity_mapping : dict, optional
        Numeric-code → purpose-string mapping applied to the survey before
        building templates (e.g. ``{1: 'work', 2: 'home', 3: 'school'}``).
        Applied to ``OTripPur`` / ``DTripPur`` columns if they hold integer
        codes rather than string labels.
    random_seed : int, optional
        RNG seed for reproducibility.
    use_time_stratification : bool
        If ``True``, build activity-specific time distributions from the survey
        and use them as a fallback when an empirical template cannot supply a
        time for a given trip position.  Default: ``True``.
    time_bin_size : int
        Bin width in minutes for time-stratified distributions (default 10).
    survey_time_col : str
        Column with numeric trip-start times in minutes (default ``'OTimin'``).
    survey_origin_col, survey_dest_col : str
        GEOID columns for origin / destination (default ``'O_GEOID20'``,
        ``'D_GEOID20'``).
    survey_origin_activity_col, survey_dest_activity_col : str
        Activity-purpose columns (default ``'OTripPur'``, ``'DTripPur'``).
    flows_origin_col, flows_work_geocode_col, flows_weight_col : str
        Column names in the LODES flows file (only used when *flows_path* is
        provided).
    census_geoid_col : str
        GEOID column in the census shapefile (default ``'GEOID20'``).
    census_crs_target : str
        CRS for the census shapefile (default ``'EPSG:4326'``).
    survey_home_geoid_lookup : dict, optional
        Maps survey household/person IDs to a home OA so activity chains and
        modes can be drawn from local diaries. Keys should match
        ``survey_home_lookup_key``.
    survey_home_lookup_key : str
        Survey column used to join ``survey_home_geoid_lookup`` (default
        ``HouseholdID``).
    survey_ward_col : str, optional
        Survey household-ward column used as the spatial fallback when an OA
        has too few diaries (default ``HHoldOSWard_B01ID``).
    person_home_ward_key : str
        Key on each synthetic person dict for home ward (default ``home_ward``).
    min_zone_templates : int
        Minimum survey person-days required to stay at OA (then ward) before
        falling back to the next coarser geography.
    continuous_start_times : bool
        If ``True`` (default), ``OTimin`` is drawn from a continuous KDE of
        the full survey start times for that destination activity. A local
        clock-time window around the diary time is used when it has enough
        trips; otherwise the activity's full-survey distribution is used.
        This keeps AM/PM peak locations and widths aligned with NTS.
    min_trip_gap_minutes : float
        Minimum minutes between consecutive trips in a chain.
    osm_buildings_path : str, optional
        OSM building footprints shapefile. Used only when *purpose_poi_pools*
        is not supplied. Destinations for education, shop, personal, leisure
        and other are then classified from OSM ``type`` / ``name``.
    purpose_poi_pools : dict, optional
        Pre-built destination sampling pools ``{purpose: {x, y, weight, geoid, n}}``.
        Prepare these outside the package (e.g. from
        ``cpca_activity_poi.gpkg`` via :func:`build_purpose_poi_pools`).
        When set, *osm_buildings_path* is ignored. Home and work still use
        the person's assigned locations.
    decay_purposes : sequence of str, optional
        Purposes that use building + NTS-length matching. Default is
        education, shop, personal, leisure, other.
    mode_mapping : dict, optional
        NTS mode-code → label map (e.g. notebook ``MODE_CODE_MAP``). Labels
        become the nested-logit alternatives. Default folds passenger codes
        into ``car``.
    mode_nests : dict, optional
        Nest name → list of mode labels (e.g. ``{"active": ["walk", "bike"],
        "private": ["car", "car_passenger"], "public": ["bus", "rail"]}``).
        If omitted, nests are inferred from the labels in ``mode_mapping``.
    mode_reference : str
        Nested-logit reference alternative (default ``car``).
    use_mode_choice_model : bool
        If True (default), trip mode is sampled from a decision tree fitted on
        NTS (age group, sex, personal and household income, adults, children,
        cars, bikes, distance, purpose, previous mode). Leaf probabilities are
        sampled so shares stay close to NTS. Nested logit is the fallback if
        scikit-learn is unavailable.
    mode_choice_cache_path : str, optional
        Path used to cache the fitted model. The tree is stored as
        ``nts_tree_mode_choice.joblib`` next to this file; nested logit
        still uses the JSON coefficients.
    ACTIVITY_MAP : dict, optional
        Same as *activity_mapping*; accepted for backward compatibility.
    persona_map : dict, optional
        ``{survey IndividualID: persona 1-9}`` (``PersonaModel.persona_map()``). When given,
        survey diaries are tagged by persona and each synthetic person (whose persona is in
        ``person_persona_key``) samples its diary and destination-length pack from its own
        persona's survey people.
    persona_fallback_survey_path : str, optional
        Wider survey (e.g. East of England) used to supplement personas that have fewer than
        ``persona_min_persons`` people in the main survey. CPCA people are not duplicated.
    persona_min_zone_templates : int
        Minimum same-persona diaries needed for an OA / ward local pool.

    Returns
    -------
    List[Dict]
        One dict per trip across all synthetic persons.  Each record contains:

        * ``synthetic_person_id``, ``household_id``, ``person_id``
        * ``trip_sequence`` (1-indexed within the person's day)
        * ``OTripPur``, ``DTripPur`` (activity purposes)
        * ``OTimin`` (trip start time, minutes from midnight; float when
          ``continuous_start_times`` is True)
        * ``OTIME`` (HH:MM:SS string)
        * ``O_GEOID20``, ``D_GEOID20``
        * ``OLON_RANDOM``, ``OLAT_RANDOM``, ``DLON_RANDOM``, ``DLAT_RANDOM``
        * ``trip_distance`` (Euclidean metres) and ``trip_distance_network_m``
        * ``home_geoid``, ``work_geoid``
        * any demographic columns present in *persons* dicts
    """
    from .data_io import (
        load_survey_csv, load_flow_csv,
        load_census_block_shapefile,
        build_centroid_lookup,
        build_lodes_destination_lookup,
        build_activity_transition_matrix,
        build_purpose_transition_matrix,
    )
    from .time_stratified import (
        filter_unrealistic_activity_chains,
        learn_activity_time_windows,
        learn_activity_duration_ranges,
        build_activity_time_distributions,
        build_activity_pair_time_distributions,
        build_purpose_start_time_mixtures,
        sample_chain_start_times_continuous,
        format_mixture_peaks,
    )

    # Merge activity_mapping / ACTIVITY_MAP
    act_map = activity_mapping

    # ------------------------------------------------------------------
    # 0. Random seed
    # ------------------------------------------------------------------
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # 1. Load and optionally recode survey
    # ------------------------------------------------------------------
    print("=" * 68)
    print("generate_travel_activities — building survey data structures")
    print("=" * 68)
    print("[1/7] Loading survey from:", survey_path)
    df_survey = load_survey_csv(survey_path)

    persona_on = bool(persona_map)
    needy_personas: set = set()
    if persona_on:
        pm = {int(k): int(v) for k, v in persona_map.items()}
        sid = persona_survey_id_col
        if sid not in df_survey.columns:
            raise ValueError(f"persona_map needs survey column '{sid}'")
        ids_num = pd.to_numeric(df_survey[sid], errors="coerce")
        df_survey["_persona"] = ids_num.map(pm).fillna(0).astype(int)
        df_survey["_is_region"] = False
        n_by_p = df_survey.drop_duplicates(sid).groupby("_persona").size()
        needy_personas = {p for p in range(1, 10) if int(n_by_p.get(p, 0)) < persona_min_persons}
        if persona_fallback_survey_path and needy_personas:
            reg = load_survey_csv(persona_fallback_survey_path)
            reg_ids = pd.to_numeric(reg[sid], errors="coerce")
            reg["_persona"] = reg_ids.map(pm).fillna(0).astype(int)
            keep = reg["_persona"].isin(needy_personas) & ~reg_ids.isin(set(ids_num.dropna()))
            reg = reg[keep].copy()
            reg["_is_region"] = True
            print(f"  Personas {sorted(needy_personas)} have < {persona_min_persons} CPCA people; "
                  f"adding {reg[sid].nunique():,} regional survey people.")
            df_survey = pd.concat([df_survey, reg], ignore_index=True)
        elif needy_personas:
            print(f"  Personas {sorted(needy_personas)} have < {persona_min_persons} CPCA people "
                  "and no fallback survey was supplied.")
    # Recode NTS numeric purpose codes (or mixed int/float/str codes) to labels
    if act_map is not None:
        expanded_act = _expand_code_map(act_map)
        known_labels = {str(v).strip().lower() for v in expanded_act.values()}

        def _map_act(v):
            mapped = _lookup_mapped(v, expanded_act, default=None)
            if mapped is not None:
                return mapped
            if isinstance(v, str) and v.strip() and v.strip().lower() not in ("nan", "none"):
                s = v.strip().lower()
                return s if s in known_labels else "other"
            return "other"

        for col in [survey_origin_activity_col, survey_dest_activity_col]:
            if col in df_survey.columns:
                df_survey[col] = df_survey[col].map(_map_act)

    # Recode survey modes to MATSim-style mode_category labels
    if survey_mode_col and survey_mode_col in df_survey.columns:
        df_survey["mode_category"] = df_survey[survey_mode_col].map(
            lambda v: _label_mode(v, mode_mapping)
        )
        n_mode = int(df_survey["mode_category"].notna().sum())
        print(f"  Mode column '{survey_mode_col}' → mode_category ({n_mode:,} labelled trips)")
        print("  ", df_survey["mode_category"].value_counts(dropna=False).head(8).to_dict())
    elif "mode_category" not in df_survey.columns:
        df_survey["mode_category"] = None

    # Ensure numeric time column
    if survey_time_col in df_survey.columns:
        if df_survey[survey_time_col].dtype not in [int, float, 'int64', 'float64']:
            def _to_min(v):
                try:
                    s = str(v).strip()
                    if ':' in s:
                        h, m = s.split(':')[:2]
                        return int(h) * 60 + int(m)
                    return int(float(s))
                except Exception:
                    return np.nan
            df_survey[survey_time_col] = df_survey[survey_time_col].apply(_to_min)

    if survey_home_geoid_lookup and survey_home_lookup_key in df_survey.columns:
        def _hh_oa(v):
            k = _norm_zone_key(v)
            if k and k in survey_home_geoid_lookup:
                return _norm_zone_key(survey_home_geoid_lookup[k])
            if v in survey_home_geoid_lookup:
                return _norm_zone_key(survey_home_geoid_lookup[v])
            try:
                iv = int(float(v))
                if iv in survey_home_geoid_lookup:
                    return _norm_zone_key(survey_home_geoid_lookup[iv])
            except (TypeError, ValueError):
                pass
            return ""

        df_survey["_survey_home_oa"] = df_survey[survey_home_lookup_key].map(_hh_oa)
        n_tag = int((df_survey["_survey_home_oa"] != "").sum())
        print(
            f"  Home OA lookup on '{survey_home_lookup_key}': "
            f"{n_tag:,} / {len(df_survey):,} trips tagged"
        )
    if survey_ward_col and survey_ward_col in df_survey.columns:
        df_survey["_survey_home_ward"] = df_survey[survey_ward_col].map(_norm_zone_key)

    df_region = None
    if persona_on:
        df_region = df_survey[df_survey["_is_region"]].copy()
        df_survey = df_survey[~df_survey["_is_region"]].copy()

    # ------------------------------------------------------------------
    # 2. Filter unrealistic activity chains
    # ------------------------------------------------------------------
    print("[2/7] Filtering unrealistic activity chains...")
    
    # Auto-detect person ID column
    person_col_candidates = ['HHPERSONID', 'IndividualID', 'PID', 'person_id', 'PersonID']
    survey_person_col = next((col for col in person_col_candidates if col in df_survey.columns), None)
    if survey_person_col is None:
        raise ValueError(f"No person ID column found. Expected one of: {person_col_candidates}. "
                        f"Available columns: {list(df_survey.columns)}")
    
    # Auto-detect day/date column
    day_col_candidates = ['ODATE', 'DayID', 'TripDate', 'date', 'day', 'Date']
    survey_day_col = next((col for col in day_col_candidates if col in df_survey.columns), None)
    if survey_day_col is None:
        raise ValueError(f"No day/date column found. Expected one of: {day_col_candidates}. "
                        f"Available columns: {list(df_survey.columns)}")
    
    print(f"  Using person column: '{survey_person_col}', day column: '{survey_day_col}'")
    
    df_filtered = filter_unrealistic_activity_chains(
        df_survey,
        survey_origin_activity_col=survey_origin_activity_col,
        survey_dest_activity_col=survey_dest_activity_col,
        survey_time_col=survey_time_col,
        survey_person_col=survey_person_col,
        survey_day_col=survey_day_col,
    )
    print(f"  Retained {len(df_filtered):,} / {len(df_survey):,} trips after filtering.")

    mode_by_purpose: Dict[str, Tuple[list, np.ndarray]] = {}
    if "mode_category" in df_filtered.columns and survey_dest_activity_col in df_filtered.columns:
        tmp_m = df_filtered[[survey_dest_activity_col, "mode_category"]].dropna()
        if len(tmp_m):
            for pur, grp in tmp_m.groupby(survey_dest_activity_col):
                vc = grp["mode_category"].value_counts(normalize=True)
                mode_by_purpose[str(pur).lower().strip()] = (vc.index.tolist(), vc.to_numpy(dtype=float))
            print(f"  Purpose-conditional mode tables: {len(mode_by_purpose)} purposes")

    mode_fit = None
    mode_spec = None
    if mode_mapping or mode_nests:
        from .mode_spec import ModeSpec
        mode_spec = ModeSpec.from_mapping(
            mode_mapping, nests=mode_nests, reference=mode_reference,
        )
        print(f"  Mode spec: {mode_spec.modes}  nests={mode_spec.nests}")
    if use_mode_choice_model:
        tree_fit = None
        try:
            from .tree_mode_choice import fit_tree_mode_choice
            print("[2b/7] Decision-tree mode choice (age group, sex, income, adults, children, cars, bikes, distance, purpose)...")
            tree_fit = fit_tree_mode_choice(
                df_filtered,
                cache_path=mode_choice_cache_path,
                spec=mode_spec,
                mode_mapping=mode_mapping,
                mode_nests=mode_nests,
            )
        except Exception as exc:
            print(f"  Warning: decision tree failed ({exc}).")
            tree_fit = None
        if tree_fit is not None:
            src = "cache" if tree_fit.get("from_cache") else "NTS fit"
            acc_te = tree_fit.get("accuracy_test")
            acc_tr = tree_fit.get("accuracy_train")
            fitted_modes = (tree_fit.get("mode_spec") or {}).get("modes") or []
            print(
                f"  {src}: n={tree_fit.get('nobs', 0):,}  "
                f"holdout accuracy={acc_te:.3f}  train={acc_tr:.3f}  modes={fitted_modes}"
                if acc_te is not None and acc_tr is not None
                else f"  {src}: n={tree_fit.get('nobs', 0):,}  modes={fitted_modes}"
            )
            nts_s = tree_fit.get("nts_share") or {}
            pred_s = tree_fit.get("pred_share") or {}
            if nts_s and pred_s:
                print("  Mode shares (NTS → tree expected):")
                for m in fitted_modes:
                    print(
                        f"    {m:16s}  {100 * nts_s.get(m, 0):5.1f}% → "
                        f"{100 * pred_s.get(m, 0):5.1f}%"
                    )
            imps = tree_fit.get("importances") or {}
            if imps:
                top = list(imps.items())[:8]
                print("  Top splits: " + ", ".join(f"{k}={v:.2f}" for k, v in top))
            mode_fit = tree_fit
        else:
            from .nested_mode_choice import fit_mode_choice_model
            print("[2b/7] Nested logit mode choice (tree unavailable; using nested logit)...")
            try:
                mode_fit = fit_mode_choice_model(
                    df_filtered,
                    cache_path=mode_choice_cache_path,
                    spec=mode_spec,
                    mode_mapping=mode_mapping,
                    mode_nests=mode_nests,
                )
                src = "cache" if mode_fit.get("from_cache") else "NTS fit"
                rho = mode_fit.get("rho2")
                acc = mode_fit.get("accuracy")
                fitted_modes = (mode_fit.get("mode_spec") or {}).get("modes") or []
                print(
                    f"  {src}: n={mode_fit.get('nobs', 0):,}  "
                    f"rho2={rho:.3f}  accuracy={acc:.3f}  modes={fitted_modes}"
                    if rho is not None and acc is not None
                    else f"  {src}: n={mode_fit.get('nobs', 0):,}  modes={fitted_modes}"
                )
                nts_s = mode_fit.get("nts_share") or {}
                pred_s = mode_fit.get("pred_share") or {}
                if nts_s and pred_s:
                    print("  Mode shares (NTS → fitted):")
                    for m in fitted_modes:
                        print(
                            f"    {m:16s}  {100 * nts_s.get(m, 0):5.1f}% → "
                            f"{100 * pred_s.get(m, 0):5.1f}%"
                        )
            except Exception as exc:
                print(f"  Warning: mode-choice model failed ({exc}). Using diary / purpose draws.")
                mode_fit = None

    # Learn time windows and duration ranges (informational / used in validation)
    learned_time_windows = learn_activity_time_windows(
        df_filtered,
        survey_origin_activity_col=survey_origin_activity_col,
        survey_time_col=survey_time_col,
    ) if len(df_filtered) > 0 else {}
    if learned_time_windows:
        print("  Learned time windows:")
        for act, (lo, hi) in sorted(learned_time_windows.items()):
            print(f"    {act:15s}  {int(lo//60):02d}:{int(lo%60):02d} – {int(hi//60):02d}:{int(hi%60):02d}")
    else:
        print("  No time windows learned (filtered data is empty or missing activity columns)")

    # ------------------------------------------------------------------
    # 3. Build time-stratified distributions (optional)
    # ------------------------------------------------------------------
    activity_time_dists     = None
    activity_pair_time_dists = None

    if use_time_stratification:
        print("[3/7] Building time-stratified distributions...")
        activity_time_dists = build_activity_time_distributions(
            df_filtered,
            time_bin_size=time_bin_size,
            survey_origin_activity_col=survey_origin_activity_col,
            survey_time_col=survey_time_col,
        )
        activity_pair_time_dists = build_activity_pair_time_distributions(
            df_filtered,
            time_bin_size=time_bin_size,
            survey_origin_activity_col=survey_origin_activity_col,
            survey_dest_activity_col=survey_dest_activity_col,
            survey_time_col=survey_time_col,
        )
        print(f"  Activity distributions: {len(activity_time_dists)} activities")
        print(f"  Activity-pair distributions: {len(activity_pair_time_dists)} pairs")
    else:
        print("[3/7] Time stratification disabled — skipping.")

    purpose_time_mixtures: Dict[str, Dict] = {}
    if continuous_start_times:
        print("[3b/7] Building full-survey continuous start-time KDEs per activity...")
        purpose_time_mixtures = build_purpose_start_time_mixtures(
            df_survey if len(df_survey) else df_filtered,
            survey_dest_activity_col=survey_dest_activity_col,
            survey_time_col=survey_time_col,
        )
        for purpose, mix in sorted(purpose_time_mixtures.items()):
            if purpose.startswith("_"):
                continue
            print(f"    {purpose:12s}  n={mix['n']:,}  {format_mixture_peaks(mix)}")
    else:
        print("[3b/7] Continuous start times disabled — using empirical template minutes.")

    # ------------------------------------------------------------------
    # 4. Extract rich empirical activity-chain templates
    # ------------------------------------------------------------------
    print("[4/7] Extracting empirical activity-chain templates from survey...")
    
    # Auto-detect trip number column
    trip_no_col_candidates = ['JourSeq', 'TRIPNO', 'trip_no', 'tripno', 'TripNumber', 'TripID']
    trip_no_col = next((col for col in trip_no_col_candidates if col in df_survey.columns), None)
    
    activity_chain_templates = _extract_rich_activity_chain_templates(
        survey_df=df_survey,                       # use unfiltered for max coverage
        person_id_col=survey_person_col,
        date_col=survey_day_col,
        trip_no_col=trip_no_col,
        trip_time_col=survey_time_col,
        origin_activity_col=survey_origin_activity_col,
        dest_activity_col=survey_dest_activity_col,
        origin_geoid_col=survey_origin_col if survey_origin_col and survey_origin_col in df_survey.columns else None,
        dest_geoid_col=survey_dest_col if survey_dest_col and survey_dest_col in df_survey.columns else None,
        home_geoid_col="_survey_home_oa" if "_survey_home_oa" in df_survey.columns else None,
        home_ward_col="_survey_home_ward" if "_survey_home_ward" in df_survey.columns else None,
        min_trips=1,
        persona_col="_persona" if persona_on else None,
    )

    persona_templates: Dict[int, Dict] = {}
    if persona_on:
        reg_templates: List[Dict] = []
        if df_region is not None and len(df_region):
            reg_templates = _extract_rich_activity_chain_templates(
                survey_df=df_region,
                person_id_col=survey_person_col,
                date_col=survey_day_col,
                trip_no_col=trip_no_col,
                trip_time_col=survey_time_col,
                origin_activity_col=survey_origin_activity_col,
                dest_activity_col=survey_dest_activity_col,
                origin_geoid_col=None,
                dest_geoid_col=None,
                home_geoid_col=None,
                home_ward_col=None,
                min_trips=1,
                persona_col="_persona",
            )
        for p in range(1, 10):
            loc = [t for t in activity_chain_templates if t.get("persona") == p]
            reg_t = [t for t in reg_templates if t.get("persona") == p]
            pool_all = loc + reg_t
            if not pool_all:
                continue
            persona_templates[p] = {
                "all": pool_all,
                "by_oa": _index_templates_by_key(loc, "home_geoid"),
                "by_ward": _index_templates_by_key(loc, "home_ward"),
                "source": "cpca+region" if reg_t else "cpca",
                "n_local": len(loc),
            }
        print("  Persona diary pools (person-days):",
              {p: (v["n_local"], len(v["all"]) - v["n_local"]) for p, v in persona_templates.items()},
              "(cpca, region)")
    templates_by_oa = _index_templates_by_key(activity_chain_templates, "home_geoid")
    templates_by_ward = _index_templates_by_key(activity_chain_templates, "home_ward")
    n_oa_t = sum(1 for t in activity_chain_templates if t.get("home_geoid"))
    n_w_t = sum(1 for t in activity_chain_templates if t.get("home_ward"))

    # Build n_trips distribution for weighted sampling (global, diagnostic)
    all_n = [t['n_trips'] for t in activity_chain_templates]
    n_trips_counts = pd.Series(all_n).value_counts(normalize=True)
    print(f"  Extracted {len(activity_chain_templates):,} person-day templates.")
    print(f"  Templates with home OA: {n_oa_t:,}  ({len(templates_by_oa):,} OAs)")
    print(f"  Templates with home ward: {n_w_t:,}  ({len(templates_by_ward):,} wards)")
    print("  Trip-count distribution:")
    for n, pct in sorted(n_trips_counts.items()):
        print(f"    {n:2d} trips: {pct*100:5.1f}%")

    # ------------------------------------------------------------------
    # 5. Build survey OD destination lookup
    # ------------------------------------------------------------------
    print("[5/7] Building OD destination lookup from filtered survey...")
    od_destination_lookup: Dict = {}
    use_trip_chaining = False
    
    if survey_origin_col is None or survey_dest_col is None:
        print("  Survey origin/dest columns are None — enabling trip chaining mode.")
        use_trip_chaining = True
    elif survey_origin_col in df_filtered.columns and survey_dest_col in df_filtered.columns:
        od_clean = df_filtered.dropna(subset=[survey_origin_col, survey_dest_col]).copy()
        if len(od_clean) > 0:
            od_clean[survey_origin_col] = od_clean[survey_origin_col].astype(str)
            od_clean[survey_dest_col]   = od_clean[survey_dest_col].astype(str)
            od_counts = od_clean.groupby([survey_origin_col, survey_dest_col]).size().reset_index(name='cnt')
            for origin, grp in od_counts.groupby(survey_origin_col):
                total = grp['cnt'].sum()
                od_destination_lookup[str(origin)] = {
                    str(row[survey_dest_col]): row['cnt'] / total
                    for _, row in grp.iterrows()
                }
            print(f"  OD lookup: {len(od_destination_lookup):,} origin zones.")
        else:
            print("  Warning: no valid OD pairs in survey — enabling trip chaining mode.")
            use_trip_chaining = True
    else:
        print("  Warning: survey origin/dest GEOID columns not found — enabling trip chaining mode.")
        use_trip_chaining = True

    # ------------------------------------------------------------------
    # 6. Build centroid lookup and LODES OD lookup (optional)
    # ------------------------------------------------------------------
    centroid_lookup:      Dict = {}
    flows_od_lookup:      Dict = {}
    activity_transition_matrix = None
    oa_to_lad: Dict[str, str] = {}

    if census_block_shp_path is not None:
        print("[6/7] Loading census blocks and building spatial lookups...")
        from .data_io import load_census_block_shapefile
        gdf = load_census_block_shapefile(census_block_shp_path, to_crs=census_crs_target)
        gdf[census_geoid_col] = gdf[census_geoid_col].astype(str)
        centroid_lookup = build_centroid_lookup(gdf, geoid_col=census_geoid_col)
        print(f"  Centroid lookup: {len(centroid_lookup):,} blocks.")
        for lad_col in ("LAD11CD", "LAD21CD", "LAD16CD"):
            if lad_col not in gdf.columns:
                continue
            oa_to_lad = {
                str(o).strip(): str(v).strip()
                for o, v in zip(gdf[census_geoid_col], gdf[lad_col])
                if pd.notna(v) and str(v).strip() not in ("", "nan", "None")
            }
            print(
                f"  OA→LAD ({lad_col}): {len(oa_to_lad):,} OAs, "
                f"{len(set(oa_to_lad.values()))} LADs"
            )
            break

        # Normalise centroid_lookup keys to match persons' GEOID format
        # (generate_population_profiles zero-pads to 15 chars)
        if centroid_lookup:
            target_len = _infer_geoid_len(centroid_lookup.keys())
            persons_geoid_len = max(
                (len(str(p.get('home_geoid', '')).strip()) for p in persons[:20]),
                default=target_len,
            )
            if persons_geoid_len != target_len:
                print(f"  Normalising centroid GEOIDs from {target_len} to {persons_geoid_len} chars")
                centroid_lookup = {
                    _normalise_geoid(k, persons_geoid_len): v
                    for k, v in centroid_lookup.items()
                }

        if flows_path is not None:
            flows_df = load_flow_csv(flows_path)
            flows_df[flows_origin_col]       = flows_df[flows_origin_col].astype(str)
            flows_df[flows_work_geocode_col] = flows_df[flows_work_geocode_col].astype(str)
            flows_od_lookup = build_lodes_destination_lookup(
                flows_df,
                h_geocode_col=flows_origin_col,
                w_geocode_col=flows_work_geocode_col,
                weight_col=flows_weight_col,
            )
            # Normalise flows_od_lookup keys to match persons' GEOID format
            if centroid_lookup and flows_od_lookup:
                norm_len = len(next(iter(centroid_lookup)))
                flows_od_lookup = {
                    _normalise_geoid(k, norm_len): {
                        _normalise_geoid(dk, norm_len): dv
                        for dk, dv in v.items()
                    }
                    for k, v in flows_od_lookup.items()
                }
            print(f"  LODES OD fallback lookup: {len(flows_od_lookup):,} origins.")

        # Build activity-transition matrix from filtered survey
        if (survey_origin_activity_col in df_filtered.columns and
                survey_dest_activity_col   in df_filtered.columns and
                survey_origin_col is not None and
                survey_dest_col   is not None and
                survey_origin_col          in df_filtered.columns and
                survey_dest_col            in df_filtered.columns):
            atm_clean = df_filtered.dropna(subset=[
                survey_origin_col, survey_dest_col,
                survey_origin_activity_col, survey_dest_activity_col,
            ]).copy()
            atm_clean[survey_origin_col] = atm_clean[survey_origin_col].astype(str)
            atm_clean[survey_dest_col]   = atm_clean[survey_dest_col].astype(str)
            activity_transition_matrix = build_activity_transition_matrix(
                atm_clean,
                o_geoid_col=survey_origin_col,
                d_geoid_col=survey_dest_col,
                o_purpose_col=survey_origin_activity_col,
                d_purpose_col=survey_dest_activity_col,
            )
            print(f"  Activity-transition matrix: {len(activity_transition_matrix)} transitions.")
        elif use_trip_chaining:
            print("  Trip chaining mode: activity-transition matrix will use flow data only.")
    else:
        print("[6/7] census_block_shp_path not provided — skipping spatial lookups.")

    # ------------------------------------------------------------------
    # 6b. OSM buildings + NTS destination lengths (draw d*, match POI)
    # ------------------------------------------------------------------
    from .buildings import (
        DECAY_PURPOSES as _DEFAULT_DECAY_PURPOSES,
        DEFAULT_DECAY_BETA,
        load_purpose_building_pools,
        summarise_pools,
    )
    from .nested_mode_choice import (
        decay_beta_by_purpose as _decay_beta_by_purpose,
        nts_distance_by_lad as _nts_distance_by_lad,
        recode_nts_trips as _recode_nts_trips,
        summarise_nts_distance_by_lad as _summarise_nts_distance_by_lad,
    )

    decay_use = tuple(decay_purposes) if decay_purposes is not None else _DEFAULT_DECAY_PURPOSES
    coords_are_projected = not str(census_crs_target).upper().endswith("4326")
    purpose_building_pools: Dict = {}
    decay_beta: Dict[str, float] = dict(DEFAULT_DECAY_BETA)
    nts_dest_dist: Dict = {}

    def _dest_packs(df_src: pd.DataFrame, verbose: bool = True):
        beta = dict(DEFAULT_DECAY_BETA)
        dcol = next(
            (c for c in ("TripDisIncSW", "TripDisExSW", "distance_mi") if c in df_src.columns),
            None,
        )
        if dcol is not None and survey_dest_activity_col in df_src.columns:
            wcol_ = "W5" if "W5" in df_src.columns else None
            tmp_ = pd.DataFrame({
                "purpose": df_src[survey_dest_activity_col].map(lambda v: str(v).strip().lower()),
                "distance_mi": pd.to_numeric(df_src[dcol], errors="coerce"),
                "weight": pd.to_numeric(df_src[wcol_], errors="coerce").fillna(1.0) if wcol_ else 1.0,
            })
            beta.update(_decay_beta_by_purpose(tmp_))
        dist = {}
        try:
            src_ = _recode_nts_trips(df_src)
            if survey_dest_activity_col in df_src.columns:
                known_ = {"work", "education", "shop", "personal", "leisure", "home", "other"}
                mapped_ = df_src[survey_dest_activity_col].map(lambda v: str(v).strip().lower())
                src_ = src_.copy()
                src_["purpose"] = [p if p in known_ else "other" for p in mapped_]
            dist = _nts_distance_by_lad(src_, purposes=dest_pack_purposes)
        except Exception as exc:
            if verbose:
                print(f"  Warning: could not build NTS destination packs ({exc}).")
        return beta, dist

    dest_pack_purposes = tuple(dict.fromkeys(list(decay_use) + ["work", "home"]))
    decay_beta, nts_dest_dist = _dest_packs(df_survey)
    print("[6b/7] NTS exponential distance-decay β (gravity fallback, per mile):")
    for p in decay_use:
        if p in decay_beta:
            print(f"    {p:12s}  β={decay_beta[p]:.3f}")
    if nts_dest_dist:
        print("[6b/7] NTS destination lengths by purpose/LAD (draw d*, match POI):")
        print(" ", _summarise_nts_distance_by_lad(nts_dest_dist))

    persona_dest: Dict[int, Tuple[Dict, Dict]] = {}
    if persona_on:
        for p in persona_templates:
            sub = df_survey[df_survey["_persona"] == p]
            if p in needy_personas and df_region is not None and len(df_region):
                sub = pd.concat([sub, df_region[df_region["_persona"] == p]], ignore_index=True)
            if len(sub):
                b_p, d_p = _dest_packs(sub, verbose=False)
                if d_p:
                    persona_dest[p] = (b_p, d_p)
        print(f"  Persona-specific destination-length packs: {sorted(persona_dest)}")
    if purpose_poi_pools:
        purpose_building_pools = purpose_poi_pools
        print("[6b/7] Using supplied activity POI pools:", summarise_pools(purpose_building_pools))
    elif osm_buildings_path:
        print("[6b/7] Classifying OSM buildings for decay destinations:", osm_buildings_path)
        oa_for_join = None
        if census_block_shp_path is not None:
            try:
                oa_for_join = gdf
            except NameError:
                oa_for_join = None
        try:
            purpose_building_pools = load_purpose_building_pools(
                osm_buildings_path,
                oa_gdf=oa_for_join,
                geoid_col=census_geoid_col,
                target_crs=census_crs_target,
            )
            print("  Building pools:", summarise_pools(purpose_building_pools))
        except Exception as exc:
            print(f"  Warning: could not load OSM buildings ({exc}). Falling back to zone sampling.")
            purpose_building_pools = {}
    else:
        print("[6b/7] No activity POI pools or osm_buildings_path — non-home/work destinations use zone lookups.")

    # ------------------------------------------------------------------
    # 7. Generate trip records for each person
    # ------------------------------------------------------------------
    print(f"[7/7] Generating activity chains for {len(persons):,} persons...")
    start_time = time.time()
    sample_mode_from_person_trip = None
    mode_rng = None
    dest_rng = np.random.default_rng(None if random_seed is None else int(random_seed) + 23)
    from .circuity import geodesic_m_to_network_m as _geo_to_net
    if mode_fit is not None:
        mode_rng = np.random.default_rng(None if random_seed is None else int(random_seed) + 17)
        if mode_fit.get("kind") == "tree":
            from .tree_mode_choice import sample_mode_from_tree as sample_mode_from_person_trip
        else:
            from .nested_mode_choice import sample_mode_from_person_trip as sample_mode_from_person_trip

    # Use default column names if None provided
    o_col = survey_origin_col if survey_origin_col is not None else 'O_GEOID20'
    d_col = survey_dest_col if survey_dest_col is not None else 'D_GEOID20'

    # Cache n_trips distributions per OA / ward so local sampling stays fast
    pool_cache: Dict[Tuple, Tuple[Dict[int, List[Dict]], np.ndarray, np.ndarray]] = {}

    def _pool_stats(level: str, key: str, pool: List[Dict], persona: int = 0):
        ck = (level, key, persona)
        if ck not in pool_cache:
            by_n: Dict[int, List[Dict]] = {}
            for tmpl in pool:
                by_n.setdefault(tmpl["n_trips"], []).append(tmpl)
            n_arr = np.array([t["n_trips"] for t in pool], dtype=int)
            vals, cnts = np.unique(n_arr, return_counts=True)
            pool_cache[ck] = (by_n, vals, (cnts.astype(float) / cnts.sum()))
        return pool_cache[ck]

    # Determine which demographic columns are in the persons dicts
    demo_keys = [k for k in (persons[0].keys() if persons else [])
                 if k not in {'synthetic_person_id', 'household_id', 'person_id',
                               'home_geoid', 'home_lon', 'home_lat',
                               'work_geoid', 'work_lon', 'work_lat'}]

    all_records: List[Dict] = []
    match_counts = {"oa": 0, "ward": 0, "persona": 0, "global": 0}

    for person in persons:
        spid       = person['synthetic_person_id']
        home_geoid = str(person.get('home_geoid', ''))
        home_lon   = float(person.get('home_lon', np.nan))
        home_lat   = float(person.get('home_lat', np.nan))
        work_geoid = str(person.get('work_geoid', ''))
        work_lon   = float(person.get('work_lon', np.nan))
        work_lat   = float(person.get('work_lat', np.nan))
        home_ward  = person.get(person_home_ward_key, "")
        person_lad = str(
            person.get("home_lad")
            or person.get("lad")
            or person.get("HHoldOSLAUA_B01ID")
            or ""
        ).strip()
        if person_lad in {"", "nan", "None"}:
            person_lad = str(
                oa_to_lad.get(home_geoid)
                or oa_to_lad.get(str(home_geoid).strip())
                or ""
            )

        # ---- Steps A/B: sample a local diary (OA → ward → all CPCA) ----
        p_id = 0
        if persona_on:
            try:
                p_id = int(person.get(person_persona_key) or 0)
            except (TypeError, ValueError):
                p_id = 0
        pt = persona_templates.get(p_id)
        if pt is not None:
            pool, match_level = _choose_persona_pool(
                home_geoid, home_ward, pt, persona_min_zone_templates,
            )
        else:
            pool, match_level = _choose_spatial_template_pool(
                home_geoid,
                home_ward,
                templates_by_oa,
                templates_by_ward,
                activity_chain_templates,
                min_zone_templates,
            )
        if not pool:
            pool, match_level = activity_chain_templates, "global"
        if not pool:
            continue
        cache_key = (
            _norm_zone_key(home_geoid) if match_level == "oa"
            else (_norm_zone_key(home_ward) if match_level == "ward" else "")
        )
        by_n, vals, probs = _pool_stats(match_level, cache_key, pool, p_id if pt is not None else 0)
        n_trips = int(np.random.choice(vals, p=probs))
        cands = by_n.get(n_trips)
        if not cands:
            n_trips = min(by_n.keys(), key=lambda x: abs(x - n_trips))
            cands = by_n[n_trips]
        template = cands[int(np.random.randint(len(cands)))]
        n_trips = int(template["n_trips"])
        match_counts[match_level] = match_counts.get(match_level, 0) + 1
        p_beta, p_dest = decay_beta, nts_dest_dist
        if p_id in persona_dest:
            p_beta, p_dest = persona_dest[p_id]

        trip_details = [dict(td) for td in template['trips']]  # make a copy so we can modify

        # ---- Optionally force first/last activity to home ----
        # Do not rewrite home→X as home→home (0 m).  Append a return instead.
        if force_home_start_end and len(trip_details) > 0:
            trip_details = _force_home_tour_ends(trip_details, home_geoid)
            n_trips = len(trip_details)

        # ---- Step C: assign / validate trip times ----
        if continuous_start_times and purpose_time_mixtures:
            dests = [
                trip_details[i]["DTripPur"] if i < len(trip_details) else "home"
                for i in range(n_trips)
            ]
            tmpl_times = [
                trip_details[i].get("time") if i < len(trip_details) else None
                for i in range(n_trips)
            ]
            trip_times = sample_chain_start_times_continuous(
                dest_purposes=dests,
                mixtures=purpose_time_mixtures,
                template_times=tmpl_times,
                min_gap_minutes=min_trip_gap_minutes,
            )
        else:
            trip_times = []
            for seq_idx in range(n_trips):
                if seq_idx < len(trip_details) and trip_details[seq_idx]['time'] is not None:
                    t = int(trip_details[seq_idx]['time'])
                else:
                    o_pur = (str(trip_details[seq_idx]['OTripPur']).lower().strip()
                             if seq_idx < len(trip_details) and trip_details[seq_idx]['OTripPur']
                             else 'home')

                    if activity_time_dists and o_pur in activity_time_dists:
                        dist    = activity_time_dists[o_pur]
                        bin_idx = np.random.choice(len(dist['probs']), p=dist['probs'])
                        t       = int(np.random.uniform(dist['bins'][bin_idx],
                                                        dist['bins'][min(bin_idx + 1,
                                                                         len(dist['bins']) - 1)]))
                    else:
                        t = int(np.random.uniform(0, 1439))

                trip_times.append(max(0, min(t, 1439)))

            for idx in range(1, len(trip_times)):
                if trip_times[idx] <= trip_times[idx - 1]:
                    trip_times[idx] = min(trip_times[idx - 1] + 15, 1439)

        # ---- Step D: build one record per trip ----
        # Track previous destination for trip chaining
        prev_dest_geoid = None
        prev_dest_lon = None
        prev_dest_lat = None
        prev_mode = None

        for seq_idx in range(n_trips):
            t_minutes = float(trip_times[seq_idx])
            t_clip = float(np.clip(t_minutes, 0.0, 24 * 60 - 1e-6))
            sec = int(round(t_clip * 60.0))
            hh_hours, rem = divmod(sec, 3600)
            hh_minutes, hh_seconds = divmod(rem, 60)
            otime_str = f"{hh_hours:02d}:{hh_minutes:02d}:{hh_seconds:02d}"
            o_pur = trip_details[seq_idx]['OTripPur'] if seq_idx < len(trip_details) else None
            d_pur = trip_details[seq_idx]['DTripPur'] if seq_idx < len(trip_details) else None
            o_geoid_tmpl = trip_details[seq_idx].get('O_GEOID20') if seq_idx < len(trip_details) else None
            d_geoid_tmpl = trip_details[seq_idx].get('D_GEOID20') if seq_idx < len(trip_details) else None

            # ---- Trip chaining logic ----
            if use_trip_chaining:
                if seq_idx == 0:
                    o_geoid_tmpl = home_geoid
                else:
                    o_geoid_tmpl = prev_dest_geoid
                d_geoid_tmpl = None

            rec: Dict = {
                'synthetic_person_id': spid,
                'household_id':        person.get('household_id'),
                'person_id':           person.get('person_id'),
                'trip_sequence':       seq_idx + 1,
                'OTripPur':            o_pur,
                'DTripPur':            d_pur,
                'OTimin':              t_minutes,
                'OTIME':               otime_str,
                'home_geoid':          home_geoid,
                'work_geoid':          work_geoid,
                'OLON_RANDOM':         np.nan,
                'OLAT_RANDOM':         np.nan,
                'DLON_RANDOM':         np.nan,
                'DLAT_RANDOM':         np.nan,
                'mode_category':       None,
                'survey_match_level':  match_level,
            }
            
            # Set origin/dest columns using the computed column names
            rec[o_col] = o_geoid_tmpl
            rec[d_col] = d_geoid_tmpl

            # Add demographic attributes
            for k in demo_keys:
                rec[k] = person.get(k)

            # ---- Step E: location assignment ----
            # For trip chaining mode, pre-set origin from previous trip
            origin_locked = (
                use_trip_chaining and seq_idx > 0
                and prev_dest_lon is not None and pd.notna(prev_dest_lon)
            )
            if origin_locked:
                rec[o_col] = prev_dest_geoid
                rec['OLON_RANDOM'] = prev_dest_lon
                rec['OLAT_RANDOM'] = prev_dest_lat

            rec = _assign_trip_locations(
                rec=rec,
                o_purpose=o_pur,
                d_purpose=d_pur,
                person_home_geoid=home_geoid,
                person_home_lon=home_lon,
                person_home_lat=home_lat,
                person_work_geoid=work_geoid,
                person_work_lon=work_lon,
                person_work_lat=work_lat,
                centroid_lookup=centroid_lookup if centroid_lookup else None,
                od_destination_lookup=od_destination_lookup,
                flows_od_lookup=flows_od_lookup if flows_od_lookup else None,
                activity_transition_matrix=activity_transition_matrix,
                survey_origin_col=o_col,
                survey_dest_col=d_col,
                purpose_building_pools=purpose_building_pools or None,
                decay_beta=p_beta,
                decay_purposes=decay_use,
                coords_are_projected=coords_are_projected,
                rng=dest_rng,
                origin_locked=origin_locked,
                nts_by_purpose=p_dest or None,
                lad=person_lad,
            )
            if rec is None:
                continue

            ox, oy = rec.get("OLON_RANDOM"), rec.get("OLAT_RANDOM")
            dx, dy = rec.get("DLON_RANDOM"), rec.get("DLAT_RANDOM")
            if pd.notna(ox) and pd.notna(oy) and pd.notna(dx) and pd.notna(dy):
                dist = float(np.hypot(float(dx) - float(ox), float(dy) - float(oy)))
            else:
                dist = np.nan
            rec["trip_distance"] = dist
            net_m = np.nan
            if pd.notna(dist) and coords_are_projected:
                net_m = float(_geo_to_net(dist))
            rec["trip_distance_network_m"] = net_m
            dur_m = net_m if pd.notna(net_m) else dist
            rec["trip_duration"] = (dur_m / default_speed_ms) if pd.notna(dur_m) and default_speed_ms else np.nan

            mode_cat = None
            if mode_fit is not None and sample_mode_from_person_trip is not None:
                try:
                    if mode_fit.get("kind") == "tree":
                        mode_cat = sample_mode_from_person_trip(
                            person=person,
                            distance_m=dist,
                            purpose=d_pur,
                            prev_mode=prev_mode,
                            fit=mode_fit,
                            rng=mode_rng,
                            coords_in_metres=coords_are_projected,
                            apply_driving_circuity=not pd.notna(net_m),
                            network_distance_m=net_m if pd.notna(net_m) else None,
                            spec=mode_fit.get("mode_spec") or mode_spec,
                        )
                    else:
                        mode_cat = sample_mode_from_person_trip(
                            person=person,
                            distance_m=dist,
                            purpose=d_pur,
                            start_min=t_minutes,
                            prev_mode=prev_mode,
                            theta=mode_fit["theta"],
                            scaler=mode_fit.get("scaler") or {},
                            rng=mode_rng,
                            coords_in_metres=coords_are_projected,
                            apply_driving_circuity=not pd.notna(net_m),
                            network_distance_m=net_m if pd.notna(net_m) else None,
                            spec=mode_fit.get("mode_spec") or mode_spec,
                        )
                except Exception:
                    mode_cat = None
            if not mode_cat:
                mode_cat = trip_details[seq_idx].get("mode_category") if seq_idx < len(trip_details) else None
            if not mode_cat:
                key = str(d_pur).lower().strip() if d_pur is not None else ""
                if key in mode_by_purpose:
                    labels, probs = mode_by_purpose[key]
                    mode_cat = str(np.random.choice(labels, p=probs))
            rec["mode_category"] = mode_cat
            prev_mode = mode_cat

            # Store destination for next trip's origin (trip chaining)
            if use_trip_chaining:
                prev_dest_geoid = rec.get(d_col)
                prev_dest_lon = rec.get('DLON_RANDOM')
                prev_dest_lat = rec.get('DLAT_RANDOM')

            all_records.append(rec)

    elapsed = time.time() - start_time
    n_persons  = len(persons)
    n_trips_total = len(all_records)
    print(f"  Done. {n_trips_total:,} trip records for {n_persons:,} persons "
          f"({n_trips_total/n_persons:.2f} trips/person avg) in {elapsed:.1f}s.")
    if (
        mode_fit is not None
        and mode_fit.get("kind") != "tree"
        and all_records
        and sample_mode_from_person_trip is not None
    ):
        try:
            from .nested_mode_choice import refine_modes_to_nts_shares
            print("  Recalibrating mode ASCs on synthetic trips to NTS shares...")
            refine_modes_to_nts_shares(
                all_records,
                mode_fit,
                rng=mode_rng,
                coords_in_metres=coords_are_projected,
            )
            before = mode_fit.get("synth_share_before") or {}
            after = mode_fit.get("synth_share_after") or {}
            nts_s = mode_fit.get("nts_share") or {}
            fitted_modes = (mode_fit.get("mode_spec") or {}).get("modes") or list(after)
            print("  Mode shares after synthetic ASC calibration (NTS → before → after):")
            for m in fitted_modes:
                print(
                    f"    {m:16s}  {100 * nts_s.get(m, 0):5.1f}% → "
                    f"{100 * before.get(m, 0):5.1f}% → {100 * after.get(m, 0):5.1f}%"
                )
        except Exception as exc:
            print(f"  Warning: synthetic mode-share calibration skipped ({exc}).")
    if n_persons:
        print(
            "  Survey diary match: "
            f"OA {match_counts.get('oa', 0):,} ({100 * match_counts.get('oa', 0) / n_persons:.1f}%)  "
            f"ward {match_counts.get('ward', 0):,} ({100 * match_counts.get('ward', 0) / n_persons:.1f}%)  "
            f"persona-wide {match_counts.get('persona', 0):,} ({100 * match_counts.get('persona', 0) / n_persons:.1f}%)  "
            f"CPCA {match_counts.get('global', 0):,} ({100 * match_counts.get('global', 0) / n_persons:.1f}%)"
        )
    print("=" * 68)
    return all_records