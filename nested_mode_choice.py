"""Distance-decay and nested logit mode choice for CPCA NTS trips.

Distance decay
--------------
``TripDisIncSW`` is trip distance in miles (including short walks).  For each
purpose we estimate exponential and power deterrence functions on a weighted
histogram of that distance.

Nested logit
------------
The choice set and nest tree come from ``ModeSpec`` (notebook
``MODE_CODE_MAP`` / ``MODE_NESTS``).  Package default is five modes::

    root
      active   : walk, bike
      private  : car
      public   : bus, rail

A sixth alternative ``car_passenger`` can sit in the private nest.

Utility uses trip characteristics (distance, purpose, peak period), traveller
attributes (age, sex, income, household vehicles), and supply / land-use
features already on the NTS file (settlement, area type, population density,
origin and destination urban flags).  Car as driver is available when the
household has a vehicle (``NumCar`` / ``NumCarVan`` / ``NumVehicles``);
driving licence is not a choice covariate and does not gate car.  At
application, walk and bike use geodesic length (mild walk circuity);
car / passenger / bus / rail use driving-network miles.  Bike is
unavailable unless the household has a bicycle (``NumBike`` > 0).  Bus is
unavailable under 2 network miles.  After fitting, ASCs are calibrated to
NTS shares on the survey sample and again on synthetic trips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


from .mode_spec import (
    DEFAULT_MODE_CODE_MAP,
    DEFAULT_MODE_SPEC,
    ModeSpec,
)

ACTIVITY_CODE_MAP = {
    1.0: "work", 2.0: "work", 3.0: "education", 4.0: "shop", 5.0: "shop",
    6.0: "personal", 7.0: "personal", 8.0: "personal", 9.0: "leisure",
    10.0: "leisure", 11.0: "leisure", 12.0: "leisure", 13.0: "leisure",
    14.0: "leisure", 15.0: "leisure", 16.0: "other", 17.0: "home",
    18.0: "work", 19.0: "work", 20.0: "education", 21.0: "shop",
    22.0: "other", 23.0: "home", -10.0: "other", -8.0: "other",
}

# Package defaults (five modes; NTS passenger codes map to car).
MODE_CODE_MAP = dict(DEFAULT_MODE_CODE_MAP)
MODES = list(DEFAULT_MODE_SPEC.modes)
NESTS: Dict[str, List[str]] = {k: list(v) for k, v in DEFAULT_MODE_SPEC.nests.items()}
NEST_ORDER = list(DEFAULT_MODE_SPEC.nest_order)
SINGLETON_NESTS = {k for k, v in NESTS.items() if len(v) == 1}

# 2011 rural–urban AreaType2 codes used on this NTS extract
_URBAN_AREA = {1, 2, 3, 4, 10, 11, 12, 13}
_MISSING = {-10, -9, -8, -1}
# Bump when utility/availability rules change so the JSON cache will not reload.
FIT_VERSION = 6
BUS_MIN_MILES = 2.0
# Walk/bike path factor on geodesic metres (driving circuity is much larger).
WALK_CIRCUITY = 1.2
WALK_MAX_MILES = 8.0
BIKE_MAX_MILES = 20.0
# Kept only to turn off adult passenger purpose penalties for children.
# NTS DrivLic_B02ID: 1 = full, 2 = provisional (can drive with a supervisor).
_LICENCE_OK = {1, 1.0, 2, 2.0}


def _resolve_spec(spec=None, frame=None, fit=None, mapping=None) -> ModeSpec:
    if isinstance(spec, ModeSpec):
        return spec
    if isinstance(spec, dict) and spec.get("modes"):
        return ModeSpec.from_dict(spec, mapping=mapping)
    if fit and fit.get("mode_spec"):
        return ModeSpec.from_dict(fit["mode_spec"], mapping=mapping or fit.get("mode_mapping"))
    if frame is not None:
        attrs = getattr(frame, "attrs", None) or {}
        if attrs.get("mode_spec"):
            return ModeSpec.from_dict(attrs["mode_spec"], mapping=mapping)
    if mapping:
        return ModeSpec.from_mapping(mapping)
    return DEFAULT_MODE_SPEC


def _lookup(val, mapping, default):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return default
    if val in mapping:
        return mapping[val]
    try:
        f = float(val)
        if f in mapping:
            return mapping[f]
        if float(f).is_integer() and int(f) in mapping:
            return mapping[int(f)]
    except (TypeError, ValueError, OverflowError):
        pass
    return default


def _valid_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where(~x.isin(_MISSING))


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return _valid_numeric(df[name])
    return pd.Series(np.nan, index=df.index)


def _household_vehicle_count(df: pd.DataFrame) -> pd.Series:
    """Household cars/vans/vehicles. Missing codes are treated as zero."""
    n = pd.Series(0.0, index=df.index, dtype=float)
    for name in ("NumCar", "NumCarVan", "NumVehicles", "n_car"):
        if name not in df.columns:
            continue
        n = np.maximum(n, _col(df, name).fillna(0.0).clip(lower=0.0))
    return pd.Series(n, index=df.index, dtype=float)


def recode_nts_trips(
    df: pd.DataFrame,
    spec: Optional[ModeSpec] = None,
    mode_mapping: Optional[Dict] = None,
) -> pd.DataFrame:
    """Add purpose, mode, land-use and traveller fields used by both analyses."""
    spec = _resolve_spec(spec, mapping=mode_mapping)
    out = df.copy()
    known_purposes = {"work", "education", "shop", "personal", "leisure", "home", "other"}
    mode_set = set(spec.modes)

    def _purpose(v):
        mapped = _lookup(v, ACTIVITY_CODE_MAP, None)
        if mapped:
            return mapped
        s = str(v).strip().lower() if v is not None else ""
        return s if s in known_purposes else "other"

    def _mode(v):
        s = str(v).strip().lower() if v is not None else ""
        if s in mode_set:
            return s
        mapped = _lookup(v, spec.mapping, None)
        if mapped:
            return str(mapped).strip().lower()
        if s in {"pt", "transit"} and "bus" in mode_set:
            return "bus"
        return spec.reference

    purp_col = "TripPurpTo_B01ID" if "TripPurpTo_B01ID" in out.columns else (
        "purpose" if "purpose" in out.columns else None
    )
    mode_col = "MainMode_B04ID" if "MainMode_B04ID" in out.columns else (
        "mode_category" if "mode_category" in out.columns else ("mode" if "mode" in out.columns else None)
    )
    out["purpose"] = out[purp_col].map(_purpose) if purp_col else "other"
    out["mode"] = out[mode_col].map(_mode) if mode_col else "car"
    out["distance_mi"] = pd.to_numeric(out["TripDisIncSW"], errors="coerce")
    out["distance_km"] = out["distance_mi"] * 1.60934
    out["travel_min"] = _col(out, "TripTravTime")
    w = _col(out, "W5").fillna(1.0).clip(lower=0.0)
    out["weight"] = w

    age_col = "Age" if "Age" in out.columns else "Age_B01ID"
    out["age"] = _col(out, age_col)
    out["female"] = (_col(out, "Sex_B01ID") == 2).astype(float)
    inc_col = "IndIncome2002_B02ID" if "IndIncome2002_B02ID" in out.columns else "HHIncome2002_B02ID"
    out["income"] = _col(out, inc_col)
    out["n_car"] = _household_vehicle_count(out)
    out["n_bike"] = _col(out, "NumBike").fillna(0).clip(lower=0)
    out["licence"] = _col(out, "DrivLic_B02ID").isin([1, 2]).astype(float)
    out["car_access"] = (out["n_car"] > 0).astype(float)
    out["hh_size"] = _col(out, "HHoldNumPeople").fillna(1).clip(lower=1)

    dens = _col(out, "HHoldPopDensity")
    dens_fill = dens.clip(lower=0)
    out["log_density"] = np.log1p(dens_fill.fillna(dens_fill.median() if dens_fill.notna().any() else 0))
    out["urban_home"] = (_col(out, "Settlement2011EW_B03ID") == 1).astype(float)
    out["urban_area_home"] = _col(out, "HHoldAreaType2_B01ID").isin(_URBAN_AREA).astype(float)
    out["urban_orig"] = _col(out, "TripOrigAreaType2_B01ID").isin(_URBAN_AREA).astype(float)
    out["urban_dest"] = _col(out, "TripDestAreaType2_B01ID").isin(_URBAN_AREA).astype(float)
    out["urban_od"] = (out["urban_orig"] + out["urban_dest"]) / 2.0

    start = _col(out, "TripStart")
    hour = start / 60.0
    out["peak"] = ((hour.between(7, 9.5)) | (hour.between(16, 18.5))).astype(float)
    out["is_work"] = (out["purpose"] == "work").astype(float)
    out["is_education"] = (out["purpose"] == "education").astype(float)
    out["is_shop"] = (out["purpose"] == "shop").astype(float)
    out["is_home"] = (out["purpose"] == "home").astype(float)
    out.attrs["mode_spec"] = spec.to_dict()
    return out


# ---------------------------------------------------------------------------
# Distance decay
# ---------------------------------------------------------------------------

def _exp_curve(d, a, beta):
    return a * np.exp(-np.clip(beta, 0.0, None) * d)


def _power_curve(d, a, gamma):
    return a * np.power(np.clip(d, 1e-3, None), -np.clip(gamma, 0.0, None))


def _r2(y, yhat) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _scale_a(y: np.ndarray, shape: np.ndarray) -> float:
    denom = float(np.sum(shape * shape))
    return float(np.sum(y * shape) / denom) if denom > 0 else 0.0


def _ols_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Intercept and slope of y ~ a + b x without LAPACK (MKL-safe)."""
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    dx = x - xm
    denom = float(np.sum(dx * dx))
    b = 0.0 if denom <= 0 else float(np.sum(dx * (y - ym)) / denom)
    return ym - b * xm, b


def _loglinear_exp(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """y = a * exp(-beta x) via OLS on log y."""
    intercept, slope = _ols_line(x, np.log(np.clip(y, 1e-12, None)))
    return float(np.exp(intercept)), float(max(-slope, 0.0))


def _loglinear_power(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """y = a * x ** -gamma via OLS on log y."""
    intercept, slope = _ols_line(np.log(np.clip(x, 1e-3, None)), np.log(np.clip(y, 1e-12, None)))
    return float(np.exp(intercept)), float(max(-slope, 0.0))


def histogram_decay(
    distance: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: Optional[np.ndarray] = None,
    d_max: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted trip counts by distance; returns bin midpoints and densities."""
    d = np.asarray(distance, dtype=float)
    w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(d) & np.isfinite(w) & (d > 0) & (d <= d_max) & (w >= 0)
    d, w = d[ok], w[ok]
    if bins is None:
        bins = np.concatenate(([0.0], np.linspace(0.5, 10.0, 20), np.linspace(12.0, d_max, 15)))
    counts, edges = np.histogram(d, bins=bins, weights=w)
    width = np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    dens = np.zeros_like(counts, dtype=float)
    pos = width > 0
    dens[pos] = counts[pos] / width[pos]
    tot = dens.sum()
    if tot > 0:
        dens = dens / tot
    return mid, dens


def fit_decay_functions(
    distance: np.ndarray,
    weights: Optional[np.ndarray] = None,
    d_max: float = 40.0,
) -> Dict:
    """Fit exponential ``exp(-β d)`` and power ``d^{-γ}`` deterrence on a histogram."""
    mid, dens = histogram_decay(distance, weights=weights, d_max=d_max)
    mask = dens > 0
    x, y = mid[mask], dens[mask]
    out = {"mid": mid, "density": dens, "exponential": None, "power": None}
    if len(x) < 4:
        return out
    try:
        _, beta = _loglinear_exp(x, y)
        shape = np.exp(-beta * x)
        a_e = _scale_a(y, shape)
        out["exponential"] = {"a": a_e, "beta": beta, "r2": _r2(y, a_e * shape)}
    except Exception:
        pass
    try:
        _, gamma = _loglinear_power(x, y)
        shape = np.power(np.clip(x, 1e-3, None), -gamma)
        a_p = _scale_a(y, shape)
        out["power"] = {"a": a_p, "gamma": gamma, "r2": _r2(y, a_p * shape)}
    except Exception:
        pass
    return out


def fit_decay_by_purpose(
    df: pd.DataFrame,
    purpose_col: str = "purpose",
    dist_col: str = "distance_mi",
    weight_col: str = "weight",
    d_max: float = 40.0,
) -> pd.DataFrame:
    rows = []
    purposes = [p for p in ["work", "education", "shop", "personal", "leisure", "home", "other"]
                if p in set(df[purpose_col].astype(str))]
    for purpose in purposes:
        sub = df.loc[df[purpose_col] == purpose]
        d = sub[dist_col].to_numpy(dtype=float)
        w = sub[weight_col].to_numpy(dtype=float) if weight_col in sub.columns else None
        fit = fit_decay_functions(d, weights=w, d_max=d_max)
        exp, pwr = fit["exponential"] or {}, fit["power"] or {}
        ok = np.isfinite(d) & (d > 0) & (d <= d_max)
        rows.append({
            "purpose": purpose,
            "n_trips": int(ok.sum()),
            "mean_mi": float(np.average(d[ok], weights=None if w is None else w[ok])) if ok.any() else np.nan,
            "median_mi": float(np.nanmedian(d[ok])) if ok.any() else np.nan,
            "exp_beta": exp.get("beta", np.nan),
            "exp_r2": exp.get("r2", np.nan),
            "power_gamma": pwr.get("gamma", np.nan),
            "power_r2": pwr.get("r2", np.nan),
        })
    return pd.DataFrame(rows)


def decay_beta_by_purpose(
    df: pd.DataFrame,
    purpose_col: str = "purpose",
    dist_col: str = "distance_mi",
    weight_col: str = "weight",
    d_max: float = 40.0,
) -> Dict[str, float]:
    """Return ``{purpose: exponential β}`` for gravity-style destination sampling."""
    tab = fit_decay_by_purpose(
        df, purpose_col=purpose_col, dist_col=dist_col,
        weight_col=weight_col, d_max=d_max,
    )
    out: Dict[str, float] = {}
    for _, row in tab.iterrows():
        b = row.get("exp_beta")
        if pd.notna(b) and float(b) > 0:
            out[str(row["purpose"])] = float(b)
    return out


def _distance_pack(miles: np.ndarray, weights: np.ndarray, d_max: float = 40.0) -> Dict:
    miles = np.asarray(miles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.size != miles.size:
        weights = np.ones(miles.size, dtype=float)
    ok = (
        np.isfinite(miles) & np.isfinite(weights)
        & (miles > 0) & (miles <= d_max) & (weights > 0)
    )
    miles, weights = miles[ok], weights[ok]
    beta = None
    if miles.size >= 8:
        fit = fit_decay_functions(miles, weights, d_max=d_max)
        raw = (fit.get("exponential") or {}).get("beta")
        if raw is not None and np.isfinite(raw) and float(raw) > 0:
            beta = float(raw)
    return {"miles": miles, "weight": weights, "beta": beta, "n": int(miles.size)}


def nts_distance_by_lad(
    df: pd.DataFrame,
    purposes: Sequence[str] = ("work", "home"),
    lad_col: str = "HHoldOSLAUA_B01ID",
    dist_col: str = "TripDisIncSW",
    weight_col: str = "W5",
    purpose_col: str = "purpose",
    d_max: float = 40.0,
) -> Dict[str, Dict]:
    """Empirical trip-length packs (miles) and exponential β, by purpose and LAD.

    ``HHoldOSLAUA_B01ID`` is the NTS household local-authority code.  Each
    pack is ``{miles, weight, beta, n}`` for sampling a reported trip length
    ``d*`` (home–work separations, and secondary destinations by purpose).
    """
    if df is None or len(df) == 0:
        return {}
    work = df
    if purpose_col not in work.columns or (
        dist_col == "TripDisIncSW" and "distance_mi" not in work.columns
        and dist_col not in work.columns
    ):
        work = recode_nts_trips(df)
        if purpose_col not in work.columns:
            purpose_col = "purpose"
        if dist_col not in work.columns and "distance_mi" in work.columns:
            dist_col = "distance_mi"
    if purpose_col not in work.columns:
        work = recode_nts_trips(df)
        purpose_col = "purpose"
    if dist_col not in work.columns:
        if "distance_mi" in work.columns:
            dist_col = "distance_mi"
        elif "TripDisIncSW" in work.columns:
            dist_col = "TripDisIncSW"
        else:
            return {}

    miles = pd.to_numeric(work[dist_col], errors="coerce").to_numpy(dtype=float)
    if weight_col in work.columns:
        weights = pd.to_numeric(work[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    elif "weight" in work.columns:
        weights = pd.to_numeric(work["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        weights = np.ones(len(work), dtype=float)
    purpose = work[purpose_col].astype(str).str.strip().str.lower().to_numpy()
    if lad_col in work.columns:
        lads = work[lad_col].astype(str).replace({"nan": "", "None": ""}).to_numpy()
    else:
        lads = np.full(len(work), "", dtype=object)

    out: Dict[str, Dict] = {}
    for purp in purposes:
        mask = purpose == str(purp).strip().lower()
        all_pack = _distance_pack(miles[mask], weights[mask], d_max=d_max)
        by_lad: Dict[str, Dict] = {}
        for lad in sorted(set(lads[mask])):
            if not lad or lad in {"nan", "None"}:
                continue
            sub = mask & (lads == lad)
            pack = _distance_pack(miles[sub], weights[sub], d_max=d_max)
            if pack["n"] >= 8:
                by_lad[lad] = pack
        out[str(purp)] = {"all": all_pack, "by_lad": by_lad}
    return out


def summarise_nts_distance_by_lad(packs: Dict[str, Dict]) -> str:
    lines = []
    for purp, block in packs.items():
        all_p = block.get("all") or {}
        n_all = int(all_p.get("n") or 0)
        beta = all_p.get("beta")
        beta_s = f"beta={beta:.3f}" if beta else "beta=NA"
        n_lads = len(block.get("by_lad") or {})
        lines.append(f"{purp}: n={n_all:,} {beta_s} across {n_lads} LADs")
    return "; ".join(lines) if lines else "none"


# ---------------------------------------------------------------------------
# Nested logit
# ---------------------------------------------------------------------------

COVARIATES = list(DEFAULT_MODE_SPEC.covariates)


def add_previous_mode(df: pd.DataFrame, spec: Optional[ModeSpec] = None) -> pd.DataFrame:
    """Lagged mode within a person-day; first trip of the day has all prev_* = 0."""
    spec = _resolve_spec(spec, frame=df)
    x = df.copy()
    pid = next((c for c in ("IndividualID", "HHPERSONID", "person_id") if c in x.columns), None)
    day = next((c for c in ("DayID", "ODATE", "TravDay") if c in x.columns), None)
    seq = next((c for c in ("JourSeq", "TripStart", "OTimin") if c in x.columns), None)
    x["prev_mode"] = np.nan
    if pid is not None:
        sort_cols = [c for c in (pid, day, seq) if c is not None]
        x = x.sort_values(sort_cols)
        grp = [c for c in (pid, day) if c is not None]
        x["prev_mode"] = x.groupby(grp, sort=False)["mode"].shift(1)
    for m in spec.modes:
        x[f"prev_{m}"] = (x["prev_mode"] == m).astype(float)
    x["prev_none"] = x["prev_mode"].isna().astype(float)
    return x.reset_index(drop=True)


def _zscore(s: pd.Series) -> Tuple[pd.Series, float, float]:
    mean = float(s.mean()) if s.notna().any() else 0.0
    std = float(s.std()) if s.notna().any() else 1.0
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return (s - mean) / std, mean, std


def _mode_availability(x: pd.DataFrame, spec: ModeSpec) -> pd.DataFrame:
    d = x["distance_mi"]
    has_car = x["n_car"] > 0
    has_bike = x["n_bike"] > 0 if "n_bike" in x.columns else True
    for m in spec.modes:
        if m in spec.active_modes:
            is_walk = m == "walk" or ("walk" in m and "bike" not in m and "cycl" not in m)
            cap = WALK_MAX_MILES if is_walk else BIKE_MAX_MILES
            ok = d <= cap
            if not is_walk:
                ok = ok & has_bike
            x[f"avail_{m}"] = ok.astype(int)
        elif spec.is_passenger(m) or spec.is_driver(m):
            x[f"avail_{m}"] = (has_car | (x["mode"] == m)).astype(int)
        elif "rail" in m:
            x[f"avail_{m}"] = ((d >= 1.0) | (x["mode"] == m)).astype(int)
        elif "bus" in m:
            # NTS bus under 2 miles is ~1%. Opening bus at 1 mile absorbs walk trips.
            x[f"avail_{m}"] = ((d >= BUS_MIN_MILES) | (x["mode"] == m)).astype(int)
        else:
            x[f"avail_{m}"] = 1
        x.loc[x["mode"] == m, f"avail_{m}"] = 1
    return x


def prepare_mode_choice_frame(
    df: pd.DataFrame,
    min_distance_mi: float = 0.05,
    max_distance_mi: float = 80.0,
    spec: Optional[ModeSpec] = None,
    mode_mapping: Optional[Dict] = None,
) -> pd.DataFrame:
    """Estimation sample with availability flags and standardised covariates."""
    spec = _resolve_spec(spec, frame=df, mapping=mode_mapping)
    x = recode_nts_trips(df, spec=spec) if "mode" not in df.columns else df.copy()
    if "prev_none" not in x.columns:
        x = add_previous_mode(x, spec=spec)
    if "is_home" not in x.columns:
        x["is_home"] = (x.get("purpose", pd.Series(index=x.index)) == "home").astype(float)
    x = x.loc[x["distance_mi"].between(min_distance_mi, max_distance_mi)].copy()
    x = x.loc[x["mode"].isin(spec.modes)].copy()
    x["log_dist"] = np.log1p(x["distance_mi"].clip(lower=0))
    age = x["age"].replace({np.nan: x["age"].median()})
    inc = x["income"].replace({np.nan: x["income"].median()})
    hh = x["hh_size"].replace({np.nan: x["hh_size"].median()})
    x["age_z"], age_mean, age_std = _zscore(age)
    x["income_z"], inc_mean, inc_std = _zscore(inc)
    x["hh_size_z"], hh_mean, hh_std = _zscore(hh)
    scaler = {
        "age_mean": age_mean, "age_std": age_std,
        "income_mean": inc_mean, "income_std": inc_std,
        "hh_size_mean": hh_mean, "hh_size_std": hh_std,
        "log_density_fill": float(x["log_density"].median()) if x["log_density"].notna().any() else 0.0,
    }

    x = _mode_availability(x, spec)
    keep = np.zeros(len(x), dtype=bool)
    for m in spec.modes:
        keep |= (x["mode"] == m) & (x[f"avail_{m}"] == 1)
    out = x.loc[keep].reset_index(drop=True)
    out.attrs["scaler"] = scaler
    out.attrs["mode_spec"] = spec.to_dict()
    return out


def _param_layout(spec: Optional[ModeSpec] = None) -> Tuple[List[str], Dict[str, int]]:
    spec = spec or DEFAULT_MODE_SPEC
    names: List[str] = []
    idx: Dict[str, int] = {}

    def add(name: str):
        idx[name] = len(names)
        names.append(name)

    for m in spec.modes:
        if m != spec.reference:
            add(f"ASC_{m}")
    for m in spec.modes:
        add(f"b_logdist_{m}")
    for cov in spec.covariates:
        for m in spec.modes:
            if m != spec.reference:
                add(f"b_{cov}_{m}")
    for nest, members in spec.nests.items():
        if len(members) > 1:
            add(f"mu_{nest}")
    return names, idx


PARAM_NAMES, PARAM_INDEX = _param_layout(DEFAULT_MODE_SPEC)


def _layout(spec: ModeSpec) -> Tuple[List[str], Dict[str, int]]:
    return _param_layout(spec)


def _utilities(theta: np.ndarray, frame: pd.DataFrame, spec: Optional[ModeSpec] = None) -> np.ndarray:
    """Return (N, J) utilities. The reference alternative has ASC = 0."""
    spec = _resolve_spec(spec, frame=frame)
    names, idx = _param_layout(spec)
    n = len(frame)
    j = len(spec.modes)
    v = np.zeros((n, j), dtype=float)
    generic = frame["log_dist"].to_numpy(dtype=float)
    for j_i, mode in enumerate(spec.modes):
        col = f"log_dist_{mode}"
        log_dist = frame[col].to_numpy(dtype=float) if col in frame.columns else generic
        if mode != spec.reference:
            v[:, j_i] += theta[idx[f"ASC_{mode}"]]
        v[:, j_i] += theta[idx[f"b_logdist_{mode}"]] * log_dist
        for cov in spec.covariates:
            if mode == spec.reference:
                continue
            if cov not in frame.columns:
                continue
            colv = frame[cov].to_numpy(dtype=float)
            # Purpose shifts on passenger were identified vs adult car drivers.
            # Unlicensed travellers (children) would otherwise be pushed onto bus
            # for education/work because b_is_education_car_passenger is large and negative.
            # Licence is not a car predictor; it only turns those adult passenger
            # purpose penalties off for children.
            if spec.is_passenger(mode) and cov.startswith("is_") and "licence" in frame.columns:
                colv = colv * frame["licence"].to_numpy(dtype=float)
            v[:, j_i] += theta[idx[f"b_{cov}_{mode}"]] * colv
    return v


def _availability(frame: pd.DataFrame, spec: Optional[ModeSpec] = None) -> np.ndarray:
    spec = _resolve_spec(spec, frame=frame)
    return np.column_stack([frame[f"avail_{m}"].to_numpy(dtype=float) for m in spec.modes])


def nested_logit_probs(
    theta: np.ndarray,
    frame: pd.DataFrame,
    spec: Optional[ModeSpec] = None,
) -> np.ndarray:
    """(N, J) choice probabilities under the nest tree in *spec*."""
    spec = _resolve_spec(spec, frame=frame)
    names, idx = _param_layout(spec)
    v = _utilities(theta, frame, spec)
    avail = _availability(frame, spec)
    mu = {}
    for nest, members in spec.nests.items():
        key = f"mu_{nest}"
        if len(members) <= 1 or key not in idx:
            mu[nest] = 1.0
        else:
            mu[nest] = float(np.clip(theta[idx[key]], 0.15, 1.5))

    n = v.shape[0]
    iv = np.zeros((n, len(spec.nest_order)))
    p_cond = {}
    large_neg = -1e3

    for k, nest in enumerate(spec.nest_order):
        members = spec.nests[nest]
        cols = [spec.modes.index(m) for m in members]
        scale = mu[nest]
        vk = v[:, cols] / scale
        vk = np.where(avail[:, cols] > 0, vk, large_neg)
        vmax = np.max(vk, axis=1, keepdims=True)
        ev = np.exp(vk - vmax) * avail[:, cols]
        denom = ev.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)
        p_cond[nest] = ev / denom
        logsum = (vmax + np.log(np.clip(denom, 1e-12, None))).ravel()
        iv[:, k] = scale * logsum
        iv[:, k] = np.where(avail[:, cols].sum(axis=1) > 0, iv[:, k], large_neg)

    iv_max = iv.max(axis=1, keepdims=True)
    eiv = np.exp(iv - iv_max)
    p_nest = eiv / eiv.sum(axis=1, keepdims=True)

    p = np.zeros_like(v)
    for k, nest in enumerate(spec.nest_order):
        members = spec.nests[nest]
        cols = [spec.modes.index(m) for m in members]
        p[:, cols] = p_cond[nest] * p_nest[:, [k]]
    row = p.sum(axis=1, keepdims=True)
    row = np.where(row <= 0, 1.0, row)
    return p / row


def nested_logit_loglik(
    theta: np.ndarray,
    frame: pd.DataFrame,
    spec: Optional[ModeSpec] = None,
) -> float:
    spec = _resolve_spec(spec, frame=frame)
    p = nested_logit_probs(theta, frame, spec)
    y = frame["mode"].map({m: i for i, m in enumerate(spec.modes)}).to_numpy(dtype=int)
    chosen = p[np.arange(len(frame)), y]
    w = frame["weight"].to_numpy(dtype=float)
    return float(np.sum(w * np.log(np.clip(chosen, 1e-12, 1.0))))


def _start_values(frame: pd.DataFrame, spec: Optional[ModeSpec] = None) -> np.ndarray:
    spec = _resolve_spec(spec, frame=frame)
    names, idx = _param_layout(spec)
    theta = np.zeros(len(names))
    w = frame["weight"].to_numpy(dtype=float)
    shares = {m: float(np.average(frame["mode"] == m, weights=w)) for m in spec.modes}
    p_ref = max(shares.get(spec.reference, 0.0), 1e-3)
    for m in spec.modes:
        if m != spec.reference:
            theta[idx[f"ASC_{m}"]] = np.log(max(shares.get(m, 0.0), 1e-4) / p_ref)
        theta[idx[f"b_logdist_{m}"]] = spec.logdist0(m)
    for nest, members in spec.nests.items():
        key = f"mu_{nest}"
        if key in idx:
            theta[idx[key]] = 0.7
    return theta


def _clip_theta(theta: np.ndarray, mnl: bool, spec: Optional[ModeSpec] = None) -> np.ndarray:
    spec = spec or DEFAULT_MODE_SPEC
    names, idx = _param_layout(spec)
    th = np.asarray(theta, dtype=float).copy()
    for nest, members in spec.nests.items():
        key = f"mu_{nest}"
        if key not in idx:
            continue
        if mnl:
            th[idx[key]] = 1.0
        else:
            th[idx[key]] = float(np.clip(th[idx[key]], 0.2, 1.0))
    return th


def _param_groups(mnl: bool, spec: Optional[ModeSpec] = None) -> List[List[int]]:
    spec = spec or DEFAULT_MODE_SPEC
    names, idx = _param_layout(spec)
    groups = [
        [idx[f"ASC_{m}"] for m in spec.modes if m != spec.reference],
        [idx[f"b_logdist_{m}"] for m in spec.modes],
    ]
    if not mnl:
        mu_idx = [idx[f"mu_{nest}"] for nest, members in spec.nests.items()
                  if len(members) > 1 and f"mu_{nest}" in idx]
        if mu_idx:
            groups.append(mu_idx)
    for cov in spec.covariates:
        groups.append([idx[f"b_{cov}_{m}"] for m in spec.modes if m != spec.reference])
    return groups


def calibrate_mode_ascs(
    theta: np.ndarray,
    frame: pd.DataFrame,
    spec: Optional[ModeSpec] = None,
    n_iter: int = 12,
    damping: float = 0.85,
    target_share: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Shift ASCs so mean predicted shares match *target_share* (or observed)."""
    spec = _resolve_spec(spec, frame=frame)
    names, idx = _param_layout(spec)
    th = np.asarray(theta, dtype=float).copy()
    if "weight" in frame.columns:
        w = frame["weight"].to_numpy(dtype=float)
    else:
        w = np.ones(len(frame), dtype=float)
    wsum = float(w.sum()) if w.size else 1.0
    if target_share:
        target = np.array([float(target_share.get(m, 0.0)) for m in spec.modes], dtype=float)
        tsum = float(target.sum())
        if tsum > 0:
            target = target / tsum
    else:
        target = np.array([
            np.average(frame["mode"] == m, weights=w) for m in spec.modes
        ], dtype=float)
    for _ in range(int(n_iter)):
        p = nested_logit_probs(th, frame, spec)
        pred = (p * w[:, None]).sum(axis=0) / max(wsum, 1e-12)
        for j, m in enumerate(spec.modes):
            if m == spec.reference:
                continue
            key = f"ASC_{m}"
            if key not in idx:
                continue
            th[idx[key]] += damping * float(np.log((target[j] + 1e-6) / (pred[j] + 1e-6)))
    return th


def _powell_block(nll, theta: np.ndarray, idx: Sequence[int], maxiter: int) -> np.ndarray:
    """Powell on a subset of parameters (no LAPACK / L-BFGS)."""
    idx = list(idx)
    if not idx:
        return theta
    x0 = theta[idx].copy()

    def sub(x):
        th = theta.copy()
        th[idx] = x
        return nll(th)

    res = minimize(sub, x0, method="Powell", options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-6})
    out = theta.copy()
    out[idx] = res.x
    return out


def fit_nested_logit(
    frame: pd.DataFrame,
    maxiter: int = 40,
    mnl: bool = False,
    n_rounds: int = 3,
    spec: Optional[ModeSpec] = None,
) -> Dict:
    """Maximum weighted likelihood via block-coordinate Powell (MKL-safe).

    Set *mnl* True to freeze nest scales at 1 (multinomial logit).
    """
    spec = _resolve_spec(spec, frame=frame)
    names, idx = _param_layout(spec)
    theta = _clip_theta(_start_values(frame, spec), mnl, spec)

    def nll(th):
        return -nested_logit_loglik(_clip_theta(th, mnl, spec), frame, spec)

    groups = _param_groups(mnl, spec)
    f0 = nll(theta)
    for _ in range(n_rounds):
        for g in groups:
            theta = _clip_theta(_powell_block(nll, theta, g, maxiter=maxiter), mnl, spec)
        f1 = nll(theta)
        if abs(f0 - f1) < 1e-3:
            break
        f0 = f1

    theta = _clip_theta(theta, mnl, spec)
    theta = calibrate_mode_ascs(theta, frame, spec)
    ll = nested_logit_loglik(theta, frame, spec)
    p = nested_logit_probs(theta, frame, spec)
    y = frame["mode"].map({m: i for i, m in enumerate(spec.modes)}).to_numpy(dtype=int)
    pred = p.argmax(axis=1)
    acc = float(np.average(pred == y, weights=frame["weight"]))
    w = frame["weight"].to_numpy(dtype=float)
    pshare = np.array([
        np.average(frame["mode"] == m, weights=w) for m in spec.modes
    ], dtype=float)
    pred_share = (p * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-12)
    ll_const = float(np.sum(w * np.log(np.clip(pshare[y], 1e-12, 1.0))))
    rho2 = 1.0 - ll / ll_const if ll_const != 0 else np.nan
    params = pd.DataFrame({"param": names, "estimate": theta})
    mus = {f"mu_{nest}": float(theta[idx[f"mu_{nest}"]])
           for nest, members in spec.nests.items()
           if f"mu_{nest}" in idx}
    return {
        "success": True,
        "message": f"block-Powell rounds finished (nll={f0:.3f})",
        "nobs": int(len(frame)),
        "loglik": ll,
        "loglik_constants": ll_const,
        "rho2": rho2,
        "accuracy": acc,
        "params": params,
        "theta": theta,
        "scaler": dict(frame.attrs.get("scaler") or {}),
        "param_names": names,
        "mnl": mnl,
        "mode_spec": spec.to_dict(),
        "probs": p,
        "nts_share": {m: float(pshare[i]) for i, m in enumerate(spec.modes)},
        "pred_share": {m: float(pred_share[i]) for i, m in enumerate(spec.modes)},
        **mus,
    }


def predict_mode_probs(
    frame: pd.DataFrame,
    theta: np.ndarray,
    spec: Optional[ModeSpec] = None,
) -> pd.DataFrame:
    spec = _resolve_spec(spec, frame=frame)
    p = nested_logit_probs(theta, frame, spec)
    out = pd.DataFrame(p, columns=[f"p_{m}" for m in spec.modes])
    out["mode_hat"] = [spec.modes[i] for i in p.argmax(axis=1)]
    return out


def sample_mode(
    frame: pd.DataFrame,
    theta: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    spec: Optional[ModeSpec] = None,
) -> np.ndarray:
    spec = _resolve_spec(spec, frame=frame)
    rng = rng or np.random.default_rng()
    p = nested_logit_probs(theta, frame, spec)
    u = rng.random(len(frame))
    cdf = np.cumsum(p, axis=1)
    idx = (u[:, None] > cdf).sum(axis=1)
    idx = np.clip(idx, 0, len(spec.modes) - 1)
    return np.array(spec.modes)[idx]


def save_mode_choice_fit(path: Union[str, Path], fit: Dict) -> Path:
    path = Path(path)
    payload = {
        "theta": np.asarray(fit["theta"], dtype=float).tolist(),
        "scaler": fit.get("scaler") or {},
        "param_names": list(fit.get("param_names") or PARAM_NAMES),
        "rho2": fit.get("rho2"),
        "accuracy": fit.get("accuracy"),
        "nobs": fit.get("nobs"),
        "mnl": bool(fit.get("mnl", False)),
        "mode_spec": fit.get("mode_spec") or DEFAULT_MODE_SPEC.to_dict(),
        "fit_version": FIT_VERSION,
        "nts_share": fit.get("nts_share") or {},
        "pred_share": fit.get("pred_share") or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_mode_choice_fit(
    path: Union[str, Path],
    spec: Optional[ModeSpec] = None,
) -> Optional[Dict]:
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = list(payload.get("param_names") or [])
    want_spec = _resolve_spec(spec)
    want_names, _ = _param_layout(want_spec)
    cached_spec = ModeSpec.from_dict(payload.get("mode_spec"), mapping=want_spec.mapping)
    if names != list(want_names) or cached_spec.modes != want_spec.modes:
        return None
    if cached_spec.nests != want_spec.nests or cached_spec.reference != want_spec.reference:
        return None
    if int(payload.get("fit_version") or 0) != FIT_VERSION:
        return None
    theta = np.asarray(payload["theta"], dtype=float)
    if theta.shape != (len(want_names),):
        return None
    return {
        "theta": theta,
        "scaler": payload.get("scaler") or {},
        "param_names": names,
        "rho2": payload.get("rho2"),
        "accuracy": payload.get("accuracy"),
        "nobs": payload.get("nobs"),
        "mnl": payload.get("mnl", False),
        "mode_spec": cached_spec.to_dict(),
        "success": True,
        "from_cache": True,
        "nts_share": payload.get("nts_share") or {},
        "pred_share": payload.get("pred_share") or {},
    }


def fit_mode_choice_model(
    df: pd.DataFrame,
    cache_path: Optional[Union[str, Path]] = None,
    max_n: int = 25000,
    mnl: bool = False,
    n_rounds: int = 2,
    maxiter: int = 25,
    spec: Optional[ModeSpec] = None,
    mode_mapping: Optional[Dict] = None,
    mode_nests: Optional[Dict] = None,
) -> Dict:
    """Fit (or reload) the nested logit used when synthesising trip modes."""
    if spec is None:
        spec = ModeSpec.from_mapping(mode_mapping, nests=mode_nests) if (mode_mapping or mode_nests) else DEFAULT_MODE_SPEC
    if cache_path:
        cached = load_mode_choice_fit(cache_path, spec=spec)
        if cached is not None:
            return cached
    frame = prepare_mode_choice_frame(df, spec=spec, mode_mapping=spec.mapping)
    scaler = dict(frame.attrs.get("scaler") or {})
    if len(frame) > max_n:
        frame = frame.sample(n=max_n, random_state=42).reset_index(drop=True)
        frame.attrs["scaler"] = scaler
        frame.attrs["mode_spec"] = spec.to_dict()
    fit = fit_nested_logit(frame, maxiter=maxiter, mnl=mnl, n_rounds=n_rounds, spec=spec)
    fit["scaler"] = scaler
    names, _ = _param_layout(spec)
    fit["param_names"] = list(names)
    fit["mode_spec"] = spec.to_dict()
    if cache_path:
        save_mode_choice_fit(cache_path, fit)
    return fit


def _person_value(person: Dict, names: Sequence[str], default=np.nan):
    for name in names:
        if name in person and person[name] is not None:
            val = person[name]
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass
            return val
    return default


def _person_float(person: Dict, names: Sequence[str], default: float = np.nan) -> float:
    val = _person_value(person, names, default=default)
    try:
        out = float(val)
        return out if np.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _person_vehicle_count(person: Dict) -> float:
    n = 0.0
    for key in ("NumCar", "NumCarVan", "NumVehicles", "n_car"):
        v = _person_float(person, (key,), default=np.nan)
        if np.isfinite(v):
            n = max(n, v)
    return max(n, 0.0)


def sample_mode_from_person_trip(
    person: Dict,
    distance_m: float,
    purpose: Optional[str],
    start_min: Optional[float],
    prev_mode: Optional[str],
    theta: np.ndarray,
    scaler: Dict,
    rng: Optional[np.random.Generator] = None,
    coords_in_metres: bool = True,
    apply_driving_circuity: bool = True,
    network_distance_m: Optional[float] = None,
    spec: Optional[ModeSpec] = None,
) -> str:
    """Draw a MATSim-style mode for one synthetic trip from the nested logit.

    ``distance_m`` is the Euclidean / geodesic OD length.  Walk and bike see
    that length times ``WALK_CIRCUITY``.  Car, passenger, bus and rail see
    driving-network miles.  Car requires a household vehicle; bike requires
    a household bicycle.  Bus is closed under ``BUS_MIN_MILES``.
    """
    spec = _resolve_spec(spec)
    rng = rng or np.random.default_rng()
    geo_m = float(distance_m) if np.isfinite(distance_m) else np.nan
    if not np.isfinite(geo_m) or geo_m < 0:
        geo_m = 800.0
    if not coords_in_metres:
        geo_m = geo_m * 111_320.0

    net_m = np.nan
    if network_distance_m is not None:
        try:
            net_m = float(network_distance_m)
        except (TypeError, ValueError):
            net_m = np.nan
    if not np.isfinite(net_m) or net_m < 0:
        if apply_driving_circuity:
            from .circuity import geodesic_m_to_network_m
            net_m = float(geodesic_m_to_network_m(geo_m))
        else:
            net_m = geo_m

    miles_motor = max(net_m / 1609.34, 0.05)
    miles_active = max(geo_m * WALK_CIRCUITY / 1609.34, 0.05)
    log_motor = float(np.log1p(miles_motor))
    log_active = float(np.log1p(miles_active))

    purpose = str(purpose or "other").strip().lower()
    hour = (float(start_min) / 60.0) if start_min is not None and np.isfinite(start_min) else 12.0
    peak = 1.0 if (7.0 <= hour <= 9.5) or (16.0 <= hour <= 18.5) else 0.0

    age = _person_float(person, ("Age", "age", "Age_B01ID", "Age_B"), default=np.nan)
    if not np.isfinite(age):
        age = float(scaler.get("age_mean", 40.0))
    income = _person_float(
        person,
        ("IndIncome2002_B02ID", "IndIncomeRng", "HHIncome2002_B02ID", "income"),
        default=np.nan,
    )
    if not np.isfinite(income):
        income = float(scaler.get("income_mean", 2.0))
    hh_size = _person_float(person, ("HHoldNumPeople", "hh_size"), default=np.nan)
    if not np.isfinite(hh_size) or hh_size <= 0:
        hh_size = float(scaler.get("hh_size_mean", 2.0))
    n_car = _person_vehicle_count(person)
    n_bike = max(_person_float(person, ("NumBike", "n_bike"), default=0.0), 0.0)
    sex = _person_float(person, ("Sex_B01ID", "Sex"), default=np.nan)
    female = 1.0 if sex == 2 else 0.0
    licence_raw = _person_float(person, ("DrivLic_B02ID", "licence"), default=0.0)
    licence = 1.0 if licence_raw in _LICENCE_OK else 0.0
    dens = _person_float(person, ("HHoldPopDensity", "log_density"), default=np.nan)
    if np.isfinite(dens) and dens > 8:
        log_density = float(np.log1p(max(dens, 0.0)))
    elif np.isfinite(dens) and dens >= 0:
        log_density = dens if "log_density" in person else float(np.log1p(dens))
    else:
        log_density = float(scaler.get("log_density_fill", 0.0))
    urban_home = 1.0 if _person_float(person, ("Settlement2011EW_B03ID", "urban_home"), default=2.0) == 1 else 0.0
    if person.get("urban_home") in (0, 1, 0.0, 1.0):
        urban_home = float(person.get("urban_home"))

    age_z = (age - float(scaler.get("age_mean", age))) / float(scaler.get("age_std", 1.0) or 1.0)
    income_z = (income - float(scaler.get("income_mean", income))) / float(scaler.get("income_std", 1.0) or 1.0)
    hh_size_z = (hh_size - float(scaler.get("hh_size_mean", hh_size))) / float(scaler.get("hh_size_std", 1.0) or 1.0)

    prev = str(prev_mode).strip().lower() if prev_mode else ""
    if miles_motor < BUS_MIN_MILES and prev in spec.public_modes:
        prev = ""
    has_car = n_car > 0
    cov = {
        "age_z": age_z,
        "female": female,
        "income_z": income_z,
        "hh_size_z": hh_size_z,
        "n_car": n_car,
        "licence": licence,
        "urban_home": urban_home,
        "log_density": log_density,
        "urban_dest": urban_home,
        "urban_od": urban_home,
        "is_work": 1.0 if purpose == "work" else 0.0,
        "is_education": 1.0 if purpose == "education" else 0.0,
        "is_shop": 1.0 if purpose == "shop" else 0.0,
        "is_home": 1.0 if purpose == "home" else 0.0,
        "peak": peak,
        "prev_none": 1.0 if prev not in spec.modes else 0.0,
    }
    for m in spec.modes:
        cov[f"prev_{m}"] = 1.0 if prev == m else 0.0

    avail = []
    log_by_mode = {}
    for m in spec.modes:
        if m in spec.active_modes:
            is_walk = m == "walk" or ("walk" in m and "bike" not in m and "cycl" not in m)
            log_by_mode[m] = log_active
            cap = WALK_MAX_MILES if is_walk else BIKE_MAX_MILES
            a = 1.0 if miles_active <= cap else 0.0
            if (not is_walk) and n_bike <= 0:
                a = 0.0
            avail.append(a)
        else:
            log_by_mode[m] = log_motor
            if spec.is_passenger(m) or spec.is_driver(m):
                avail.append(1.0 if has_car else 0.0)
            elif "rail" in m:
                avail.append(1.0 if miles_motor >= 1.0 else 0.0)
            elif "bus" in m:
                avail.append(1.0 if miles_motor >= BUS_MIN_MILES else 0.0)
            else:
                avail.append(1.0)
    avail = np.asarray(avail, dtype=float)
    if avail.sum() <= 0:
        avail[:] = 1.0

    row = pd.DataFrame({
        "log_dist": [log_motor],
        **{f"log_dist_{m}": [log_by_mode[m]] for m in spec.modes},
        **{name: [cov.get(name, 0.0)] for name in spec.covariates},
        "licence": [licence],
        **{f"avail_{m}": [avail[i]] for i, m in enumerate(spec.modes)},
    })
    row.attrs["mode_spec"] = spec.to_dict()
    p = nested_logit_probs(theta, row, spec)[0]
    p = np.where(np.isfinite(p), p, 0.0)
    p = p * avail
    s = p.sum()
    if s <= 0:
        p = avail / avail.sum()
    else:
        p = p / s
    idx = int(rng.choice(len(spec.modes), p=p))
    return spec.modes[idx]


def _numeric_series(df: pd.DataFrame, names: Sequence[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def synthetic_records_to_mode_frame(
    records: Sequence[Dict],
    scaler: Dict,
    spec: Optional[ModeSpec] = None,
    coords_in_metres: bool = True,
) -> pd.DataFrame:
    """Application design matrix: walk/bike on geodesic miles, motor on network."""
    spec = _resolve_spec(spec)
    x = pd.DataFrame.from_records(list(records))
    if x.empty:
        return x
    geo = _numeric_series(x, ("trip_distance",)).to_numpy(dtype=float)
    net = _numeric_series(x, ("trip_distance_network_m",)).to_numpy(dtype=float)
    geo = np.where(np.isfinite(geo) & (geo >= 0), geo, 800.0)
    if not coords_in_metres:
        geo = geo * 111_320.0
    net = np.where(np.isfinite(net) & (net >= 0), net, geo)
    miles_act = np.clip(geo * WALK_CIRCUITY / 1609.34, 0.05, None)
    miles_mot = np.clip(net / 1609.34, 0.05, None)
    log_act = np.log1p(miles_act)
    log_mot = np.log1p(miles_mot)

    purpose = (
        x["DTripPur"].astype(str).str.strip().str.lower()
        if "DTripPur" in x.columns else pd.Series("other", index=x.index)
    )
    start = _numeric_series(x, ("OTimin", "OTIME"), default=12 * 60).fillna(12 * 60)
    hour = start / 60.0
    peak = ((hour.between(7.0, 9.5)) | (hour.between(16.0, 18.5))).astype(float)

    age = _numeric_series(x, ("Age", "age", "Age_B01ID", "Age_B")).fillna(
        float(scaler.get("age_mean", 40.0))
    )
    income = _numeric_series(
        x, ("IndIncome2002_B02ID", "IndIncomeRng", "HHIncome2002_B02ID", "income")
    ).fillna(float(scaler.get("income_mean", 2.0)))
    hh = _numeric_series(x, ("HHoldNumPeople", "hh_size")).fillna(
        float(scaler.get("hh_size_mean", 2.0))
    ).clip(lower=1)
    n_car = np.zeros(len(x), dtype=float)
    for name in ("NumCar", "NumCarVan", "NumVehicles", "n_car"):
        if name in x.columns:
            n_car = np.maximum(n_car, _numeric_series(x, (name,)).fillna(0).clip(lower=0).to_numpy())
    n_bike = _numeric_series(x, ("NumBike", "n_bike")).fillna(0).clip(lower=0).to_numpy()
    sex = _numeric_series(x, ("Sex_B01ID", "Sex"))
    female = (sex == 2).astype(float)
    lic_raw = _numeric_series(x, ("DrivLic_B02ID", "licence"))
    licence = lic_raw.isin(list(_LICENCE_OK)).astype(float)
    dens = _numeric_series(x, ("HHoldPopDensity", "log_density"))
    log_density = pd.Series(float(scaler.get("log_density_fill", 0.0)), index=x.index)
    dens_ok = dens.notna()
    log_density.loc[dens_ok & (dens > 8)] = np.log1p(dens.loc[dens_ok & (dens > 8)].clip(lower=0))
    log_density.loc[dens_ok & (dens >= 0) & (dens <= 8)] = dens.loc[dens_ok & (dens >= 0) & (dens <= 8)]
    urban_home = (_numeric_series(x, ("Settlement2011EW_B03ID",), default=2.0) == 1).astype(float)
    if "urban_home" in x.columns:
        uh = pd.to_numeric(x["urban_home"], errors="coerce")
        urban_home = uh.where(uh.isin([0, 1, 0.0, 1.0]), urban_home).astype(float)

    age_z = (age - float(scaler.get("age_mean", 40.0))) / float(scaler.get("age_std", 1.0) or 1.0)
    income_z = (income - float(scaler.get("income_mean", 2.0))) / float(scaler.get("income_std", 1.0) or 1.0)
    hh_size_z = (hh - float(scaler.get("hh_size_mean", 2.0))) / float(scaler.get("hh_size_std", 1.0) or 1.0)

    mode_col = "mode_category" if "mode_category" in x.columns else "mode"
    pid = x["synthetic_person_id"] if "synthetic_person_id" in x.columns else pd.Series(0, index=x.index)
    seq = _numeric_series(x, ("trip_sequence",), default=0).fillna(0)
    order = np.lexsort((seq.to_numpy(), pid.astype(str).to_numpy()))
    prev = pd.Series(np.nan, index=x.index, dtype=object)
    sorted_pid = pid.astype(str).to_numpy()[order]
    sorted_mode = x[mode_col].to_numpy()[order] if mode_col in x.columns else np.array([None] * len(x))
    shifted = np.empty(len(x), dtype=object)
    shifted[:] = None
    for i in range(1, len(order)):
        if sorted_pid[i] == sorted_pid[i - 1]:
            shifted[i] = sorted_mode[i - 1]
    prev.iloc[order] = shifted

    has_car = n_car > 0
    out = pd.DataFrame({
        "log_dist": log_mot,
        "weight": 1.0,
        "mode": x[mode_col] if mode_col in x.columns else spec.reference,
        "licence": licence.to_numpy(dtype=float),
        "age_z": age_z.to_numpy(dtype=float),
        "female": female.to_numpy(dtype=float),
        "income_z": income_z.to_numpy(dtype=float),
        "hh_size_z": hh_size_z.to_numpy(dtype=float),
        "n_car": n_car,
        "urban_home": urban_home.to_numpy(dtype=float),
        "log_density": log_density.to_numpy(dtype=float),
        "urban_dest": urban_home.to_numpy(dtype=float),
        "urban_od": urban_home.to_numpy(dtype=float),
        "is_work": (purpose == "work").astype(float).to_numpy(),
        "is_education": (purpose == "education").astype(float).to_numpy(),
        "is_shop": (purpose == "shop").astype(float).to_numpy(),
        "is_home": (purpose == "home").astype(float).to_numpy(),
        "peak": peak.to_numpy(dtype=float),
    })
    prev_s = prev.astype(str).str.strip().str.lower()
    out["prev_none"] = (~prev_s.isin(spec.modes)).astype(float)
    for m in spec.modes:
        out[f"prev_{m}"] = (prev_s == m).astype(float)
        out[f"log_dist_{m}"] = log_act if m in spec.active_modes else log_mot
        if m in spec.active_modes:
            is_walk = m == "walk" or ("walk" in m and "bike" not in m and "cycl" not in m)
            cap = WALK_MAX_MILES if is_walk else BIKE_MAX_MILES
            ok = miles_act <= cap
            if not is_walk:
                ok = ok & (n_bike > 0)
            out[f"avail_{m}"] = ok.astype(float)
        elif spec.is_passenger(m) or spec.is_driver(m):
            out[f"avail_{m}"] = has_car.astype(float)
        elif "rail" in m:
            out[f"avail_{m}"] = (miles_mot >= 1.0).astype(float)
        elif "bus" in m:
            out[f"avail_{m}"] = (miles_mot >= BUS_MIN_MILES).astype(float)
        else:
            out[f"avail_{m}"] = 1.0
    out.attrs["mode_spec"] = spec.to_dict()
    out.attrs["scaler"] = dict(scaler or {})
    return out


def resample_record_modes(
    records: List[Dict],
    theta: np.ndarray,
    scaler: Dict,
    spec: Optional[ModeSpec] = None,
    rng: Optional[np.random.Generator] = None,
    coords_in_metres: bool = True,
) -> None:
    """Rewrite ``mode_category`` in place, keeping diary order within person."""
    spec = _resolve_spec(spec)
    rng = rng or np.random.default_rng()
    by_pid: Dict = {}
    for i, rec in enumerate(records):
        by_pid.setdefault(rec.get("synthetic_person_id"), []).append(i)
    for idxs in by_pid.values():
        idxs.sort(key=lambda i: records[i].get("trip_sequence") or 0)
        prev = None
        for i in idxs:
            rec = records[i]
            net_m = rec.get("trip_distance_network_m")
            rec["mode_category"] = sample_mode_from_person_trip(
                person=rec,
                distance_m=rec.get("trip_distance"),
                purpose=rec.get("DTripPur"),
                start_min=rec.get("OTimin"),
                prev_mode=prev,
                theta=theta,
                scaler=scaler,
                rng=rng,
                coords_in_metres=coords_in_metres,
                apply_driving_circuity=not pd.notna(net_m),
                network_distance_m=net_m if pd.notna(net_m) else None,
                spec=spec,
            )
            prev = rec["mode_category"]


def refine_modes_to_nts_shares(
    records: List[Dict],
    fit: Dict,
    rng: Optional[np.random.Generator] = None,
    coords_in_metres: bool = True,
    n_rounds: int = 1,
    max_cal_persons: int = 8000,
) -> np.ndarray:
    """Shift ASCs on synthetic trips so predicted shares match NTS, then resample.

    Returns the updated theta (also written onto *fit*).
    """
    spec = _resolve_spec(fit.get("mode_spec"), fit=fit)
    scaler = fit.get("scaler") or {}
    theta = np.asarray(fit["theta"], dtype=float).copy()
    target = fit.get("nts_share") or {}
    if not records or not target:
        return theta
    rng = rng or np.random.default_rng()
    pids = list(dict.fromkeys(r.get("synthetic_person_id") for r in records))
    n_cal = min(int(max_cal_persons), len(pids))

    def _share(recs):
        s = pd.Series([r.get("mode_category") for r in recs]).value_counts(normalize=True)
        return {m: float(s.get(m, 0.0)) for m in spec.modes}

    before = _share(records)
    for _ in range(int(n_rounds)):
        idx = rng.choice(len(pids), size=n_cal, replace=False)
        take = {pids[int(i)] for i in idx}
        cal = [r for r in records if r.get("synthetic_person_id") in take]
        frame = synthetic_records_to_mode_frame(
            cal, scaler, spec, coords_in_metres=coords_in_metres,
        )
        if len(frame) == 0:
            break
        theta = calibrate_mode_ascs(theta, frame, spec, target_share=target, n_iter=10, damping=0.8)
        resample_record_modes(
            records, theta, scaler, spec, rng=rng, coords_in_metres=coords_in_metres,
        )
    fit["theta"] = theta
    fit["synth_share_before"] = before
    fit["synth_share_after"] = _share(records)
    return theta

