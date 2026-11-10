"""Decision-tree mode choice on NTS socio-demographics, distance, and purpose.

Splits use age group, sex, personal and household income, adults, children,
cars, bikes, trip distance, and purpose. Previous mode is included so a
person-day stays coherent (same role as in the nested logit). Modes are
drawn from leaf class probabilities, not argmax, so predicted shares stay
close to NTS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .mode_spec import DEFAULT_MODE_SPEC, ModeSpec
from .nested_mode_choice import (
    BIKE_MAX_MILES,
    BUS_MIN_MILES,
    WALK_CIRCUITY,
    WALK_MAX_MILES,
    _household_vehicle_count,
    _person_float,
    _person_vehicle_count,
    _resolve_spec,
    add_previous_mode,
    recode_nts_trips,
)

TREE_VERSION = 1
NUMERIC_FEATURES = [
    "age_group", "sex", "p_income", "hh_income",
    "n_adults", "n_children", "n_car", "n_bike", "distance_mi",
]
PURPOSE_LEVELS = ["work", "education", "shop", "personal", "leisure", "home", "other"]
# NTS Age_B01ID bands (used as age_group). Fallback when Age_B01ID is missing.
AGE_BINS = [0, 1, 3, 5, 11, 16, 17, 18, 19, 20, 21, 26, 30, 40, 50, 60, 65, 70, 75, 80, 85, 200]
AGE_CODES = list(range(1, 22))
_MISSING = {-10, -9, -8, -1, -10.0, -9.0, -8.0, -1.0}


def _tree_cache_path(logit_cache: Optional[Union[str, Path]]) -> Optional[Path]:
    if not logit_cache:
        return None
    p = Path(logit_cache)
    return p.with_name("nts_tree_mode_choice.joblib")


def _missing_to_nan(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where(~x.isin(_MISSING))


def _age_group_from_age(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BINS, labels=AGE_CODES, right=False).astype(float)


def _age_group_value(age: float, age_b: float) -> float:
    if np.isfinite(age_b) and age_b not in _MISSING and age_b > 0:
        return float(age_b)
    if np.isfinite(age) and age >= 0:
        for hi, code in zip(AGE_BINS[1:], AGE_CODES):
            if age < hi:
                return float(code)
        return float(AGE_CODES[-1])
    return np.nan


def prepare_tree_frame(
    df: pd.DataFrame,
    spec: Optional[ModeSpec] = None,
) -> pd.DataFrame:
    spec = _resolve_spec(spec, frame=df)
    x = recode_nts_trips(df, spec=spec)
    ag = _missing_to_nan(x["Age_B01ID"]) if "Age_B01ID" in x.columns else pd.Series(np.nan, index=x.index)
    age = _missing_to_nan(x["Age"]) if "Age" in x.columns else pd.Series(np.nan, index=x.index)
    x["age_group"] = ag.fillna(_age_group_from_age(age))
    x["sex"] = _missing_to_nan(x["Sex_B01ID"]) if "Sex_B01ID" in x.columns else x.get("female")
    if "IndIncome2002_B02ID" in x.columns:
        x["p_income"] = _missing_to_nan(x["IndIncome2002_B02ID"])
    else:
        x["p_income"] = _missing_to_nan(x["income"]) if "income" in x.columns else np.nan
    x["hh_income"] = _missing_to_nan(x["HHIncome2002_B02ID"]) if "HHIncome2002_B02ID" in x.columns else np.nan
    x["n_adults"] = _missing_to_nan(x["HHoldNumAdults"]) if "HHoldNumAdults" in x.columns else np.nan
    x["n_children"] = _missing_to_nan(x["HHoldNumChildren"]) if "HHoldNumChildren" in x.columns else np.nan
    x["n_car"] = _household_vehicle_count(x)
    if "NumBike" in x.columns:
        x["n_bike"] = _missing_to_nan(x["NumBike"]).fillna(0).clip(lower=0)
    elif "n_bike" not in x.columns:
        x["n_bike"] = 0.0
    x = add_previous_mode(x, spec=spec)
    x["prev_label"] = x["prev_mode"].fillna("none").astype(str).str.strip().str.lower()
    x["prev_label"] = x["prev_label"].where(x["prev_label"].isin(spec.modes), "none")
    x = x.loc[x["mode"].isin(spec.modes)].copy()
    x = x.loc[x["distance_mi"].between(0.05, 80.0)].copy()
    x.attrs["mode_spec"] = spec.to_dict()
    return x.reset_index(drop=True)


def _fill_medians(frame: pd.DataFrame) -> Dict[str, float]:
    med = {}
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(frame[col], errors="coerce")
        med[col] = float(s.median()) if s.notna().any() else 0.0
    return med


def design_matrix(
    frame: pd.DataFrame,
    spec: ModeSpec,
    medians: Optional[Dict[str, float]] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    medians = dict(medians or _fill_medians(frame))
    parts = []
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(frame[col], errors="coerce").fillna(medians.get(col, 0.0))
        parts.append(s.rename(col))
    purpose = frame["purpose"].astype(str).str.strip().str.lower() if "purpose" in frame.columns else "other"
    for p in PURPOSE_LEVELS:
        parts.append((purpose == p).astype(float).rename(f"purp_{p}"))
    prev = frame["prev_label"] if "prev_label" in frame.columns else pd.Series("none", index=frame.index)
    prev_levels = ["none"] + list(spec.modes)
    for lab in prev_levels:
        parts.append((prev == lab).astype(float).rename(f"prev_{lab}"))
    X = pd.concat(parts, axis=1)
    names = list(feature_names or X.columns)
    for name in names:
        if name not in X.columns:
            X[name] = 0.0
    X = X[names]
    return X, names, medians


def _person_split(frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    pid = None
    for col in ("IndividualID", "person_id", "HHPERSONID"):
        if col in frame.columns:
            pid = frame[col].astype(str)
            break
    if pid is None:
        n = len(frame)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(n * (1.0 - test_size))
        tr = np.zeros(n, dtype=bool)
        tr[perm[:cut]] = True
        return tr, ~tr
    uniq = pid.drop_duplicates()
    rng = np.random.default_rng(seed)
    u = uniq.to_numpy()
    rng.shuffle(u)
    cut = int(len(u) * (1.0 - test_size))
    train_ids = set(u[:cut])
    tr = pid.isin(train_ids).to_numpy()
    return tr, ~tr


def fit_tree_mode_choice(
    df: pd.DataFrame,
    cache_path: Optional[Union[str, Path]] = None,
    spec: Optional[ModeSpec] = None,
    mode_mapping: Optional[Dict] = None,
    mode_nests: Optional[Dict] = None,
    max_depth: int = 8,
    min_samples_leaf: int = 80,
) -> Optional[Dict]:
    """Fit a CART classifier. Returns None if scikit-learn is unavailable."""
    try:
        from sklearn.tree import DecisionTreeClassifier
        import joblib
    except ImportError:
        return None

    if spec is None:
        spec = ModeSpec.from_mapping(mode_mapping, nests=mode_nests) if (mode_mapping or mode_nests) else DEFAULT_MODE_SPEC

    tree_path = _tree_cache_path(cache_path)
    if tree_path and tree_path.exists():
        try:
            payload = joblib.load(tree_path)
            if int(payload.get("tree_version") or 0) == TREE_VERSION:
                cached_spec = ModeSpec.from_dict(payload.get("mode_spec"), mapping=spec.mapping)
                if cached_spec.modes == spec.modes:
                    payload["from_cache"] = True
                    payload["mode_spec"] = cached_spec.to_dict()
                    payload["kind"] = "tree"
                    return payload
        except Exception:
            pass

    frame = prepare_tree_frame(df, spec=spec)
    if len(frame) < 200:
        return None
    X, names, medians = design_matrix(frame, spec)
    y = frame["mode"].astype(str).to_numpy()
    w = pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0).clip(lower=0).to_numpy()
    tr, te = _person_split(frame)
    clf_cv = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
    )
    clf_cv.fit(X.loc[tr], y[tr], sample_weight=w[tr])
    acc_tr = float(clf_cv.score(X.loc[tr], y[tr], sample_weight=w[tr]))
    acc_te = float(clf_cv.score(X.loc[te], y[te], sample_weight=w[te])) if te.any() else acc_tr

    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
    )
    clf.fit(X, y, sample_weight=w)
    classes = [str(c) for c in clf.classes_]
    p = clf.predict_proba(X)
    pred_share = {
        m: float((p[:, classes.index(m)] * w).sum() / max(float(w.sum()), 1e-12))
        if m in classes else 0.0
        for m in spec.modes
    }
    nts_share = {
        m: float(np.average(y == m, weights=w)) for m in spec.modes
    }
    importances = {
        names[i]: float(clf.feature_importances_[i])
        for i in np.argsort(clf.feature_importances_)[::-1]
        if clf.feature_importances_[i] > 1e-6
    }
    payload = {
        "kind": "tree",
        "tree_version": TREE_VERSION,
        "clf": clf,
        "feature_names": names,
        "classes": classes,
        "medians": medians,
        "mode_spec": spec.to_dict(),
        "accuracy": acc_te,
        "accuracy_train": acc_tr,
        "accuracy_test": acc_te,
        "nobs": int(len(frame)),
        "nts_share": nts_share,
        "pred_share": pred_share,
        "importances": importances,
        "success": True,
        "from_cache": False,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }
    if tree_path:
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, tree_path)
    return payload


def load_tree_mode_choice(
    cache_path: Optional[Union[str, Path]],
    spec: Optional[ModeSpec] = None,
) -> Optional[Dict]:
    try:
        import joblib
    except ImportError:
        return None
    tree_path = _tree_cache_path(cache_path)
    if not tree_path or not tree_path.exists():
        return None
    try:
        payload = joblib.load(tree_path)
    except Exception:
        return None
    if int(payload.get("tree_version") or 0) != TREE_VERSION:
        return None
    spec = spec or DEFAULT_MODE_SPEC
    cached_spec = ModeSpec.from_dict(payload.get("mode_spec"), mapping=spec.mapping)
    if cached_spec.modes != spec.modes:
        return None
    payload["from_cache"] = True
    payload["kind"] = "tree"
    return payload


def _feature_row(
    person: Dict,
    miles: float,
    purpose: str,
    prev_mode: Optional[str],
    fit: Dict,
    spec: ModeSpec,
) -> pd.DataFrame:
    age = _person_float(person, ("Age", "age"), default=np.nan)
    age_b = _person_float(person, ("Age_B", "Age_B01ID", "age_group"), default=np.nan)
    sex = _person_float(person, ("Sex_B01ID", "Sex", "sex"), default=np.nan)
    p_inc = _person_float(
        person, ("IndIncomeRng", "IndIncome2002_B02ID", "p_income", "income"), default=np.nan,
    )
    hh_inc = _person_float(person, ("HHIncome2002_B02ID", "hh_income"), default=np.nan)
    n_ad = _person_float(person, ("HHoldNumAdults", "n_adults"), default=np.nan)
    n_ch = _person_float(person, ("HHoldNumChildren", "n_children"), default=np.nan)
    hh = _person_float(person, ("HHoldNumPeople", "hh_size"), default=np.nan)
    if not np.isfinite(n_ad) or not np.isfinite(n_ch):
        if np.isfinite(hh) and hh > 0:
            child = 1.0 if (np.isfinite(age) and age < 17) else 0.0
            n_ch = n_ch if np.isfinite(n_ch) else child
            n_ad = n_ad if np.isfinite(n_ad) else max(hh - (n_ch if np.isfinite(n_ch) else 0.0), 1.0)
        else:
            n_ad = 1.0 if not np.isfinite(n_ad) else n_ad
            n_ch = 0.0 if not np.isfinite(n_ch) else n_ch
    n_car = _person_vehicle_count(person)
    n_bike = max(_person_float(person, ("NumBike", "n_bike"), default=0.0), 0.0)
    prev = str(prev_mode).strip().lower() if prev_mode else "none"
    if prev not in spec.modes:
        prev = "none"
    row = {
        "age_group": [_age_group_value(age, age_b)],
        "sex": [sex],
        "p_income": [p_inc],
        "hh_income": [hh_inc],
        "n_adults": [n_ad],
        "n_children": [n_ch],
        "n_car": [n_car],
        "n_bike": [n_bike],
        "distance_mi": [miles],
        "purpose": [str(purpose or "other").strip().lower()],
        "prev_label": [prev],
    }
    frame = pd.DataFrame(row)
    X, _, _ = design_matrix(
        frame, spec, medians=fit.get("medians") or {}, feature_names=fit.get("feature_names"),
    )
    return X


def _availability(
    spec: ModeSpec,
    miles_motor: float,
    miles_active: float,
    n_car: float,
    n_bike: float,
) -> np.ndarray:
    has_car = n_car > 0
    avail = []
    for m in spec.modes:
        if m in spec.active_modes:
            is_walk = m == "walk" or ("walk" in m and "bike" not in m and "cycl" not in m)
            cap = WALK_MAX_MILES if is_walk else BIKE_MAX_MILES
            a = 1.0 if miles_active <= cap else 0.0
            if (not is_walk) and n_bike <= 0:
                a = 0.0
            avail.append(a)
        elif spec.is_passenger(m) or spec.is_driver(m):
            avail.append(1.0 if has_car else 0.0)
        elif "rail" in m:
            avail.append(1.0 if miles_motor >= 1.0 else 0.0)
        elif "bus" in m:
            avail.append(1.0 if miles_motor >= BUS_MIN_MILES else 0.0)
        else:
            avail.append(1.0)
    a = np.asarray(avail, dtype=float)
    if a.sum() <= 0:
        a[:] = 1.0
    return a


def sample_mode_from_tree(
    person: Dict,
    distance_m: float,
    purpose: Optional[str],
    prev_mode: Optional[str],
    fit: Dict,
    rng: Optional[np.random.Generator] = None,
    coords_in_metres: bool = True,
    apply_driving_circuity: bool = True,
    network_distance_m: Optional[float] = None,
    spec: Optional[ModeSpec] = None,
) -> str:
    """Draw a mode from the tree's leaf probabilities, with availability."""
    spec = _resolve_spec(spec or fit.get("mode_spec"), fit=fit)
    rng = rng or np.random.default_rng()
    geo_m = float(distance_m) if distance_m is not None and np.isfinite(distance_m) else np.nan
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
    n_car = _person_vehicle_count(person)
    n_bike = max(_person_float(person, ("NumBike", "n_bike"), default=0.0), 0.0)
    if miles_motor < BUS_MIN_MILES and prev_mode and str(prev_mode).strip().lower() in spec.public_modes:
        prev_mode = None

    X = _feature_row(person, miles_motor, str(purpose or "other"), prev_mode, fit, spec)
    clf = fit["clf"]
    proba = np.asarray(clf.predict_proba(X)[0], dtype=float)
    classes = [str(c) for c in (fit.get("classes") or clf.classes_)]
    p_by_mode = {m: 0.0 for m in spec.modes}
    for i, m in enumerate(classes):
        if m in p_by_mode:
            p_by_mode[m] = float(proba[i]) if i < len(proba) else 0.0
    avail = _availability(spec, miles_motor, miles_active, n_car, n_bike)
    p = np.array([p_by_mode[m] for m in spec.modes], dtype=float) * avail
    s = p.sum()
    if s <= 0:
        p = avail / max(avail.sum(), 1.0)
    else:
        p = p / s
    idx = int(rng.choice(len(spec.modes), p=p))
    return spec.modes[idx]
