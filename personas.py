"""Transport personas: classify survey people, profile residential density, assign spatially.

The nine segments follow the DfT "Meet the Personas" pack (``transport-personas-pack.pdf``,
v3 June 2023). The pack is attitudinal, so the NTS cannot reproduce it exactly. People are
instead scored against each segment's published profile (age, sex, mobility difficulty,
household income) plus soft structural cues taken from the segment narratives (children,
cars, urban/rural, employment, living alone). Every number lives in the tables below so
it can be audited and edited.

Workflow::

    model = build_persona_model(cpca_trips_df, region_trips_df)   # classify + density profile
    assign_personas_to_persons(persons, model, density_by_oa, rng, survey_person_cols)
    generate_travel_activities(..., persona_map=model.persona_map(), ...)

People aged under 16 are not segmented in the pack; they inherit the persona of the adults
in their household.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PERSONA_NAMES: Dict[int, str] = {
    1: "Less mobile, car reliant",
    2: "Young urban families",
    3: "Older, less affluent",
    4: "Comfortable empty-nesters",
    5: "Suburban families",
    6: "Heavy car users, frequent flyers",
    7: "Elderly, low income, no car",
    8: "Urban professionals, no car",
    9: "Young, low income, no car",
}
PERSONA_IDS = tuple(PERSONA_NAMES)
CAR_OWNING = (1, 2, 3, 4, 5, 6)
NON_CAR = (7, 8, 9)

# ---------------------------------------------------------------------------
# Published segment profiles (percentages from the persona pack, segments 1-9)
# ---------------------------------------------------------------------------
AGE_BANDS = [(16, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 200)]
SEG_AGE = {
    1: [3, 12, 15, 13, 14, 17, 26],
    2: [12, 37, 30, 14, 7, 0, 0],
    3: [0, 1, 2, 4, 30, 32, 31],
    4: [0, 0, 0, 1, 21, 40, 38],
    5: [1, 11, 30, 37, 19, 2, 0],
    6: [4, 20, 29, 22, 17, 6, 1],
    7: [0, 0, 0, 0, 22, 30, 48],
    8: [5, 20, 24, 19, 17, 11, 4],
    9: [12, 30, 29, 22, 5, 2, 0],
}
SEG_MALE = {1: 52, 2: 46, 3: 53, 4: 47, 5: 48, 6: 52, 7: 42, 8: 47, 9: 55}
# Segment 6 is not legible in the pack; 9's printed "61%" is read as 6% (young urban adults).
SEG_MOBILITY = {1: 93, 2: 3, 3: 0, 4: 0, 5: 6, 6: 2, 7: 52, 8: 9, 9: 6}
# Household finances: under 25k / 25-60k / over 60k (excludes don't know / refused)
SEG_INCOME = {
    1: [45, 32, 14], 2: [36, 38, 8], 3: [57, 29, 2], 4: [39, 39, 7], 5: [19, 38, 30],
    6: [14, 41, 31], 7: [78, 10, 0], 8: [48, 24, 7], 9: [66, 12, 3],
}

# ---------------------------------------------------------------------------
# Soft structural cues from the segment narratives (probabilities, not published stats)
# ---------------------------------------------------------------------------
SEG_CHILDREN = {1: .30, 2: .95, 3: .10, 4: .02, 5: .90, 6: .90, 7: .05, 8: .20, 9: .30}
# household cars given ownership: P(exactly one car); segment 6 "2-3 cars", segment 2 "one car"
SEG_ONE_CAR = {1: .60, 2: .90, 3: .70, 4: .55, 5: .35, 6: .10}
SEG_URBAN = {1: .60, 2: .90, 3: .60, 4: .40, 5: .70, 6: .50, 7: .75, 8: .95, 9: .90}
# employed / retired-or-inactive / student-or-other
SEG_ECON = {
    1: [.35, .55, .10], 2: [.75, .12, .13], 3: [.80, .15, .05], 4: [.20, .78, .02],
    5: [.90, .07, .03], 6: [.93, .02, .05], 7: [.03, .95, .02], 8: [.80, .05, .15],
    9: [.45, .10, .45],
}
SEG_ALONE = {1: .15, 2: .03, 3: .15, 4: .10, 5: .02, 6: .02, 7: .60, 8: .30, 9: .20}
# Probability of mismatching the car-ownership split (segments 1-6 own a car, 7-9 do not)
CAR_SPLIT_LEAK = 1e-6
PROB_FLOOR = 0.005
# Mid-points of the pack's approximate size bands (5% or less -> 3, 6-10% -> 8, 11-20% -> 15.5)
SEG_SIZE_MID = {1: 15.5, 2: 15.5, 3: 8.0, 4: 15.5, 5: 15.5, 6: 15.5, 7: 3.0, 8: 8.0, 9: 3.0}

COL_ID = "IndividualID"
COL_HH = "HouseholdID"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _norm_rows(table: Mapping[int, Sequence[float]]) -> np.ndarray:
    arr = np.array([table[s] for s in PERSONA_IDS], dtype=float)
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    return np.maximum(arr, PROB_FLOOR)


def person_table_from_trips(trips: pd.DataFrame, id_col: str = COL_ID) -> pd.DataFrame:
    """One row per surveyed person from trip-level rows."""
    return trips.drop_duplicates(id_col).reset_index(drop=True).copy()


def classify_personas(
    persons: pd.DataFrame,
    priors: Optional[Mapping[int, float]] = None,
    return_scores: bool = False,
):
    """Assign each person aged 16+ to the most likely persona (1-9); others get 0.

    Naive-Bayes scoring on published segment profiles; the car-owner / non-car split is
    near-deterministic. Unknown attributes (NTS codes below zero) contribute nothing.
    """
    n = len(persons)
    age = _num(persons, "Age").to_numpy()
    male = (_num(persons, "Sex_B01ID").to_numpy() == 1)
    sex_known = np.isin(_num(persons, "Sex_B01ID").to_numpy(), (1, 2))

    mob_raw = _num(persons, "MobDiffSum_B01ID").to_numpy()
    lim_raw = _num(persons, "LimitAct_B01ID").to_numpy()
    mob_known = (mob_raw >= 1) | (lim_raw >= 1)
    mobile_difficulty = (mob_raw >= 2) | (lim_raw == 1)

    inc_raw = _num(persons, "HHIncQDSEngTS_B01ID").to_numpy()
    inc_cls = np.where(inc_raw <= 2, 0, np.where(inc_raw <= 4, 1, 2))
    inc_known = np.isin(inc_raw, (1, 2, 3, 4, 5))

    cars = _num(persons, "NumCarVan")
    if cars.isna().all():
        cars = _num(persons, "NumVehicles")
    cars = cars.to_numpy()
    cars_known = np.isfinite(cars) & (cars >= 0)
    owns_car = cars > 0
    one_car = cars == 1

    kids_raw = _num(persons, "HHoldNumChildren").to_numpy()
    kids_known = np.isfinite(kids_raw) & (kids_raw >= 0)
    kids = kids_raw > 0

    urban_raw = _num(persons, "Settlement2011EW_B03ID").to_numpy()
    urban_known = np.isin(urban_raw, (1, 2))
    urban = urban_raw == 1

    eco_raw = _num(persons, "EcoStat_B03ID").to_numpy()
    eco_known = np.isin(eco_raw, (1, 2, 3, 4))
    eco_cls = np.where(eco_raw <= 2, 0, np.where(eco_raw == 3, 1, 2))

    hh_size = _num(persons, "HHoldNumPeople").to_numpy()
    alone_known = np.isfinite(hh_size) & (hh_size >= 1)
    alone = hh_size == 1

    age_band = np.full(n, -1)
    for b, (lo, hi) in enumerate(AGE_BANDS):
        age_band[(age >= lo) & (age <= hi)] = b

    p_age = _norm_rows(SEG_AGE)
    p_inc = _norm_rows(SEG_INCOME)
    p_econ = _norm_rows(SEG_ECON)
    male_p = np.array([SEG_MALE[s] / 100 for s in PERSONA_IDS])
    mob_p = np.maximum(np.array([SEG_MOBILITY[s] / 100 for s in PERSONA_IDS]), PROB_FLOOR)
    mob_p = np.minimum(mob_p, 1 - PROB_FLOOR)

    def _bern(p_seg: Mapping[int, float], flag, known):
        p = np.array([p_seg.get(s, np.nan) for s in PERSONA_IDS])
        out = np.where(flag[:, None], p[None, :], 1 - p[None, :])
        out = np.where(np.isnan(out), 1.0, out)
        return np.where(known[:, None], np.log(np.maximum(out, PROB_FLOOR)), 0.0)

    score = np.zeros((n, len(PERSONA_IDS)))
    ab = np.where(age_band >= 0, age_band, 0)
    score += np.where((age_band >= 0)[:, None], np.log(p_age[:, ab].T), 0.0)
    score += np.where(
        sex_known[:, None],
        np.log(np.where(male[:, None], male_p[None, :], 1 - male_p[None, :])),
        0.0,
    )
    score += np.where(
        mob_known[:, None],
        np.log(np.where(mobile_difficulty[:, None], mob_p[None, :], 1 - mob_p[None, :])),
        0.0,
    )
    ic = np.where(inc_known, inc_cls, 0)
    score += np.where(inc_known[:, None], np.log(p_inc[:, ic].T), 0.0)
    ec = np.where(eco_known, eco_cls, 0)
    score += np.where(eco_known[:, None], np.log(p_econ[:, ec].T), 0.0)
    score += _bern(SEG_CHILDREN, kids, kids_known)
    score += _bern(SEG_URBAN, urban, urban_known)
    score += _bern(SEG_ALONE, alone, alone_known)

    owner_seg = np.array([s in CAR_OWNING for s in PERSONA_IDS])
    p_owner_match = np.where(
        owns_car[:, None], np.where(owner_seg[None, :], 1 - CAR_SPLIT_LEAK, CAR_SPLIT_LEAK),
        np.where(owner_seg[None, :], CAR_SPLIT_LEAK, 1 - CAR_SPLIT_LEAK),
    )
    score += np.where(cars_known[:, None], np.log(p_owner_match), 0.0)
    one_p = np.array([SEG_ONE_CAR.get(s, np.nan) for s in PERSONA_IDS])
    one_term = np.where(one_car[:, None], one_p[None, :], 1 - one_p[None, :])
    score += np.where(
        (cars_known & owns_car)[:, None] & ~np.isnan(one_p)[None, :],
        np.log(np.maximum(np.nan_to_num(one_term, nan=1.0), PROB_FLOOR)),
        0.0,
    )
    if priors:
        score += np.log(np.array([max(float(priors.get(s, 1.0)), 1e-9) for s in PERSONA_IDS]))[None, :]

    persona = np.array(PERSONA_IDS)[np.argmax(score, axis=1)]
    persona = np.where((age >= 16) & np.isfinite(age), persona, 0)
    out = pd.Series(persona.astype(int), index=persons.index, name="persona")
    return (out, score) if return_scores else out


def calibrate_priors(
    score: np.ndarray,
    adult: np.ndarray,
    weights: np.ndarray,
    target_sizes: Mapping[int, float] = SEG_SIZE_MID,
    n_iter: int = 60,
    step: float = 0.7,
) -> Dict[int, float]:
    """Find segment priors so predicted adult shares follow the pack's published size bands.

    The car-owner / non-car split is fixed by the data, so targets are rescaled within each group.
    """
    ids = np.array(PERSONA_IDS)
    owner_seg = np.isin(ids, CAR_OWNING)
    log_prior = np.zeros(len(ids))
    w = weights[adult]
    sc = score[adult]
    for _ in range(n_iter):
        pred = np.bincount(np.argmax(sc + log_prior, axis=1), weights=w, minlength=len(ids))
        pred = pred / pred.sum()
        target = np.array([target_sizes[s] for s in PERSONA_IDS], dtype=float)
        for grp in (owner_seg, ~owner_seg):
            target[grp] = target[grp] / target[grp].sum() * pred[grp].sum()
        log_prior += step * (np.log(np.maximum(target, 1e-6)) - np.log(np.maximum(pred, 1e-6)))
        log_prior -= log_prior.mean()
    return {s: float(np.exp(lp)) for s, lp in zip(PERSONA_IDS, log_prior)}


def inherit_household_persona(
    persons: pd.DataFrame, persona: pd.Series, hh_col: str = COL_HH
) -> pd.Series:
    """Give persona-0 people (children) the most common persona among adults in their household."""
    out = persona.copy()
    if hh_col not in persons.columns:
        return out
    adults = pd.DataFrame({"hh": persons[hh_col], "p": persona})
    adults = adults[adults["p"] > 0]
    if adults.empty:
        return out
    mode = adults.groupby("hh")["p"].agg(lambda s: s.value_counts().sort_index().idxmax())
    child = persona == 0
    out.loc[child] = persons.loc[child, hh_col].map(mode).fillna(0).astype(int)
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, v)


@dataclass
class PersonaModel:
    """Classified survey people, persona density profiles and the density→persona posterior."""

    person_table: pd.DataFrame
    shares: pd.Series
    source: Dict[int, str]
    bin_edges: np.ndarray
    posterior: np.ndarray                      # (n_bins, n_personas), rows sum to 1
    density_profile: pd.DataFrame
    density_quantiles: np.ndarray              # NTS (CPCA) density at percentiles 0..1
    id_col: str = COL_ID
    cpca_ids: set = field(default_factory=set)
    priors: Dict[int, float] = field(default_factory=dict)

    def persona_map(self, adults_only: bool = True) -> Dict:
        """{survey person id: persona}; with ``adults_only`` under-16s map to 0 (no diary pool)."""
        t = self.person_table
        pers = t["persona"].astype(int)
        if adults_only and "Age" in t.columns:
            pers = pers.where(_num(t, "Age") >= 16, 0)
        return dict(zip(t[self.id_col], pers))

    def bins_for_density(self, density: np.ndarray, match: str = "quantile") -> np.ndarray:
        """Bin index per home density; ``quantile`` matches ranks, ``absolute`` uses raw values."""
        d = np.asarray(density, dtype=float)
        ok = np.isfinite(d)
        eq = d.copy()
        if match == "quantile" and ok.any():
            n = ok.sum()
            ranks = pd.Series(d[ok]).rank(method="average").to_numpy()
            pct = (ranks - 0.5) / n
            grid = np.linspace(0, 1, len(self.density_quantiles))
            eq[ok] = np.interp(pct, grid, self.density_quantiles)
        b = np.searchsorted(self.bin_edges, eq, side="right") - 1
        b = np.clip(b, 0, self.posterior.shape[0] - 1)
        return np.where(ok, b, -1)

    def persona_probabilities(self, density: np.ndarray, match: str = "quantile") -> np.ndarray:
        """P(persona | home density) per entry; unknown density falls back to the overall mix."""
        b = self.bins_for_density(density, match)
        base = self.shares.reindex(PERSONA_IDS).fillna(0).to_numpy()
        base = base / base.sum()
        return np.where((b >= 0)[:, None], self.posterior[np.maximum(b, 0)], base[None, :])

    def pool_for(self, persona: int) -> pd.DataFrame:
        t = self.person_table
        sub = t[t["persona"] == persona]
        if self.source.get(persona) == "cpca":
            sub = sub[sub[self.id_col].isin(self.cpca_ids)]
        return sub


def build_persona_model(
    cpca_trips: pd.DataFrame,
    region_trips: Optional[pd.DataFrame] = None,
    min_cpca_persons: int = 150,
    n_bins: int = 10,
    density_col: str = "HHoldPopDensity",
    id_col: str = COL_ID,
    weight_col: str = "W2",
    smoothing: float = 1.0,
    priors: Optional[Mapping[int, float]] = None,
    calibrate: bool = True,
) -> PersonaModel:
    """Classify survey people and learn persona residential-density distributions.

    Persona shares come from the CPCA sample. A persona's density distribution (and later its
    diaries and demographics) comes from CPCA when at least ``min_cpca_persons`` of that persona
    are present; otherwise from the wider ``region_trips`` (e.g. East of England). With
    ``calibrate`` the segment priors are tuned so adult shares follow the pack's size bands.
    """
    cpca_ids = set(cpca_trips[id_col].unique())
    base = cpca_trips if region_trips is None else region_trips
    if region_trips is not None:
        extra = cpca_trips[~cpca_trips[id_col].isin(set(region_trips[id_col].unique()))]
        base = pd.concat([region_trips, extra], ignore_index=True) if len(extra) else region_trips
    people = person_table_from_trips(base, id_col)

    if priors is None and calibrate:
        _, score = classify_personas(people, return_scores=True)
        w_all = _num(people, weight_col).fillna(1.0).clip(lower=0).to_numpy()
        adult = _num(people, "Age").to_numpy() >= 16
        priors = calibrate_priors(score, adult, w_all)
    raw = classify_personas(people, priors=priors)
    people["persona"] = inherit_household_persona(people, raw).to_numpy()
    people["is_cpca"] = people[id_col].isin(cpca_ids)
    people["_w"] = _num(people, weight_col).fillna(1.0).clip(lower=0).to_numpy()
    people["_dens"] = _num(people, density_col).clip(lower=0).to_numpy()

    cp = people[people["is_cpca"] & (people["persona"] > 0)]
    counts = cp.groupby("persona").size().reindex(PERSONA_IDS).fillna(0).astype(int)
    source = {p: ("cpca" if counts[p] >= min_cpca_persons else "region") for p in PERSONA_IDS}
    if region_trips is None:
        source = {p: "cpca" for p in PERSONA_IDS}

    shares = cp.groupby("persona")["_w"].sum().reindex(PERSONA_IDS).fillna(0)
    shares = shares / shares.sum()

    ref = cp[np.isfinite(cp["_dens"])]
    qgrid = np.linspace(0, 1, 201)
    dens_q = _weighted_quantile(ref["_dens"].to_numpy(), ref["_w"].to_numpy(), qgrid)
    edges = np.unique(_weighted_quantile(
        ref["_dens"].to_numpy(), ref["_w"].to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1]
    ))
    edges = np.concatenate([[-np.inf], edges])
    nb = len(edges)

    f = np.zeros((nb, len(PERSONA_IDS)))
    prof_rows = []
    for j, p in enumerate(PERSONA_IDS):
        pool = people[(people["persona"] == p) & np.isfinite(people["_dens"])]
        if source[p] == "cpca":
            pool = pool[pool["is_cpca"]]
        b = np.clip(np.searchsorted(edges, pool["_dens"].to_numpy(), side="right") - 1, 0, nb - 1)
        w = np.bincount(b, weights=pool["_w"].to_numpy(), minlength=nb)
        f[:, j] = (w + smoothing) / (w.sum() + smoothing * nb)
        d, ww = pool["_dens"].to_numpy(), pool["_w"].to_numpy()
        q = _weighted_quantile(d, ww, np.array([.25, .5, .75])) if len(d) else [np.nan] * 3
        prof_rows.append({
            "persona": p, "name": PERSONA_NAMES[p], "n_cpca": int(counts[p]),
            "n_pool": int(len(pool)), "source": source[p], "share_cpca": float(shares[p]),
            "density_p25": q[0], "density_median": q[1], "density_p75": q[2],
            "density_mean": float(np.average(d, weights=ww)) if len(d) else np.nan,
            **{f"bin{k}": f[k, j] for k in range(nb)},
        })
    post = f * shares.reindex(PERSONA_IDS).to_numpy()[None, :]
    post = post / post.sum(axis=1, keepdims=True)

    return PersonaModel(
        person_table=people.drop(columns=["_w", "_dens"]),
        shares=shares,
        source=source,
        bin_edges=edges,
        posterior=post,
        density_profile=pd.DataFrame(prof_rows).set_index("persona"),
        density_quantiles=dens_q,
        id_col=id_col,
        cpca_ids=cpca_ids,
        priors=dict(priors or {}),
    )


def density_by_oa_from_cells(
    cells: pd.DataFrame,
    oa_col: str = "OA11CD",
    pop_col: str = "pop",
    cell_area_ha: float = 100.0,
    oa_xy: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Mean residential density (persons/ha) of the populated 1 km raster cells in each OA.

    OAs with no raster-cell centre inside them (small OAs) take the density of the nearest
    populated cell to their centroid when ``oa_xy`` ({OA: (x, y)} in the cells' CRS) is given.
    """
    oa = cells[oa_col].astype(str)
    dens = pd.to_numeric(cells[pop_col], errors="coerce") / cell_area_ha
    out = dens[oa.str.len() > 0].groupby(oa[oa.str.len() > 0]).mean().to_dict()
    if oa_xy:
        ok = dens.notna() & (dens > 0)
        cxy = cells.loc[ok, ["x", "y"]].to_numpy(dtype=float)
        vals = dens[ok].to_numpy()
        missing = [k for k in oa_xy if str(k) not in out]
        for k in missing:
            x, y = oa_xy[k]
            out[str(k)] = float(vals[int(np.argmin((cxy[:, 0] - x) ** 2 + (cxy[:, 1] - y) ** 2))])
    return out


def assign_personas_to_persons(
    persons: List[Dict],
    model: PersonaModel,
    density_by_oa: Mapping[str, float],
    rng: np.random.Generator,
    survey_person_cols: Optional[Mapping[str, str]] = None,
    home_key: str = "home_geoid",
    match: str = "quantile",
    resample_demographics: bool = True,
) -> Dict:
    """Draw each synthetic person's persona from P(persona | home density), in place.

    With ``resample_demographics`` the survey-derived attributes in ``survey_person_cols`` are
    redrawn from surveyed people of the same persona (CPCA, or the wider region when the CPCA
    sample is small), so age, income, cars and children are consistent with the persona.
    """
    n = len(persons)
    dens = np.array([density_by_oa.get(str(p.get(home_key) or ""), np.nan) for p in persons], dtype=float)
    probs = model.persona_probabilities(dens, match=match)
    cum = probs.cumsum(axis=1)
    u = rng.random(n)[:, None]
    idx = np.minimum((u > cum).sum(axis=1), len(PERSONA_IDS) - 1)
    pids = np.array(PERSONA_IDS)[idx]
    bins = model.bins_for_density(dens, match)

    pools = {}
    if resample_demographics and survey_person_cols:
        for p in PERSONA_IDS:
            pool = model.pool_for(p)
            pool = pool[(pool["persona"] == p) & (_num(pool, "Age") >= 16)]
            if len(pool):
                w = _num(pool, "W2").fillna(1.0).clip(lower=0).to_numpy()
                w = w / w.sum() if w.sum() > 0 else np.full(len(pool), 1 / len(pool))
                pools[p] = (pool, w)
    cols = {o: s for o, s in (survey_person_cols or {}).items()}
    for p in PERSONA_IDS:
        members = np.flatnonzero(pids == p)
        if not len(members) or p not in pools:
            continue
        pool, w = pools[p]
        pick = rng.choice(len(pool), size=len(members), p=w)
        chosen = pool.iloc[pick]
        for out_col, src in cols.items():
            if src in chosen.columns:
                vals = chosen[src].to_numpy()
                for m, v in zip(members, vals):
                    persons[m][out_col] = v
    for i, person in enumerate(persons):
        person["persona"] = int(pids[i])
        person["persona_name"] = PERSONA_NAMES[int(pids[i])]
        person["home_density_ha"] = float(dens[i]) if np.isfinite(dens[i]) else np.nan
        person["density_bin"] = int(bins[i])
    counts = pd.Series(pids).value_counts(normalize=True).reindex(PERSONA_IDS).fillna(0)
    return {"n": n, "share": counts, "n_density_missing": int((bins < 0).sum())}


def compare_persona_shares(persons: pd.DataFrame, model: PersonaModel) -> pd.DataFrame:
    """Synthetic vs surveyed persona shares (survey = CPCA sample)."""
    syn = persons["persona"].value_counts(normalize=True).reindex(PERSONA_IDS).fillna(0)
    out = pd.DataFrame({"name": pd.Series(PERSONA_NAMES), "survey_cpca": model.shares, "synthetic": syn})
    out["diff"] = out["synthetic"] - out["survey_cpca"]
    return out


def persona_density_summary(persons: pd.DataFrame, density_col: str = "home_density_ha") -> pd.DataFrame:
    """Synthetic persona × residential-density summary (median, quartiles, mean)."""
    g = persons.groupby("persona")[density_col]
    out = g.describe(percentiles=[.25, .5, .75])[["count", "mean", "25%", "50%", "75%"]]
    out.insert(0, "name", [PERSONA_NAMES.get(int(i), "") for i in out.index])
    return out


def extract_region_survey(
    src_csv: str,
    dst_csv: str,
    gor_col: str = "HHoldGOR_B02ID",
    gor_values: Iterable = (6,),
    chunksize: int = 200_000,
) -> int:
    """Stream a national NTS trip CSV and keep only rows for the given government-office region(s).

    East of England is code 6 in ``HHoldGOR_B02ID``. Returns the number of rows written.
    """
    keep = set(gor_values)
    first, n = True, 0
    for chunk in pd.read_csv(src_csv, chunksize=chunksize, low_memory=False):
        sub = chunk[pd.to_numeric(chunk[gor_col], errors="coerce").isin(keep)]
        if len(sub):
            sub.to_csv(dst_csv, mode="w" if first else "a", header=first, index=False)
            first = False
            n += len(sub)
    return n
