"""Evaluate synthetic activity chains and origin–destination distance decay.

NTS comparison uses trip weight ``W5``. Synthetic Euclidean metres between
``OLON_RANDOM``/``OLAT_RANDOM`` and ``DLON_RANDOM``/``DLAT_RANDOM`` (BNG) are
converted to driving-network miles with the MSOA circuity curve so they match
NTS ``TripDisIncSW``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .nested_mode_choice import (
    ACTIVITY_CODE_MAP,
    _exp_curve,
    fit_decay_functions,
    recode_nts_trips,
)

PURPOSE_ORDER = ["work", "education", "shop", "personal", "leisure", "home", "other"]
METERS_PER_MILE = 1609.34
KNOWN_PURPOSES = set(PURPOSE_ORDER)

NTS_BLUE = "#4e79a7"
SYN_ORANGE = "#f28e2b"


def map_purpose(val) -> str:
    if val is None:
        return "other"
    s = str(val).strip().lower()
    if s in KNOWN_PURPOSES:
        return s
    if val in ACTIVITY_CODE_MAP:
        return ACTIVITY_CODE_MAP[val]
    try:
        f = float(val)
        if f in ACTIVITY_CODE_MAP:
            return ACTIVITY_CODE_MAP[f]
        if float(f).is_integer() and int(f) in ACTIVITY_CODE_MAP:
            return ACTIVITY_CODE_MAP[int(f)]
    except (TypeError, ValueError, OverflowError):
        pass
    return "other"


def od_distance_miles(
    df: pd.DataFrame,
    ox_col: str = "OLON_RANDOM",
    oy_col: str = "OLAT_RANDOM",
    dx_col: str = "DLON_RANDOM",
    dy_col: str = "DLAT_RANDOM",
    fallback_m_col: str = "trip_distance",
    as_network: bool = True,
) -> pd.Series:
    """Trip length in miles from origin and destination coordinates.

    Euclidean metres in EPSG:27700 are converted to driving-network miles
    (``as_network=True``) so they are comparable with NTS ``TripDisIncSW``.
    """
    from .circuity import geodesic_m_to_network_m

    ox = pd.to_numeric(df[ox_col], errors="coerce") if ox_col in df.columns else np.nan
    oy = pd.to_numeric(df[oy_col], errors="coerce") if oy_col in df.columns else np.nan
    dx = pd.to_numeric(df[dx_col], errors="coerce") if dx_col in df.columns else np.nan
    dy = pd.to_numeric(df[dy_col], errors="coerce") if dy_col in df.columns else np.nan
    metres = np.hypot(dx - ox, dy - oy)
    if fallback_m_col in df.columns:
        fb = pd.to_numeric(df[fallback_m_col], errors="coerce")
        metres = metres.where(np.isfinite(metres), fb)
    if as_network:
        converted = pd.Series(geodesic_m_to_network_m(np.asarray(metres, dtype=float)), index=df.index)
        net_col = "trip_distance_network_m"
        if net_col in df.columns:
            stored = pd.to_numeric(df[net_col], errors="coerce")
            metres = stored.where(np.isfinite(stored), converted)
        else:
            metres = converted
    return metres / METERS_PER_MILE


def prepare_synthetic_trips(trips: pd.DataFrame) -> pd.DataFrame:
    out = trips.copy()
    ocol = "OTripPur" if "OTripPur" in out.columns else "activity_type"
    dcol = "DTripPur" if "DTripPur" in out.columns else "activity_type"
    out["o_purpose"] = out[ocol].map(map_purpose)
    out["d_purpose"] = out[dcol].map(map_purpose)
    out["purpose"] = out["d_purpose"]
    out["miles"] = od_distance_miles(out)
    if "OTimin" in out.columns:
        out["tmin"] = pd.to_numeric(out["OTimin"], errors="coerce")
    out["weight"] = 1.0
    return out


def prepare_nts_trips(survey: pd.DataFrame) -> pd.DataFrame:
    out = recode_nts_trips(survey)
    out["o_purpose"] = (
        out["TripPurpFrom_B01ID"].map(map_purpose)
        if "TripPurpFrom_B01ID" in out.columns
        else out["purpose"]
    )
    out["d_purpose"] = out["purpose"]
    out["miles"] = pd.to_numeric(out.get("distance_mi", out.get("TripDisIncSW")), errors="coerce")
    out["tmin"] = pd.to_numeric(out.get("TripStart"), errors="coerce")
    if "weight" not in out.columns:
        wcol = "W5" if "W5" in out.columns else None
        out["weight"] = (
            pd.to_numeric(out[wcol], errors="coerce").fillna(1.0).clip(lower=0)
            if wcol else 1.0
        )
    return out


def activity_chain_table(
    trips: pd.DataFrame,
    person_col: str,
    seq_col: str,
    origin_purpose_col: str = "o_purpose",
    dest_purpose_col: str = "d_purpose",
    day_col: Optional[str] = None,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """One row per person-day: purpose sequence, trip count, home start/end."""
    cols = [person_col, seq_col, origin_purpose_col, dest_purpose_col]
    if day_col and day_col in trips.columns:
        cols.append(day_col)
    if weight_col in trips.columns:
        cols.append(weight_col)
    df = trips[cols].copy()
    df[seq_col] = pd.to_numeric(df[seq_col], errors="coerce")
    df = df.dropna(subset=[person_col, seq_col, dest_purpose_col])
    sort_keys = [person_col] + ([day_col] if day_col and day_col in df.columns else []) + [seq_col]
    df = df.sort_values(sort_keys)
    group_keys = [person_col] + ([day_col] if day_col and day_col in df.columns else [])

    dest_str = df.groupby(group_keys, sort=False)[dest_purpose_col].agg(
        lambda s: "-".join(str(x) for x in s)
    )
    first_o = df.groupby(group_keys, sort=False)[origin_purpose_col].first()
    n_trips = df.groupby(group_keys, sort=False).size().rename("n_trips")
    last_d = df.groupby(group_keys, sort=False)[dest_purpose_col].last()
    if weight_col in df.columns:
        w = df.groupby(group_keys, sort=False)[weight_col].mean().rename("weight")
    else:
        w = pd.Series(1.0, index=n_trips.index, name="weight")

    chain = (first_o.astype(str) + "-" + dest_str.astype(str)).rename("chain")
    out = pd.concat([chain, n_trips, first_o.rename("first_purpose"), last_d.rename("last_purpose"), w], axis=1)
    out = _close_open_home_tours(out)
    out["starts_home"] = out["first_purpose"].eq("home")
    out["ends_home"] = out["last_purpose"].eq("home")
    out["tour_type"] = out["chain"].map(_tour_type)
    return out.reset_index()


def _close_open_home_tours(out: pd.DataFrame) -> pd.DataFrame:
    """Match ``force_home_start_end``: a 1-trip day becomes a 2-trip home-based tour.

    ``home-X`` (no return) and ``X-home`` (day starts away) are counted as
    ``home-X-home``.  In particular ``home-leisure`` and ``leisure-home`` join
    ``home-leisure-home``.  Longer tours that start at home but do not end
    there get an implied return-home trip.
    """
    closed = out.copy()
    chains = []
    ntrips = []
    firsts = []
    lasts = []
    for chain, n, first, last in zip(
        closed["chain"].astype(str),
        closed["n_trips"].to_numpy(),
        closed["first_purpose"].astype(str),
        closed["last_purpose"].astype(str),
    ):
        parts = [p for p in str(chain).split("-") if p]
        n = int(n)
        if n == 1 and len(parts) == 2:
            origin, dest = parts
            if origin == "home" and dest != "home":
                chain, n, first, last = f"home-{dest}-home", 2, "home", "home"
            elif dest == "home" and origin != "home":
                chain, n, first, last = f"home-{origin}-home", 2, "home", "home"
        elif chain in {"home-leisure", "leisure-home"}:
            chain, n, first, last = "home-leisure-home", max(n + 1, 2), "home", "home"
        elif first == "home" and last != "home":
            chain = f"{chain}-home"
            n = n + 1
            last = "home"
        chains.append(chain)
        ntrips.append(n)
        firsts.append(first)
        lasts.append(last)
    closed["chain"] = chains
    closed["n_trips"] = ntrips
    closed["first_purpose"] = firsts
    closed["last_purpose"] = lasts
    return closed


def _tour_type(chain: str) -> str:
    parts = [p for p in str(chain).split("-") if p]
    if not parts:
        return "other"
    away = [p for p in parts[1:] if p != "home"]
    uniq = []
    for p in away:
        if p not in uniq:
            uniq.append(p)
    starts = parts[0] == "home"
    ends = parts[-1] == "home"
    if starts and ends and uniq == ["work"]:
        return "H-W-H"
    if starts and ends and uniq == ["education"]:
        return "H-E-H"
    if starts and ends and uniq == ["shop"]:
        return "H-S-H"
    if starts and ends and uniq == ["leisure"]:
        return "H-L-H"
    if starts and ends and len(uniq) == 1:
        return "H-X-H"
    if starts and ends and len(uniq) >= 2:
        return "complex"
    return "other"


def _weighted_share(values: pd.Series, weights: pd.Series) -> pd.Series:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0)
    tab = (
        pd.DataFrame({"v": values, "w": w})
        .groupby("v", dropna=False)["w"].sum()
    )
    tot = float(tab.sum())
    if tot <= 0:
        return tab * 0
    return tab / tot


def compare_chain_lengths(nts_chains: pd.DataFrame, syn_chains: pd.DataFrame) -> pd.DataFrame:
    """Share of person-days by trip count. One-trip days are omitted (they are
    recoded to two trips when an implied home leg is added)."""
    nts_use = nts_chains.loc[pd.to_numeric(nts_chains["n_trips"], errors="coerce") >= 2]
    syn_use = syn_chains.loc[pd.to_numeric(syn_chains["n_trips"], errors="coerce") >= 2]
    nts = _weighted_share(nts_use["n_trips"], nts_use["weight"]).rename("nts")
    syn = _weighted_share(syn_use["n_trips"], syn_use["weight"]).rename("synthetic")
    out = pd.concat([nts, syn], axis=1).fillna(0.0).sort_index()
    out.index.name = "n_trips"
    out["diff"] = out["synthetic"] - out["nts"]
    return out.reset_index()


def compare_chain_patterns(
    nts_chains: pd.DataFrame,
    syn_chains: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    nts = _weighted_share(nts_chains["chain"], nts_chains["weight"]).rename("nts")
    syn = _weighted_share(syn_chains["chain"], syn_chains["weight"]).rename("synthetic")
    top = nts.nlargest(top_n).index.union(syn.nlargest(min(top_n, 8)).index)
    out = pd.concat([nts, syn], axis=1).fillna(0.0)
    out = out.loc[out.index.isin(top)].copy()
    out["rank_nts"] = nts.reindex(out.index).fillna(0).rank(ascending=False, method="min")
    out = out.sort_values("nts", ascending=False)
    out.index.name = "chain"
    out["diff"] = out["synthetic"] - out["nts"]
    return out.reset_index()


def compare_tour_types(nts_chains: pd.DataFrame, syn_chains: pd.DataFrame) -> pd.DataFrame:
    nts = _weighted_share(nts_chains["tour_type"], nts_chains["weight"]).rename("nts")
    syn = _weighted_share(syn_chains["tour_type"], syn_chains["weight"]).rename("synthetic")
    out = pd.concat([nts, syn], axis=1).fillna(0.0)
    order = ["H-W-H", "H-E-H", "H-S-H", "H-L-H", "H-X-H", "complex"]
    out = out.reindex([k for k in order if k in out.index])
    for col in ("nts", "synthetic"):
        tot = float(out[col].sum())
        if tot > 0:
            out[col] = out[col] / tot
    out.index.name = "tour_type"
    out["diff"] = out["synthetic"] - out["nts"]
    return out.reset_index()


def chain_home_flags(nts_chains: pd.DataFrame, syn_chains: pd.DataFrame) -> pd.DataFrame:
    def _rate(df, col):
        w = pd.to_numeric(df["weight"], errors="coerce").fillna(0).to_numpy()
        x = df[col].astype(bool).to_numpy()
        return float(np.average(x, weights=w)) if w.sum() > 0 else float(x.mean())

    return pd.DataFrame([
        {
            "metric": "starts at home",
            "nts": _rate(nts_chains, "starts_home"),
            "synthetic": _rate(syn_chains, "starts_home"),
        },
        {
            "metric": "ends at home",
            "nts": _rate(nts_chains, "ends_home"),
            "synthetic": _rate(syn_chains, "ends_home"),
        },
        {
            "metric": "mean trips / person-day",
            "nts": float(np.average(nts_chains["n_trips"], weights=nts_chains["weight"])),
            "synthetic": float(np.average(syn_chains["n_trips"], weights=syn_chains["weight"])),
        },
        {
            "metric": "person-days",
            "nts": float(len(nts_chains)),
            "synthetic": float(len(syn_chains)),
        },
    ])


def purpose_share_table(nts_trips: pd.DataFrame, syn_trips: pd.DataFrame) -> pd.DataFrame:
    nts = _weighted_share(nts_trips["purpose"], nts_trips["weight"]).rename("nts")
    syn = _weighted_share(syn_trips["purpose"], syn_trips["weight"]).rename("synthetic")
    out = pd.concat([nts, syn], axis=1).fillna(0.0)
    out = out.reindex([p for p in PURPOSE_ORDER if p in out.index])
    out.index.name = "purpose"
    out["diff"] = out["synthetic"] - out["nts"]
    return out.reset_index()


def distance_summary(
    nts_trips: pd.DataFrame,
    syn_trips: pd.DataFrame,
    d_max: float = 40.0,
    purposes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    purposes = list(purposes) if purposes is not None else ["all"] + PURPOSE_ORDER
    rows = []
    for purpose in purposes:
        if purpose == "all":
            s, y, title = nts_trips, syn_trips, "all trips"
        else:
            s = nts_trips.loc[nts_trips["purpose"] == purpose]
            y = syn_trips.loc[syn_trips["purpose"] == purpose]
            title = purpose
        s = s.loc[s["miles"].gt(0) & s["miles"].le(d_max)]
        y = y.loc[y["miles"].gt(0) & y["miles"].le(d_max)]
        if len(s) == 0 and len(y) == 0:
            continue
        fit_s = fit_decay_functions(s["miles"].to_numpy(), weights=s["weight"].to_numpy(), d_max=d_max) if len(s) else {}
        fit_y = fit_decay_functions(y["miles"].to_numpy(), d_max=d_max) if len(y) else {}
        beta_s = (fit_s.get("exponential") or {}).get("beta", np.nan)
        beta_y = (fit_y.get("exponential") or {}).get("beta", np.nan)
        rows.append({
            "purpose": title,
            "n_nts": int(len(s)),
            "n_syn": int(len(y)),
            "median_mi_nts": float(s["miles"].median()) if len(s) else np.nan,
            "median_mi_syn": float(y["miles"].median()) if len(y) else np.nan,
            "mean_mi_nts": float(np.average(s["miles"], weights=s["weight"])) if len(s) else np.nan,
            "mean_mi_syn": float(y["miles"].mean()) if len(y) else np.nan,
            "share_lt_100m_syn": float((y["miles"] * METERS_PER_MILE < 100).mean()) if len(y) else np.nan,
            "share_lt_1mi_nts": float((s["miles"] < 1).mean()) if len(s) else np.nan,
            "share_lt_1mi_syn": float((y["miles"] < 1).mean()) if len(y) else np.nan,
            "exp_beta_nts": beta_s,
            "exp_beta_syn": beta_y,
        })
    return pd.DataFrame(rows)


def person_commute_miles(persons: pd.DataFrame, as_network: bool = True) -> pd.Series:
    from .circuity import geodesic_m_to_network_m

    hx = pd.to_numeric(persons["home_lon"], errors="coerce")
    hy = pd.to_numeric(persons["home_lat"], errors="coerce")
    wx = pd.to_numeric(persons["work_lon"], errors="coerce")
    wy = pd.to_numeric(persons["work_lat"], errors="coerce")
    metres = np.hypot(wx - hx, wy - hy)
    if as_network:
        metres = geodesic_m_to_network_m(metres.to_numpy())
    return pd.Series(np.asarray(metres, dtype=float) / METERS_PER_MILE, index=persons.index)


def plot_activity_chains(
    nts_chains: pd.DataFrame,
    syn_chains: pd.DataFrame,
    top_n: int = 12,
    outfile: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    length = compare_chain_lengths(nts_chains, syn_chains)
    patterns = compare_chain_patterns(nts_chains, syn_chains, top_n=top_n)
    tours = compare_tour_types(nts_chains, syn_chains)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    x = length["n_trips"].to_numpy(dtype=float)
    w = 0.38
    ax.bar(x - w / 2, length["nts"], width=w, color=NTS_BLUE, label="NTS survey")
    ax.bar(x + w / 2, length["synthetic"], width=w, color=SYN_ORANGE, label="Synthetic")
    ax.set_xlabel("Trips per person-day", fontweight="bold")
    ax.set_ylabel("Share of person-days", fontweight="bold")
    ax.set_title("Activity-chain length", fontweight="bold")
    ax.set_xlim(0.4, min(12.6, float(x.max()) + 0.6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    ax = axes[1]
    ypos = np.arange(len(patterns))
    ax.barh(ypos - 0.18, patterns["nts"], height=0.36, color=NTS_BLUE, label="NTS survey")
    ax.barh(ypos + 0.18, patterns["synthetic"], height=0.36, color=SYN_ORANGE, label="Synthetic")
    ax.set_yticks(ypos)
    ax.set_yticklabels(patterns["chain"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of person-days", fontweight="bold")
    ax.set_title(f"Top {len(patterns)} purpose sequences", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ypos = np.arange(len(tours))
    ax.barh(ypos - 0.18, tours["nts"], height=0.36, color=NTS_BLUE)
    ax.barh(ypos + 0.18, tours["synthetic"], height=0.36, color=SYN_ORANGE)
    ax.set_yticks(ypos)
    ax.set_yticklabels(tours["tour_type"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of person-days", fontweight="bold")
    ax.set_title("Tour type", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig


def plot_od_distance_decay(
    nts_trips: pd.DataFrame,
    syn_trips: pd.DataFrame,
    d_max: float = 40.0,
    outfile: Optional[str] = None,
):
    """Histogram + exponential decay by destination purpose, using OD miles."""
    import matplotlib.pyplot as plt

    purposes = [p for p in PURPOSE_ORDER if (nts_trips["purpose"] == p).any() or (syn_trips["purpose"] == p).any()]
    panels = ["all"] + purposes
    n = len(panels)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    bins_nts = np.concatenate(([0.0], np.linspace(0.5, 10.0, 20), np.linspace(12.0, d_max, 15)))
    xline = np.linspace(0.2, d_max, 200)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    rows = []

    for ax, purpose in zip(axes, panels):
        if purpose == "all":
            s, y, title = nts_trips, syn_trips, "all trips"
        else:
            s = nts_trips.loc[nts_trips["purpose"] == purpose]
            y = syn_trips.loc[syn_trips["purpose"] == purpose]
            title = purpose
        s = s.loc[s["miles"].gt(0) & s["miles"].le(d_max)]
        y = y.loc[y["miles"].gt(0) & y["miles"].le(d_max)]

        ax.hist(
            s["miles"], bins=bins_nts, density=True, weights=s["weight"],
            alpha=0.8, color=NTS_BLUE, label="NTS survey", edgecolor="blue",
        )
        ax.hist(
            y["miles"], bins=bins_nts, density=True,
            alpha=0.45, color=SYN_ORANGE, label="Synthetic", edgecolor="orange",
        )
        fit_s = fit_decay_functions(s["miles"].to_numpy(), weights=s["weight"].to_numpy(), d_max=d_max) if len(s) else {}
        fit_y = fit_decay_functions(y["miles"].to_numpy(), d_max=d_max) if len(y) else {}
        beta_s = (fit_s.get("exponential") or {}).get("beta", np.nan)
        beta_y = (fit_y.get("exponential") or {}).get("beta", np.nan)
        if fit_s.get("exponential"):
            e = fit_s["exponential"]
            ax.plot(xline, _exp_curve(xline, e["a"], e["beta"]), color=NTS_BLUE, lw=3)
        if fit_y.get("exponential"):
            e = fit_y["exponential"]
            ax.plot(xline, _exp_curve(xline, e["a"], e["beta"]), color="#d67d2f", lw=3, ls="--")
        ax.set_title(
            f"{title}  (n_nts={len(s):,}, n_syn={len(y):,})\n"
            f"beta_nts={beta_s:.2f}  beta_syn={beta_y:.2f}",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlim(0, d_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        rows.append({
            "purpose": title,
            "n_nts": len(s), "n_syn": len(y),
            "median_mi_nts": float(s["miles"].median()) if len(s) else np.nan,
            "median_mi_syn": float(y["miles"].median()) if len(y) else np.nan,
            "mean_mi_nts": float(np.average(s["miles"], weights=s["weight"])) if len(s) else np.nan,
            "mean_mi_syn": float(y["miles"].mean()) if len(y) else np.nan,
            "exp_beta_nts": beta_s, "exp_beta_syn": beta_y,
        })

    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", prop={"weight": "bold", "size": 11})
    for ax in axes[(nrows - 1) * ncols:(nrows - 1) * ncols + ncols]:
        ax.set_xlabel("Trip distance (miles, origin to destination)", fontsize=12, fontweight="bold")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("Density", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig, pd.DataFrame(rows)


def plot_purpose_shares(nts_trips: pd.DataFrame, syn_trips: pd.DataFrame, outfile: Optional[str] = None):
    import matplotlib.pyplot as plt

    tab = purpose_share_table(nts_trips, syn_trips)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(tab))
    w = 0.38
    ax.bar(x - w / 2, tab["nts"], width=w, color=NTS_BLUE, label="NTS survey")
    ax.bar(x + w / 2, tab["synthetic"], width=w, color=SYN_ORANGE, label="Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(tab["purpose"], rotation=0)
    ax.set_ylabel("Share of trips", fontweight="bold")
    ax.set_title("Destination activity shares", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig, tab
