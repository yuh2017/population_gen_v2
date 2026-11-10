"""Travel-mode labels, nests, and NTS code maps used by the nested logit.

The synthesizer default is five modes with NTS passenger codes folded into
``car``.  A notebook can pass a richer ``MODE_CODE_MAP`` (e.g. ``car_passenger``)
into ``generate_travel_activities(mode_mapping=..., mode_nests=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

PREFERRED_MODE_ORDER = ["walk", "bike", "car", "car_passenger", "bus", "rail"]

DEFAULT_MODE_CODE_MAP = {
    1.0: "walk", 2.0: "bike", 3.0: "car", 4.0: "car", 5.0: "car", 6.0: "car",
    7.0: "bus", 8.0: "bus", 9.0: "bus", 10.0: "rail", 11.0: "rail",
    12.0: "car", 13.0: "bus", -10.0: "car", -8.0: "car",
}

DEFAULT_NESTS = {
    "active": ["walk", "bike"],
    "private": ["car"],
    "public": ["bus", "rail"],
}

BASE_COVARIATES = [
    "age_z", "female", "income_z", "hh_size_z", "n_car",
    "urban_home", "log_density", "urban_dest", "urban_od",
    "is_work", "is_education", "is_shop", "is_home", "peak",
]

_LOGDIST0 = {
    "walk": -2.2,
    "bike": -1.1,
    "car": -0.3,
    "car_passenger": -0.35,
    "bus": -0.5,
    "rail": 0.4,
}


def expand_mode_map(mapping: Optional[Dict]) -> Dict:
    """Accept int, float, and string forms of the same NTS mode code."""
    if not mapping:
        return {}
    out = dict(mapping)
    for k, v in list(mapping.items()):
        label = str(v).strip().lower()
        out[k] = label
        try:
            f = float(k)
            out[f] = label
            if float(f).is_integer():
                out[int(f)] = label
        except (TypeError, ValueError, OverflowError):
            pass
        out[str(k).strip()] = label
        out[str(k).strip().lower()] = label
    return out


def unique_mode_labels(mapping: Dict) -> List[str]:
    seen = []
    got = set()
    for v in mapping.values():
        s = str(v).strip().lower()
        if not s or s in got:
            continue
        got.add(s)
        seen.append(s)
    ordered = [m for m in PREFERRED_MODE_ORDER if m in got]
    ordered.extend(m for m in seen if m not in ordered)
    return ordered


def infer_nests(modes: Sequence[str]) -> Dict[str, List[str]]:
    active, private, public = [], [], []
    for m in modes:
        if m in {"walk", "bike"} or "walk" in m or "bike" in m or "cycl" in m:
            active.append(m)
        elif m in {"bus", "rail", "pt"} or "bus" in m or "rail" in m or "transit" in m:
            public.append(m)
        else:
            private.append(m)
    nests: Dict[str, List[str]] = {}
    if active:
        nests["active"] = active
    if private:
        nests["private"] = private
    if public:
        nests["public"] = public
    return nests


def filter_nests(nests: Dict[str, List[str]], modes: Sequence[str]) -> Dict[str, List[str]]:
    mode_set = set(modes)
    out: Dict[str, List[str]] = {}
    placed = set()
    for name, members in nests.items():
        keep = [m for m in members if m in mode_set]
        if keep:
            out[name] = keep
            placed.update(keep)
    leftover = [m for m in modes if m not in placed]
    if leftover:
        out.setdefault("private", [])
        out["private"] = list(out["private"]) + leftover
    return out


@dataclass
class ModeSpec:
    """Choice-set, nest tree, and NTS code map for one synthesizer run."""

    mapping: Dict
    modes: List[str]
    nests: Dict[str, List[str]]
    nest_order: List[str]
    reference: str = "car"

    @property
    def covariates(self) -> List[str]:
        prev = ["prev_none"] + [f"prev_{m}" for m in self.modes if m != self.reference]
        return list(BASE_COVARIATES) + prev

    @property
    def active_modes(self) -> List[str]:
        return list(self.nests.get("active") or [])

    @property
    def public_modes(self) -> List[str]:
        return list(self.nests.get("public") or [])

    @property
    def motor_modes(self) -> List[str]:
        return [m for m in self.modes if m not in self.active_modes]

    def is_passenger(self, mode: str) -> bool:
        return "passenger" in str(mode)

    def is_driver(self, mode: str) -> bool:
        m = str(mode)
        return (m == "car" or m.startswith("car")) and not self.is_passenger(m)

    def logdist0(self, mode: str) -> float:
        return float(_LOGDIST0.get(mode, -0.4))

    def to_dict(self) -> Dict:
        return {
            "modes": list(self.modes),
            "nests": {k: list(v) for k, v in self.nests.items()},
            "nest_order": list(self.nest_order),
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict], mapping: Optional[Dict] = None) -> "ModeSpec":
        if not data:
            return default_mode_spec(mapping)
        modes = [str(m).strip().lower() for m in (data.get("modes") or [])]
        nests = {
            str(k): [str(x).strip().lower() for x in v]
            for k, v in (data.get("nests") or {}).items()
        }
        if not modes:
            return cls.from_mapping(mapping or DEFAULT_MODE_CODE_MAP, nests=nests or None)
        nests = filter_nests(nests or infer_nests(modes), modes)
        order = list(data.get("nest_order") or [k for k in ("active", "private", "public") if k in nests])
        order = [k for k in order if k in nests] + [k for k in nests if k not in order]
        ref = str(data.get("reference") or "car").strip().lower()
        if ref not in modes:
            ref = "car" if "car" in modes else modes[0]
        return cls(
            mapping=expand_mode_map(mapping or DEFAULT_MODE_CODE_MAP),
            modes=modes,
            nests=nests,
            nest_order=order,
            reference=ref,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Optional[Dict] = None,
        nests: Optional[Dict[str, List[str]]] = None,
        reference: str = "car",
    ) -> "ModeSpec":
        mapping = expand_mode_map(mapping or DEFAULT_MODE_CODE_MAP)
        modes = unique_mode_labels(mapping)
        if not modes:
            modes = list(PREFERRED_MODE_ORDER[:5])
        raw_nests = nests or infer_nests(modes)
        nests_f = filter_nests(raw_nests, modes)
        order = [k for k in ("active", "private", "public") if k in nests_f]
        order.extend(k for k in nests_f if k not in order)
        ref = str(reference or "car").strip().lower()
        if ref not in modes:
            ref = "car" if "car" in modes else modes[0]
        return cls(
            mapping=mapping,
            modes=modes,
            nests=nests_f,
            nest_order=order,
            reference=ref,
        )


def default_mode_spec(mapping: Optional[Dict] = None) -> ModeSpec:
    if mapping:
        return ModeSpec.from_mapping(mapping)
    return ModeSpec.from_mapping(DEFAULT_MODE_CODE_MAP, nests=DEFAULT_NESTS)


DEFAULT_MODE_SPEC = default_mode_spec()
