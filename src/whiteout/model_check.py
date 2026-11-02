"""Find where the drawing and the 3D model disagree.

A small STEP AP242 reader (pure Python, regex over the DATA section) pulls the semantic PMI out of the model:
dimensional sizes and locations with their values and tolerances, geometric tolerances with datum systems and
modifiers, datums, and cylindrical holes for inspection planning. compare() then matches the model PMI one to one
with the characteristics read from the drawing and reports what is only in the model, only on the drawing, and what
is in both with a different value or tolerance.

uv run python -m whiteout.model_check MODEL.stp JOB_DIR_OR_PDF
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# STEP Part 21 parsing


@dataclass(frozen=True)
class Ref:
    id: int


@dataclass(frozen=True)
class Enum:
    name: str


@dataclass(frozen=True)
class Typed:
    name: str
    value: object


_RECORD = re.compile(r"#(\d+)\s*=\s*((?:[^;']|'(?:[^']|'')*')*);", re.S)
_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*')|#(?P<ref>\d+)|\.(?P<enum>[A-Z_0-9]+)\.|"
    r"(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)|(?P<name>[A-Z_][A-Z_0-9]*)|(?P<p>[(),$*]))"
)


def _decode(s: str) -> str:
    """Decode a STEP string literal body (\\X2\\...\\X0\\ unicode, \\S\\ high bit, \\X\\hh)."""
    s = s.replace("''", "'")
    if "\\" not in s:
        return s

    def x2(m):
        h = m.group(1)
        return "".join(chr(int(h[i:i + 4], 16)) for i in range(0, len(h), 4))

    s = re.sub(r"\\X2\\([0-9A-Fa-f]*)\\X0\\", x2, s)
    s = re.sub(r"\\X4\\([0-9A-Fa-f]*)\\X0\\",
               lambda m: "".join(chr(int(m.group(1)[i:i + 8], 16)) for i in range(0, len(m.group(1)), 8)), s)
    s = re.sub(r"\\X\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"\\S\\(.)", lambda m: chr(ord(m.group(1)) + 128), s)
    s = re.sub(r"\\P[A-I]\\", "", s)
    return s.replace("\\\\", "\\")


def _parse_args(text: str, pos: int = 0) -> tuple[list, int]:
    """Parse '(a, b, (c), NAME(d))' starting at the opening parenthesis; returns (values, end position)."""
    out: list = []
    m = _TOKEN.match(text, pos)
    if not m or m.group("p") != "(":
        raise ValueError(f"expected '(' at {pos}: {text[pos:pos + 40]!r}")
    pos = m.end()
    while True:
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"bad token at {pos}: {text[pos:pos + 40]!r}")
        if m.group("str") is not None:
            out.append(_decode(m.group("str")[1:-1]))
            pos = m.end()
        elif m.group("ref") is not None:
            out.append(Ref(int(m.group("ref"))))
            pos = m.end()
        elif m.group("enum") is not None:
            out.append(Enum(m.group("enum")))
            pos = m.end()
        elif m.group("num") is not None:
            out.append(float(m.group("num")))
            pos = m.end()
        elif m.group("name") is not None:
            name = m.group("name")
            inner, pos = _parse_args(text, m.end())
            out.append(Typed(name, inner[0] if len(inner) == 1 else inner))
        else:
            p = m.group("p")
            if p == "(":
                inner, pos = _parse_args(text, m.start("p"))
                out.append(inner)
            elif p == ")":
                return out, m.end()
            elif p == "$":
                out.append(None)
                pos = m.end()
            elif p == "*":
                out.append("*")
                pos = m.end()
            else:  # comma
                pos = m.end()


class Step:
    """Entity store. Each instance is a dict {TYPE: args}; a complex instance has several types."""

    def __init__(self, text: str):
        self.header = text[: text.find("DATA;")] if "DATA;" in text else ""
        body = text[text.find("DATA;") + 5:] if "DATA;" in text else text
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        self.raw: dict[int, str] = {}
        self.type_of: dict[int, tuple[str, ...]] = {}
        self.by_type: dict[str, list[int]] = {}
        for m in _RECORD.finditer(body):
            i, rhs = int(m.group(1)), m.group(2).strip()
            self.raw[i] = rhs
            if rhs.startswith("("):
                names = tuple(re.findall(r"(?:^\(|\))\s*([A-Z_][A-Z_0-9]*)\s*\(", rhs))
            else:
                names = (rhs[: rhs.find("(")].strip(),)
            self.type_of[i] = names
            for n in names:
                self.by_type.setdefault(n, []).append(i)
        self._cache: dict[int, dict[str, list]] = {}

    @classmethod
    def read(cls, path: str | Path) -> Step:
        return cls(Path(path).read_text(errors="replace"))

    def get(self, ref: Ref | int | None) -> dict[str, list]:
        if ref is None:
            return {}
        i = ref.id if isinstance(ref, Ref) else ref
        if i in self._cache:
            return self._cache[i]
        rhs = self.raw.get(i)
        if rhs is None:
            return {}
        ent: dict[str, list] = {}
        if rhs.startswith("("):
            pos = 1
            while True:
                m = re.compile(r"\s*([A-Z_][A-Z_0-9]*)\s*").match(rhs, pos)
                if not m:
                    break
                args, pos = _parse_args(rhs, m.end())
                ent[m.group(1)] = args
        else:
            name = rhs[: rhs.find("(")].strip()
            ent[name] = _parse_args(rhs, rhs.find("("))[0]
        self._cache[i] = ent
        return ent

    def types(self, *names: str) -> list[int]:
        out: set[int] = set()
        for n in names:
            out.update(self.by_type.get(n, []))
        return sorted(out)

    def users(self, i: int) -> list[int]:
        """Instances that reference #i (built on first call)."""
        if not hasattr(self, "_users"):
            self._users: dict[int, list[int]] = {}
            for j, rhs in self.raw.items():
                for r in re.findall(r"#(\d+)", rhs):
                    self._users.setdefault(int(r), []).append(j)
        return self._users.get(i, [])


def _first_ref(v) -> Ref | None:
    if isinstance(v, Ref):
        return v
    if isinstance(v, list):
        for x in v:
            r = _first_ref(x)
            if r is not None:
                return r
    return None


def _refs(v) -> list[Ref]:
    if isinstance(v, Ref):
        return [v]
    if isinstance(v, Typed):
        return _refs(v.value)
    if isinstance(v, list):
        return [r for x in v for r in _refs(x)]
    return []


def _vec(v) -> list[float]:
    return [float(x) for x in v]


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a, b) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _sub(a, b) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def _add(a, b, k: float = 1.0) -> list[float]:
    return [x + k * y for x, y in zip(a, b, strict=True)]


# ---------------------------------------------------------------------------------------------------------------
# Semantic PMI

GDT_SYMBOLS = {
    "position": "⌖", "flatness": "⏥", "perpendicularity": "⊥", "parallelism": "∥", "angularity": "∠",
    "surface_profile": "⌓", "line_profile": "⌒", "straightness": "⏤", "roundness": "○", "cylindricity": "⌭",
    "circular_runout": "↗", "total_runout": "⌰", "concentricity": "◎", "coaxiality": "◎", "symmetry": "⌯",
}
_GT_ENTITIES = {k.upper() + "_TOLERANCE": k for k in GDT_SYMBOLS}
MODIFIER_SYMBOLS = {
    "MAXIMUM_MATERIAL_REQUIREMENT": "Ⓜ", "LEAST_MATERIAL_REQUIREMENT": "Ⓛ", "FREE_STATE": "Ⓕ",
    "TANGENT_PLANE": "Ⓣ", "STATISTICAL_TOLERANCE": "<ST>", "SEPARATE_REQUIREMENT": "SEP REQT",
    "RECIPROCITY_REQUIREMENT": "Ⓡ", "UNEQUALLY_DISPOSED": "Ⓤ", "COMMON_ZONE": "CZ",
    "ALL_AROUND": "ALL AROUND", "ALL_OVER": "ALL OVER", "DIAMETER": "⌀",
}
_SIZE_SYMBOL = {"diameter": "⌀", "spherical diameter": "S⌀", "radius": "R", "spherical radius": "SR"}
# Descriptive text that is bookkeeping, not a requirement
_ADMIN_NAMES = {"datum references", "dimensional note", "equivalent unicode string", "filesavetime", "filename",
                "mtime", "attr_version", "materialmissingassignments", "materialmultipleassigned", "cad_source"}


def _fmt(v: float | None, places: int | None = None) -> str:
    if v is None:
        return ""
    if places is not None:
        s = f"{v:.{places}f}"
    else:
        s = f"{round(v, 6):.6f}".rstrip("0").rstrip(".")
    if s.startswith("0.") and places is not None:
        pass
    return "0" if s in ("-0", "") else s


class _Reader:
    def __init__(self, step: Step):
        self.s = step
        self._unit_cache: dict[int, tuple[str, str, float]] = {}
        self.length_unit, self.length_factor = self._global_length_unit()

    # ---- units ------------------------------------------------------------------------------------------------
    def unit(self, ref: Ref | None) -> tuple[str, str, float]:
        """(quantity, name, factor) where factor converts to mm for lengths and to degrees for angles."""
        if ref is None:
            return ("", "", 1.0)
        if ref.id in self._unit_cache:
            return self._unit_cache[ref.id]
        e = self.s.get(ref)
        out = ("", "", 1.0)
        if "SI_UNIT" in e:
            prefix, name = e["SI_UNIT"]
            p = {"MILLI": 1e-3, "CENTI": 1e-2, "DECI": 1e-1, "KILO": 1e3, "MICRO": 1e-6}.get(
                prefix.name if isinstance(prefix, Enum) else "", 1.0)
            n = name.name if isinstance(name, Enum) else str(name)
            if n == "METRE":
                out = ("length", "mm" if p == 1e-3 else "m", p * 1000.0)
            elif n == "RADIAN":
                out = ("angle", "rad", p * 180.0 / math.pi)
            else:
                out = (n.lower(), n.lower(), p)
        elif "CONVERSION_BASED_UNIT" in e:
            name, mref = e["CONVERSION_BASED_UNIT"]
            val, uref = self._raw_measure(mref)
            q, _, f = self.unit(uref)
            nm = str(name).lower()
            if "LENGTH_UNIT" in e or q == "length":
                nm = "in" if nm.startswith("inch") else "mm" if nm.startswith("milli") else nm
                out = ("length", nm, (val or 1.0) * f)
            elif "PLANE_ANGLE_UNIT" in e or q == "angle":
                out = ("angle", "deg", (val or 1.0) * f)
            else:
                out = (q, nm, (val or 1.0) * f)
        self._unit_cache[ref.id] = out
        return out

    def _global_length_unit(self) -> tuple[str, float]:
        for i in self.s.types("GLOBAL_UNIT_ASSIGNED_CONTEXT"):
            for r in _refs(self.s.get(i)["GLOBAL_UNIT_ASSIGNED_CONTEXT"]):
                q, name, f = self.unit(r)
                if q == "length":
                    return name, f
        return "mm", 1.0

    def _raw_measure(self, ref: Ref | None) -> tuple[float | None, Ref | None]:
        e = self.s.get(ref)
        for k in ("MEASURE_WITH_UNIT", "LENGTH_MEASURE_WITH_UNIT", "PLANE_ANGLE_MEASURE_WITH_UNIT",
                  "MEASURE_REPRESENTATION_ITEM", "UNCERTAINTY_MEASURE_WITH_UNIT", "RATIO_MEASURE_WITH_UNIT"):
            args = e.get(k)
            if args and isinstance(args[0], Typed) and isinstance(args[0].value, float):
                return args[0].value, args[1] if len(args) > 1 and isinstance(args[1], Ref) else None
        for args in e.values():
            for a in args:
                if isinstance(a, Typed) and isinstance(a.value, float) and a.name.endswith("MEASURE"):
                    i = args.index(a)
                    return a.value, args[i + 1] if i + 1 < len(args) and isinstance(args[i + 1], Ref) else None
        return None, None

    def measure(self, ref: Ref | None) -> dict:
        """Value in file units (length) or degrees (angle), with its label and qualifiers."""
        e = self.s.get(ref)
        val, uref = self._raw_measure(ref)
        q, _, f = self.unit(uref)
        if val is not None:
            if q == "length":
                val = val * f / self.length_factor
            elif q == "angle":
                val = val * f
            val = round(val, 9)
        label = ""
        if e.get("REPRESENTATION_ITEM"):
            label = e["REPRESENTATION_ITEM"][0] or ""
        quals: list[str] = []
        places = None
        for qr in _refs(e.get("QUALIFIED_REPRESENTATION_ITEM", [])):
            qe = self.s.get(qr)
            if "VALUE_FORMAT_TYPE_QUALIFIER" in qe:
                m = re.search(r"NR2\s*\d+\.(\d+)", qe["VALUE_FORMAT_TYPE_QUALIFIER"][0] or "")
                if m:
                    places = int(m.group(1))
            elif "TYPE_QUALIFIER" in qe:
                quals.append(str(qe["TYPE_QUALIFIER"][0]))
        if q == "angle" or "PLANE_ANGLE_MEASURE_WITH_UNIT" in e:
            q = "angle"
        return {"value": val, "quantity": q or "length", "label": label, "qualifiers": quals, "places": places}

    # ---- shape aspects and the faces they point at -------------------------------------------------------------
    def faces(self, aspect: Ref | int | None, depth: int = 0) -> list[int]:
        if aspect is None or depth > 6:
            return []
        a = aspect.id if isinstance(aspect, Ref) else aspect
        out: list[int] = []
        ta = self.s.type_of.get(a, ())
        for u in self.s.users(a):
            tu = self.s.type_of.get(u, ())
            if "GEOMETRIC_ITEM_SPECIFIC_USAGE" in tu or "ITEM_IDENTIFIED_REPRESENTATION_USAGE" in tu:
                args = list(self.s.get(u).values())[0]
                if _first_ref(args[2:3]) and _first_ref(args[2:3]).id == a:
                    out += [r.id for r in _refs(args[4:5])]
            elif ("SHAPE_ASPECT_RELATIONSHIP" in tu or "SHAPE_ASPECT_DERIVING_RELATIONSHIP" in tu) and \
                    any(t.startswith(("COMPOSITE", "ALL_AROUND", "CENTRE_OF_SYMMETRY", "DERIVED_SHAPE_ASPECT",
                                      "DATUM_FEATURE", "BETWEEN", "CONTINUOUS")) for t in ta) and \
                    not any("DIMENSIONAL" in t or "TOLERANCE" in t for t in tu):
                args = self.s.get(u).get("SHAPE_ASPECT_RELATIONSHIP") or list(self.s.get(u).values())[0]
                if isinstance(args[2], Ref) and args[2].id == a and isinstance(args[3], Ref):
                    if "DATUM" in self.s.type_of.get(args[3].id, ()):
                        continue
                    out += self.faces(args[3], depth + 1)
        return list(dict.fromkeys(out))

    def surface_key(self, face: int) -> tuple:
        e = self.s.get(face)
        if "ADVANCED_FACE" not in e and "FACE_SURFACE" not in e:
            return ("item", face)
        args = e.get("ADVANCED_FACE") or e.get("FACE_SURFACE")
        surf = args[2]
        se = self.s.get(surf)
        if "CYLINDRICAL_SURFACE" in se or "CONICAL_SURFACE" in se or "TOROIDAL_SURFACE" in se:
            name = next(k for k in ("CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "TOROIDAL_SURFACE") if k in se)
            p, z, _ = self.placement(se[name][1])
            r = round(float(se[name][2]), 4)
            if name == "CYLINDRICAL_SURFACE":
                z = _canon(z)
                p = _add(p, z, -_dot(p, z))  # closest point of the axis line to the origin
                return ("cyl", tuple(round(x, 3) for x in z), tuple(round(x, 3) for x in p), r)
            return (name, tuple(round(x, 3) for x in p), r)
        if "SPHERICAL_SURFACE" in se:
            p, _, _ = self.placement(se["SPHERICAL_SURFACE"][1])
            return ("sph", tuple(round(x, 3) for x in p), round(float(se["SPHERICAL_SURFACE"][2]), 4))
        if "PLANE" in se:
            p, z, _ = self.placement(se["PLANE"][1])
            z = _canon(z)
            return ("plane", tuple(round(x, 3) for x in z), round(_dot(p, z), 3))
        return ("face", face)

    def placement(self, ref: Ref) -> tuple[list[float], list[float], list[float]]:
        e = self.s.get(ref).get("AXIS2_PLACEMENT_3D")
        if not e:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        p = _vec(self.s.get(e[1])["CARTESIAN_POINT"][1])
        z = _norm(_vec(self.s.get(e[2])["DIRECTION"][1])) if isinstance(e[2], Ref) else [0.0, 0.0, 1.0]
        x = _vec(self.s.get(e[3])["DIRECTION"][1]) if len(e) > 3 and isinstance(e[3], Ref) else \
            ([1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
        x = _norm(_add(x, z, -_dot(x, z)))
        return p, z, x

    def count(self, aspect) -> tuple[int, list[int]]:
        faces = self.faces(aspect)
        if not faces:
            return 1, []
        keys = {self.surface_key(f) for f in faces}
        return max(1, len(keys)), faces


def _canon(v: list[float]) -> list[float]:
    """Direction with a fixed sign: the largest component is positive."""
    i = max(range(3), key=lambda k: abs(v[k]))
    return [x if v[i] >= 0 else -x for x in v]


_DIM_TYPES = ("DIMENSIONAL_SIZE", "DIMENSIONAL_LOCATION", "ANGULAR_SIZE", "ANGULAR_LOCATION",
              "DIMENSIONAL_SIZE_WITH_PATH", "DIMENSIONAL_LOCATION_WITH_PATH", "DIRECTED_DIMENSIONAL_LOCATION",
              "DIMENSIONAL_SIZE_WITH_DATUM_FEATURE")


def _dimensions(rd: _Reader) -> list[dict]:
    s = rd.s
    reps: dict[int, list[int]] = {}
    for i in s.types("DIMENSIONAL_CHARACTERISTIC_REPRESENTATION"):
        d, r = s.get(i)["DIMENSIONAL_CHARACTERISTIC_REPRESENTATION"]
        reps.setdefault(d.id, []).append(r.id)
    tols: dict[int, int] = {}
    for i in s.types("PLUS_MINUS_TOLERANCE"):
        rng, d = s.get(i)["PLUS_MINUS_TOLERANCE"]
        if isinstance(d, Ref):
            tols[d.id] = i
    out = []
    for i in s.types(*_DIM_TYPES):
        e = s.get(i)
        types = s.type_of[i]
        is_loc = any("LOCATION" in t for t in types)
        angular = any(t.startswith("ANGULAR") for t in types)
        if is_loc:
            args = e.get("DIMENSIONAL_LOCATION") or next(v for k, v in e.items() if "LOCATION" in k)
            name, aspects = (args[0] or ""), [a for a in args[2:4] if isinstance(a, Ref)]
        else:
            args = e.get("DIMENSIONAL_SIZE") or next(v for k, v in e.items() if "SIZE" in k)
            name, aspects = (args[1] or ""), [args[0]] if isinstance(args[0], Ref) else []
        d = {"id": f"#{i}", "entity": "location" if is_loc else "size", "name": str(name), "nominal": None,
             "upper": None, "lower": None, "plus": None, "minus": None, "tolerance": "none", "fit": "",
             "places": None, "qualifiers": [], "notes": [], "angular": angular, "count": 1, "faces": []}
        for r in reps.get(i, []):
            items = _refs(list(s.get(r).values())[0][1:2])
            for it in items:
                ie = s.get(it)
                if "DESCRIPTIVE_REPRESENTATION_ITEM" in ie:
                    nm, txt = ie["DESCRIPTIVE_REPRESENTATION_ITEM"]
                    d["notes" if (nm or "") != "dimensional note" else "qualifiers"].append(str(txt))
                    continue
                if "COMPOUND_REPRESENTATION_ITEM" in ie:
                    for sub in _refs(ie["COMPOUND_REPRESENTATION_ITEM"]):
                        se = s.get(sub)
                        if "DESCRIPTIVE_REPRESENTATION_ITEM" in se:
                            d["qualifiers"].append(str(se["DESCRIPTIVE_REPRESENTATION_ITEM"][1]))
                    continue
                m = rd.measure(it)
                if m["value"] is None:
                    continue
                if m["quantity"] == "angle":
                    d["angular"] = True
                lab = m["label"].lower()
                if "upper" in lab:
                    d["upper"] = m["value"]
                elif "lower" in lab:
                    d["lower"] = m["value"]
                elif d["nominal"] is None:
                    d["nominal"] = m["value"]
                d["qualifiers"] += m["qualifiers"]
                if m["places"] is not None and d["places"] is None:
                    d["places"] = m["places"]
        if d["upper"] is not None and d["lower"] is not None:
            d["tolerance"] = "limits"
            if d["nominal"] is None:
                d["nominal"] = round((d["upper"] + d["lower"]) / 2, 9)
        if i in tols:
            rng = s.get(tols[i])["PLUS_MINUS_TOLERANCE"][0]
            re_ = s.get(rng)
            if "TOLERANCE_VALUE" in re_:
                lo, hi = (rd.measure(x) for x in re_["TOLERANCE_VALUE"][:2])
                if lo["value"] is not None and hi["value"] is not None:
                    # some writers store the bounds the other way round
                    d["minus"], d["plus"] = min(lo["value"], hi["value"]), max(lo["value"], hi["value"])
                    d["tolerance"] = "plus_minus"
                    if hi["places"] is not None:
                        d["tol_places"] = hi["places"]
                    if d["nominal"] is not None:
                        d["lower"] = round(d["nominal"] + d["minus"], 9)
                        d["upper"] = round(d["nominal"] + d["plus"], 9)
            elif "LIMITS_AND_FITS" in re_:
                parts = [str(x) for x in re_["LIMITS_AND_FITS"][:3] if x and str(x).lower() not in ("hole", "shaft")]
                d["fit"] = "".join(parts).strip()
                d["tolerance"] = "fit"
        qs = [q.lower() for q in d["qualifiers"]]
        n = d["name"].lower()
        if "theoretical" in qs:
            d["type"] = "basic"
        elif "auxiliary" in qs:
            d["type"] = "reference"
        elif d["angular"] or "angle" in n:
            d["type"] = "angle"
        elif "radius" in n:
            d["type"] = "radius"
        elif is_loc:
            d["type"] = "location"
        else:
            d["type"] = "size"
        d["symbol"] = _SIZE_SYMBOL.get(n, "")
        if "controlled radius" in qs:
            d["symbol"] = "CR"
        count_from = aspects if not is_loc else aspects[1:] or aspects
        counts = [rd.count(a) for a in count_from]
        d["count"] = max([c for c, _ in counts] or [1])
        d["faces"] = [f"#{f}" for _, fs in counts for f in fs]
        d["unit"] = "deg" if d["angular"] else rd.length_unit
        d["text"] = _dim_text(d)
        out.append(d)
    return out


def _dim_text(d: dict) -> str:
    p = d.get("places")
    nom = _fmt(d["nominal"], p)
    deg = "°" if d.get("angular") else ""
    head = (f"{d['count']}X " if d.get("count", 1) > 1 else "") + d.get("symbol", "")
    if d["type"] == "basic":
        return f"{head}[{nom}{deg}]"
    if d["type"] == "reference":
        return f"{head}({nom}{deg})"
    if d["tolerance"] == "plus_minus":
        tp = d.get("tol_places")
        if abs((d["plus"] or 0) + (d["minus"] or 0)) < 1e-9:
            return f"{head}{nom}{deg} ±{_fmt(d['plus'], tp)}{deg}"
        return f"{head}{nom}{deg} +{_fmt(d['plus'], tp)}/{_fmt(d['minus'], tp)}{deg}"
    if d["tolerance"] == "limits":
        return f"{head}{_fmt(d['upper'], p)}/{_fmt(d['lower'], p)}{deg}"
    if d["tolerance"] == "fit":
        return f"{head}{nom}{deg} {d['fit']}"
    return f"{head}{nom}{deg}"


def _datum_label(rd: _Reader, base) -> str:
    s = rd.s
    if isinstance(base, Typed) and base.name == "COMMON_DATUM_LIST":
        parts = []
        for r in _refs(base.value):
            e = s.get(r)
            args = next(iter(e.values()))
            parts.append(_datum_label(rd, args[4]) + _datum_mods(args[5] if len(args) > 5 else None))
        return "-".join(parts)
    if isinstance(base, Ref):
        e = s.get(base)
        if "DATUM" in e:
            return str(e["DATUM"][4])
        args = next(iter(e.values()))
        if len(args) > 4:
            return _datum_label(rd, args[4])
    return "?"


def _datum_mods(mods) -> str:
    out = []
    for m in mods or []:
        if isinstance(m, Typed):
            v = m.value
            name = v.name if isinstance(v, Enum) else str(v)
            out.append({"MAXIMUM_MATERIAL_REQUIREMENT": "Ⓜ", "LEAST_MATERIAL_REQUIREMENT": "Ⓛ",
                        "FREE_STATE": "Ⓕ", "TRANSLATION": "▷", "PROJECTED": "Ⓟ"}.get(name, name.lower()))
    return "".join(out)


def _datum_refs(rd: _Reader, systems: list[Ref]) -> list[dict]:
    s = rd.s
    out = []
    for ds in systems:
        e = s.get(ds)
        if "DATUM_SYSTEM" in e:
            for c in _refs(e["DATUM_SYSTEM"][4]):
                args = next(iter(s.get(c).values()))
                out.append({"label": _datum_label(rd, args[4]), "modifiers": _datum_mods(args[5])})
        elif "DATUM_REFERENCE" in e:  # AP242 edition 1 style: (precedence, datum)
            out.append({"label": _datum_label(rd, e["DATUM_REFERENCE"][1]), "modifiers": "",
                        "precedence": e["DATUM_REFERENCE"][0]})
    if out and "precedence" in out[0]:
        out.sort(key=lambda r: r["precedence"])
    return out


def _tolerances(rd: _Reader) -> list[dict]:
    s = rd.s
    zones: dict[int, dict] = {}
    for z in s.types("TOLERANCE_ZONE"):
        args = s.get(z)["TOLERANCE_ZONE"]
        form = s.get(args[5]).get("TOLERANCE_ZONE_FORM", [""])[0] if isinstance(args[5], Ref) else ""
        for g in _refs(args[4]):
            zones.setdefault(g.id, {})["form"] = form
            zones[g.id]["zone"] = z
    for p in s.types("PROJECTED_ZONE_DEFINITION"):
        args = s.get(p)["PROJECTED_ZONE_DEFINITION"]
        zone = args[0].id if isinstance(args[0], Ref) else None
        for zd in zones.values():
            if zd.get("zone") == zone:
                zd["projected"] = rd.measure(args[3])["value"]
    composite: dict[int, int] = {}
    for r in s.types("GEOMETRIC_TOLERANCE_RELATIONSHIP"):
        nm, _, a, b = s.get(r)["GEOMETRIC_TOLERANCE_RELATIONSHIP"][:4]
        if str(nm).lower() == "composite" and isinstance(a, Ref) and isinstance(b, Ref):
            composite[b.id] = a.id
    out = []
    ids = set()
    for name in list(_GT_ENTITIES) + ["GEOMETRIC_TOLERANCE"]:
        ids.update(s.by_type.get(name, []))
    for i in sorted(ids):
        e = s.get(i)
        kind = next((_GT_ENTITIES[t] for t in s.type_of[i] if t in _GT_ENTITIES), "")
        if not kind:
            continue
        base = e.get("GEOMETRIC_TOLERANCE") or e[next(t for t in s.type_of[i] if t in _GT_ENTITIES)]
        name, _, mag, aspect = base[:4]
        systems = _refs(e.get("GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE", [])) or \
            (_refs(base[4]) if len(base) > 4 else [])
        mods = [m.name for m in (e.get("GEOMETRIC_TOLERANCE_WITH_MODIFIERS") or [[]])[0] if isinstance(m, Enum)]
        if not mods:
            for k, v in e.items():
                if k.endswith("_TOLERANCE") and len(v) > 4 and isinstance(v[-1], list) and v[-1] and \
                        isinstance(v[-1][0], Enum):
                    mods = [m.name for m in v[-1]]
        m = rd.measure(mag) if isinstance(mag, Ref) else {"value": None, "places": None}
        z = zones.get(i, {})
        form = str(z.get("form", "")).lower()
        zone = "S⌀" if "spherical" in form else ("⌀" if "cylindrical" in form or "circular" in form else "")
        t = {"id": f"#{i}", "type": "gdt", "characteristic": kind, "symbol": GDT_SYMBOLS[kind],
             "name": str(name or ""), "value": m["value"], "places": m.get("places"), "zone": zone,
             "modifiers": [MODIFIER_SYMBOLS.get(x, x.lower()) for x in mods],
             "datums": _datum_refs(rd, systems), "unit": rd.length_unit}
        if "GEOMETRIC_TOLERANCE_WITH_DEFINED_UNIT" in e:
            u = e["GEOMETRIC_TOLERANCE_WITH_DEFINED_UNIT"][0]
            unit = {"size": rd.measure(u)["value"] if isinstance(u, Ref) else None}
            if "GEOMETRIC_TOLERANCE_WITH_DEFINED_AREA_UNIT" in e:
                at, second = e["GEOMETRIC_TOLERANCE_WITH_DEFINED_AREA_UNIT"][:2]
                unit["area"] = at.name.lower() if isinstance(at, Enum) else str(at)
                unit["second"] = rd.measure(second)["value"] if isinstance(second, Ref) else None
            t["per_unit"] = unit
        if "UNEQUALLY_DISPOSED_GEOMETRIC_TOLERANCE" in e:
            t["unequal"] = rd.measure(e["UNEQUALLY_DISPOSED_GEOMETRIC_TOLERANCE"][0])["value"]
        if "GEOMETRIC_TOLERANCE_WITH_MAXIMUM_TOLERANCE" in e:
            t["maximum"] = rd.measure(e["GEOMETRIC_TOLERANCE_WITH_MAXIMUM_TOLERANCE"][0])["value"]
        if z.get("projected") is not None:
            t["projected"] = z["projected"]
        if i in composite:
            t["composite_of"] = f"#{composite[i]}"
        t["count"], faces = rd.count(aspect)
        t["faces"] = [f"#{f}" for f in faces]
        t["all_around"] = isinstance(aspect, Ref) and "ALL_AROUND_SHAPE_ASPECT" in s.type_of.get(aspect.id, ())
        t["text"] = _gdt_text(t)
        out.append(t)
    return out


def _gdt_text(t: dict) -> str:
    val = t["zone"] + _fmt(t["value"], t.get("places"))
    mods = "".join(m for m in t["modifiers"] if len(m) == 1)
    if t.get("unequal") is not None:
        val += "Ⓤ" + _fmt(t["unequal"])
    if t.get("projected") is not None:
        val += "Ⓟ" + _fmt(t["projected"])
    cells = [t["symbol"], val + mods]
    if t.get("per_unit"):
        u = t["per_unit"]
        cells[1] += "/" + _fmt(u.get("size")) + (f" X {_fmt(u['second'])}" if u.get("second") else "")
    cells += [d["label"] + d["modifiers"] for d in t["datums"]]
    return "|".join(cells)


def _datums(rd: _Reader) -> dict:
    s = rd.s
    labels = sorted({str(s.get(i)["DATUM"][4]) for i in s.types("DATUM")})
    targets = []
    for i in s.types("PLACED_DATUM_TARGET_FEATURE", "DATUM_TARGET"):
        args = next(iter(s.get(i).values()))
        if len(args) > 4 and isinstance(args[4], str):
            targets.append(args[4])
    return {"labels": labels, "targets": sorted(set(targets))}


def _texts(rd: _Reader) -> dict:
    s = rd.s
    notes, fcf = [], []
    for i in s.types("DESCRIPTIVE_REPRESENTATION_ITEM"):
        args = s.get(i)["DESCRIPTIVE_REPRESENTATION_ITEM"]
        nm, txt = (str(args[0] or "").strip(), str(args[1] or "").strip()) if len(args) > 1 else ("", "")
        if nm.lower() == "equivalent unicode string":
            fcf.append(txt.replace("\\w", "|").replace("\\n", " / "))
        elif nm.lower() not in _ADMIN_NAMES and len(txt) >= 8 and " " in txt:
            notes.append(txt)
    return {"notes": list(dict.fromkeys(notes)), "equivalent_strings": list(dict.fromkeys(fcf))}


# ---------------------------------------------------------------------------------------------------------------
# Holes for inspection planning


def _face_points(rd: _Reader, face: int, frame: tuple) -> tuple[list[list[float]], set[int]]:
    """Points on the face boundary and the 10 degree bins around the cylinder axis that its circular edges cover."""
    s = rd.s
    c0, z0, x0 = frame
    y0 = _cross(z0, x0)
    pts: list[list[float]] = []
    bins: set[int] = set()
    args = s.get(face).get("ADVANCED_FACE") or []
    for b in _refs(args[1:2]):
        loop = next(iter(s.get(b).values()))[1]
        for oe in _refs(s.get(loop).get("EDGE_LOOP", [None, []])[1]):
            oargs = s.get(oe).get("ORIENTED_EDGE")
            if not oargs:
                continue
            ec = s.get(oargs[3]).get("EDGE_CURVE")
            if not ec:
                continue
            vs = [_vec(s.get(s.get(v)["VERTEX_POINT"][1])["CARTESIAN_POINT"][1]) for v in ec[1:3]]
            pts += vs
            ce = s.get(ec[3])
            if "CIRCLE" not in ce:
                continue
            c, z, x = rd.placement(ce["CIRCLE"][1])
            r = float(ce["CIRCLE"][2])
            y = _cross(z, x)

            def ang(p, c=c, x=x, y=y):
                d = _sub(p, c)
                return math.atan2(_dot(d, y), _dot(d, x))

            a1, a2 = (ang(vs[0]), ang(vs[1])) if ec[4] != Enum("F") else (ang(vs[1]), ang(vs[0]))
            sweep = (a2 - a1) % (2 * math.pi)
            if sweep < 1e-6:
                sweep = 2 * math.pi
            for k in range(73):
                a = a1 + sweep * k / 72
                p = _add(_add(c, x, r * math.cos(a)), y, r * math.sin(a))
                d = _sub(p, c0)
                bins.add(int((math.degrees(math.atan2(_dot(d, y0), _dot(d, x0))) % 360) // 10) % 36)
    return pts, bins


def _holes(rd: _Reader, pmi_by_face: dict[int, list[str]]) -> dict:
    s = rd.s
    feats: dict[tuple, dict] = {}
    for f in s.types("ADVANCED_FACE"):
        args = s.get(f)["ADVANCED_FACE"]
        se = s.get(args[2])
        if "CYLINDRICAL_SURFACE" not in se:
            continue
        key = rd.surface_key(f)
        p, z, x = rd.placement(se["CYLINDRICAL_SURFACE"][1])
        ft = feats.setdefault(key, {"faces": [], "inner": 0, "outer": 0, "pts": [], "bins": set(),
                                    "radius": float(se["CYLINDRICAL_SURFACE"][2]), "frame": (p, z, x)})
        ft["faces"].append(f)
        ft["inner" if args[3] == Enum("F") else "outer"] += 1
        pts, bins = _face_points(rd, f, ft["frame"])
        ft["pts"] += pts
        ft["bins"] |= bins
    holes, partial, bosses = [], 0, 0
    k = rd.length_factor
    for ft in feats.values():
        full = len(ft["bins"]) >= 34
        if not full:
            partial += 1
            continue
        if ft["outer"] > ft["inner"]:
            bosses += 1
            continue
        p, z, _ = ft["frame"]
        z = _canon(z)
        ts = [_dot(_sub(q, p), z) for q in ft["pts"]] or [0.0]
        top, bottom = _add(p, z, max(ts)), _add(p, z, min(ts))
        pmi = sorted({x for f in ft["faces"] for x in pmi_by_face.get(f, [])}, key=lambda v: int(v[1:]))
        holes.append({"xyz": [round(v, 4) for v in top], "bottom": [round(v, 4) for v in bottom],
                      "axis": [round(v, 6) for v in z], "diameter": round(2 * ft["radius"], 4),
                      "depth": round(max(ts) - min(ts), 4), "faces": [f"#{f}" for f in ft["faces"]], "pmi": pmi})
    holes.sort(key=lambda h: (-h["diameter"], h["xyz"]))
    groups: dict[tuple, dict] = {}
    for n, h in enumerate(holes, 1):
        h["id"] = f"H{n}"
        g = groups.setdefault((h["diameter"], tuple(round(v, 3) for v in h["axis"])),
                              {"diameter": h["diameter"], "axis": h["axis"], "count": 0, "holes": []})
        g["count"] += 1
        g["holes"].append(h["id"])
    return {"holes": holes, "hole_groups": list(groups.values()),
            "cylinders": {"features": len(feats), "holes": len(holes), "bosses": bosses, "partial": partial},
            "unit": rd.length_unit, "mm_per_unit": k}


# ---------------------------------------------------------------------------------------------------------------
# Public reader


def read_step(path: str | Path) -> dict:
    """Semantic PMI, datums, notes and holes from a STEP AP242 file."""
    step = Step.read(path)
    rd = _Reader(step)
    schema = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", step.header)
    dims = _dimensions(rd)
    tols = _tolerances(rd)
    pmi_by_face: dict[int, list[str]] = {}
    for x in dims + tols:
        for f in x["faces"]:
            pmi_by_face.setdefault(int(f[1:]), []).append(x["id"])
    holes = _holes(rd, pmi_by_face)
    return {
        "file": str(path),
        "schema": schema.group(1).split()[0] if schema else "",
        "units": {"length": rd.length_unit, "angle": "deg", "mm_per_unit": rd.length_factor},
        "dimensions": dims,
        "tolerances": tols,
        "datums": _datums(rd),
        **_texts(rd),
        **{k: v for k, v in holes.items() if k not in ("unit", "mm_per_unit")},
        "counts": {"entities": len(step.raw), "dimensions": len(dims), "tolerances": len(tols),
                   "semantic_pmi": len(dims) + len(tols)},
    }


# ---------------------------------------------------------------------------------------------------------------
# Drawing side: characteristics to comparable entries

_GDT_WORDS = [
    (r"total\s*runout|⌰|↗↗", "total_runout"),
    (r"runout|↗", "circular_runout"),
    (r"profile\s*of\s*a\s*line|line\s*profile|⌒", "line_profile"),
    (r"profile|⌓|⏒", "surface_profile"),
    (r"position|⌖|⊕", "position"),
    (r"flat|⏥|▱", "flatness"),
    (r"perpendic|⊥|⏊", "perpendicularity"),
    (r"parallel|∥|//", "parallelism"),
    (r"angular|∠", "angularity"),
    (r"straight|⏤|^\s*\u2014", "straightness"),
    (r"cylindric|⌭", "cylindricity"),
    (r"circular|round|○", "roundness"),
    (r"concentric|coaxial|◎", "concentricity"),
    (r"symmetr|⌯", "symmetry"),
]
_DIM_REQ_TYPES = {"size", "location", "depth", "radius", "angle", "chamfer", "basic", "reference"}
_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)"


def gdt_kind(text: str) -> str:
    t = (text or "").lower()
    for pat, kind in _GDT_WORDS:
        if re.search(pat, t):
            return "concentricity" if kind == "coaxiality" else kind
    return ""


def _places(text: str) -> int | None:
    ds = [len(m.group(1)) for m in re.finditer(r"\d*\.(\d+)", text or "")]
    if ds:
        return max(ds)
    return 0 if re.search(r"\d", text or "") else None


def _material(text_or_list) -> str:
    """Material condition written as Ⓜ, Ⓛ or empty, from a modifier list or printed text."""
    t = " ".join(text_or_list) if isinstance(text_or_list, list) else str(text_or_list or "")
    tu = t.upper()
    if "Ⓜ" in t or "(M)" in tu or re.search(r"\bMM[CB]\b|MAXIMUM", tu):
        return "Ⓜ"
    if "Ⓛ" in t or "(L)" in tu or re.search(r"\bLM[CB]\b|LEAST", tu):
        return "Ⓛ"
    return ""


def _datum_cell(cell: str) -> dict | None:
    """'BⓂ', 'B(M)', 'E-F', 'HⓂ-JⓂ' to a label without modifiers plus the modifiers in order."""
    t = re.sub(r"\(([ML])\)", lambda m: "Ⓜ" if m.group(1) == "M" else "Ⓛ", str(cell).strip())
    t = re.sub(r"\s*-\s*", "-", t).split(" ")[0] if t else ""
    label = re.sub(r"[^A-Z-]", "", t.upper().replace("Ⓜ", "").replace("Ⓛ", "").replace("Ⓕ", "")).strip("-")
    if not re.fullmatch(r"[A-Z]{1,2}(?:-[A-Z]{1,2})*", label):
        return None
    return {"label": label, "modifiers": "".join(c for c in t if c in "ⓂⓁ")}


def _datum_cells(datums: list[str]) -> list[dict]:
    return [x for x in (_datum_cell(d) for d in datums or []) if x]


_DATUM_TOKEN = re.compile(r"^[A-Z][ⓂⓁⒻ]*(?:-[A-Z][ⓂⓁⒻ]*)*$")


def _frame_from_text(text: str) -> tuple[str, list[dict]] | None:
    """(tolerance cell, datum cells) from a printed frame such as '⌓ 0.5 D E(M) F(M)' or '⌖|⌀0.2Ⓜ|A|B'."""
    t = re.sub(r"\(([MLFP])\)", lambda m: {"M": "Ⓜ", "L": "Ⓛ", "F": "Ⓕ", "P": "Ⓟ"}[m.group(1)], text or "")
    t = re.sub(r"^\s*\d+\s*[Xx](?=\s|[^\w.])", "", t)  # feature count, not the tolerance
    toks = t.replace("|", " ").split()
    vi = next((i for i, x in enumerate(toks) if re.search(r"\d", x)), None)
    if vi is None:
        return None
    i, value = vi + 1, [toks[vi]]
    while i < len(toks):
        # a circled modifier read without its circle (.03 F T A): a lone F, T, P or U before more datum letters
        lone = ("F", "T", "P", "U", "M", "L") if i == vi + 1 else ("F", "T", "P", "U")
        if toks[i] in lone and i + 1 < len(toks) and _DATUM_TOKEN.match(toks[i + 1]):
            value.append({"F": "Ⓕ", "T": "Ⓣ", "P": "Ⓟ", "U": "Ⓤ", "M": "Ⓜ", "L": "Ⓛ"}[toks[i]])
            i += 1
            continue
        if _DATUM_TOKEN.match(toks[i]):
            break
        if not re.fullmatch(r"[ⓂⓁⒻⓉⓊⓅ/X]+|(?:S?[⌀Ø∅])?[-+]?\d*\.?\d+[ⓂⓁⒻⓉⓊⓅ]*|MAX|<?ST>?|CZ", toks[i]):
            break
        value.append(toks[i])
        i += 1
    datums: list[dict] = []
    while i < len(toks):
        x = toks[i]
        if x in ("Ⓜ", "Ⓛ") and datums:
            datums[-1]["modifiers"] += x
        elif _DATUM_TOKEN.match(x) and x not in ("X", "ST", "CZ", "MAX", "MIN", "LD", "MD", "PD", "ACS", "UF"):
            datums.append(_datum_cell(x))
        else:
            break
        i += 1
    return " ".join(value), datums


def _parse_dim_text(t: str) -> tuple[float | None, float | None, float | None]:
    from .tolerance import parse_limits

    s = t.replace("−", "-").replace("–", "-").replace("°", "")
    s = re.sub(r"^\s*\d+\s*[Xx]\s*", "", s)
    s = re.sub(r"^[\s(\[]*(?:S⌀|SØ|SR|CR|⌀|Ø|∅|R|⌴|⌵|↧)+\s*", "", s)
    s = s.strip(" )]")
    sep = r"(?:\s*/\s*|\s+|(?=[-+]))"
    m = re.match(rf"\s*({_NUM}){sep}([-+]?{_NUM}){sep}([-+]?{_NUM})\s*$", s)
    if m and (any(x[0] in "+-" for x in m.groups()[1:]) or float(m.group(2)) == 0 or float(m.group(3)) == 0):
        # stacked unequal tolerance: 20 +0.05/-0.10, 35 0/-0.2, 35 +0.2 0
        n, a, b = (float(x) for x in m.groups())
        return n, n + min(a, b), n + max(a, b)
    m = re.match(rf"\s*({_NUM})(?:\s*/\s*|\s+)({_NUM})\s*$", s)
    if m:  # stacked limits without the line: 35.2 34.8
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and 0.5 < b / a < 2:
            return (a + b) / 2, min(a, b), max(a, b)
    n, lo, hi, _ = parse_limits(s)
    return n, lo, hi


def requirements_from_text(text: str, kind: str = "dimension") -> list:
    """Requirements for a printed callout when no model split is available (answer keys, manual items)."""
    from .types import Requirement

    t = (text or "").strip()
    if "|" in t:
        cells = [c.strip() for c in t.split("|")]
        v = re.search(_NUM, cells[1] if len(cells) > 1 else "")
        val = float(v.group(0)) if v else None
        mods = ["diameter"] if re.search("[⌀Ø∅]", cells[1] if len(cells) > 1 else "") else []
        mat = _material(cells[1] if len(cells) > 1 else "")
        mods += [mat] if mat else []
        return [Requirement(type="gdt", text=t, gdt_symbol=cells[0], nominal=val, lower=0.0, upper=val,
                            datums=cells[2:], modifiers=mods, tolerance_source="printed")]
    if kind in ("note", "surface_finish", "thread"):
        return []
    if "↧" in t and not t.lstrip().startswith("↧"):  # hole size and depth are two lines
        a, b = t.split("↧", 1)
        return requirements_from_text(a, kind) + requirements_from_text("↧" + b, kind)
    parts = re.split(r"\s+X\s+", t)
    if len(parts) == 2 and all(re.search(r"\d", x) for x in parts) and re.search(r"[±+]", parts[1]):
        # countersink or slot: '⌵⌀20 ±0.3 X 90° ±1°', '.375 ±.008 X 1.500 ±.012'
        return requirements_from_text(parts[0], kind) + requirements_from_text(parts[1], kind)
    n, lo, hi = _parse_dim_text(t)
    if n is None:
        return []
    if kind == "basic_dimension" or re.fullmatch(r"\s*(?:\d+X\s*)?[⌀Ø]?\[.*\]\s*", t):
        rtype = "basic"
    elif kind == "reference_dimension" or re.fullmatch(r"\s*(?:\d+X\s*)?\(.*\)\s*", t):
        rtype = "reference"
    elif kind == "angle" or "°" in t:
        rtype = "angle"
    elif kind == "radius" or re.match(r"^\s*(?:\d+X\s*)?(?:S|C)?R", t):
        rtype = "radius"
    elif "↧" in t:
        rtype = "depth"
    else:
        rtype = "size"
    has_tol = lo is not None and rtype not in ("basic", "reference")
    return [Requirement(type=rtype, text=t, nominal=n, lower=lo if has_tol else None, upper=hi if has_tol else None,
                        tolerance_source="printed" if has_tol else "none")]


def _shape(text: str) -> str:
    """Size symbol in front of the first number: ⌀, S⌀, R or SR ('' for a plain length or a depth)."""
    t = re.sub(r"^\s*\d+\s*[Xx]\s*", "", text or "")
    m = re.search(r"(↧|↓|S\s*[⌀Ø∅]|[⌀Ø∅]|DIA|SR|CR|R)?\s*\(?\s*[.\d]", t)
    sym = (m.group(1) or "").replace(" ", "") if m else ""
    return {"S⌀": "S⌀", "SØ": "S⌀", "S∅": "S⌀", "⌀": "⌀", "Ø": "⌀", "∅": "⌀", "DIA": "⌀", "SR": "SR",
            "CR": "R", "R": "R"}.get(sym, "")


def _not_a_dimension(text: str, fits: set[str] = frozenset()) -> bool:
    """Datum target sizes (K1) and taper or slope ratios (1.00 : 2.00) are not compared with model dimensions;
    a fit the model carries (G6) is not a datum target."""
    if re.search(r"\d\s*:\s*\d", text):
        return True
    if any(f and f in text for f in fits):
        return False
    return bool(re.search(r"(?<![\w.])[A-Z]\d{1,2}(?![\w.])", text)) and not re.search(r"[±+]", text)


def _drawing_entries(chars: list, fits: set[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    entries, skipped = [], []
    for c in chars:
        if getattr(c, "ignored", False):
            continue
        reqs = list(c.requirements) or requirements_from_text(c.text, c.kind)
        used = False
        for k, r in enumerate(reqs):
            rt = r.text or c.text
            base = {"char": c, "req": k, "count": int(c.multiplier or 1), "text": rt}
            is_frame = r.type == "gdt" or (c.kind == "gdt_frame" and r.type not in _DIM_REQ_TYPES)
            if is_frame:
                cells = [x for x in re.split(r"[|]", rt) if x.strip()]
                kinds = {gdt_kind(r.gdt_symbol), gdt_kind(cells[0] if cells else rt), gdt_kind(c.text[:3])} - {""}
                val = r.nominal if r.nominal is not None else r.upper
                if not kinds or val is None:
                    continue
                mods = " ".join(r.modifiers)
                if "|" in rt and len(cells) >= 2:  # written cell by cell: ⌖|⌀0.2Ⓜ|A|B|C
                    parsed = (cells[1], _datum_cells(cells[2:]))
                else:
                    parsed = _frame_from_text(rt)
                whole = _frame_from_text(c.text) if rt != c.text else None
                zone_text = parsed[0] if parsed else (cells[1] if "|" in rt and len(cells) > 1 else rt)
                zone_text = zone_text.split("/")[0]  # after the slash is a unit length or area
                if parsed and parsed[1]:
                    datums, material = parsed[1], _material(zone_text)
                elif r.datums:
                    datums = _datum_cells(r.datums)
                    material = _material(zone_text) or ("" if any(x["modifiers"] for x in datums)
                                                        else _material(r.modifiers))
                else:
                    datums = whole[1] if whole else []
                    material = _material(zone_text) if parsed else _material(r.modifiers)
                zone = "S⌀" if re.search(r"S\s*[⌀Ø∅]", zone_text) or "spher" in mods.lower() else \
                    ("⌀" if re.search("[⌀Ø∅]", zone_text) or "diam" in mods.lower() else "")
                entries.append({**base, "cat": "gdt", "kinds": kinds, "value": val, "datums": datums, "zone": zone,
                                "zone_text": parsed[0] if parsed else zone_text,
                                "material": material, "places": _places(zone_text.split()[0] if zone_text else "")})
                used = True
            elif r.type in _DIM_REQ_TYPES and r.nominal is not None and not _not_a_dimension(rt, fits):
                nom, lo, hi, src = r.nominal, r.lower, r.upper, r.tolerance_source
                if src == "model" or (lo is not None and hi is not None and not lo - 1e-9 <= nom <= hi + 1e-9) or \
                        (src == "none" and r.type not in ("basic", "reference")):
                    # limits not taken from the printed text: read the text again in code
                    n2, lo2, hi2 = _parse_dim_text(rt)
                    if n2 is not None and lo2 is not None:
                        nom, lo, hi, src = n2, lo2, hi2, "printed"
                        base["reparsed"] = True
                    elif src == "model":
                        lo = hi = None
                entries.append({**base, "cat": "dim", "nominal": nom, "lower": lo, "upper": hi,
                                "tol_source": src, "angle": r.type == "angle" or "°" in rt,
                                "basic": r.type == "basic" or c.kind == "basic_dimension",
                                "reference": r.type == "reference" or c.kind == "reference_dimension",
                                "shape": _shape(rt) if rt != c.text or k else _shape(c.text),
                                "places": _places(rt)})
                used = True
        if not used:
            skipped.append({"id": c.id, "kind": c.kind, "text": c.text, "sheet": c.sheet})
    return entries, skipped


# ---------------------------------------------------------------------------------------------------------------
# Matching


def _model_entries(model: dict, scale: float) -> list[dict]:
    """Model PMI as comparable entries, lengths scaled into drawing units."""
    out = []
    for d in model["dimensions"]:
        if d["nominal"] is None:
            continue
        k = 1.0 if d["angular"] else scale

        def f(v, k=k):
            return None if v is None else v * k

        out.append({"cat": "dim", "src": d, "nominal": f(d["nominal"]), "lower": f(d["lower"]),
                    "upper": f(d["upper"]), "fit": d.get("fit", ""), "angle": d["angular"],
                    "basic": d["type"] == "basic", "reference": d["type"] == "reference",
                    "shape": {"S⌀": "S⌀", "⌀": "⌀", "R": "R", "SR": "SR", "CR": "R"}.get(d.get("symbol", ""), ""),
                    "count": d["count"], "places": d["places"], "scaled": k != 1.0})
    for t in model["tolerances"]:
        if t["value"] is None:
            continue
        mat = "".join(m for m in t["modifiers"] if m in ("Ⓜ", "Ⓛ"))
        out.append({"cat": "gdt", "src": t, "kind": t["characteristic"] if t["characteristic"] != "coaxiality"
                    else "concentricity", "value": t["value"] * scale, "zone": t["zone"], "material": mat,
                    "datums": [_datum_cell(x["label"] + x["modifiers"]) or x for x in t["datums"]],
                    "count": t["count"], "places": t.get("places"), "scaled": scale != 1.0})
    return out


def _close(a, b, eps) -> bool:
    return a is not None and b is not None and abs(a - b) <= eps


def _eps(places: int | None, ref: float | None) -> float:
    """Half a unit of the last printed decimal (inclusive, drawings round halves either way)."""
    if places is None:
        return max(1e-6, abs(ref or 0) * 1e-4)
    return 0.5 * 10 ** (-places) + 1e-9


def _same(mv, dv, places: int | None, mplaces: int | None, scaled: bool) -> bool:
    """Model value equals the printed one: it rounds to it, and the model shows no digits the drawing lacks."""
    if not _close(mv, dv, _eps(places, dv)):
        return False
    if scaled or places is None or abs(mv - dv) < 1e-9:
        return True
    if mplaces is None:
        mplaces = len(f"{mv:.6f}".rstrip("0").split(".")[1])
    return mplaces <= places


def _tol_str(lo, hi, nom, places=None) -> str:
    if lo is None or hi is None:
        return "none"
    if nom is not None:
        p, m = hi - nom, lo - nom
        if abs(p + m) < 1e-9:
            return f"±{_fmt(abs(p), places)}"
        return f"+{_fmt(p, places)}/{_fmt(m, places)}"
    return f"{_fmt(hi, places)}/{_fmt(lo, places)}"


def _pair(m: dict, d: dict) -> tuple[int, float, list[dict]] | None:
    """(tier, weight, differing fields): tier 0 agrees, 1 same requirement with a different value; None if unrelated."""
    if m["cat"] != d["cat"]:
        return None
    fields: list[dict] = []
    weight = 0.0
    if m["cat"] == "dim":
        if m["angle"] != d["angle"]:
            return None
        nom_eq = _same(m["nominal"], d["nominal"], d["places"], m["places"], m["scaled"])
        m_tol = m["lower"] is not None and m["upper"] is not None
        d_tol = d["lower"] is not None and d["upper"] is not None
        if m_tol and d_tol:  # tolerance amounts, so a moved nominal is not also a tolerance change
            tol_eq = all(_same(a, b, d["places"], None, True) and abs(a - b) <= _eps(d["places"], b) / 2
                         for a, b in ((m["lower"] - m["nominal"], d["lower"] - d["nominal"]),
                                      (m["upper"] - m["nominal"], d["upper"] - d["nominal"])))
        elif m["fit"]:
            tol_eq = m["fit"].replace(" ", "").upper() in d["text"].replace(" ", "").upper()
        elif d_tol and d["tol_source"] == "general":
            tol_eq = True  # untoleranced in the model, title block tolerance on the drawing
        else:
            tol_eq = m_tol == d_tol
        flag_eq = m["basic"] == d["basic"] and m["reference"] == d["reference"]
        shape_eq = m["shape"] == d["shape"]
        rel = abs(m["nominal"] - d["nominal"]) / max(abs(d["nominal"]), 1e-9)
        if not nom_eq:
            fields.append({"field": "nominal", "model": _fmt(m["nominal"], d["places"]),
                           "drawing": _fmt(d["nominal"], d["places"])})
        if not tol_eq:
            fields.append({"field": "tolerance",
                           "model": m["fit"] or _tol_str(m["lower"], m["upper"], m["nominal"], d["places"]),
                           "drawing": _tol_str(d["lower"], d["upper"], d["nominal"], d["places"])})
        if not flag_eq:
            fields.append({"field": "basic" if m["basic"] != d["basic"] else "reference",
                           "model": "basic" if m["basic"] else ("reference" if m["reference"] else "toleranced"),
                           "drawing": "basic" if d["basic"] else ("reference" if d["reference"] else "toleranced")})
        if not shape_eq:
            fields.append({"field": "symbol", "model": m["shape"] or "none", "drawing": d["shape"] or "none"})
        weight = 3 * nom_eq + 2 * tol_eq + flag_eq + 3 * shape_eq + (m["count"] == d["count"]) + \
            (1 - min(1.0, rel / 0.2))
        if nom_eq and tol_eq and flag_eq and shape_eq:
            return 0, weight, []
        if nom_eq and (shape_eq or tol_eq):
            return 1, weight, fields
        has_tol = m_tol and d_tol and (m["upper"] - m["lower"]) > 0
        if not nom_eq and tol_eq and has_tol and flag_eq and shape_eq and rel <= 0.2:
            return 1, weight, fields
        return None
    # geometric tolerance
    if m["kind"] not in d["kinds"]:
        # a different symbol with everything else the same is still the same frame
        p = _pair({**m, "kind": next(iter(sorted(d["kinds"])))}, d)
        if p and p[0] == 0:
            return 1, p[1] - 3, [{"field": "characteristic", "model": m["kind"].replace("_", " "),
                                  "drawing": " or ".join(sorted(k.replace("_", " ") for k in d["kinds"]))}]
        return None
    val_eq = _same(m["value"], d["value"], d["places"], m["places"], m["scaled"])
    dat_eq = [x["label"] for x in m["datums"]] == [x["label"] for x in d["datums"]]
    dmod_eq = [x["modifiers"] for x in m["datums"]] == [x["modifiers"] for x in d["datums"]]
    mat_eq = m["material"] == d["material"]
    zone_eq = (m["zone"] == "") == (d["zone"] == "")
    if not val_eq:
        fields.append({"field": "tolerance value", "model": _fmt(m["value"], d["places"]),
                       "drawing": _fmt(d["value"], d["places"])})
    if not dat_eq or not dmod_eq:
        fields.append({"field": "datums", "model": "|".join(x["label"] + x["modifiers"] for x in m["datums"]) or "none",
                       "drawing": "|".join(x["label"] + x["modifiers"] for x in d["datums"]) or "none"})
    if not mat_eq:
        fields.append({"field": "material condition", "model": m["material"] or "none",
                       "drawing": d["material"] or "none"})
    if not zone_eq:
        fields.append({"field": "zone", "model": m["zone"] or "width", "drawing": d["zone"] or "width"})
    def others(mods: str) -> set[str]:
        return {x for x in "ⒻⓉⓅ" if x in mods}

    weight = 3 * val_eq + 2 * dat_eq + dmod_eq + mat_eq + zone_eq + (m["count"] == d["count"]) + \
        0.5 * (others("".join(m["src"]["modifiers"]) + ("Ⓟ" if m["src"].get("projected") else "")) ==
               others(d.get("zone_text", "")))
    if not fields:
        return 0, weight, []
    if (val_eq and len(fields) == 1) or (dat_eq and len(fields) <= 2):
        return 1, weight, fields
    return None


def _model_view(m: dict, scale: float, unit: str) -> dict:
    s = m["src"]
    out = {"id": s["id"], "type": s["type"], "text": s["text"], "count": s["count"], "unit": s["unit"]}
    if m["cat"] == "dim":
        out.update({"name": s["name"], "nominal": s["nominal"], "lower": s["lower"], "upper": s["upper"]})
    else:
        out.update({"characteristic": s["characteristic"], "value": s["value"],
                    "datums": [x["label"] + x["modifiers"] for x in s["datums"]], "modifiers": s["modifiers"]})
    if scale != 1.0 and m["cat"] == "dim" and not m["angle"]:
        out["in_drawing_units"] = {"nominal": m["nominal"], "lower": m["lower"], "upper": m["upper"], "unit": unit}
    elif scale != 1.0 and m["cat"] == "gdt":
        out["in_drawing_units"] = {"value": m["value"], "unit": unit}
    return out


def _drawing_view(d: dict) -> dict:
    c = d["char"]
    r = c.requirements[d["req"]] if d["req"] < len(c.requirements) else None
    return {"id": c.id, "number": c.number, "text": c.text, "box": list(c.box), "sheet": c.sheet, "zone": c.zone,
            "kind": c.kind, "requirement": d["req"], "requirement_text": d["text"],
            "nominal": d.get("nominal", d.get("value")), "lower": d.get("lower"), "upper": d.get("upper"),
            "count": d["count"], "tolerance_source": d.get("tol_source", r.tolerance_source if r else ""),
            "reread": bool(d.get("reparsed"))}


def _match(ms: list[dict], ds: list[dict]) -> tuple[list, list, set, set]:
    cands = []
    for i, m in enumerate(ms):
        for j, d in enumerate(ds):
            p = _pair(m, d)
            if p:
                cands.append((p[0], -p[1], i, j, p[2]))
    cands.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_m, used_d, agree, differ = set(), set(), [], []
    for tier, _, i, j, fields in cands:
        if i in used_m or j in used_d:
            continue
        used_m.add(i)
        used_d.add(j)
        (agree if tier == 0 else differ).append((i, j, fields))
    return agree, differ, used_m, used_d


SCALES = {("mm", "in"): 1 / 25.4, ("in", "mm"): 25.4}


def compare(step_path: str | Path | dict, chars: list, drawing_units: str | None = None) -> dict:
    """Match model PMI with drawing characteristics one to one and report where they disagree."""
    model = step_path if isinstance(step_path, dict) else read_step(step_path)
    chars = [c if not isinstance(c, dict) else _char(c) for c in chars]
    ds, skipped = _drawing_entries(chars, {d["fit"] for d in model["dimensions"] if d.get("fit")})
    mu = model["units"]["length"]
    options = {1.0: mu}
    for (a, b), k in SCALES.items():
        if a == mu:
            options[k] = b
    # The drawing's unit is the one under which most dimensions agree (geometric tolerance values are small and
    # say little about units); the title block's unit, as read, only breaks ties.
    runs = {}
    for k in options:
        ms_k = _model_entries(model, k)
        runs[k] = (ms_k, _match(ms_k, ds))

    def agree_in(k: float, cat: str) -> int:
        ms_k, res_k = runs[k]
        return sum(1 for i, _, _ in res_k[0] if ms_k[i]["cat"] == cat)

    scale = max(options, key=lambda k: (agree_in(k, "dim"), len(runs[k][1][0]), options[k] == drawing_units,
                                        k == 1.0))
    unit = options[scale]
    ms, (agree, differ, used_m, used_d) = runs[scale]
    warnings = []
    for cat, label in (("dim", "dimension"), ("gdt", "geometric tolerance")):
        alt = max(options, key=lambda k: agree_in(k, cat))
        if alt != scale and agree_in(alt, cat) >= max(2, 2 * agree_in(scale, cat)):
            how = (f"if the model's numbers are taken as {unit} with no conversion (the model labels them {mu})"
                   if alt == 1.0 else f"if the drawing is read in {options[alt]}")
            warnings.append(f"{label} values agree with the drawing only {how}: {agree_in(alt, cat)} agree that "
                            f"way, {agree_in(scale, cat)} in {unit}. The model's units for them look wrong.")

    gdt_scale = scale
    if warnings and any(w.startswith("geometric") for w in warnings):
        gdt_scale = max(options, key=lambda k: agree_in(k, "gdt"))
        ms = [m for m in ms if m["cat"] == "dim"] + [m for m in _model_entries(model, gdt_scale) if m["cat"] == "gdt"]
        agree, differ, used_m, used_d = _match(ms, ds)
        bad = {i for i, m in enumerate(ms) if m["cat"] == "gdt"}
        moved = [(i, j, []) for i, j, _ in agree if i in bad]
        agree = [x for x in agree if x[0] not in bad]
        differ = differ + moved
        for i, j, f in differ:
            if i in bad:
                f.insert(0, {"field": "unit", "model": f"{_fmt(ms[i]['src']['value'])} {mu}",
                             "drawing": f"{_fmt(ds[j]['value'])} {unit}"})

    def exact(a: dict, b: dict) -> bool:
        p = _pair(a, b)
        return bool(p) and p[0] == 0

    matched_m = {i: j for i, j, _ in agree}
    matched_d = {j: i for i, j, _ in agree}
    repeats_model = [i for i in range(len(ms)) if i not in used_m and any(exact(ms[i], ds[j]) for j in matched_d)]
    repeats_drawing = [j for j in range(len(ds)) if j not in used_d and any(exact(ms[i], ds[j]) for i in matched_m)]
    def view_m(i: int) -> dict:
        return _model_view(ms[i], gdt_scale if ms[i]["cat"] == "gdt" else scale, unit)

    out = {
        "step": model.get("file", ""),
        "model_units": mu,
        "drawing_units": unit,
        "scale": scale,
        "matched": [{"model": view_m(i), "drawing": _drawing_view(ds[j]),
                     **({"count": {"model": ms[i]["count"], "drawing": ds[j]["count"]}}
                        if ms[i]["count"] != ds[j]["count"] else {})} for i, j, _ in agree],
        "differs": [{"model": view_m(i), "drawing": _drawing_view(ds[j]), "fields": f} for i, j, f in differ],
        "model_only": [view_m(i) for i in range(len(ms)) if i not in used_m and i not in repeats_model],
        "drawing_only": [_drawing_view(ds[j]) for j in range(len(ds)) if j not in used_d and j not in repeats_drawing],
        "repeats": {"model": [view_m(i) for i in repeats_model],
                    "drawing": [_drawing_view(ds[j]) for j in repeats_drawing]},
        "warnings": warnings,
        "model_without_value": [{"id": d["id"], "type": d["type"], "name": d["name"]}
                                for d in model["dimensions"] if d["nominal"] is None],
        "not_compared": skipped,
    }
    out["summary"] = {"model_pmi": len(ms), "drawing_requirements": len(ds), "matched": len(out["matched"]),
                      "differs": len(out["differs"]), "model_only": len(out["model_only"]),
                      "drawing_only": len(out["drawing_only"]), "repeats_model": len(repeats_model),
                      "repeats_drawing": len(repeats_drawing), "not_compared": len(skipped)}
    return out


def _char(d: dict):
    from .types import Characteristic

    return Characteristic(**d)


# ---------------------------------------------------------------------------------------------------------------
# Loading drawing characteristics and the command line


def to_json(obj) -> str:
    """JSON with symbols kept readable; a long dash (a misread straightness symbol) is written escaped."""
    import json

    return json.dumps(obj, indent=1, ensure_ascii=False).replace("\u2014", "\\u2014")


def load_chars(src: str | Path) -> tuple[list, str | None]:
    """Characteristics and drawing units from a job dir (job.json), a CLI out dir (result.json) or that JSON file."""
    import json

    from .types import Characteristic

    p = Path(src)
    if p.is_dir():
        p = next((p / n for n in ("job.json", "result.json") if (p / n).exists()), p / "job.json")
    data = json.loads(p.read_text())
    chars, units = [], None
    for s in data.get("sheets", []):
        units = units or (s.get("info", {}).get("units") if s.get("info", {}).get("units") in ("in", "mm") else None)
        chars += [Characteristic(**c) for c in s.get("chars", [])]
    return chars, units


async def read_pdf(path: str | Path, passes: int = 0) -> tuple[list, str | None]:
    """Run the pipeline on a drawing (first read by default; cached calls cost nothing)."""
    from . import ingest
    from .model_client import ModelConfig, VisionClient
    from .pipeline import process_sheet

    client = VisionClient(ModelConfig())
    chars, units, general = [], None, None
    for i, sh in enumerate(ingest.load(path)):
        r = await process_sheet(client, sh, 0, audit_passes=passes, general_fallback=general)
        general = general or list(r.info.general_tolerances)
        units = units or (r.info.units if r.info.units in ("in", "mm") else None)
        for n, c in enumerate(r.chars):
            c.id, c.sheet = f"s{i}c{n}", i
        chars += r.chars
    print(f"model calls: {client.usage.as_dict()}")
    return chars, units


def main(argv: list[str] | None = None) -> None:
    import argparse
    import asyncio
    import json

    ap = argparse.ArgumentParser(prog="python -m whiteout.model_check",
                                 description="Compare a STEP AP242 model's PMI with a drawing's characteristics.")
    ap.add_argument("step")
    ap.add_argument("drawing", nargs="?", help="job dir, result.json, or a drawing PDF/image to read")
    ap.add_argument("--units", choices=["in", "mm"], help="drawing units if the title block does not say")
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args(argv)
    if not a.drawing:
        r = read_step(a.step)
        print(json.dumps({k: r[k] for k in ("schema", "units", "counts", "datums", "cylinders")}, indent=1))
        for x in r["dimensions"] + r["tolerances"]:
            print(f"{x['id']:>8}  {x['type']:<10} {x['text']}")
        return
    p = Path(a.drawing)
    if p.is_dir() or p.suffix.lower() == ".json":
        chars, units = load_chars(p)
    else:
        chars, units = asyncio.run(read_pdf(p))
    res = compare(a.step, chars, a.units or units)
    if a.json:
        Path(a.json).write_text(to_json(res))
    s = res["summary"]
    print(f"model {res['model_units']}, drawing {res['drawing_units']}: {s['model_pmi']} model PMI, "
          f"{s['drawing_requirements']} drawing requirements")
    print(f"matched {s['matched']}, differ {s['differs']}, model only {s['model_only']}, "
          f"drawing only {s['drawing_only']}")
    for w in res["warnings"]:
        print(f"  WARNING  {w}")
    for x in res["differs"]:
        f = "; ".join(f"{y['field']}: model {y['model']}, drawing {y['drawing']}" for y in x["fields"])
        print(f"  DIFFERS  {x['model']['id']:>8} {x['model']['text']:<28} | sheet {x['drawing']['sheet'] + 1} "
              f"{x['drawing']['id']} {x['drawing']['text']!r}: {f}")
    for x in res["model_only"]:
        print(f"  MODEL    {x['id']:>8} {x['text']}")
    for x in res["drawing_only"]:
        part = f" (in {x['text']!r})" if x["requirement_text"] != x["text"] else ""
        print(f"  DRAWING  sheet {x['sheet'] + 1} {x['id']} {x['requirement_text']!r}{part}")


if __name__ == "__main__":
    main()
