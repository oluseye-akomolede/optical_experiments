"""The time-domain (``td``) form: parse, lay out and cap-check one request.

``pc_form``'s sibling for the FDTD tool. Everything the worker would refuse is
refused HERE, before a row exists, with the message the solver itself writes:
the grid, source and probe placement come from ``fdtd_layout.plan`` (the
worker's own layout rule without the voxeliser), the source from
``fdtd.Source``, and the limits from ``fdtd.check_limits`` on
``fdtd_layout.estimate``. No second copy of any of those rules lives here.

Every refusal leaves as ``TDRequestError``, including the ones nobody wrote a
``raise`` for (a mesh library failing on a strange file, a TypeError from a
value that parsed but makes no sense). Counting raise sites is not enough, so
the geometry step converts ANY exception into a refusal that names it; the
route turns ``TDRequestError`` into a flash, never a 500.

Lengths on the form (dx, the source position, the probes) are in the form's
own length unit and are measured from the object's minimum corner, the frame
``fdtd_layout`` places them in. Frequencies are in THz and durations in fs;
the stored request is SI (Hz, s, m), the worker's ``REQUEST_KEYS``.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

MODE = "td"

BOUNDARIES = ("metal", "open")
SOURCE_KINDS = ("point", "plane")
PROFILES = ("cw", "pulse", "drawn")
POLARISATIONS = ("x", "y", "z")
DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")
LENGTH_UNITS = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3}

#: The worker's REQUEST_KEYS, in its order. tests/test_td_form.py pins this
#: against worker/td_solve.py, so the two lists cannot drift.
REQUEST_KEYS = ("boundary", "source_kind", "profile", "frequency_hz",
                "bandwidth", "position", "direction", "polarisation",
                "samples", "sample_duration_s", "duration_s", "probes", "dx_m")

#: Field-name prefix for each part's material, shared with the cavity form.
MATERIAL_PREFIX = "part_index."

#: The most energy a drawn waveform may put above the grid's resolvable
#: frequency (the spec's 1%).
MAX_DRAWN_ENERGY_ABOVE = 0.01

THZ = 1e12
FS = 1e-15


class TDRequestError(Exception):
    """A readable refusal of a td request; the message is shown as is."""


def _solver():
    """``fdtd``, ``fdtd_layout`` and ``mse_parts``, reached the way
    ``pc_form`` reaches ``pc_geometry``: a late import with the solver
    directory on the path for a checkout (in the image they sit in /opt/app).
    All three are numpy/trimesh only; none imports scipy or cupy at load."""
    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import fdtd  # noqa: PLC0415 - deliberately late, so tier-1 tests stay cheap
    import fdtd_layout  # noqa: PLC0415
    import mse_parts  # noqa: PLC0415

    return fdtd, fdtd_layout, mse_parts


def _field(form, name: str) -> str:
    return (form.get(name) or "").strip()


def _choice(form, name: str, label: str, choices) -> str:
    value = _field(form, name)
    if value not in choices:
        raise TDRequestError(f"Please choose the {label}: one of "
                             f"{', '.join(choices)}.")
    return value


def _number(raw: str, label: str, *, positive=True) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise TDRequestError(f"The {label} must be a number, got {raw!r}.") from None
    if not math.isfinite(value):
        raise TDRequestError(f"The {label} must be a finite number, got {raw!r}.")
    if positive and not value > 0:
        raise TDRequestError(f"The {label} must be greater than zero, got {raw!r}.")
    return value


def _numbers(raw: str, label: str) -> list:
    words = [w for w in re.split(r"[\s,;]+", raw) if w]
    return [_number(w, label, positive=False) for w in words]


def _point(raw: str, label: str, scale: float) -> list:
    values = _numbers(raw, label)
    if len(values) != 3:
        raise TDRequestError(f"The {label} is three numbers, x y z; got {raw!r}.")
    return [v * scale for v in values]


def _probes(raw: str, scale: float) -> list:
    probes = []
    for line in [ln for ln in raw.splitlines() if ln.strip()]:
        words = [w for w in re.split(r"[\s,;]+", line.strip()) if w]
        if len(words) != 4 or words[3] not in POLARISATIONS:
            raise TDRequestError(
                f"Each probe line is x y z and a component (x, y or z), "
                f"got {line.strip()!r}.")
        probes.append(_point(" ".join(words[:3]), "probe position", scale)
                      + [words[3]])
    if not probes:
        raise TDRequestError("Add at least one probe: a line of x y z and the "
                             "field component it records.")
    return probes


def parse_request(form) -> dict:
    """The keyed request (SI units) from the form's fields, or a refusal.

    Only the fields the chosen source uses are read; every other key holds
    null, as the worker's contract requires.
    """
    unit = _choice(form, "td_unit", "length unit", tuple(LENGTH_UNITS))
    scale = LENGTH_UNITS[unit]
    req = dict.fromkeys(REQUEST_KEYS)
    req["boundary"] = _choice(form, "td_boundary", "boundary", BOUNDARIES)
    req["source_kind"] = _choice(form, "td_source_kind", "source kind",
                                 SOURCE_KINDS)
    req["profile"] = _choice(form, "td_profile", "time profile", PROFILES)
    req["polarisation"] = _choice(form, "td_polarisation", "polarisation",
                                  POLARISATIONS)
    req["frequency_hz"] = _number(_field(form, "td_frequency_thz"),
                                  "frequency (THz)") * THZ
    req["duration_s"] = _number(_field(form, "td_duration_fs"),
                                "run duration (fs)") * FS
    req["dx_m"] = _number(_field(form, "td_dx"), f"cell size dx ({unit})") * scale
    if req["profile"] == "pulse":
        req["bandwidth"] = _number(_field(form, "td_bandwidth_thz"),
                                   "pulse bandwidth (THz)") * THZ
    if req["profile"] == "drawn":
        req["samples"] = _numbers(_field(form, "td_samples"), "drawn sample")
        req["sample_duration_s"] = _number(_field(form, "td_sample_duration_fs"),
                                           "drawn waveform's duration (fs)") * FS
    if req["source_kind"] == "point":
        req["position"] = _point(_field(form, "td_position"),
                                 f"source position ({unit})", scale)
    else:
        req["direction"] = _choice(form, "td_direction", "direction",
                                   DIRECTIONS)
    req["probes"] = _probes(form.get("td_probes") or "", scale)
    return req


def named_materials(fdtd=None) -> tuple:
    """The materials a part may name instead of an index: ``pec`` and the
    solver's dispersive metals. The page's per-part pickers list exactly
    these, and ``parse_materials`` accepts exactly these."""
    if fdtd is None:
        fdtd = _solver()[0]
    return ("pec",) + tuple(fdtd.METALS)


def parse_materials(form, fdtd):
    """``{part name: n or metal name or 'pec'}`` from the part fields, or None
    when the form sent none (the one-part path, which uses ``n_index``)."""
    fields = {k[len(MATERIAL_PREFIX):]: (v or "").strip()
              for k, v in form.items() if k.startswith(MATERIAL_PREFIX)}
    if not fields:
        return None
    named = named_materials(fdtd)
    table = {}
    for part, raw in fields.items():
        if raw in named:
            table[part] = raw
            continue
        try:
            n = float(raw)
        except ValueError:
            raise TDRequestError(
                f"The material for part {part!r} must be a refractive index "
                f"(1 or more) or one of {', '.join(named)}; got {raw!r}.") from None
        if not (math.isfinite(n) and n >= 1.0):
            raise TDRequestError(f"The refractive index for part {part!r} must "
                                 f"be 1 or more, got {raw!r}.")
        table[part] = n
    return table


def _n_max(table, fdtd, frequency: float) -> float:
    """The largest |n| in the scene: dielectric n, and each metal's
    |n| = sqrt(|eps|) at the run's reference frequency (the solver's backstop
    knows only the dielectric part; this is the check that knows the metals)."""
    n = 1.0
    for kind in table.values():
        if isinstance(kind, str):
            if kind in fdtd.METALS:
                n = max(n, math.sqrt(abs(complex(fdtd.permittivity(kind, frequency)))))
        else:
            n = max(n, float(kind))
    return n


def plan_request(form, data: bytes, units_scale_m: float) -> dict:
    """Parse, lay out and cap-check one td request against its upload.

    ``units_scale_m`` is the upload's metres per model unit times the owner's
    scale (the worker's ``length_scale``). Returns ``tds`` (the stored request),
    ``materials`` (the table, or None), ``estimate`` and ``n_index``. Raises
    ``TDRequestError`` for every refusal.
    """
    fdtd, fdtd_layout, mse_parts = _solver()
    request = parse_request(form)
    materials = parse_materials(form, fdtd)
    n_index = _number(_field(form, "td_n_index") or "1", "refractive index")
    if n_index < 1.0:
        raise TDRequestError(f"The refractive index must be 1 or more, got "
                             f"{n_index:g}.")
    try:
        names, parts = mse_parts.split_obj_parts(data)
        if materials is None and len(names) > 1:
            # Never one material for every part: the layout would refuse this
            # anyway, but in words meant for a developer.
            raise TDRequestError(
                f"This object has {len(names)} parts "
                f"({', '.join(map(repr, names))}), and a time-domain run needs "
                f"a material for each one. Choose one for every part in the "
                f"list the page shows once it has read the file.")
        if materials is not None:
            missing = [n for n in names if n not in materials]
            unknown = [n for n in materials if n not in names]
            if missing or unknown:
                raise TDRequestError(
                    "The materials do not match this upload's parts"
                    + (f"; no material for {', '.join(map(repr, missing))}"
                       if missing else "")
                    + (f"; not in the upload: {', '.join(map(repr, unknown))}"
                       if unknown else "")
                    + f". Its parts are {', '.join(map(repr, names))}.")
        planned = fdtd_layout.plan(names, parts, request,
                                   length_scale=units_scale_m,
                                   materials=materials, n_index=n_index)
        fdtd_layout.refuse_conductor_points(planned, parts,
                                            length_scale=units_scale_m)
        est = fdtd_layout.estimate(planned, request)
        fdtd.check_limits(est)
        if request["source_kind"] == "plane":
            place = {"kind": "plane", "direction": request["direction"],
                     "box": planned["box"]}
        else:
            place = {"cell": planned["cell"]}
        fdtd.Source(frequency=request["frequency_hz"], profile=request["profile"],
                    bandwidth=request["bandwidth"],
                    component=request["polarisation"],
                    samples=request["samples"],
                    duration=request["sample_duration_s"], **place)
        if request["profile"] == "drawn":
            _check_drawn(request, planned, est, fdtd)
    except TDRequestError:
        raise
    except fdtd.FDTDError as exc:
        raise TDRequestError(_sentence(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        raise TDRequestError(
            f"This upload could not be laid out for a time-domain run: "
            f"{_sentence(str(exc) or type(exc).__name__)}") from exc
    return {"tds": request, "materials": materials, "estimate": est,
            "n_index": n_index}


def _check_drawn(request, planned, est, fdtd) -> None:
    dt, duration = float(est["dt"]), float(request["sample_duration_s"])
    if duration < 2 * dt:
        raise TDRequestError(
            f"The drawn waveform lasts {duration / FS:.3g} fs, under two time "
            f"steps ({2 * dt / FS:.3g} fs): the grid would read nothing of it. "
            f"Make it longer.")
    f_res = fdtd.resolvable_frequency(
        planned["spacings"],
        _n_max(planned["table"], fdtd, request["frequency_hz"]))
    above = fdtd.drawn_energy_above(request["samples"], duration, dt, f_res)
    if above > MAX_DRAWN_ENERGY_ABOVE:
        raise TDRequestError(
            f"{above:.1%} of the drawn waveform's energy lies above "
            f"{f_res / THZ:.4g} THz, the highest frequency this grid resolves "
            f"(the limit is {MAX_DRAWN_ENERGY_ABOVE:.0%}). Smooth the drawing's "
            f"sharp edges, or make dx smaller.")


def _sentence(text: str) -> str:
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def label(planned: dict, filename: str) -> str:
    """The job's name in the list: the file plus what was asked of it."""
    req = planned["tds"]
    return (f"{filename} (time domain: {req['profile']} "
            f"{req['source_kind']} source, {req['frequency_hz'] / THZ:g} THz)")
