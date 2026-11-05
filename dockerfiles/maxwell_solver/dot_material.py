"""Cited material parameters for the paper-1 InAsP/InP nanowire dot (task 1.9).

OWNER DECISION (option 1): the dot's values come from the literature, every one
cited with its page/table, shown as ASSUMED, and where sources disagree the
range is given rather than a single number.

The distinction this module is built to preserve
------------------------------------------------
The InP **host** constants are measured and sourced (Ioffe/NSM). The InAs_xP_1-x
**alloy** constants for the dot are NOT sourced: no bowing formula and no alloy
effective masses were retrieved. So the dot values below are produced by
linear interpolation between the two sourced endpoints, and that interpolation
is OUR MODELLING ASSUMPTION. It is not a published rule, and it does not become
one by being written in a numpy expression.

Every constant carries ``verified`` and ``basis``; anything derived from the
unsourced interpolation carries ``assumed=True``. See
``scripts/paper1_material_sources.md`` for the retrieval record.

Sources
-------
InP host, 300 K, from the Ioffe/NSM compilation:
    eps_s  = 12.5   InP/basic.html, "Basic Parameters at 300 K",
                    row "Dielectric constant (static)".
                    Underlying: Shmidt, Handbook Series on Semiconductor
                    Parameters vol. 1, pp. 169-190 (World Scientific, 1996).
    eps_inf = 9.61  same table, row "Dielectric constant (high frequency)".
                    Cross-check: n = 3.1 (InP/optic.html, "Infrared refractive
                    index"), n^2 = 9.61. Self-consistent.
    m_e*   = 0.08   InP/basic.html "Effective electron mass"; bandstr.html
                    "Effective Masses", Gamma row m_Gamma = 0.08 m0.
    m_hh*  = 0.6    InP/basic.html "Effective hole masses m_h 0.6 m0";
                    bandstr.html "Holes: Heavy m_h = 0.6 m0".

InAs endpoint, same compilation:
    m_e*  = 0.023   InAs/basic.html "Effective electron mass"
    m_hh* = 0.41    InAs/basic.html "Effective hole masses m_h 0.41 m0"

Dot geometry (measured, from TEM):
    height h ~ 4 nm, diameter D ~ 18 nm, x = 0.20
        Cygorek, Korkusinski & Hawrylak, Phys. Rev. B 101, 075307 (2020),
        Sec. V.1: "a diameter of D ~ 18 nm, a height of h ~ 4 nm, and an As
        concentration of x = 20%". NOTE: that is a theory paper describing a
        grown dot, citing Dalacu et al., Nano Lett. 12, 5919 (2012) for the
        growth, which we did not read.
    height range measured 3-7 nm on the same 18 nm diameter
        Haffouz et al., Nano Lett. 18, 3047 (2018), TEM analysis, discussion of
        Fig. 5: "the dot thickness from h ~ 3 nm to h ~ 7 nm ... h/D_dot = 7/18".

What is NOT sourced, and therefore marked ASSUMED:
    * Any InAs_xP_1-x band-gap bowing. Consequence: this module does not
      report a dot band gap at all, and nothing downstream may quote one.
      A bounded retrieval on 2026-11-05 for the usual compilation
      (Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001)) came
      back EMPTY: the publisher landing page returns HTTP 403 to automated
      retrieval, and searches for the InP/InAs gap table and the InAsP bowing
      parameter returned no usable hits. Nothing here was filled in from
      memory. To close this, someone needs table-level access to that paper
      (or an equivalent compilation) and must cite section and table for
      each of: the InP gap, the InAs gap, the InP/InAs band-offset split, and
      the InAsP bowing parameter.
    * Any InAs_xP_1-x effective mass. The values below are a linear
      interpolation in x between the InP and InAs endpoints. No source states
      this rule.
    * Any wurtzite value. The Ioffe numbers are zincblende; the nanowire is
      wurtzite, and the strain shift is not accounted for. See
      ``WURZITE_CAVEAT``.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Constant:
    """One number, with where it came from and whether we assume it."""

    name: str
    value: float
    unit: str
    basis: str
    """The citation, including the table/page/row. Never a bare title."""
    verified: bool
    """True only if read directly in a retrieved source."""
    assumed: bool = False
    """True if this number is derived by an unsourced rule of ours."""
    note: str = ""

    def cite(self) -> str:
        tag = "ASSUMED" if self.assumed else ("cited" if self.verified else "UNVERIFIED")
        return f"{self.name} = {self.value:g} {self.unit} [{tag}] {self.basis}"


_NOUR_SRC = "not found in any retrieved source"


# ---------------------------------------------------------------- InP host

INP_EPS_S = Constant(
    name="InP eps_s",
    value=12.5,
    unit="1",
    basis=(
        "Ioffe/NSM InP/basic.html, 'Basic Parameters at 300 K', row "
        "'Dielectric constant (static)'. Original: Shmidt, Handbook Series on "
        "Semiconductor Parameters vol.1, pp.169-190 (1996)."
    ),
    verified=True,
    note="A web-search summary suggested 12.8-12.9; that is not a retrieved "
    "primary source and is deliberately not used.",
)

INP_EPS_INF = Constant(
    name="InP eps_inf",
    value=9.61,
    unit="1",
    basis=(
        "Ioffe/NSM InP/basic.html, row 'Dielectric constant (high frequency)'. "
        "Cross-check: n = 3.1 (InP/optic.html, 'Infrared refractive index'), "
        "n^2 = 9.61."
    ),
    verified=True,
)

INP_M_E = Constant(
    name="InP m_e*",
    value=0.08,
    unit="m0",
    basis=(
        "Ioffe/NSM InP/basic.html 'Effective electron mass'; bandstr.html "
        "'Effective Masses', Gamma-valley row m_Gamma = 0.08 m0."
    ),
    verified=True,
)

INP_M_HH = Constant(
    name="InP m_hh*",
    value=0.6,
    unit="m0",
    basis=(
        "Ioffe/NSM InP/basic.html 'Effective hole masses m_h 0.6 m0'; "
        "bandstr.html 'Holes: Heavy m_h = 0.6 m0'."
    ),
    verified=True,
    note="Ioffe labels this 'Heavy' but does not separate m_hh from the other "
    "zincblende hole masses.",
)

# ---------------------------------------------------------------- InAs endpoint

INAS_M_E = Constant(
    name="InAs m_e*",
    value=0.023,
    unit="m0",
    basis="Ioffe/NSM InAs/basic.html 'Effective electron mass'.",
    verified=True,
)

INAS_M_HH = Constant(
    name="InAs m_hh*",
    value=0.41,
    unit="m0",
    basis=(
        "Ioffe/NSM InAs/basic.html 'Effective hole masses m_h 0.41 m0'; "
        "bandstr.html 'Heavy m_h = 0.41 m0'."
    ),
    verified=True,
)

# ---------------------------------------------------------------- geometry

DOT_HEIGHT_NM = Constant(
    name="dot height",
    value=4.0,
    unit="nm",
    basis=(
        "Cygorek, Korkusinski & Hawrylak, Phys. Rev. B 101, 075307 (2020), "
        "Sec. V.1: 'a height of h ~ 4 nm'. Theory paper describing a grown dot."
    ),
    verified=True,
    note="Measured range 3-7 nm on an 18 nm diameter: Haffouz et al., Nano "
    "Lett. 18, 3047 (2018), TEM, discussion of Fig. 5.",
)

DOT_DIAMETER_NM = Constant(
    name="dot diameter",
    value=18.0,
    unit="nm",
    basis=(
        "Cygorek et al., Phys. Rev. B 101, 075307 (2020), Sec. V.1: "
        "'a diameter of D ~ 18 nm'."
    ),
    verified=True,
)

DOT_X = Constant(
    name="As fraction x",
    value=0.20,
    unit="1",
    basis=(
        "Cygorek et al., Phys. Rev. B 101, 075307 (2020), Sec. V.1: "
        "'an As concentration of x = 20%'."
    ),
    verified=True,
)

WURZITE_CAVEAT = (
    "The Ioffe constants above are ZINCBLENDE. The nanowire dot is wurtzite and "
    "strained, and neither the wurtzite values nor the strain shift were "
    "retrieved. Using the zincblende numbers is a further ASSUMED approximation "
    "and any downstream result inherits it."
)


def interpolate(name: str, a: Constant, b: Constant, x: float) -> Constant:
    """Linear interpolation in ``x`` between two sourced endpoints.

    The RULE is ours. No retrieved source gives an InAsS_xP_1-x interpolation
    rule for the effective mass, so the returned Constant is marked assumed.
    It is deliberately NOT usable for a band gap: no bowing parameter was
    retrieved, so the absence of that half is not a licence to guess it.
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x={x} outside [0, 1]")
    value = a.value * (1.0 - x) + b.value * x
    return Constant(
        name=name,
        value=value,
        unit=a.unit,
        basis=(
            f"LINEAR INTERPOLATION in x={x:g} between {a.name}={a.value:g} and "
            f"{b.name}={b.value:g}. The endpoints are cited; the interpolation "
            f"rule is OUR ASSUMPTION, not a published relation."
        ),
        verified=False,
        assumed=True,
        note=f"endpoints from: {a.basis} | {b.basis}",
    )


def paper1_dot_at(x: float = DOT_X.value) -> tuple[Constant, Constant]:
    """The effective masses of the InAs_xP_1-x dot at ``x``, marked ASSUMED."""
    m_e = interpolate("InAsP m_e*", INP_M_E, INAS_M_E, x)
    m_hh = interpolate("InAsP m_hh*", INP_M_HH, INAS_M_HH, x)
    return m_e, m_hh


def report() -> str:
    """Human-readable dump: every constant with its citation and status."""
    rows = [
        INP_EPS_S, INP_EPS_INF, INP_M_E, INP_M_HH,
        INAS_M_E, INAS_M_HH,
        DOT_HEIGHT_NM, DOT_DIAMETER_NM, DOT_X,
    ]
    out = ["Paper-1 InAsP/InP dot material constants", "=" * 45, ""]
    for c in rows:
        out.append("  " + c.cite())
        if c.note:
            out.append("      note: " + c.note)
    m_e, m_hh = paper1_dot_at()
    out += ["", "Dot values at x = %g (derived):" % DOT_X.value, ""]
    for c in (m_e, m_hh):
        out.append("  " + c.cite())
    out += ["", "Caveat:", "  " + WURZITE_CAVEAT]
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
