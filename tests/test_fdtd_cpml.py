"""The CPML's reflection, measured on the solver's own layer.

``scripts/fdtd_cpml_reflection.py`` runs the exact 1-D reduction of the 3-D
update for a plane wave at an angle, with (b, c) from ``fdtd._pml_profiles``.
A line is a few hundred cells, so these are tier-1 runs.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fdtd_cpml_reflection", ROOT / "scripts" / "fdtd_cpml_reflection.py")
instrument = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(instrument)


def test_the_instrument_sees_a_bare_wall():
    """Positive control: with no layer the wall sends the whole pulse back.
    Without this, a broken instrument would read every layer as perfect."""
    assert instrument.reflection(0, 0.0, 20.0) > 0.5


def test_a_ten_cell_layer_barely_reflects_at_normal_incidence():
    """E and H read ONE sigma(x) at their own positions. When E's ramp was
    measured over node thicknesses and H's over cells, the two halves of the
    update saw different layers and 10 cells reflected 2.7e-2; one profile
    reads 3.0e-4."""
    assert instrument.reflection(10, 0.0, 20.0) < 1e-3


def test_a_ten_cell_layer_holds_near_grazing():
    """At 75 degrees the old grading (order 3, R = 1e-6) reflected 1.4e-2;
    order 2 at R = 1e-12 reads 4.4e-4 (20 cells per wavelength)."""
    assert instrument.reflection(10, 75.0, 20.0) < 2e-3
