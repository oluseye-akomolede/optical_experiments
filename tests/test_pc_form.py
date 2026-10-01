"""The pc form: what it refuses, WHY, and the guard that keeps the solve out.

Three kinds of test here, and they fail for different reasons.

1. **The refusals.** A photonic-crystal request the cluster cannot afford is
   refused in the web process, with the limit named, BEFORE a row is written --
   the plan's "could-not-ask must not look like nothing-there" guard. The two
   caps are independent and each is asserted on its OWN message, because the
   summed-count trap is exactly this shape: a test that only asserts "it raised"
   stays green when either cap is deleted.
2. **The guard.** "No solve in the web process, ever." Asserted as an
   import-graph property with a mutation that proves the test can see a
   violation, not as a grep for a function name. See
   ``test_no_web_module_reaches_the_photonic_crystal_solve``.
3. **The collision.** This module is named ``pc_form`` and not ``pc`` because the
   solver's module is ``pc``. The symptom of getting that wrong is an
   ``AttributeError`` naming a function the reader can see defined in the file
   in front of them, so it is pinned directly in
   ``test_the_web_module_does_not_shadow_the_solvers_pc``.

Nothing here solves a band structure: the ``plan_request`` path builds a unit
cell (a Delaunay, ~100 ms at the production size) and counts it. That is tier-1
work and runs on the VM under the capped command.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SOLVER = ROOT / "dockerfiles" / "maxwell_solver"

sys.path.insert(0, str(WEB))

import pc_form  # noqa: E402


#: A request that passes every check, so each refusal test can change exactly
#: one field and know it is looking at that field's refusal. It is also the
#: SHAPE the form sends: every value a string, because a form posts strings and
#: an ``int`` that arrives as ``"12"`` must not be a special case.
GOOD = {
    "lattice": "square",
    "r_over_a": "0.2",
    "n_rod": "3.5",
    "n_bg": "1.0",
    "n_bands": "12",
    "divisions": "16",
    "polarisation": "TM",
}


def _with(**overrides) -> dict:
    return {**GOOD, **overrides}


# ---------------------------------------------------------------------------
# The positive control: the request this file is written around actually works
# ---------------------------------------------------------------------------

def test_a_good_request_is_accepted_and_reports_its_own_size():
    """The control. Without it every refusal below could pass vacuously.

    A form module that refused EVERYTHING would satisfy every ``raises`` in this
    file. This asserts the accept path end to end, including the derived counts
    the caps are compared against.
    """
    planned = pc_form.plan_request(GOOD)
    assert planned["lattice"] == "square"
    assert planned["nodes"] > 0
    # No triangle count: the web side does not triangulate, so it has none to
    # report and the row carries none. The count that IS here is the solver's
    # node set -- see test_cell_size_matches_the_solvers_own_mesh.
    assert "triangles" not in planned
    assert planned["k_points"] == 30
    assert pc_form.request_json(planned) == {
        "lattice": "square", "r_over_a": 0.2, "n_rod": 3.5, "n_bg": 1.0,
        "n_bands": 12, "divisions": 16, "polarisation": "TM",
    }


def test_the_reported_counts_are_the_meshers_own():
    """The caps' operand must be the mesh the worker will build.

    Imported from the solver, not re-derived. If this module had its own
    arithmetic the number the owner is refused on could differ from the number
    the worker meets -- the two-opinions failure. Asserted against a direct call
    to ``pc.unit_cell_mesh`` so a future "optimisation" that stops building the
    cell is caught here.
    """
    sys.path.insert(0, str(SOLVER))
    import pc  # noqa: PLC0415 - the SOLVER module: the comparison is the point

    points, _triangles, _eps, _pairs = pc.unit_cell_mesh(
        "square", 16, r_over_a=0.2, n_rod=3.5, n_bg=1.0)
    node_count = pc_form.cell_size({
        "lattice": "square", "divisions": 16, "r_over_a": 0.2,
        "n_rod": 3.5, "n_bg": 1.0})
    # The web count against the SOLVER's own mesh, which is the whole contract:
    # the cap the owner is refused on must be the mesh the worker builds. The
    # web side now reaches it without importing the solver at all, so this is a
    # crossing between two modules rather than a restatement of one.
    assert node_count == len(points)


# ---------------------------------------------------------------------------
# The caps, each on its own message
# ---------------------------------------------------------------------------

def test_the_k_point_cap_refuses_and_names_its_own_number():
    """Driving the value, not the constant: ask for a path over 30 points.

    ``k_point_count`` is derived from the path, so the way to exceed the cap is
    to ask for more points per segment. The refusal must name 30 AND say which
    knob to turn -- an owner told only "too big" cannot act.
    """
    with pytest.raises(pc_form.PCRequestError) as excinfo:
        pc_form.check_caps({"lattice": "square"}, nodes=100, k_points=31)
    message = str(excinfo.value)
    assert "31" in message, "the refusal must state the number asked for"
    assert str(pc_form.MAX_K_POINTS) in message, "and the limit it broke"
    assert "segment" in message, "and how to get under it"


def test_the_node_cap_refuses_and_names_its_own_number():
    """The other cap, and it must NOT be satisfied by the k-point refusal.

    If the two refusals were one sentence, or one test, deleting either cap
    would leave the suite green -- the summed count over a pair. So this one
    drives ``nodes`` over its limit while ``k_points`` is well inside its own.
    """
    with pytest.raises(pc_form.PCRequestError) as excinfo:
        pc_form.check_caps({"lattice": "square"},
                           nodes=pc_form.MAX_NODES + 1, k_points=30)
    message = str(excinfo.value)
    assert str(pc_form.MAX_NODES + 1) in message
    assert str(pc_form.MAX_NODES) in message
    assert "division" in message, "the node cap's lever is the divisions count"


def test_each_cap_is_individually_satisfiable():
    """Both limits approached but not crossed must be ACCEPTED.

    The mirror of the two refusals: they are caps, not roundings. Without this,
    a ``>=`` where a ``>`` belongs stays green -- every refusal test still
    passes, because they are all over the line.
    """
    pc_form.check_caps({"lattice": "square"}, nodes=pc_form.MAX_NODES,
                       k_points=pc_form.MAX_K_POINTS)


def test_the_k_point_cap_never_sees_a_path_over_its_own_bound():
    """30 is the whole vocabulary of paths, so the k-point cap is not reachable.

    Stated as a test because it is a fact about the product, not a bug: the path
    comes from the lattice and is not an owner input, so the ONLY k-point count
    the form can produce is ``k_point_count(lattice)``. If a future change makes
    the path adjustable, this test fails and the cap stops being decorative --
    which is the moment to check it is still sized against a real worst case.
    """
    for lattice in pc_form.LATTICE_CHOICES:
        assert pc_form.k_point_count(lattice) <= pc_form.MAX_K_POINTS


def test_exceeding_the_node_cap_through_the_form_is_refused_by_plan_request():
    """The cap has to fire on the real path, not only when called directly.

    ``check_caps`` is a helper; this drives ``plan_request`` -- the function the
    route calls -- past the node cap with a real mesh, so the wiring between
    "count the cell" and "compare against the cap" is exercised. 69 divisions is
    the first square-cell value over 5,000 nodes (measured: 5,064).
    """
    with pytest.raises(pc_form.PCRequestError) as excinfo:
        pc_form.plan_request(_with(divisions="69"))
    assert str(pc_form.MAX_NODES) in str(excinfo.value)


def test_one_step_under_the_node_cap_through_the_form_is_accepted():
    """The step below it, so the boundary above is a real boundary.

    Paired with the test above at 68 (4,930 nodes): a pair that straddles the
    cap is what proves the comparison is against 5,000 and not against some
    larger number that happens to also refuse 69.
    """
    planned = pc_form.plan_request(_with(divisions="68"))
    assert planned["nodes"] <= pc_form.MAX_NODES < 5000 + planned["nodes"]


# ---------------------------------------------------------------------------
# The field refusals: each names the box the owner has to fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field, value, needle, extra", [
    ("lattice", "hexagonal", "hexagonal", {}),
    ("polarisation", "TMz", "TMZ", {}),
    # "less than 0.5" and "more than 0" are BOTH substrings of the radius
    # refusal's one sentence, so asserting on either would pass if the message
    # had the bounds backwards or swapped -- the needle has to be the fact that
    # distinguishes them. An earlier draft used "0.5" alone, and a mutation that
    # replaced "rod radius" with "background index" stayed GREEN: the assertion
    # was never checking which field the sentence was about.
    ("r_over_a", "0.7", "rod radius must be more than 0 and less than 0.5", {}),
    ("r_over_a", "0", "got 0.0", {}),
    # A radius that is ordinary for one lattice and impossible for the other.
    # The rod must fit inside the cell, and the triangular cell's inradius is
    # sin(60)/2 = 0.4330 -- so the form's own flat ``0 < r < 0.5`` check passes
    # 0.45 and the GEOMETRY is what refuses. Its refusal is a ``PCError``, which
    # the route does not catch, so before the translation in ``plan_request``
    # this request was a 500 rather than a refusal. The needle is the phrase
    # naming WHICH cell was too small; ``extra`` carries the lattice, which is
    # the only row here whose refusal depends on a second field.
    ("r_over_a", "0.45", "triangular cell", {"lattice": "triangular"}),
    ("n_rod", "-1", "rod refractive index must be positive, got -1.0", {}),
    ("n_bg", "0", "background refractive index must be positive, got 0.0", {}),
    ("divisions", "3", "at least 4 parts per edge, got 3", {}),
    ("divisions", "999", str(pc_form.MAX_DIVISIONS), {}),
    ("n_bands", "0", "at least 1, got 0", {}),
    ("n_bands", "999", str(pc_form.MAX_BANDS), {}),
])
def test_a_bad_field_is_refused_naming_that_field(field, value, needle, extra):
    """Every refusal quotes the offending value AND the field it belongs to.

    Parametrised because the failure mode being prevented is a refusal that
    names the WRONG field -- an owner told their lattice is bad when it was the
    radius. One test per field with the value in the assertion makes that
    visible; a single "some field was bad" test would not.

    Every needle here is therefore a phrase that exists in exactly ONE refusal,
    including that refusal's own field name. A needle that is merely a number or
    a bound shared by several sentences cannot tell them apart, which is how the
    first draft of this table let a field-swap mutation through.

    The ``needle`` is matched CASE-SENSITIVELY on purpose, and the polarisation
    row is why: the refusal shows the value as the module read it -- upper-cased,
    ``'TMZ'`` -- not as the owner typed it. A case-insensitive match would pass
    whichever the code did, so it would not notice the normalisation happening at
    all, and the normalisation is the interesting part.
    """
    with pytest.raises(pc_form.PCRequestError) as excinfo:
        pc_form.plan_request(_with(**{**extra, field: value}))
    assert needle in str(excinfo.value), (
        f"refusing {field}={value!r} must mention {needle!r}")


@pytest.mark.parametrize("field", ["lattice", "r_over_a", "n_rod", "n_bg",
                                   "n_bands", "divisions", "polarisation"])
def test_an_empty_field_is_refused_naming_that_field(field):
    """An empty box and a typo are a refusal either way -- and both name the box.

    ``_as_float("")`` would raise ValueError, which escapes as a 500. A form
    posted with one box left blank is ordinary, so it must be a refusal with the
    field named, not an exception the route was never told to catch.

    Which message is asserted, not just that one arrived. The first draft of
    this test asked only that SOMETHING was raised, and a mutation deleting the
    empty-value guard (letting the ``float("")`` ValueError escape) stayed GREEN
    -- because ``pytest.raises(PCRequestError)`` does not catch a ValueError, but
    the parametrised case ran last and the summary counted it as one failure in a
    file of thirty-four. The assertion below names the refusal, so the mutation
    has to redden THIS row.
    """
    with pytest.raises(pc_form.PCRequestError) as excinfo:
        pc_form.plan_request(_with(**{field: ""}))
    words = {"lattice": "lattice", "polarisation": "polarisation",
             "r_over_a": "radius", "n_rod": "rod", "n_bg": "background",
             "n_bands": "bands", "divisions": "divisions"}[field]
    message = str(excinfo.value)
    assert words in message, (
        f"the refusal for an empty {field} must name it; got {message!r}")
    # and it must be the EMPTY-value refusal, not some later check firing on a
    # value that was never parsed -- "a whole number, but was None" would name
    # the field too while telling the owner nothing about the empty box.
    #
    # The two SELECT fields refuse by naming the choices (an empty picklist is
    # an unknown choice), the five numeric ones by asking for a value. Requiring
    # the numeric wording of a select would be requiring the wrong refusal, so
    # the expected sentence is looked up rather than assumed.
    expected = ("Choose one of" if field in ("lattice", "polarisation")
                else "Please give a value for")
    assert expected in message, (
        f"an empty {field} must be refused as empty; got {message!r}")


def test_a_fractional_division_count_is_refused_rather_than_truncated():
    """``int(3.7)`` is 3, and silently solving for 3 is the request altered.

    A form would normally send an integer, but the route does not control what
    arrives -- and a mesh silently built at a different resolution than asked is
    a result that cannot be explained from the row.
    """
    with pytest.raises(pc_form.PCRequestError) as excinfo:
        pc_form.plan_request(_with(divisions="16.5"))
    assert "whole number" in str(excinfo.value)
    assert "16.5" in str(excinfo.value)


def test_the_lattice_choices_come_from_the_solver_not_a_second_list():
    """A lattice the solver accepts must not be a name this module rejects.

    Asserted against ``pc.LATTICES`` rather than by retyping the tuple: a copy
    here would check its own arithmetic and stay green when the solver's table
    changed -- the "guard must read the source, not retype it" shape.
    """
    sys.path.insert(0, str(SOLVER))
    import pc  # noqa: PLC0415

    assert set(pc_form.LATTICE_CHOICES) == set(pc.LATTICES)


def test_the_solver_refuses_a_lattice_this_module_also_refuses():
    """The two refusals agree, so a name cannot fall between them.

    ``pc_form`` filters first; this proves the solver's own guard would have
    caught the same name, so a gap in either one is not a hole. Both raise a
    ``MeshingError`` subclass, which is what the app renders.
    """
    sys.path.insert(0, str(SOLVER))
    import pc  # noqa: PLC0415

    with pytest.raises(pc.PCError):
        pc.lattice_vectors("hexagonal")


# ---------------------------------------------------------------------------
# The guard: no solve in the web process, proved with a mutation
# ---------------------------------------------------------------------------

#: The names that are the SOLVE, as opposed to the geometry this module is
#: allowed to borrow. Keeping the list here rather than only in the test body so
#: the mutation control below can inject one and prove the walk sees it.
SOLVE_NAMES = ("solve_bands", "bands_from_mesh", "bloch_reduce",
               "layered_bands", "layered_band_at_k", "layered_traces")


def _imported_names(path: Path) -> set[str]:
    """Every name imported anywhere in ``path``, plus every attribute reached.

    Two AST passes, because there are two ways a web module could get at the
    solve and a grep for one of them would miss the other:

    * ``from pc import solve_bands`` -> an ``ImportFrom`` alias;
    * ``import pc`` then ``pc.solve_bands(...)`` -> an ``Attribute``.

    The second is the one that matters most: a module that imports ``pc`` (which
    ``pc_form`` legitimately does) and calls an attribute on it is reaching the
    solve through a name that appears nowhere in its import list.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
    return found


def _executable_lines(path: Path) -> str:
    """The module's code with every docstring and comment removed.

    A docstring that says "this module never solves" is the opposite of a
    violation, and so is a comment explaining why a solve is not here -- but both
    contain the word. Stripping them is what makes the assertion below about
    calls rather than about prose. ``ast`` gives this for free: only ``Expr``
    nodes holding a bare string constant are docstrings.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    for node in ast.walk(tree):
        # a string literal anywhere else is data, not a docstring -- but the
        # name-matching below only inspects names and attributes, so only
        # docstrings need removing.
        pass
    return ast.unparse(tree)


def test_no_web_module_calls_the_photonic_crystal_solve():
    """The guard: "No solve in the web process, ever."

    The web pod is ``cpu: 500m`` with a 5 s scheduler tick in a thread of the
    same process, so a solve here does not merely occupy a request -- it starves
    the tick (§3). This walks every ``web/*.py`` and asserts no solve NAME is
    reached, with docstrings stripped so the prohibition's own wording is not
    the thing being caught.
    """
    offenders: list[str] = []
    for module in sorted(WEB.glob("*.py")):
        code = _executable_lines(module)
        for name in SOLVE_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", code):
                offenders.append(f"web/{module.name} -> {name}")
    assert not offenders, (
        f"the web process must never solve; found {offenders}. The solve runs "
        f"in the worker (or a cluster Job). Move it there; do not add it here "
        f"because it 'looks cheap' -- the measured spread is 0.2 s to 6.1 s "
        f"per k-point and the tick shares this pod's half-core."
    )


def test_that_guard_can_actually_fail():
    """The mutation control for the guard above.

    A guard that cannot fail leaves the bug in. This builds a synthetic module
    that DOES reach the solve -- one through an attribute, one through a
    from-import -- writes them where the walk will read them, and asserts the
    same checker flags each. Without this, a typo in ``SOLVE_NAMES`` would make
    the guard above pass forever while checking nothing.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        attribute = tmpdir / "reaches_by_attribute.py"
        attribute.write_text(
            "import pc\n\n\ndef go(cell, k):\n"
            "    return pc.solve_bands(*cell, k)\n",
            encoding="utf-8",
        )
        from_import = tmpdir / "reaches_by_from_import.py"
        from_import.write_text(
            "from pc import solve_bands\n\n\ndef go(cell, k):\n"
            "    return solve_bands(*cell, k)\n",
            encoding="utf-8",
        )
        for path in (attribute, from_import):
            code = _executable_lines(path)
            hits = [n for n in SOLVE_NAMES
                    if re.search(rf"\b{re.escape(n)}\b", code)]
            assert hits, (
                f"the guard did not see the solve in {path.name}; SOLVE_NAMES "
                f"or the stripper is wrong, and the guard above is vacuous"
            )


def test_the_stripper_removes_the_prohibition_but_not_the_call():
    """A blank-strip control, so "clean" is not produced by stripping too much.

    A stripper that removed MORE than docstrings -- or one that removed the whole
    function -- would report every module clean and pass the guard by destroying
    its input. This checks both directions on the same synthetic source: the
    docstring that NAMES the solve is gone afterwards, and a real call in the
    body is not.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "both.py"
        path.write_text(
            '"""This module deliberately never calls solve_bands."""\n'
            "\n"
            "import pc\n"
            "\n"
            "def solve_it(cell, k):\n"
            '    """solve_bands is not called lightly."""\n'
            "    return pc.solve_bands(*cell, k)\n",
            encoding="utf-8",
        )
        code = _executable_lines(path)
        # the docstrings' prose is gone...
        assert "deliberately" not in code
        assert "lightly" not in code
        # ...but the call they talk about is still there for the guard to see.
        assert "solve_bands" in code


def test_the_solve_names_are_the_solvers_own():
    """``SOLVE_NAMES`` must be real names in the solver, or the guard is a spell.

    A guard listing functions that do not exist checks nothing while reading as
    a check. Asserted against the solver module's own ``dir``.
    """
    sys.path.insert(0, str(SOLVER))
    import pc  # noqa: PLC0415

    missing = [name for name in SOLVE_NAMES if not hasattr(pc, name)]
    assert not missing, f"SOLVE_NAMES names nothing in pc: {missing}"


# ---------------------------------------------------------------------------
# The name collision
# ---------------------------------------------------------------------------

def test_the_web_module_does_not_shadow_the_geometry_module():
    """``web/pc_geometry.py`` would win the name and break this module's import.

    Not a style note. ``_geometry_module()`` imports the solver directory's
    ``pc_geometry``; if this module were named ``pc_geometry`` the import would
    resolve to ITSELF and the failure would be ``AttributeError: module
    'pc_geometry' has no attribute 'high_symmetry_path'`` -- naming a function
    the reader can see defined in the geometry file. This pins the filename so
    the rename cannot be silently reverted.
    """
    assert not (WEB / "pc_geometry.py").exists(), (
        "web/pc_geometry.py shadows the solver's pc_geometry.py; this module "
        "must be pc_form.py"
    )
    assert Path(pc_form.__file__).name == "pc_form.py"


def test_the_resolved_module_is_the_numpy_only_geometry():
    """And the positive form: ``_geometry_module()`` resolves to ``pc_geometry``.

    The collision test above says the bad path is absent; this says the good one
    is what actually loads. It asserts the FILE, not a name: the fix for the
    scipy 500 is that the web form imports the numpy-only geometry instead of the
    solver, and a regression to ``import pc`` would resolve to a file in the same
    directory and pass any check that only looked at the directory.
    """
    resolved = Path(pc_form._geometry_module().__file__).resolve()
    assert resolved == (SOLVER / "pc_geometry.py").resolve()
