"""The graded-grid builder keeps its own rules (task 3.0).

The rules are restated from the spec (>= 4 cells across a layer thinner than
the coarse cell or across a skin depth, neighbour ratio <= 1.3), not read
from the builder's constants, so a builder that loosened its own constants
fails here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402
import fdtd_grid as grid  # noqa: E402

SPEC_RATIO = 1.3
SPEC_CELLS = 4

#: (length, coarse, layers): thin films, a film touching an end, overlapping
#: layers, a skin-depth spacing, and nothing at all.
AXES = [
    (1.0, 0.05, [grid.Layer(0.40, 0.41)]),
    (1.0, 0.05, [grid.Layer(0.0, 0.003), grid.Layer(0.5, 0.5004)]),
    (2.0, 0.10, [grid.Layer(0.70, 0.90), grid.Layer(0.85, 0.86)]),
    (1.0, 0.05, [grid.Layer(0.6, 0.7, 0.002)]),
    (0.8, 0.07, []),
]


def _cells_across(nodes, lo, hi):
    tol = 1e-9 * (nodes[-1] - nodes[0])
    return int(np.sum((nodes >= lo - tol) & (nodes <= hi + tol))) - 1


@pytest.mark.parametrize("length,coarse,layers", AXES)
def test_the_builder_keeps_the_spec_rules(length, coarse, layers):
    nodes = grid.graded_axis(length, coarse, layers)
    widths = np.diff(nodes)
    # each fixture grades in under 100 nodes; a builder whose repairs keep
    # refining the fine end (once: 288 M nodes) breaks this before the RAM
    assert nodes.size < 200, nodes.size
    assert nodes[0] == 0.0 and nodes[-1] == pytest.approx(length, rel=1e-12)
    assert np.all(widths > 0)
    ratios = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
    assert ratios.max() <= SPEC_RATIO + 1e-9, ratios.max()
    assert widths.max() <= coarse * (1 + 1e-9)
    for layer in layers:
        assert np.min(np.abs(nodes - layer.lo)) < 1e-9 * length, "edge not a node"
        assert np.min(np.abs(nodes - layer.hi)) < 1e-9 * length, "edge not a node"
        if layer.hi - layer.lo < coarse:
            assert _cells_across(nodes, layer.lo, layer.hi) >= SPEC_CELLS
        if layer.spacing is not None:
            inside = nodes[(nodes >= layer.lo - 1e-12) & (nodes <= layer.hi + 1e-12)]
            assert np.diff(inside).max() <= layer.spacing * (1 + 1e-9)
    assert grid.check_axis(nodes, coarse, layers) == []


def test_the_coarse_region_stays_coarse():
    # the grading must relax back to the coarse cell, or a film costs a fine
    # grid everywhere and the estimate is wrong the other way
    nodes = grid.graded_axis(1.0, 0.05, [grid.Layer(0.40, 0.41)])
    far = np.diff(nodes)[(nodes[:-1] > 0.75) | (nodes[1:] < 0.1)]
    assert far.size and far.min() > 0.6 * 0.05
    assert nodes.size < 80          # uniform at 0.01 / 4 would be 400


def test_the_rule_checker_sees_each_broken_rule():
    # positive control for check_axis: each rule, broken alone, is reported
    good = np.linspace(0.0, 1.0, 21)
    assert grid.check_axis(good, 0.05) == []
    jump = np.array([0.0, 0.01, 0.05, 0.1])
    assert any("ratio" in p for p in grid.check_axis(jump, 0.05))
    assert any("coarse" in p for p in grid.check_axis(good, 0.04))
    thin = [grid.Layer(0.40, 0.41)]
    two = np.sort(np.concatenate([good, [0.405, 0.41]]))
    assert any("fewer than 4" in p for p in grid.check_axis(two, 0.05, thin))
    assert any("not a node" in p
               for p in grid.check_axis(good, 0.05, [grid.Layer(0.401, 0.6)]))
    assert any("> 0.002" in p for p in
               grid.check_axis(good, 0.05, [grid.Layer(0.4, 0.6, 0.002)]))


def test_build_grid_gives_simulate_its_spacings():
    built = grid.build_grid((1.0, 0.6, 0.3), 0.05,
                            ([grid.Layer(0.4, 0.41)], [], [grid.Layer(0.1, 0.104)]))
    assert built["shape"] == tuple(n.size for n in built["nodes"])
    assert len(set(built["shape"])) == 3          # non-square on purpose
    for nodes, spacing in zip(built["nodes"], built["spacings"]):
        np.testing.assert_allclose(spacing, np.diff(nodes))
    eps = np.ones(built["shape"], dtype=np.float32)
    run = fdtd.simulate(eps, fdtd.Source((5, 5, 3), frequency=3e9, bandwidth=1e9),
                        2, spacings=built["spacings"])
    assert run["steps"] == 2


def test_skin_depth_matches_the_published_table():
    # refractiveindex.info Rakic-LD Au row 593.93 nm: k = 2.7930, so the
    # field's 1/e length is lambda / (2 pi k) = 33.844 nm
    wavelength = 593.93e-9
    expected = wavelength / (2 * np.pi * 2.7930)
    got = grid.skin_depth("Au", fdtd.C0 / wavelength)
    assert got == pytest.approx(expected, rel=1e-3)


def test_a_curved_coating_costs_more_than_a_flat_one():
    flat = grid.coating_layers(0.3, 0.6, 0.004, flat=True)
    curved = grid.coating_layers(0.3, 0.6, 0.004, flat=False)
    assert [(l.lo, l.hi) for l in flat] == [(0.296, 0.3), (0.6, 0.604)]
    assert curved[0].spacing == pytest.approx(0.001)
    n_flat = grid.graded_axis(1.0, 0.05, flat).size
    n_curved = grid.graded_axis(1.0, 0.05, curved).size
    assert n_curved > 4 * n_flat
    nodes = grid.graded_axis(1.0, 0.05, curved)
    assert np.diff(nodes[(nodes >= 0.296) & (nodes <= 0.604)]).max() <= 0.001 + 1e-12
    with pytest.raises(grid.GridError, match="positive thickness"):
        grid.coating_layers(0.3, 0.6, 0.0)


def test_a_cut_cell_takes_the_volume_average():
    # a plane at x = 0.43 through a uniform grid of 0.1: the node at 0.4 owns
    # 0.35..0.45, of which 0.35..0.43 is eps 4 and 0.43..0.45 is eps 1
    nodes = (np.linspace(0, 1, 11), np.linspace(0, 0.6, 7), np.linspace(0, 0.3, 4))
    index_of = lambda p: (p[:, 0] < 0.43).astype(int)
    eps, _ = grid.average_eps(nodes, index_of, [1.0, 4.0], samples=10)
    assert eps.shape == (11, 7, 4)
    np.testing.assert_allclose(eps[4], 0.8 * 4.0 + 0.2 * 1.0, rtol=1e-12)
    np.testing.assert_allclose(eps[2], 4.0)
    np.testing.assert_allclose(eps[6], 1.0)


def test_a_cut_cell_on_y_and_z_lands_on_its_own_axis():
    # planes on y and z, not x: each x slab is sampled as one block, so a
    # scrambled sample order inside the slab is invisible to an x-only cut.
    # y < 0.23 is eps 4 (node y=0.2 owns 0.15..0.25: 0.8 of it), z < 0.13
    # adds 2 (node z=0.1 owns 0.05..0.15: 0.8 of it); 5 x 7 x 4, non-square
    nodes = (np.linspace(0, 0.4, 5), np.linspace(0, 0.6, 7), np.linspace(0, 0.3, 4))
    index_of = lambda p: (p[:, 1] < 0.23).astype(int) + 2 * (p[:, 2] < 0.13)
    eps, _ = grid.average_eps(nodes, index_of, [1.0, 4.0, 3.0, 6.0], samples=10)
    y_part = np.array([1, 1, 0.8, 0, 0, 0, 0])
    z_part = np.array([1, 0.8, 0, 0])
    # eps is 1 + 3 [y cut] + 2 [z cut] (index 3 = both = 6), linear in each
    # indicator, so the average is linear in each node's two fractions
    want = 1.0 + 3.0 * y_part[:, None] + 2.0 * z_part[None, :]
    for i in range(5):
        np.testing.assert_allclose(eps[i], want, rtol=1e-12)


def test_a_layer_thinner_than_a_cell_still_counts():
    # a 0.02-thick eps-9 film on a 0.1 grid: no node sits in it, but the
    # volume-weighted eps over the grid still holds the film's volume
    nodes = (np.linspace(0, 1, 11), np.linspace(0, 0.6, 7), np.linspace(0, 0.3, 4))
    index_of = lambda p: ((p[:, 0] > 0.51) & (p[:, 0] < 0.53)).astype(int)
    eps, _ = grid.average_eps(nodes, index_of, [1.0, 9.0], samples=50)
    weight = np.diff(np.concatenate([[0], 0.5 * (nodes[0][1:] + nodes[0][:-1]), [1]]))
    excess = float(np.sum((eps[:, 0, 0] - 1.0) * weight))
    assert excess == pytest.approx(0.02 * 8.0, rel=1e-6)


def test_a_metal_node_is_a_majority_of_its_cell():
    nodes = (np.linspace(0, 1, 11), np.linspace(0, 0.6, 7), np.linspace(0, 0.3, 4))
    index_of = lambda p: np.where(p[:, 0] < 0.43, 2, 0)
    eps, masks = grid.average_eps(nodes, index_of, [1.0, 4.0, 1.0],
                                  samples=10, metals=(2,))
    assert masks[2][4].all() and not masks[2][5].any()
    np.testing.assert_allclose(eps[4], 1.0)       # the dielectric part only
    np.testing.assert_allclose(eps[2], 1.0)       # all metal: no dielectric


def test_a_minority_of_metal_leaves_the_node_dielectric():
    # x < 0.47: node x=0.5 owns 0.45..0.55, so 0.2 of it is metal; it stays a
    # dielectric node, and its eps averages the dielectric samples only
    nodes = (np.linspace(0, 1, 11), np.linspace(0, 0.6, 7), np.linspace(0, 0.3, 4))
    index_of = lambda p: np.where(p[:, 0] < 0.47, 1, 0)
    eps, masks = grid.average_eps(nodes, index_of, [4.0, 1.0],
                                  samples=10, metals=(1,))
    assert masks[1][4].all() and not masks[1][5].any()
    np.testing.assert_allclose(eps[5], 4.0)


def test_an_end_node_samples_only_inside_the_domain():
    # a region just outside x = 0 must not leak into node 0's average
    nodes = (np.linspace(0, 1, 11), np.linspace(0, 0.6, 7), np.linspace(0, 0.3, 4))
    index_of = lambda p: np.where(p[:, 0] < 0.0, 1, 0)
    eps, _ = grid.average_eps(nodes, index_of, [1.0, 9.0], samples=4)
    np.testing.assert_allclose(eps, 1.0)


def test_a_grading_that_cannot_converge_is_refused_by_name(monkeypatch):
    # growth 0.5 per cell cannot meet a 1.3 ratio; the repairs must stop with
    # a named error, not refine the fine end until the RAM runs out
    monkeypatch.setattr(grid, "GROWTH", 0.5)
    with pytest.raises(grid.GridError, match="did not converge"):
        grid.graded_axis(1.0, 0.05, [grid.Layer(0.40, 0.41)])
