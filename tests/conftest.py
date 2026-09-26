"""Test configuration for the MSE solver.

**Heavy tests do not run by default.** The full solver suite (TetGen
tetrahedralization plus eigen solves) used all the RAM on the VM when it was run
natively and took the machine down. Per the operator rule, solvers, validation
and heavy tests run ONLY as bounded Kubernetes Jobs on the worker nodes.

So the suite is split in two tiers:

* **unit** (default) -- element matrices, the gradient invariant, meshing
  validation, shift selection. Pure numpy, no tetgen, no eigen solve. Safe
  locally, and small enough to run under ``ulimit -v 4194304``.
* **heavy** (opt in with ``MSE_RUN_HEAVY=1``) -- anything that calls TetGen or
  ``eigsh``. Run these on the cluster, not here.

    MSE_RUN_HEAVY=1 python -m pytest tests/ -m heavy

Collect the heavy set without running it:

    python -m pytest tests/ -m heavy --collect-only -q
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: needs TetGen and/or an eigen solve; run on the cluster, not the VM",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``heavy`` tests unless the operator explicitly opts in."""
    if os.getenv("MSE_RUN_HEAVY") == "1":
        return
    skip = pytest.mark.skip(
        reason="heavy test: set MSE_RUN_HEAVY=1 and run on a cluster Job, not the VM"
    )
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)
