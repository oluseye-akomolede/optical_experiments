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


def pytest_collect_file(parent, file_path):
    """Make the JavaScript tests part of ``pytest tests/``.

    ``tests/js/`` holds the viewer's lazy-load rule (``modecache.js``), which is
    plain ESM and needs node, not a browser. Without this hook it is only
    collected when someone remembers to run node by hand -- and a test nobody
    runs is a test that is already failing. Skips (rather than errors) where
    node is absent, so a checkout without node is not a broken build.
    """
    if file_path.suffix == ".mjs" and file_path.parent.name == "js":
        return NodeTestFile.from_parent(parent, path=file_path)
    return None


class NodeTestFile(pytest.File):
    def collect(self):
        import shutil

        runner = shutil.which("node")
        yield NodeTestItem.from_parent(self, name=self.path.name, runner=runner)


class NodeTestItem(pytest.Item):
    def __init__(self, *, runner, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner

    def runtest(self):
        import subprocess

        if self.runner is None:
            raise pytest.skip.Exception("node is not installed", _use_item_location=True)
        proc = subprocess.run(
            [self.runner, "--test", str(self.path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise NodeTestFailure(proc.stdout + proc.stderr)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, NodeTestFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, f"node --test {self.path.name}"


class NodeTestFailure(Exception):
    """node --test reported at least one failing assertion."""
