"""Load the project's real JobStore in a test, without importing Flask.

The script next to this one calls ``JobStore``'s real methods on purpose, but
``import db`` from ``web/`` drags in the app's dependencies. This loader reads
``web/db.py`` as a module by path instead, so a test can exercise the real SQL
without standing up the web app.

Lives in ``tests/`` deliberately: under ``web/`` it would be baked into the
mse-web image, where nothing imports it.
"""
import importlib.util
import sys
from pathlib import Path

_DB_PY = Path(__file__).resolve().parent.parent / "web" / "db.py"


def load_db_module():
    """``web/db.py`` as a module, so module-level helpers are reachable too.

    ``releases_are_terminal`` lives at MODULE level, not on ``JobStore``: it is
    the bound expressed as a predicate, and a check that could only reach it as
    a method would fail on a correct implementation that chose a plain function.
    """
    spec = importlib.util.spec_from_file_location("mse_db_real", _DB_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jobstore():
    return load_db_module().JobStore


_APP_PY = Path(__file__).resolve().parent.parent / "web" / "app.py"


def load_app_module():
    """``web/app.py`` as a module, so a leg can drive the REAL tick.

    ``_fail_dead_jobs`` is where the dispatch rollback lived, and a leg that
    called a copy of it would test the copy. Importing the real module also
    means the leg fails loudly if ``app`` stops importing at all, which is a
    separate bug the deploy cares about.

    ``web/`` and the solver directory go on ``sys.path`` first: the app imports
    its siblings by bare name (``import config``), exactly as it does inside the
    image where ``/opt/app`` is the working directory.
    """
    root = _APP_PY.parent.parent
    for extra in (root / "web", root / "dockerfiles" / "maxwell_solver"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location("mse_app_real", _APP_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
