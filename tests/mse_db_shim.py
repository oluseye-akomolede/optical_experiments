"""Load the project's real JobStore in a test, without importing Flask.

The script next to this one calls ``JobStore``'s real methods on purpose, but
``import db`` from ``web/`` drags in the app's dependencies. This loader reads
``web/db.py`` as a module by path instead, so a test can exercise the real SQL
without standing up the web app.

Lives in ``tests/`` deliberately: under ``web/`` it would be baked into the
mse-web image, where nothing imports it.
"""
import importlib.util
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
