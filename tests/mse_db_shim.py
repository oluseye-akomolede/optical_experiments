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


def load_jobstore():
    spec = importlib.util.spec_from_file_location("mse_db_real", _DB_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JobStore
