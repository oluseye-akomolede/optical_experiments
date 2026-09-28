"""JobStore.create_job builds its SQL parameters for real.

The route tests replace the whole store with a fake, so the parameter tuple in
create_job was never evaluated by the suite. The multi-part branch called
json.dumps without importing json, and every multi-part upload 500'd in
production (8cc1ed5) while the suite stayed green. These tests call the real
method through a connection that records what it would have sent.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

import db as DB  # noqa: E402


class _Cursor:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self, sink):
        self.sink = sink
        self.autocommit = True

    def cursor(self):
        return _Cursor(self.sink)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _RecordingStore(DB.JobStore):
    def __init__(self):
        self.sent = []

    def _open(self, autocommit=True):
        return _Conn(self.sent)


def _insert_params(store):
    inserts = [p for sql, p in store.sent if "INSERT" in sql.upper()]
    assert len(inserts) == 1, store.sent
    return inserts[0]


def test_create_job_with_materials_sends_them_as_json():
    store = _RecordingStore()
    materials = {"wall": 1.0, "core": 2.0}
    store.create_job(uuid.uuid4(), "twobox.obj", "mm", 1.0, 10,
                     materials=materials)
    params = _insert_params(store)
    assert json.loads(params[-1]) == materials


def test_create_job_without_materials_sends_null():
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10)
    assert _insert_params(store)[-1] is None
