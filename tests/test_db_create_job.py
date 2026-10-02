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

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "worker"))

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
    # ``-2`` and not ``-1``: ``pcs`` (the photonic-crystal request) was appended
    # after ``materials``, so the trailing slot stopped being the materials one.
    # This test used to say ``[-1]`` and would have gone green against a
    # ``pcs`` value -- it only failed here because the materials value happened
    # to be non-None and json.loads caught it. Pinned by NAME below so the next
    # column appended cannot repeat that.
    assert json.loads(params[_db_param_index(store, "materials")]) == materials


def test_create_job_without_materials_sends_null():
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10)
    assert _insert_params(store)[_db_param_index(store, "materials")] is None


def _db_param_index(store, column: str) -> int:
    """Which position of the parameter tuple holds ``column``.

    Read out of the INSERT statement rather than counted by hand: the column
    list and the tuple are two hand-maintained lists that must agree, and the
    moment they stop agreeing the failure is a silent off-by-one that binds one
    value to the wrong column. Naming the position here means a test says what
    it means after the next column is added.
    """
    sql = next(sql for sql, _ in store.sent if "INSERT" in sql.upper())
    head = sql.split("VALUES", 1)[0]
    columns = [c.strip() for c in
               head[head.index("(") + 1:head.rindex(")")].split(",")]
    return columns.index(column)


def test_create_job_defaults_to_the_cavity_kind():
    """A caller that predates photonic crystals writes 'cavity', not NULL.

    The column is NOT NULL DEFAULT 'cavity', so this is what the database would
    have stored anyway -- but only if the INSERT omits it. It does not: the
    statement names the column explicitly, so a missing default here would send
    None into a NOT NULL column and every upload would 500. Asserted by VALUE
    rather than by "it did not raise", because there is no database in this
    test to raise.
    """
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10)
    params = _insert_params(store)
    assert params[_db_param_index(store, "kind")] == DB.CAVITY


def test_create_job_with_a_pc_request_sends_it_as_json_and_says_pc():
    store = _RecordingStore()
    pcs = {"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5, "n_bg": 1.0,
           "n_bands": 12, "divisions": 16, "polarisation": "TM"}
    store.create_job(uuid.uuid4(), None, "mm", 1.0, 12,
                     kind=DB.PC, pcs=pcs)
    params = _insert_params(store)
    assert params[_db_param_index(store, "kind")] == DB.PC
    assert json.loads(params[_db_param_index(store, "pcs")]) == pcs


#: A complete td request: every key the worker reads. Built from the worker's
#: own list so this fixture cannot fall behind it.
def _td_request():
    import td_solve
    values = {"boundary": "open", "source_kind": "point", "profile": "pulse",
              "frequency_hz": 4.0e14, "bandwidth": 0.5,
              "position": [0.0, 0.0, 0.0], "direction": None,
              "polarisation": "z", "samples": None, "sample_duration_s": None,
              "duration_s": 2.0e-13, "probes": [[1e-7, 0.0, 0.0, "z"]],
              "dx_m": 2.0e-8}
    assert set(values) == set(td_solve.REQUEST_KEYS)
    return values


def test_create_job_with_a_td_request_sends_it_as_json_and_says_td():
    store = _RecordingStore()
    tds = _td_request()
    store.create_job(uuid.uuid4(), "box.obj", "nm", 1.0, 0, kind=DB.TD, tds=tds)
    params = _insert_params(store)
    assert params[_db_param_index(store, "kind")] == DB.TD
    assert json.loads(params[_db_param_index(store, "tds")]) == tds
    assert params[_db_param_index(store, "pcs")] is None


@pytest.mark.parametrize("kind", [DB.CAVITY, DB.PC, DB.TD])
def test_create_job_accepts_every_known_kind(kind):
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10, kind=kind)
    assert _insert_params(store)[_db_param_index(store, "kind")] == kind


@pytest.mark.parametrize("kind", ["TD", "fdtd", "", None])
def test_create_job_refuses_an_unknown_kind_before_writing(kind):
    store = _RecordingStore()
    with pytest.raises(ValueError, match="unknown job kind"):
        store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10, kind=kind)
    assert not store.sent


def test_create_job_without_a_td_request_sends_null():
    """NULL, not {}, for the reason ``pcs`` is NULL on a cavity row."""
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10, kind=DB.PC,
                     pcs={"lattice": "square"})
    assert _insert_params(store)[_db_param_index(store, "tds")] is None


def test_create_job_without_a_pc_request_sends_null():
    """NULL, not {}: {} claims a request with every field missing."""
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10, kind=DB.CAVITY)
    assert _insert_params(store)[_db_param_index(store, "pcs")] is None


def test_the_insert_names_a_column_for_every_value_it_binds():
    """The parameter count and the column list must not drift apart.

    This is the failure mode the materials bug belonged to, one step earlier:
    psycopg2 raises on a count mismatch, so it fails loudly at runtime -- but it
    fails on EVERY upload, and the count is assembled by hand in three places
    (the column list, the VALUES list and the tuple). Counting them here is
    cheaper than a 500.
    """
    store = _RecordingStore()
    store.create_job(uuid.uuid4(), "box.obj", "mm", 1.0, 10)
    sql, params = [p for p in store.sent if "INSERT" in p[0].upper()][0]
    head, values = sql.split("VALUES", 1)
    columns = head[head.index("(") + 1:head.rindex(")")].split(",")
    assert len(columns) == len(values.split(",")) == len(params), sql


class _ClaimStore(DB.JobStore):
    """A store whose SELECT returns one canned row, to read the CLAIM path.

    ``claim_jetson`` builds its result dict from row POSITIONS, so a column
    appended to the query is silently dropped unless the mapping is extended
    too -- exactly what M3/M4 found: the kinds and the pc request could fall out
    of the claim payload with every test green. The rows here are wide enough to
    make an off-by-one in that mapping a FAILURE rather than a wrong value.
    """

    def __init__(self, row):
        self.row = row
        self.statements = []

    def _open(self, autocommit=True):
        return _ClaimConn(self.row, self.statements, autocommit)


class _ClaimConn:
    def __init__(self, row, statements, autocommit):
        self.statements = statements
        # The ROW, not the store: _RecordingStore/_ClaimStore keep their state
        # on themselves, and an earlier version of this class was handed the
        # store while reading `row` off it -- an AttributeError inside
        # claim_jetson that pytest reported as a failed test rather than as a
        # broken fake. Bound to the row factory directly so there is one thing
        # to hand it.
        self.row = row

    def cursor(self):
        return _ClaimCursor(self.row, self.statements)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ClaimCursor:
    def __init__(self, row, statements):
        self.row = row
        # Sink for what was executed, owned by the STORE: the test asserts on
        # the SQL, not on the cursor, and the cursor is created per call.
        self.statements = statements

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        self._rows = [[None] for _ in range(20)]
        if "SELECT" in sql.upper():
            # Only the claim query returns a candidate; UPDATE ... RETURNING
            # does not exist here, and the row is consumed once.
            self._rows = [self.row()]

    def fetchone(self):
        if self._rows:
            return self._rows.pop(0)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _claim_row(**over):
    row = ["11111111-1111-1111-1111-111111111111", "mm", 1.0, 10, None, 1.0,
           None, DB.CAVITY, None, None]
    for name, value in over.items():
        row[_CLAIM_COLUMNS.index(name)] = value
    return row


#: The claim SELECT's column order. Read from the SQL below by a test, so this
#: list cannot silently disagree with the query.
_CLAIM_COLUMNS = ["job_id", "units", "n_index", "n_modes", "min_feature_m",
                  "scale", "materials", "kind", "pcs", "tds"]


def _claim_select_columns(store):
    sql = next(sql for sql, _ in store.statements if "SELECT" in sql.upper())
    head = sql.split("FROM")[0]
    body = head[head.index("SELECT") + len("SELECT"):]
    return [c.strip() for c in body.split(",")]


def test_the_claim_select_columns_match_this_test_s_position_table():
    """The position table above must be the query's own column order.

    Without this, adding a column to the claim SELECT would leave these tests
    green while `claim_jetson` read the wrong position -- a test that agrees
    with itself rather than with the code.
    """
    store = _ClaimStore(_claim_row)
    store.claim_jetson("w1", 300)
    assert _claim_select_columns(store) == _CLAIM_COLUMNS


def test_claim_jetson_reports_the_kind_and_the_pc_request():
    pcs = {"lattice": "triangular", "r_over_a": 0.3, "n_rod": 3.5, "n_bg": 1.0,
           "n_bands": 8, "divisions": 20, "polarisation": "TE"}
    store = _ClaimStore(lambda: _claim_row(kind=DB.PC, pcs=pcs))
    job = store.claim_jetson("w1", 300)
    assert job["kind"] == DB.PC
    assert job["pcs"] == pcs


def test_claim_jetson_reports_the_kind_and_the_td_request():
    tds = _td_request()
    store = _ClaimStore(lambda: _claim_row(kind=DB.TD, tds=tds))
    job = store.claim_jetson("w1", 300)
    assert job["kind"] == DB.TD
    assert job["tds"] == tds
    assert job["pcs"] is None


def test_claim_jetson_reports_cavity_and_null_for_a_cavity_job():
    store = _ClaimStore(_claim_row)
    job = store.claim_jetson("w1", 300)
    assert job["kind"] == DB.CAVITY
    assert job["pcs"] is None
    assert job["tds"] is None
    assert job["materials"] is None


# ---- The read side: get_job / list_jobs through _job_row --------------------
#
# The claim tests above pin claim_jetson's positions; nothing pinned the job
# readers. A reader that took tds from the pcs slot, or a list SELECT that
# dropped tds (so _job_row's row[21] raises and the job list page 500s), stayed
# green. These fakes answer each SELECT with a row built FROM THAT QUERY'S OWN
# column list, every column holding a value no other column holds, so a wrong
# index returns the wrong value and a dropped column shortens the row.


def _select_columns(sql: str) -> list[str]:
    head = sql.split("FROM")[0]
    body = head[head.index("SELECT") + len("SELECT"):]
    return [c.strip() for c in body.split(",")]


def _distinct_value(column: str):
    if column in ("created_at", "finished_at"):
        return None  # _iso formats these; None passes through as None
    if column in ("materials", "pcs", "tds"):
        return {"column": column}  # jsonb: a dict, and a DIFFERENT dict each
    return f"<{column}>"


class _ReadCursor:
    def __init__(self, statements):
        self.statements = statements

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        cols = _select_columns(sql)
        self._rows = [tuple(_distinct_value(c) for c in cols)]

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ReadConn(_Conn):
    def cursor(self):
        return _ReadCursor(self.sink)


class _ReadStore(DB.JobStore):
    def __init__(self):
        self.sent = []

    def _open(self, autocommit=True):
        return _ReadConn(self.sent)


def _read_job(reader: str) -> tuple[dict, list[str]]:
    store = _ReadStore()
    if reader == "get_job":
        job = store.get_job(uuid.uuid4())
    else:
        (job,) = store.list_jobs()
    (sql, _params), = store.sent
    return job, _select_columns(sql)


@pytest.mark.parametrize("reader", ["get_job", "list_jobs"])
def test_a_job_reader_returns_each_column_from_its_own_position(reader):
    """Every selected column comes back under its own name, with its own value.

    pcs and tds hold different dicts, so reading tds from the pcs slot fails
    here; a SELECT that leaves out tds makes the row one short of what
    _job_row indexes, and the read raises instead of passing.
    """
    job, cols = _read_job(reader)
    for name in ("kind", "pcs", "tds", "materials"):
        assert name in cols, f"{reader}'s SELECT does not select {name}"
    for name in cols:
        assert job[name] == _distinct_value(name), (reader, name, job[name])
    assert job["tds"] == {"column": "tds"}
    assert job["pcs"] == {"column": "pcs"}


@pytest.mark.parametrize("reader", ["get_job", "list_jobs"])
def test_a_job_reader_selects_every_field_the_row_reader_returns(reader):
    """The SELECT and _job_row's dict agree column for column, in order."""
    job, cols = _read_job(reader)
    assert cols == list(job)
