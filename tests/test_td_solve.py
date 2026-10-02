"""The ``td`` request contract: the keys the worker reads, and its refusals.

Tiny and pure: no store, no solver. The phase-5 form will be pinned against
``REQUEST_KEYS`` the way ``test_pc_solve`` pins ``pc_form``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

import td_solve  # noqa: E402


def _complete():
    return {k: f"value-{k}" for k in td_solve.REQUEST_KEYS}


@pytest.mark.parametrize("missing", td_solve.REQUEST_KEYS)
def test_a_missing_key_is_refused_by_name(missing):
    tds = _complete()
    del tds[missing]
    with pytest.raises(td_solve.TDSolveError) as caught:
        td_solve.request_from({"tds": tds})
    assert missing in str(caught.value)


def test_every_missing_key_is_named_not_just_the_first():
    tds = _complete()
    gone = td_solve.REQUEST_KEYS[:2] + td_solve.REQUEST_KEYS[-1:]
    for key in gone:
        del tds[key]
    message = str(pytest.raises(td_solve.TDSolveError,
                                td_solve.request_from, {"tds": tds}).value)
    for key in gone:
        assert key in message


@pytest.mark.parametrize("tds", [None, [], "{}", 3])
def test_a_request_that_is_not_an_object_is_refused(tds):
    with pytest.raises(td_solve.TDSolveError, match="'tds' request object"):
        td_solve.request_from({"tds": tds})


def test_an_empty_object_is_refused_naming_every_key():
    message = str(pytest.raises(td_solve.TDSolveError,
                                td_solve.request_from, {"tds": {}}).value)
    for key in td_solve.REQUEST_KEYS:
        assert key in message


def test_null_values_are_kept_and_extra_keys_dropped():
    """Present-but-null is a real value (``direction`` for a point source);
    keys outside the contract (results written back) never reach the solve."""
    tds = _complete()
    tds["direction"] = None
    tds["bandwidth"] = 0
    tds["spectrum"] = {"written": "back"}
    request = td_solve.request_from({"tds": tds})
    assert request["direction"] is None
    assert request["bandwidth"] == 0
    assert "spectrum" not in request
    assert list(request) == list(td_solve.REQUEST_KEYS)
