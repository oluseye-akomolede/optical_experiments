// Tests for the lazy-load rule in modecache.js. Run with:
//   node --test tests/js/*.mjs     (the glob matters -- `node --test tests/js/`
//   finds zero tests and exits 0, so it looks green having checked nothing)
//
// These exist because the viewer's data flow changed from "one response holds
// every mode" to "an index plus a fetch per mode", and none of that logic had
// any coverage at all -- viewer.js could not be imported in a test, because it
// pulls in three.js and touches the DOM at module scope. The decision was moved
// out to modecache.js precisely so it could be asserted here.

import test from 'node:test';
import assert from 'node:assert/strict';

import { wantsFetch, modeFieldUrl, storeFields, MAX_IN_FLIGHT }
  from '../../web/static/modecache.js';

test('a mode with nothing in hand is fetched, once', () => {
  const mode = { index: 3, available: true };
  assert.equal(wantsFetch(0, mode, 0), true);
  mode.loading = Promise.resolve();
  assert.equal(wantsFetch(0, mode, 0), false,
               'a request already in flight must not be started again');
});

test('a mode already in hand is never refetched', () => {
  const mode = { index: 1, fields: { slices: [] } };
  assert.equal(wantsFetch(0, mode, 0), false);
});

test('a mode the index marks unavailable is never requested', () => {
  // The request would 404 every time. Fetching it would turn "one mode is
  // missing" into "the viewer is broken" for the person looking at it.
  const mode = { index: 2, available: false };
  assert.equal(wantsFetch(0, mode, 0), false);
});

test('an absent mode is not fetched', () => {
  assert.equal(wantsFetch(0, undefined, 0), false);
  assert.equal(wantsFetch(0, null, 0), false);
});

test('requests per page are capped so a burst cannot open everything at once', () => {
  const mode = { index: 5, available: true };
  assert.equal(wantsFetch(4, mode, MAX_IN_FLIGHT - 1), true);
  assert.equal(wantsFetch(4, mode, MAX_IN_FLIGHT), false);
});

test('the URL uses the mode index the server sent, not the position in the list', () => {
  // The two differ whenever a job's modes do not start at 1, which is why the
  // index is carried through rather than recomputed as position + 1.
  assert.equal(modeFieldUrl('abc-123', 7), '/jobs/abc-123/field/7.bin');
  assert.equal(modeFieldUrl('abc-123', 0), '/jobs/abc-123/field/0.bin');
});

test('a late arrival lands against its own mode, not the selected one', () => {
  // The bug this guards: the user selects mode 3, then mode 5, and mode 3's
  // response lands afterwards. If it were stored against `current`, mode 5
  // would show mode 3's field and mode 3 would look permanently unloaded.
  const three = { index: 3, available: true, loading: Promise.resolve() };
  const five = { index: 5, available: true, loading: Promise.resolve() };
  const payload = { index: 3, slices: ['three'] };

  storeFields(three, payload);

  assert.deepEqual(three.fields, payload);
  assert.equal(three.loading, null, 'the in-flight marker must be cleared');
  assert.equal(five.fields, undefined, 'the other mode must be untouched');
});

// ---------------------------------------------------------------------------
// The browser's own parser, on bytes a real route produced
// ---------------------------------------------------------------------------
// The bug this exists for: the server sent NaN in the point cloud, `json.dumps`
// wrote the bare token, and the browser's JSON.parse rejected the whole
// payload. Every server-side test passed, because Python's json reads NaN
// happily -- the instrument could not see the failure. So this asserts the
// property the browser cares about: strict JSON, with no non-finite tokens.
//
// The fixture is written by tests/test_web_routes.py from a live route response
// (see `_write_viewer_fixture`). If it is missing the test fails loudly rather
// than skipping: a skip here would recreate the original blindness.
