// The rule for *when* a mode's field data is fetched, kept apart from the
// viewer so it can be tested without a DOM, a canvas or three.js.
//
// The viewer used to receive every mode's field in one 11 MB response. The
// server now sends an index and one payload per mode, and this decides whether
// a given selection costs a fetch. Getting this wrong is not a crash -- it is
// either a slow page (refetching a mode already in hand) or a blank one
// (never fetching, or fetching while a request is already in flight), which is
// exactly the kind of silent regression a screenshot does not catch.

/**
 * Does selecting `mode` require a network request right now?
 *
 * False when we already hold the fields, when a request for them is already in
 * flight, when the mode does not exist, or when the index told us the server
 * cannot serve it. A mode the index marked unavailable must never be requested:
 * the request is guaranteed to 404, and the failure would read to the user as a
 * broken app rather than as one missing mode.
 *
 * `>= MAX_MODES` is deliberately not a limit on how many modes load in total --
 * the cache is unbounded and every mode loads on first view. It only bounds how
 * many requests a single page can have outstanding.
 */
export const MAX_IN_FLIGHT = 3;

export function wantsFetch(index, mode, inFlightCount = 0) {
  if (!mode) return false;
  if (mode.fields) return false;
  if (mode.loading) return false;
  if (mode.available === false) return false;
  if (inFlightCount >= MAX_IN_FLIGHT) return false;
  return true;
}

/**
 * The URL for a mode's field payload.
 *
 * `index` comes from the server's mode table and is used verbatim, not as
 * `mode.index + 1`: the table's numbering is the solver's, and a job whose
 * modes do not start at 1 must still fetch the right files.
 */
export function modeFieldUrl(jobId, index) {
  return `/jobs/${jobId}/slices/${index}.json`;
}

/**
 * Record an arrival. Kept here rather than inline in the viewer so the
 * "late arrival after the user moved on" case is a thing that can be asserted:
 * the payload is stored against its own mode, not against whatever is selected
 * when it lands.
 */
export function storeFields(mode, payload) {
  mode.fields = payload;
  mode.loading = null;
  return mode;
}
