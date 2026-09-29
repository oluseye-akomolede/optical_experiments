# results-api

## ADDED Requirements

### Requirement: Results are listed from the store, not a database index

The results API SHALL decide what a finished job offers by LISTING the job's
store prefix, not by reading a database index. A field SHALL be advertised as
available only when its artifact is present in the store. Source:
`web/app.py`, `web/store.py`.

#### Scenario: a field's artifact is absent from the store
- **WHEN** the results API builds the list of a job's fields and an artifact is
  not in the store
- **THEN** that field is not advertised as available

#### Scenario: the store cannot be listed
- **WHEN** the store listing fails
- **THEN** the API does not advertise fields it could not confirm

### Requirement: The index and the fetcher resolve one filename

The list route and the fetch route SHALL agree on the exact filename of every
artifact they name. A field advertised as available SHALL be fetchable by the
same job and component, or the two routes have disagreed. Source: `web/app.py`.

#### Scenario: an advertised field is fetched
- **WHEN** a client requests a field the list route advertised as available
- **THEN** the fetch succeeds and returns that artifact
- **AND** the two routes resolve the same object name

### Requirement: Field artifacts are served as opaque binary

A field artifact SHALL be served with its bytes intact and a content type that
does not invite the browser to reinterpret it. Source: `web/app.py`.

#### Scenario: a client fetches a mode field
- **WHEN** a field artifact exists
- **THEN** the response body is the artifact's bytes and the client parses it,
  not a placeholder

### Requirement: The surface artifact is separate from the field artifacts

The job's surface geometry (for the viewer's object) SHALL be served as its own
artifact, distinct from the mode field data, so a client that draws no field can
still draw the object. Source: `web/app.py`, `web/store.py`.

#### Scenario: a client asks only for the surface
- **WHEN** a client requests the job's surface artifact
- **THEN** it receives the geometry without any field data
