# results-api

## ADDED Requirements

### Requirement: A time-domain job's object outline SHALL be its own artifact, derived from the upload

A time-domain job's object outline SHALL be served as its own artifact from the
job's own stored upload, and a finished time-domain job's status SHALL name
that artifact's URL. The artifact SHALL be the upload in metres, in the frame
the job's field movie draws in (its `units` and `scale` applied, nothing else),
and SHALL be refused -- never served in some other frame -- when the job's
upload, units or scale cannot be read, with the reason written to the server
log. It SHALL be served only for a finished job, and SHALL NOT be written by the
solver.

Why: the solver writes no such file, so serving one derived on request is what
gives jobs already solved their outline; the artifact is separate from the
field and frame artifacts for the same reason the cavity tool's surface is --
a client that draws no field can still draw the object.

#### Scenario: a client asks for a finished job's outline

- **WHEN** a finished time-domain job's outline is requested
- **THEN** it is served as geometry in metres, in the movie's own frame, and
  the job's status names its URL

#### Scenario: an unfinished or unreadable job

- **WHEN** the job is not finished, has no stored upload, or its units or scale
  cannot be read
- **THEN** the response is a 404 -- a 502 when the store itself fails -- and never
  geometry in some other frame; when it is the upload's units or scale that could
  not be read, the reason is written to the server log, which is where a reader
  finds why
