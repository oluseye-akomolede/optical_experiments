# mesh-upload

## ADDED Requirements

### Requirement: The app accepts `.obj` uploads only

The app SHALL accept an upload named `meshfile` with suffix `.obj` and no other
format. Source: `web/meshing.py`, `web/config.py`.

#### Scenario: a non-obj file is uploaded
- **WHEN** the uploaded file's suffix is not `.obj`
- **THEN** the upload is refused with a message naming the supported format

#### Scenario: the file is empty
- **WHEN** the uploaded file has no bytes
- **THEN** the upload is refused as empty

#### Scenario: the file is over the size limit
- **WHEN** the uploaded file exceeds 50 MB
- **THEN** the upload is refused with the limit in the message

### Requirement: Units and scale are applied once, at the form

The object's units SHALL come from a fixed table
(`m`=1, `cm`=1e-2, `mm`=1e-3, `um`=1e-6, `nm`=1e-9, `in`=0.0254). A free scale
SHALL multiply every vertex before anything measures the object. The declared
smallest feature SHALL carry its OWN unit, chosen from a shorter list
(`nm`, `um`, `mm`) independent of the object's units. Source:
`dockerfiles/maxwell_solver/mse_common.py`, `web/app.py`, `web/config.py`.

#### Scenario: an object in mm with a feature in um
- **WHEN** a 10 mm object declares a 10 um feature
- **THEN** the feature's metres are computed with its OWN unit (1e-5 m), not the
  object's

#### Scenario: a scale is applied
- **WHEN** a scale factor is supplied
- **THEN** it multiplies the vertices once
- **AND** the estimate computes size x scale x unit_scale, never applying the
  unit twice

### Requirement: A multi-part `.obj` is split by named groups

The app SHALL split a multi-part upload by its `o`/`g` group names, and SHALL
refuse an upload whose named groups and closed parts do not correspond
one-for-one. The split SHALL use connected components (union-find on face
adjacency), not a graph engine. Source: `web/mse_parts.py`.

#### Scenario: a group name is repeated
- **WHEN** the `.obj` names the same group more than once
- **THEN** the upload is refused, naming the duplicate

#### Scenario: faces outside any group
- **WHEN** the file has faces belonging to no named group
- **THEN** the upload is refused, telling the user to add an `o <name>` line

#### Scenario: unseen faces before the first group
- **WHEN** faces appear before any `o`/`g` line
- **THEN** they are a part named `(unnamed)`, and the count check still applies

### Requirement: Every part needs a refractive index

For a multi-part upload, the app SHALL require a refractive index for every
part, named by the part's name (not its index), and SHALL refuse an unknown part
name. Indices SHALL be numbers between 1.0 and 10.0. Source: `web/app.py`.

#### Scenario: a part is missing its index
- **WHEN** an upload names parts but omits one's refractive index
- **THEN** the upload is refused, naming the part(s) without an index

#### Scenario: a form names a part not in the upload
- **WHEN** the form supplies an index for a name not among the upload's parts
- **THEN** the upload is refused, listing the upload's actual parts

#### Scenario: an index is out of range
- **WHEN** a part's refractive index is below 1 or above 10
- **THEN** the upload is refused with the allowed range

### Requirement: An open surface is refused before solving

The app SHALL refuse an upload whose surface is not watertight, with a message
explaining that the object is treated as a cavity and must be sealed. Source:
`web/meshing.py`.

#### Scenario: the uploaded object has holes
- **WHEN** the surface is not closed
- **THEN** the upload is refused before any job is created

### Requirement: The upload is estimated and routed, or refused as too detailed

The app SHALL estimate the element count from the ENCLOSED volume, at the
effective edge length, with a conservative coefficient, and SHALL:

- ROUTE the job to the GPU worker when the estimate reaches the GPU threshold;
- otherwise ROUTE it to the cluster;
- REFUSE with a plain-English message when the estimate exceeds the refusal
  threshold (twice the real cap, because the estimate can be up to twice high).

The estimate SHALL return `0`, meaning "do not route on this", when the surface
is unusable or the formula fails -- not a small number. Source: `web/meshing.py`,
`dockerfiles/maxwell_solver/mse_common.py`, `web/config.py`.

#### Scenario: the object is too detailed
- **WHEN** the estimate exceeds the refusal threshold
- **THEN** the upload is refused, naming the estimate and the cap, and asking
  for a simpler model

#### Scenario: the estimate could not be computed
- **WHEN** the surface is unusable or the formula raises
- **THEN** the estimate is `0` and the job is not routed on that value

#### Scenario: a multi-part upload is estimated
- **WHEN** the upload has multiple parts
- **THEN** the estimate sums the FILLED volumes (a nested core is not
  double-counted) and uses the smallest effective edge

### Requirement: Refusals reach the user as plain English

Every refusal SHALL be a plain-English message shown to the user (a flash or a
JSON error), and SHALL NOT leak a stack trace or an internal token. Source:
`web/app.py`, `web/meshing.py`.

#### Scenario: a refusal is shown
- **WHEN** any upload validation fails
- **THEN** the user sees a message explaining what to change and how
