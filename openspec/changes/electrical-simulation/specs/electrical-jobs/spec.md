## ADDED Requirements

### Requirement: An electrical job SHALL be stored as kind `elec` with a keyed request
An `elec` job SHALL be stored in the jobs table with `kind = 'elec'` and its request in an
`elecs` JSON column, NULL for every other kind. The request keys SHALL be exactly the worker's
`ELEC_REQUEST_KEYS` list: `contact` (`gate` | `pin`), `parts` (per uploaded part: `role` one
of `semiconductor`, `dot`, `oxide`, `electrode`, `air`; `eps_static`, which an electrode may
omit; for `pin`, `doping_cm3` signed, + for donors), `electrodes` (part name → the swept
terminal or `ground`), `voltages_v` (the sweep, at most 41 values), `pulse` (null, or a
step/pulse waveform for `pin`), `dot` (centre in the upload's units, the dot material and the
dipole orientation) and `spectrum` (null, or the stored name of an attached
`purcell_spectrum.json`). Every key SHALL be present. The worker SHALL refuse a missing key,
naming every missing key, and an `elecs` that is not an object. Keys outside the list SHALL be
ignored. The column SHALL be added idempotently by `db/001_schema.sql` and dropped by a
rollback script of its own.

#### Scenario: A missing request key is named
- **WHEN** a claimed `elec` job's `elecs` lacks one or more keys
- **THEN** the worker refuses it with a message naming every missing key

#### Scenario: A pin request without doping is refused at upload
- **WHEN** the form posts `contact = pin` and a semiconductor part has no `doping_cm3`
- **THEN** the upload is refused with a message naming the part, and no row is written

### Requirement: The upload page SHALL offer an electrical arm that posts exactly what the form reads
The mode selector SHALL offer an electrical choice. Its arm SHALL show the shared multi-part
`.obj` upload (file, units, scale), a role, ε_s and (for p-i-n) doping per part, the contact
type, the electrode assignment, the voltage sweep, the optional pulse, the dot centre and an
optional Purcell spectrum file. The names it posts SHALL be the names the server reads.

#### Scenario: Every posted field is read
- **WHEN** the electrical arm is submitted with every field filled
- **THEN** the server's parsed request has a value for every posted name, and the posted name
  set equals the set the server reads

### Requirement: An `elec` job SHALL run on the cluster as its own bounded Job shape
An `elec` row SHALL be created with `target = cluster`. The launcher SHALL build an
electrical Job for it with requests equal to limits, at most `cpu: 4` and `memory: 16Gi`,
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to the CPU limit, and an
`activeDeadlineSeconds`. Before solving, the worker SHALL price the run with the electrostatics
and drift-diffusion budget rules and SHALL fail the row terminally, with a message naming the
estimate and the limit, when the estimate exceeds the Job's memory limit. It SHALL never retry
a priced refusal at a smaller resolution on its own.

#### Scenario: The launcher picks the shape by kind
- **WHEN** the launcher is handed an `elec` row
- **THEN** it creates the electrical Job, not a cavity Job, and the Job's requests equal its
  limits

#### Scenario: An oversized request is refused before solving
- **WHEN** an `elec` request's estimated memory exceeds the Job's limit
- **THEN** the row fails with a message naming both figures, and no solve starts

### Requirement: A finished `elec` job's results SHALL be flat files listed from the store
The worker SHALL write `elec_manifest.json` (the request echo, the mesh node counts, the solve
path per stage and the peak RSS), `sweep.json` (per voltage: V, the field at the dot,
E_X, λ_X, the overlap and, when a spectrum is attached, F and Γ, otherwise null), and, for
`pin`, `iv.json` (V, the current per contact) and, when a pulse was given, `timing.json`
(t, g(t), P(t), the mean delay and the rise time). The results API SHALL advertise only the
files present in the store, by the existing listing rule.

#### Scenario: No spectrum, no rate
- **WHEN** an `elec` job without an attached spectrum finishes
- **THEN** `sweep.json` holds E_X and λ_X at every voltage and null F and Γ, and the page
  says that no Purcell spectrum was attached

#### Scenario: A gate job writes no I-V
- **WHEN** a `gate` job finishes
- **THEN** the store holds no `iv.json`, and the results API does not advertise one

### Requirement: A finished `elec` job's page SHALL plot the sweep, the I-V and the timing
The page SHALL plot E_X (and λ_X on a second axis) against V, F and Γ against V when present,
the I-V on a log-current axis for `pin`, and P(t) with g(t) when present. Each plot SHALL have
one point per stored value. The page SHALL show the manifest's node counts and solve paths.

#### Scenario: The plotted points are the stored values
- **WHEN** a finished `elec` job's page is rendered from a fixture `sweep.json` of 7 voltages
- **THEN** the E_X plot carries 7 points whose values are the stored E_X, in the stored order

#### Scenario: A plot read from the wrong key fails the page test
- **WHEN** the page's E_X plot is fed from the `field` key instead of `E_X`
- **THEN** the page test fails on the plotted values
