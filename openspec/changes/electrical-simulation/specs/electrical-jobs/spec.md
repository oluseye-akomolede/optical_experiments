## ADDED Requirements

### Requirement: An electrical job SHALL accept any device, any parts and any contacts
An `elec` request SHALL describe a general electrical problem, not a fixed device. Every
uploaded part SHALL carry exactly one of:

- a **material**: a name from the shipped materials library, or an explicit
  `eps_static` (a real ≥ 1), and, when the part is a semiconductor, its carrier parameters
  (n_i, electron and hole mobility, electron and hole lifetime, radiative coefficient);
- a **doped material**: a material plus `doping_cm3` (signed, + for donors), which makes the
  part a semiconductor region;
- a **contact**: a conductor held at a voltage, which may be fixed or swept.

There SHALL be at least one contact. The number of contacts SHALL be arbitrary and is not
limited to a gate/pin pair. A part that is neither a material, a doped material nor a contact,
or a contact that also carries a material, SHALL be refused with `ValueError` naming the part.

#### Scenario: A MOS capacitor is a valid request
- **WHEN** a request is one semiconductor slab, one oxide slab and two contact parts
- **THEN** the request is accepted, and it is priced as electrostatics (no doping present)

#### Scenario: Any number of contacts
- **WHEN** a request has four contact parts at four voltages
- **THEN** the request is accepted and the capacitance matrix is 4 × 4

#### Scenario: A part with no material and no contact is refused
- **WHEN** a part carries neither a material entry nor a contact voltage
- **THEN** the request is refused with `ValueError` naming that part

### Requirement: The analysis kind SHALL follow the inputs, or be chosen explicitly
The worker SHALL select the analysis kind from the request: **electrostatics only** when no
part carries `doping_cm3`; **drift-diffusion** when at least one part does. A request MAY set
`analysis` to `"electrostatics"` or `"drift-diffusion"` to override that choice. An explicit
`"drift-diffusion"` on a request with no doping SHALL be refused, naming the missing doping;
an explicit `"electrostatics"` on a doped request SHALL be accepted, and the result SHALL
carry a caveat that carriers were not solved.

#### Scenario: No doping anywhere is electrostatics
- **WHEN** a request has no `doping_cm3` on any part
- **THEN** the job solves electrostatics only and writes no carrier maps

#### Scenario: Doping selects drift-diffusion
- **WHEN** one part carries `doping_cm3`
- **THEN** the job solves drift-diffusion, and the carrier densities are written as maps

#### Scenario: An explicit drift-diffusion with no doping is refused
- **WHEN** `analysis = "drift-diffusion"` and no part carries doping
- **THEN** the job fails with a message naming the missing doping, before any solve

### Requirement: Multi-terminal biasing SHALL sweep one terminal at a time
`contacts` SHALL map each contact part name to either a fixed voltage in `voltages_v` or
`ground`. Exactly one contact SHALL be the swept terminal; every other contact SHALL be fixed.
`voltages_v` SHALL be the swept terminal's voltages, ascending, at most 41 values. The worker
SHALL solve one linear system per voltage and SHALL report the current into **every** contact
at each voltage, so one sweep gives the full four-terminal response of the fixed contacts as
well. A request with no fixed contact other than the swept one SHALL be refused, because the
problem is then ungauged; the grounded terminal may be the swept one only if another contact
is fixed.

#### Scenario: A sweep reports every contact's current
- **WHEN** a four-contact request sweeps contact A while B, C and D are fixed
- **THEN** `sweep.json` holds A's, B's, C's and D's currents at every voltage

#### Scenario: Two swept terminals are refused
- **WHEN** two contacts are marked swept
- **THEN** the request is refused, naming both

### Requirement: An electrical job SHALL be stored as kind `elec` with a keyed request
An `elec` job SHALL be stored in the jobs table with `kind = 'elec'` and its request in an
`elecs` JSON column, NULL for every other kind. The request keys SHALL be exactly the worker's
`ELEC_REQUEST_KEYS` list: `parts` (per uploaded part: `name`, `material` or `eps_static`, and
for a semiconductor its carrier parameters; `doping_cm3` optional; `contact` optional — the
voltage in `voltages_v` or `ground`), `contacts` (contact part name → the swept terminal or a
fixed voltage or `ground`), `voltages_v` (the sweep of the single swept terminal, at most 41
values), `pulse` (null, or a step/pulse waveform), `dot` (null, or the add-on's dot: centre in
the upload's units, material and dipole orientation) and `spectrum` (null, or the stored name
of an attached `purcell_spectrum.json`). Every key SHALL be present. The worker SHALL refuse a
missing key, naming every missing key, and an `elecs` that is not an object. Keys outside the
list SHALL be ignored. The column SHALL be added idempotently by `db/001_schema.sql` and
dropped by a rollback script of its own.

#### Scenario: A missing request key is named
- **WHEN** a claimed `elec` job's `elecs` lacks one or more keys
- **THEN** the worker refuses it with a message naming every missing key

#### Scenario: A doped part without carrier parameters is refused at upload
- **WHEN** the form posts a part with `doping_cm3` and a material whose carrier parameters are
  absent and not overridden
- **THEN** the upload is refused with a message naming the part, and no row is written

### Requirement: The upload page SHALL offer an electrical arm that posts exactly what the form reads
The mode selector SHALL offer an electrical choice. Its arm SHALL show the shared multi-part
`.obj` upload (file, units, scale); a per-part row choosing material (from the shipped library)
or an explicit ε_s, with the carrier parameters editable; an optional doping; an optional
contact at a voltage or ground; the single swept terminal and its voltage list; the optional
pulse; and the optional quantum-dot add-on's dot centre and Purcell spectrum file. The names
it posts SHALL be the names the server reads.

#### Scenario: Every posted field is read
- **WHEN** the electrical arm is submitted with every field filled
- **THEN** the server's parsed request has a value for every posted name, and the posted name
  set equals the set the server reads

### Requirement: An `elec` job SHALL run on the cluster as its own bounded Job shape
An `elec` row SHALL be created with `target = cluster`. The launcher SHALL build an
electrical Job for it with requests equal to limits, at most `cpu: 4` and `memory: 16Gi`,
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to the CPU limit, and an
`activeDeadlineSeconds`. Before solving, the worker SHALL size the run with the budget rules
for the selected kind and SHALL fail the row terminally, with a message naming the estimate
and the limit, when the estimate exceeds the Job's memory limit. The size check SHALL price
the run from the allocator's own arrays (node and edge counts of the mesh it would build),
not from a table of typical sizes. It SHALL never retry a refused run at a smaller resolution
on its own.

#### Scenario: The launcher picks the shape by kind
- **WHEN** the launcher is handed an `elec` row
- **THEN** it creates the electrical Job, not a cavity Job, and the Job's requests equal its
  limits

#### Scenario: An oversized request is refused before solving
- **WHEN** an `elec` request's estimated memory exceeds the Job's limit
- **THEN** the row fails with a message naming both figures, and no solve starts

#### Scenario: The estimate comes from the mesh, not a table
- **WHEN** the size check prices two requests that differ only in their uploaded geometry
- **THEN** the two estimates differ, because each is computed from its own mesh's arrays

### Requirement: A finished `elec` job's results SHALL be flat files listed from the store
The worker SHALL write `elec_manifest.json` (the request echo, the analysis kind, the mesh node
and tet counts, the solve path per stage and the peak RSS), the field maps
(`potential`, `field` and, for drift-diffusion, `charge_density`, as per-node arrays over the
mesh, stored so the app can colour the surface), and `sweep.json` (per voltage: V, the
potential and field sampled at the requested points or the dot, every contact's current and
capacitance). For drift-diffusion it SHALL additionally write `iv.json` (V, the current per
contact). When a dot is marked it SHALL write the add-on's `stark.json` (V, E_X, λ_X, the
overlap and, when a spectrum is attached, F and Γ, otherwise null) and, when a pulse was
given, `timing.json` (t, g(t), P(t), the mean delay and the rise time). The results API SHALL
advertise only the files present in the store, by the existing listing rule.

#### Scenario: No dot, no Stark file
- **WHEN** an `elec` job with no `dot` finishes
- **THEN** the store holds no `stark.json` and no `timing.json`, and the results API advertises
  neither

#### Scenario: An electrostatics job writes no carrier maps
- **WHEN** an electrostatics-only job finishes
- **THEN** the store holds no `charge_density` map and no `iv.json`

### Requirement: A finished `elec` job's page SHALL plot the sweep, the maps and, when present, the add-on
The page SHALL render the potential, field and (when present) charge-density maps over the
mesh's surface with the existing mode viewer's colour scale. It SHALL plot each contact's
current against V, the capacitance matrix, the Stark sweep (E_X and λ_X on a second axis) and
F and Γ against V when present, and P(t) with g(t) when present. Each plot SHALL have one point
per stored value. The page SHALL show the manifest's analysis kind, node counts and solve
paths.

#### Scenario: The plotted points are the stored values
- **WHEN** a finished `elec` job's page is rendered from a fixture `sweep.json` of 7 voltages
- **THEN** the current plot carries 7 points whose values are the stored currents, in the
  stored order

#### Scenario: A map read from the wrong key fails the page test
- **WHEN** the potential map is fed from the `field` key instead of `potential`
- **THEN** the page test fails on the plotted values
