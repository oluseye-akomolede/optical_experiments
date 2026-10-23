## MODIFIED Requirements

### Requirement: Dispersive metals SHALL follow their published model

Gold, silver, aluminium, copper and chromium SHALL be modelled by the
Lorentz-Drude fits of Rakić et al., Applied Optics 37, 5271 (1998), with
ε∞ = 1 and the paper's parameters in eV (ω_p, f_j, Γ_j, ω_j), stepped with
auxiliary differential equations. Gold SHALL also be available as the models fitted to Palik's 1985 table
and to Johnson & Christy's 1972 table (see "A second gold model SHALL fit
Palik's table" and "A third gold model SHALL fit Johnson & Christy's
table"), chosen by name, and
the Rakić fits stay the default so an unaltered run models gold as it did
before. ADE storage SHALL cover only the metal cells'
bounding box. An E sample on a metal face SHALL take the metal. A run whose
pulse band leaves the fitted range of the model it selected for a metal in the
scene SHALL be refused before it is queued, the refusal naming that model's
source.

#### Scenario: the stepped metal has the model's susceptibility
- **WHEN** a metal's term is driven by a sinusoidal field at 20 time steps
  per radian (Au at 1 and 3 eV, Ag and Cr at 2 eV)
- **THEN** the polarisation it builds matches the model's χ(ω) within 1.5%

#### Scenario: a face sample takes the metal and eps_inf = 1
- **WHEN** a single metal node sits next to a dielectric
- **THEN** each E component takes the metal at exactly the two samples whose
  segment touches the node
- **AND** those samples step with ε∞ = 1 whatever the neighbouring permittivity

#### Scenario: a metal near a plane wave's box is refused
- **WHEN** a metal region lies within 3 cells of a plane wave's box faces
- **THEN** the run is refused with a message naming a dispersive metal and
  its node

#### Scenario: the material table matches the paper
- **WHEN** ε(ω) is computed from the library at 12 wavelengths per metal,
  spread through the published table from its shortest to its longest
- **THEN** n + ik matches an independently published Rakić Lorentz-Drude n, k
  dataset (cited in the test) within 1e-3 (relative) at every one, tight
  enough that a 10% typo in a single model parameter fails

#### Scenario: benchmark V4
- **WHEN** a pulse reflects at normal incidence from a gold slab at least 10
  skin depths thick, with at least 4 cells per skin depth, on an `open` travel
  axis with periodic transverse faces (so the slab is infinite, as the Fresnel
  result assumes)
- **THEN** the reflectance, measured as |scattered/incident|² at a
  scattered-field probe, is within 0.02 (absolute) of the analytic Fresnel
  reflectance for the same model, over the band where the incident spectrum is
  at least 1e-2 of its peak

#### Scenario: the V4 bench judges only what it measured
- **WHEN** the V4 bench runs
- **THEN** it takes the incident spectrum from a reference run with the same
  metal term over an empty region (so both runs share one time step), and
  transforms both traces without a window
- **AND** it fails if either trace still holds more than 1e-3 of its peak in
  its last 5% (a cut-off record), or if the ratio is not finite in the band
- **AND** it exits 2 without running when the pulse band leaves the metal's
  fit, and exits 2 rather than passing when the setup has fewer than 4 cells
  per skin depth or a slab under 10 skin depths
- **AND** it fails unless the judged bins span the band it checked the metal
  and skin depths over, each end within one bin and with no hole, since a
  shorter record gives fewer, wider bins over the same band
- **AND** it runs with the solver's default 10-cell CPML

#### Scenario: a metal slab across periodic faces is uniform
- **WHEN** a plane wave meets a metal slab that spans every periodic axis
- **THEN** E inside the slab is the same at every transverse sample within
  1e-6 of its peak (the metal term advances every sample the E update does,
  including the last row of a periodic axis)

### Requirement: The bench SHALL choose the gold model by name

`scripts/paper1_nanowire.py` SHALL take `--gold-data {rakic,palik,jc}`,
defaulting to `rakic` so earlier runs reproduce, SHALL use the choice wherever
gold is selected (the scene's mirror material, the reported skin depth, and the
estimate), and SHALL log which model the run used.

#### Scenario: the flag selects the material

- **WHEN** the script is parsed with and without `--gold-data palik`
- **THEN** the scene's mirror material is `Au-palik` and `Au` respectively, the
  estimate prices the same material, and the report names the model it ran
- **AND** with `--gold-data jc` the mirror, the layout's metals and the report
  name `Au-jc`, and a pulse outside that model's fit is refused in its words

## ADDED Requirements

### Requirement: A third gold model SHALL fit Johnson & Christy's table

The metal library SHALL offer `Au-jc`, a Lorentz-Drude `Metal` fitted to
P. B. Johnson and R. W. Christy's 1972 gold table, so the owner-approved
paper-1 run can be answered with the table the owner chose. The model SHALL be
held to the same criteria as Palik's: smooth as well as accurate, fitted under
a damping floor and judged on a dense grid between the table's rows, because a
pole that fits every row can ring between two of them. The table's 9 rows
between 600 and 1300 nm (616.8-1216.0 nm, in `data/jc_au_nk.csv`) have no gap
and every one of them SHALL be fitted; the fit's range SHALL be the rows it is
fitted to (1.02-2.01 eV). The fit SHALL be regenerated by the existing
`scripts/fit_palik_gold.py`, generalised with a data, name and source argument
rather than copied, and the printed literal SHALL be the library entry.
Adding the model SHALL NOT change the Rakić or Palik entries, and every
place that accepts a library metal SHALL accept the new name.

#### Scenario: the fit reproduces the table

- **WHEN** the model's complex permittivity is compared with every copied table
  point
- **THEN** every point is within 3% relative error, and the test states the
  measured maximum, where it falls, and the fitted range

#### Scenario: the model is smooth between the table's rows

- **WHEN** the model is evaluated every 2 nm over 600-1400 nm
- **THEN** Re eps is negative and Im eps positive at every point, and between
  the table's own rows the model's |eps| is within 1.5x of the linear
  interpolation between the neighbouring rows
- **AND** every pole has Gamma >= 0.05 eV, so nothing in the model can ring
  between two table rows

#### Scenario: the model is lossy, positive and stable

- **WHEN** the model is evaluated across the fitted band and its ADE term is
  checked at the bench's spacings
- **THEN** Im eps is positive at every point, every pole has f > 0 and
  Gamma > 0, and the growth factor is below 1 + `GROWTH_TOLERANCE`

#### Scenario: a band outside the fit is refused naming its source

- **WHEN** a pulse band leaves `Au-jc`'s fitted range, at the bench or in the
  app
- **THEN** the refusal names Johnson & Christy's table and its 1.02-2.01 eV
  range, not Palik's or Rakić's

#### Scenario: the two gold tables are distinguished where the paper uses them

- **WHEN** the two fitted models are compared at the paper's 905 nm
- **THEN** `Au-jc`'s Im eps is below `Au-palik`'s, by the measured factor the
  test states, and the comparison is recorded as a property of the tables, not
  as a change to an answered question

