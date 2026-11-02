## ADDED Requirements

### Requirement: A shipped materials library SHALL supply cited parameters, overridable per part
The solver SHALL ship a materials library with at least: **Si, GaAs, InP, SiC, SiO₂, Al₂O₃,
HfO₂, Au, Ag** and **air**. Each entry SHALL record, with its source, the static relative
permittivity ε_s and, for a semiconductor, the intrinsic carrier density n_i at 300 K, the
electron and hole mobilities, the electron and hole lifetimes and the radiative recombination
coefficient. Every parameter SHALL carry a citation to the source it was read from; a parameter
that is a common approximation rather than a measured value SHALL say so. A request SHALL be
able to override any parameter per part, and the manifest SHALL record which values came from
the library and which were overridden.

The values themselves are NOT specified here: each SHALL be read from its cited source before
it is written into the library, and a test SHALL check a sampled subset against that source.
A number quoted from memory SHALL NOT ship.

#### Scenario: A library material needs no parameters from the uploader
- **WHEN** a part names a semiconductor from the library and provides nothing else
- **THEN** the request is accepted and the manifest records the library as each parameter's source

#### Scenario: A per-part override wins and is recorded
- **WHEN** a part names a library material and also gives its own ε_s
- **THEN** the solve uses the given ε_s, and the manifest records that part's ε_s as overridden

#### Scenario: An uncited library entry is refused by its own test
- **WHEN** a library entry lacks a source for one of its parameters
- **THEN** the materials test fails naming the material and the parameter

#### Scenario: An unknown material is refused
- **WHEN** a part names a material that is not in the library and gives no parameters
- **THEN** the request is refused with `ValueError` naming the material and listing the library
