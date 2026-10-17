# time-domain

## ADDED Requirements

### Requirement: A run SHALL report the permittivity its update applies at the point source

`fdtd.simulate` SHALL return `"source_eps"`, computed by
`fdtd.source_permittivity` from the run's own `eps_r`, `ade_terms` and
`boundary`: for a point source, the relative permittivity the E update
applies at the source's own component sample; `None` for a plane wave. The
readers (`delivered_power`, `delivered_power_spectrum`, `purcell_spectrum`,
`dipole_moment`) SHALL keep their scalar `eps_r`, to which it is passed.

#### Scenario: a homogeneous dielectric
- **WHEN** a point source runs in a homogeneous eps 1, 2.25 or 3.17^2 grid
- **THEN** `"source_eps"` equals `eps_r[cell]` to the bit, and
  `delivered_power` read with it equals the scalar reading to the bit

#### Scenario: a dielectric face
- **WHEN** a CW Ez dipole on a 37^3 open grid (dx 10 nm, 16 cells per
  wavelength) sits on the top node layer of an eps 4 block, its sample
  reaching into vacuum
- **THEN** `"source_eps"` is 4, and `delivered_power` read with it equals the
  phasor box's surface power within 1% (measured +0.0023; vacuum +0.0016)

#### Scenario: a metal face
- **WHEN** the same dipole sits in eps 4 with a lossless Lorentz node above
  it along its own axis
- **THEN** `"source_eps"` is 1, `delivered_power` read with it equals the
  surface power within 1% (measured +0.0008), and read with the cell's 4 it
  is four times it (measured +3.0032)

### Requirement: The applied permittivity SHALL be the node value unless a metal owns the sample

In a dielectric `fdtd.source_permittivity` SHALL return `eps_r[cell]`
exactly. On a sample a Lorentz-Drude metal owns -- either node of the
segment metal, by the one rule `LorentzDrude.bind` rescales with
(`LorentzDrude.samples`) -- it SHALL return that metal's `eps_inf`, 1.

#### Scenario: which samples a metal claims
- **WHEN** a gold node is at the far end of the source's own segment, or at
  its own node
- **THEN** the value is 1; with the gold beside the segment, or at the
  segment below, it is the node value

### Requirement: The paper-1 arm SHALL read with the applied permittivity and print both

`scripts/paper1_nanowire.py` SHALL pass `fdtd.source_permittivity`'s value
to its readings and print it beside the layout's node value on its `source`
line (`eps_r <cell> (cell), <applied> (applied at E<axis>)`); the host index
SHALL stay the square root of the node value.

#### Scenario: a gold node beside the source
- **WHEN** the source's x sample has a gold node at its far end
- **THEN** the line reads the InP value as the cell's and 1 as the applied
  one, and the arm's `eps_r` is 1; with the gold beside it along y both are
  InP's
