## ADDED Requirements

### Requirement: The Purcell factor at the Stark-tuned wavelength SHALL be read from a recorded spectrum
`emitter_coupling.purcell_at(spectrum, wavelength_nm)` SHALL read a Purcell spectrum file
(`purcell_spectrum.json`: `wavelength_nm` and `F` arrays of equal length, ascending wavelength,
the source `dipole` orientation, and the run that produced it). The paper-1 bench SHALL write
this file from the F it already prints per wavelength bin. The reader SHALL interpolate F
linearly in wavelength. It SHALL refuse with `ValueError` a wavelength outside the recorded
range (never extrapolate), a spectrum whose arrays differ in length or are not ascending, and a
spectrum whose `dipole` differs from the emitter's.

#### Scenario: A recorded point is returned exactly
- **WHEN** `purcell_at` is asked for a wavelength that is one of the recorded ones
- **THEN** it returns that F exactly

#### Scenario: Out of band is refused
- **WHEN** the Stark-tuned wavelength lies outside the recorded range
- **THEN** `purcell_at` raises `ValueError` naming the wavelength and the range

#### Scenario: A mismatched dipole is refused
- **WHEN** the spectrum was recorded with a z dipole and the emitter is in-plane
- **THEN** `purcell_at` raises `ValueError` naming both orientations

### Requirement: The radiative rate SHALL be the Purcell factor times the free rate, scaled by the overlap
`emitter_coupling.rate(V)` SHALL return Γ(V) = F(λ_X(V))·Γ₀·O(V)/O(0), where λ_X = hc/E_X from
the Stark sweep, O is the electron-hole overlap and Γ₀ = n·ω³·|d|²/(3πε₀ħc³) is the rate of
a dipole d in a homogeneous medium of index n. |d| at zero field and n are inputs with a cited
source, read from the paper before the value is written down.

#### Scenario: The free rate in a homogeneous medium
- **WHEN** Γ₀ is computed for given n, ω and |d|
- **THEN** it equals the closed form to 1e-12 relative, and doubles when n doubles

#### Scenario: Unit Purcell factor and zero field give the free rate
- **WHEN** F = 1 everywhere and V = 0
- **THEN** Γ(0) = Γ₀ exactly

### Requirement: The photon timing SHALL be the injection profile convolved with the decay
`emitter_coupling.timing(g, t, Gamma)` SHALL return the photon-emission density
P(t) = (g ⊛ Γe^{−Γt})(t), normalised so that ∫P dt equals ∫g dt, together with its mean delay
and its 10–90 % rise time.

#### Scenario: An instant injection gives the bare decay
- **WHEN** g is a single-bin impulse at t = 0
- **THEN** P(t) is Γe^{−Γt} within 1 % at every sample after the first

#### Scenario: An exponential injection gives the exact two-rate curve
- **WHEN** g(t) = a·e^{−at} with a ≠ Γ
- **THEN** P(t) = aΓ/(a − Γ)·(e^{−Γt} − e^{−at}) within 1 %

