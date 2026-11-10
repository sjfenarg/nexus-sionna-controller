# MSC reflecting-surface implementation

The MSC kernel now follows the training engine's recommended
`reflecting_surface_per_lobe` mode. It supersedes the literal manuscript kernel.
Only VV is implemented; HH, VH and HV entries in the target Jones adapter are zero.

## Parameter convention

`MSCCenterParameters.from_component(component, reference_frequency_hz)` converts
caller-supplied training component dictionaries. `mu_phi_0` is azimuth and
`mu_theta_0` is elevation, both in degrees. Elevation 0 is horizontal; +90 is +Z.
The normal is `(cos(e)*cos(a), cos(e)*sin(a), sin(e))`. These values are not
Sionna zenith angles or Euler angles. Widths and lobe offsets are also degrees
in these dictionaries. Direct `MSCCenterParameters` fields use radians.

At exactly vertical preferred directions, the normal alone loses the aperture's
orientation in the XY plane. The component importer retains `mu_phi_0` as
`pole_azimuth_rad`. The specular aperture uses the horizontal radial axis
`(cos(phi), sin(phi), 0)` for its elevation-width axis at a pole. This matches
the trained float32 engine's look-at axes, up to signs that cancel in the even
sinc factors, without relying on the residual of `cos(pi/2)`. Non-pole frames
and side/back lobe formulas are unchanged. Direct parameter callers omitting
this optional azimuth retain the legacy pole fallback.

`MSCSensingTarget.from_components(name, components, reference_frequency_hz, ...)`
uses target-local `mu` positions in meters and zero point Euler orientations.
Its component normals and the engine's Duff/look-at frames are then evaluated
in the target coordinate system. Target attitude rotates positions and frames
together. Full supplied constellations are neither mirrored nor deduplicated.
The component factory defaults cuboid length, width and height to twice the
maximum absolute center coordinate on each corresponding axis. Zero-extent
axes keep their template size; explicit dimensions override these defaults.
The target origin and trained center positions stay unchanged.
No trained model file is bundled, selected, opened or imported automatically.
Omitted custom parameters still generate explicitly uncalibrated prototypes.

## Field and frequency behavior

The specular weight is fixed squared cosine of illumination colatitude with
logistic constants 1.3 and 0.3. Side/back illumination kernels use unnormalized
sinc squared with fixed widths 20 and 15. Four signed field lobes at one physical
center each receive their own direction-cosine diffraction and visibility factor.
Aperture sinc is normalized, with HPBW constant 0.44294647068945243. The visibility
factor appears once per lobe. The default factory's frequency is 12.5 GHz, but
supplied component reference frequency is preserved independently of scene carrier.

The RCS callable returns A squared. A per-center VV Jones adapter supplies sign(A),
so the solver's sqrt(RCS) multiplication restores signed A. Other Jones entries
are zero. The solver supplies propagation phase exactly once. Training's
approximate c=3e8 and Sionna's 299792458 must be accounted for in phase comparisons.
A local VV projection does not define a calibrated full polarization model.

RF and IFFT modes retain absolute-frequency solver sweeps. Carrier path/bin modes
freeze frequency-dependent amplitudes at the carrier. Many frequency bins and
centers can be expensive; path-geometry reuse is not implemented.

## Directionality and validation

MSC batches disable reciprocal link canonicalization and preserve requested TX/RX
roles, including reverse UE/BS links. The trained empirical kernel is not generally
symmetric under TX/RX exchange. Ordinary 3GPP/background-only scenes retain their
existing reciprocal behavior when no MSC target is present.

Private validation includes signed sidelobes, elevation conversion and independent
training-engine parity with synthetic component dictionaries. Set MSC_ENGINE_REPO
to a source checkout to enable that optional CPU comparison. No fitted values or
measurement data are required. Solver checks require the installed extension and
an approved Sionna runtime. Source validation alone is not a runtime parity result.
