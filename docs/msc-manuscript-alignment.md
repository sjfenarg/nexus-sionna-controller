# MSC manuscript alignment

The current kernel uses the manuscript's literal squared cosine, including its
smoothly attenuated rear response. The legacy serialized `n` field must equal 2;
older parameter sets with other exponents must be refitted, not silently reused.
Side/back lobes and aperture factors use unnormalized sin(x)/x.

Shaping offsets are evaluated in a specular-aligned orthonormal frame. The
`shaping_axis` vector, expressed in the center frame, defines the first transverse
axis by projection. At parallel alignment a deterministic Cartesian fallback is
used. This frame convention must also be used during parameter fitting. A globally
continuous transverse frame at every direction is not guaranteed at this degeneracy.

`MSCSensingTarget(center_positions=..., center_orientations=..., parameters=...)`
supports arbitrary target-frame center locations and counts through the Python API.
Positions have shape (P, 3) in meters; orientations have shape (P, 3) in radians.
Normals and shaping axes are in the individual center frames. Omitting positions
retains the one/five-point geometry templates. This is not a fitted-parameter file
import feature in the GUI or request JSON.

RCSSolver supplies physical propagation phase and delay. The RCS callable returns
power and must not add geometric phase again. Absolute path delays and the paper's
common reference differ by a common phase only for the same VV target response.
Frequency-resolved controller output reruns the solver at each RF frequency;
carrier CIR/path-bin outputs freeze aperture amplitudes at the carrier.

Polarization ratios use a shared target realization, unit VV, independent cross
ratio draws, and positive HH/VV CPR. This is a scalar-profile approximation to the
paper's effective whole-response model. Direction-dependent polarization bases,
per-path mode classification and multipath can prevent exact whole-response power
ratios. Frequency-subband law selection and spatially correlated dynamic
polarization are not implemented. Do not interpret shared center CPMs as fitted
center-specific polarimetry.

All default parameters remain uncalibrated. The manuscript contains unresolved
wording: it mentions a variable exponent although the equation specifies 2, and
frequency-independent amplitudes although the shaping kernel depends on frequency.
Its approximate effective-size formula is implemented literally; for unnormalized
sinc it does not give an exact half-power width. Clarify these points before fitting.
