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

MSC is VV-only. The Sionna Jones adapter has a unit VV entry and zero VH,
HV and HH entries in the scattering-point basis. No random polarization law or
polarization ratio is evaluated or exported. The controller disables the generic
polarization randomization options for MSC (they still apply to 3GPP targets).
Direct MSC API calls requesting random polarization fail explicitly. Rotated
antenna/point bases and intervening propagation still require care: this adapter
is a VV projection in the point basis, not a calibrated full polarimetric model.

All default parameters remain uncalibrated. The manuscript contains unresolved
wording: it mentions a variable exponent although the equation specifies 2, and
frequency-independent amplitudes although the shaping kernel depends on frequency.
Its approximate effective-size formula is implemented literally; for unnormalized
sinc it does not give an exact half-power width. Clarify these points before fitting.
