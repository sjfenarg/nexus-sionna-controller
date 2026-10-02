# MSC sensing model

MSC adds a VV reference scattering-center kernel and a shared, VV-referenced
polarization matrix to Sionna RT 2.2.0. The current parameters are seeded random
prototypes, not measurement fits. The manuscript's operating band is 10–15 GHz.

## Install and run

From this controller repository, using its existing uv environment:

```powershell
uv sync --extra gui --extra dev --link-mode=copy
uv run --no-sync python scripts/install_msc_extension.py
uv run --no-sync python scripts/run_msc.py gui
```

The installer supports **Sionna RT 2.2.0 only**. It copies the versioned sources
from `sionna_extensions/msc/` into the selected Python environment, exports them
through `sionna.rt.rcs`, and installs three small compatibility hooks. It backs
up every affected installed file under the parent `tmp/msc-implementation/`.
Reapply it after rebuilding or reinstalling the environment. Keep extension
changes in the versioned sources and rerun the installer.

On Windows, Dr.Jit requires an LLVM runtime. `run_msc.py` uses an existing
`DRJIT_LIBLLVM_PATH`, or the optional local runtime at
`../tmp/msc-implementation/runtime/llvm22/bin/LLVM-C.dll`. It also puts the JIT
cache under the parent `tmp/`. On another machine, configure an appropriate
LLVM runtime or use the CUDA backend. The launcher does not download software.

To run a saved request:

```powershell
uv run --no-sync python scripts/run_msc.py cli --config ../tmp/msc-implementation/controller-msc.json
```

The GUI object selector includes:

| Object name | Centers | Dimensions: length × width × height (m) |
|---|---:|---|
| `HUMAN_MSC` | 1 | 0.5 × 0.5 × 1.75 |
| `HUMAN_MULTI_MSC` | 5 | 0.5 × 0.5 × 1.75 |
| `CAR_MSC` | 5 | 5 × 2 × 1.6 |
| `CAR_SP_MSC` | 1 | 5 × 2 × 1.6 |
| `AGV_MSC` | 5 | 1 × 0.5 × 0.5 |
| `AGV_SP_MSC` | 1 | 1 × 0.5 × 0.5 |

Set the GUI band to, for example, `12.4:12.6:65:MSC`. RCS lobe previews use the
mean configured frequency. The default application band is outside the MSC
manuscript band and will produce a warning when MSC is simulated.

## Controller options

Existing requests use the same `sensing` object:

```json
{
  "object_name": "CAR_MSC",
  "sensing": {
    "msc_parameter_seed": 42,
    "random_phases": false,
    "random_xpr": false,
    "random_cpr": false
  }
}
```

`msc_parameter_seed` selects fixed prototype parameters and participates in the
polarization realization. The simulation seed selects stochastic draws. Flags
default to false, matching the deterministic default of the TR38901 workflow.
Enable all three flags for stochastic XPR, CPR and relative phases. VV keeps a
unit CPM entry and its propagation phase.

`random_sigma_s` is rejected for MSC because that factor is absent from the MSC
formulation. The legacy `model_type` must remain at its default of 2. Mesh and
dimension overrides follow the existing target geometry workflow.

## Native Sionna API

```python
from sionna.rt.rcs import MSCSensingTarget, RCSSolver

car = MSCSensingTarget(
    "car", "vehicle-multi-sp", parameter_seed=42,
    length=5, width=2, height=1.6, position=[0, 0, 0.8],
)
scene.add(car)
paths = RCSSolver()(scene, max_depth=1, samples_per_sp=1, seed=42)
a, tau = paths.cir(normalize_delays=False, out_type="numpy")
```

Native object types are `human`, `human-single-sp`, `human-multi-sp`,
`vehicle-single-sp`, `vehicle-multi-sp`, `agv-single-sp`, and `agv-multi-sp`.
Vehicle/AGV point positions and frames match the native TR38901 templates.
TR38901 has no five-point human; MSC supplies an additional face-center template.

`MSCRCS(k_i, k_s, seed)` returns nonnegative VV RCS in square meters.
`MSCCPM(k_i, k_s, seed)` returns real and imaginary 2×2 matrices.
`MSCScatteringModel` composes them through Sionna's `ScatteringPoints` class.
`MSCSensingTarget` accepts `parameters` and `polarization_laws` overrides for
future fits; `scattering_model.parameter_dict()` records the current values.

## Frequency and interpretation

The installed solver hook binds MSC kernels to `scene.frequency` at each solve.
In the controller, frequency-domain and both IFFT modes solve **each absolute RF
bin** so that the aperture factor changes with wavelength. This costs a solve
per bin. MSC IFFT modes use an exact IFFT of that sweep, including when the
requested mode is `pdp_ifft_gridded`; HDF5 records the actual renderer and delay
spacing. Path coefficients, path delays, and binned modes refer to the carrier.

Whole-band monostatic/bistatic ratio laws are used in this prototype; subband
fits and spatially correlated polarization evolution are future refinements.
All centers share a target's polarization realization. Independent VH/HV
amplitude draws use the same XPR law; monostatic AGV/car share their relative
cross-channel phase. HH/VV CPR has the positive dB amplitude sign.

The specular cosine is clamped to zero beyond 90° to support fractional
exponents. Aperture offsets are the scattered direction's local azimuth and
elevation minus those of the reflected incident direction. These explicit
conventions should be checked when importing fitted manuscript parameters.
The solver supplies geometric phase once, using Sionna's phase convention.

HDF5 metadata includes `msc_calibrated=false`, the prototype seed, actual center
and polarization parameters, and whether the response used an RF sweep.

## Validation from the parent workspace

```powershell
.\nexus-sionna-controller\.venv\Scripts\python.exe -B validation\msc\run_tests.py
.\nexus-sionna-controller\.venv\Scripts\python.exe -B validation\msc\compare_cir.py
```

The parent `validation/msc/` folder holds numerical checks, actual solver tests,
controller/HDF5 tests and the car comparison. Generated data stays under parent
`tmp/`; the report and plots stay under parent `reports/`.
