# MSC sensing model

MSC adds a signed reflecting-surface VV scattering-center kernel to
Sionna RT 2.2.0. The `AGV_MSC_K5` target uses the fitted five-center AGV model from run
`20261110T092624418248Z_6d75dba4b4e5`. Other MSC targets use seeded prototypes. The manuscript's operating band is 10–15 GHz.

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
| `AGV_MSC_K5` | 5 | 0.379915 × 0.748699 × 0.595169 |
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

`msc_parameter_seed` selects fixed prototype parameters for untrained targets.
The fitted AGV parameters are independent of this seed. MSC uses deterministic
VV only; keep polarization randomization flags disabled.

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
`MSCSensingTarget` accepts custom `parameters` or `from_components` input; `scattering_model.parameter_dict()` records the current values.

## Frequency and interpretation

The installed solver hook binds MSC kernels to `scene.frequency` at each solve.
In the controller, frequency-domain and both IFFT modes solve **each absolute RF
bin** so that the aperture factor changes with wavelength. This costs a solve
per bin. MSC IFFT modes use an exact IFFT of that sweep, including when the
requested mode is `pdp_ifft_gridded`; HDF5 records the actual renderer and delay
spacing. Path coefficients, path delays, and binned modes refer to the carrier.

The signed per-lobe aperture uses direction cosines and fixed exponent two,
with a VV Jones sign adapter. The solver supplies geometric phase once.
See [kernel and frame conventions](msc-manuscript-alignment.md) for the exact
training alignment, elevation convention and vertical-normal frame handling.
HDF5 metadata records the actual center parameters and RF sweep behavior.

## Validation from the parent workspace

```powershell
.\nexus-sionna-controller\.venv\Scripts\python.exe -B validation\msc\run_tests.py
.\nexus-sionna-controller\.venv\Scripts\python.exe -B validation\msc\compare_cir.py
```

The parent `validation/msc/` folder holds numerical checks, actual solver tests,
controller/HDF5 tests and the car comparison. Generated data stays under parent
`tmp/`; the report and plots stay under parent `reports/`.


## Trained AGV K5 in GUI and CLI

Select `AGV_MSC_K5` in the GUI object list, or pass
`--object agv:AGV_MSC_K5:0,0,0.29758435278799084` to the CLI. The position is
always the target center; this example places the fitted cuboid bottom at zero.
Existing `AGV_MSC` names resolve to the same fitted target and are normalized
when loaded from JSON or the CLI. There is no random-parameter fallback for it.

The package includes the five original component dictionaries, reference
frequency and source model/engine hashes in `data/msc/agv_k5.json`. Optimizer
histories and measurements are not included. Source model SHA-256:
`2b6b78ac33280ec3b66b3bea2df608bfad3ecd0094716ebfae09505f7a1d9c4f`.
GUI RCS previews and simulations use the same factory. The fitted positions,
weights, widths and normals are unchanged. Cuboid size is inferred from the
positions; custom dimensions or meshes are rejected for this fixed target.
`msc_parameter_seed` does not randomize the trained model. Only VV is used;
the source file's HH entry is not imported.

HDF5 `msc_parameters_json` records `trained=true`, actual coefficients and
`provenance` with the source hashes/run. `calibrated=false` and
`measurement_validated=false` distinguish fitted coefficients from independent
measurement validation; the other MSC targets remain marked as prototypes.

## WSL CPU/GPU selection

Keep the WSL environment separate from Windows using `UV_PROJECT_ENVIRONMENT`.
The launcher selects the optional Windows LLVM DLL only on Windows. GPU
selection probes an empty Mitsuba scene to establish OptiX readiness before
Sionna initializes its material classes. A CUDA context without functioning
OptiX emits a warning and falls back to LLVM CPU; result metadata records the
actual Mitsuba variant. `--use-gpu off` explicitly selects CPU.

On the checked WSL installation, NVIDIA-SMI detects the RTX 5070 but the exposed
`libnvoptix.so.1` is a loader lacking `optixQueryFunctionTable`; GPU ray tracing
is unavailable. Merely setting `DRJIT_LIBOPTIX_PATH` to that loader does not fix
it. See the [Mitsuba WSL OptiX instructions](https://github.com/mitsuba-renderer/mitsuba3/blob/master/docs/src/optix_setup.rst)
for driver-library compatibility; no Windows driver or system library changes
are made by this launcher.
