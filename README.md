# ISAC 6D Sampler

`isac-6d-sampler` is a Python package for building ISAC scenes, running Sionna RT channel simulations, and exporting reference-style HDF5 datasets. It is designed to be used both from a GUI and as a backend API for scripted scene generation.

The package supports:

- Base stations, UEs, and dynamic scene objects.
- Static, linear, and polyline UE/object trajectories.
- XY radiomap grids with a configurable UE template.
- Sionna RT tracing with LOS, specular, diffuse, refraction, GPU, batching, and path-count controls.
- Sionna antenna patterns plus the custom `isac_horn_77_81` pattern.
- Frequency-domain channels, CIR paths, binned PDP, and IFFT delay-domain renderers.
- A PySide6/OpenGL 3D GUI with mesh visualization, transform gizmos, radiomap handles, antenna diagrams, and a channel visualizer.

## Repository Layout

```text
src/isac_6d_sampler/
  core/   Request model, validation, frequencies, trajectories, scenarios, antenna patterns
  sim/    Dry-run simulator, Sionna backend, planner, channel rendering, power helpers
  io/     Reference HDF5 writer and schema validator
  gui/    PySide6 GUI, 3D scene view, channel visualizer, worker thread
tests/    Unit and integration tests
scenarios/
  Outdoor6D_w_car.xml
  meshes/*.ply
docs/     Architecture notes
```

Generated HDF5 outputs, Python caches, local environments, raw `.mat` measurements, and debug folders are ignored by git.

## Setup

Use Python 3.11 or 3.12. The project is managed with `uv`.

```powershell
uv sync --extra gui --extra dev --link-mode=copy
```

Run the test suite:

```powershell
uv run pytest
```

### MSC development

The `msc-model-implementation-in-sionna` branch adds an MSC prototype with
one or five scattering centers for humans, AGVs and cars. Install the extension
into this repository's environment before using an MSC target:

```powershell
uv run --no-sync python scripts/install_msc_extension.py
uv run --no-sync python scripts/run_msc.py gui
```

See [MSC model setup and API](docs/msc-model.md) for the supported Sionna version,
target names, parameters, frequency workflow and validation commands.

## GUI

Launch the GUI with:

```powershell
uv run isac6d-gui
```

or:

```powershell
uv run python -m isac_6d_sampler.gui.app
```

The GUI lets you:

- Load a scenario XML and inspect the scene meshes in 3D.
- Add, select, move, and rotate BSs, UEs, and the car object.
- Edit BS and UE antenna panels separately.
- Use local transform gizmos for position/orientation and trajectory endpoints.
- Configure radiomap bounds by dragging the two rectangle corners.
- Preview antenna diagrams in 3D.
- Run simulations while moving trajectory entities through the currently computed timeframe.
- Open a channel visualizer for stored or just-computed channels.

When a simulation is running, trajectory entities move only in a temporary preview copy. The real configured scene is restored when the worker finishes or fails.

## CLI

Create a dry-run HDF5 sample:

```powershell
uv run isac6d --dry-run --output-dir output --band 77:81:1024:77-81GHz
```

Run a minimal Sionna smoke test:

```powershell
uv run isac6d --output-dir output_sionna_smoke --band 77:77.001:2:smoke --samples-per-src 1 --max-num-paths-per-src 1 --max-depth 0 --diffuse off --sample-id s_smoke
```

Inspect the planned run size without simulating:

```powershell
uv run isac6d --plan-only --radiomap on --rm-x-min 0 --rm-x-max 4 --rm-y-min 0 --rm-y-max 2 --rm-x-spacing 1 --rm-y-spacing 1
```

Create a request template:

```powershell
uv run isac6d --write-template configs/example_request.json
```

Run from a saved GUI/JSON request:

```powershell
uv run isac6d --config configs/example_request.json --samples-per-src 500000 --max-depth 3 --diffuse on
```

Validate an HDF5 output:

```powershell
uv run isac6d --validate-h5 output\scene_20260716_230022.h5
```

## Frequency Bands

Bands use this syntax:

```text
START_GHZ:STOP_GHZ:POINTS[:NAME]
```

Examples:

```powershell
uv run isac6d --dry-run --band 77:78:512:low --band 80:81:512:high
```

The GUI accepts multiple bands separated by semicolons:

```text
77:78:512:low; 80:81:512:high
```

Multi-band requests are supported for `frequency_domain` and `cir_paths`. Delay-bin and IFFT modes require one contiguous uniformly sampled band.

## TX Power

Transmit power is global for every transmitter:

- GUI: `Channel Output -> TX power dBm`
- CLI: `--tx-power-dbm`
- JSON: `sionna.tx_power_dbm`

Default is `44.0 dBm`, matching Sionna RT's default transmitter power.

Stored `h` and `a` coefficients are scaled by `sqrt(Ptx_W)`, so plotted/stored power from `abs(h)^2` or `abs(a)^2` scales linearly with transmit power. A `+10 dB` TX power change raises plotted channel power by `+10 dB`.

## Antenna Patterns

Supported pattern names:

- `iso`
- `dipole`
- `hw_dipole`
- `tr38901`
- `isac_horn_77_81`

The custom `isac_horn_77_81` pattern is registered inside Sionna RT and used as a single-element pattern by `PlanarArray`. It models a 77-81 GHz horn with:

- 10.3 dBi maximum gain
- 56 degree horizontal 3 dB beamwidth
- 28 degree vertical 3 dB beamwidth

Sionna antenna patterns are complex field patterns. Directional power gain is `|C_theta|^2 + |C_phi|^2`; the custom horn returns field amplitude `sqrt(10^(gain_db/10))`, so the gain is part of the simulation, not a post-processing scalar.

Entity orientation from the GUI is passed directly to Sionna. The backend does not call `look_at()` for horn links, so yaw/pitch/roll and roll around boresight affect the simulated H/V response.

## Radiomap

Radiomap mode replaces explicit UEs for the simulation with a generated UE grid:

- The grid lies in the XY plane at the configured height.
- Bounds and spacing define the positions.
- The radiomap UE template defines antenna panel, pattern, polarization, and orientation.
- If objects have trajectories, the full radiomap grid is generated for each object state.

In the GUI, enabling radiomap shows draggable rectangle corners in the 3D view. Explicit UE entities are ignored by the simulation while radiomap is enabled.

## 3GPP Sensing Targets

Sionna RT 2.2 sensing targets (3GPP TR 38.901 clause 7.9.2) are added like any other object, by `object_name`:

| `object_name` | TR 38.901 type | Default size L x W x H [m] |
|---|---|---|
| `HUMAN_3GPP` | `human` | 0.5 x 0.5 x 1.75 |
| `CAR_3GPP` / `CAR_SP_3GPP` | `vehicle-multi-sp` / `vehicle-single-sp` | 5.0 x 2.0 x 1.6 |
| `AGV_3GPP` / `AGV_SP_3GPP` | `agv-multi-sp` / `agv-single-sp` | 1.0 x 0.5 x 0.5 (see below) |
| `UAV_SMALL_3GPP` / `UAV_LARGE_3GPP` | `uav-small-size` / `uav-large-size` | 0.3 x 0.4 x 0.2 / 1.6 x 1.5 x 0.7 |

```json
{"id": "human0", "object_name": "HUMAN_3GPP",
 "trajectory": {"kind": "linear", "points": [[0, 3, 0.875], [4, 3, 0.875]], "samples": 9},
 "sensing": {"model_type": 2, "random_sigma_s": false, "random_phases": false, "random_xpr": false}}
```

- The optional `sensing` block also takes `dimensions` (cuboid L x W x H) or `mesh` (an `.obj` in `scenarios/objects`, e.g. `DRONE_obj`); the mesh bounding box then places the scattering points.
- Sizes are length (along the local x-axis, which the front faces) x width x height. The AGV is 1.0 m long and 0.5 m wide because TR 38.901 clause 7.9.2.1 states that "the front of the AGV is the short edge of AGV in horizontal direction"; Sionna RT 2.2's own default (0.5 x 1.0) would put the front on the long side, so the catalog sizes are always passed to Sionna explicitly.
- Positions are the bounding-box center, so a 1.75 m human stands on the ground at z = 0.875 m.
- A target's surface is an absorber: it shadows the background channel and responds only through its scattering points. `CAR_3GPP` is therefore a different model from the ray-traced, calibrated `CAR_obj`; do not place both at the same pose.
- Each solve runs `PathSolver` (background) and `RCSSolver` (paths through scattering points) and concatenates them. `sionna.sensing_channel` selects `combined` (default), `sensing_only` or `background_only`; `rcs_max_depth` (default: `max_depth`, counting the scattering event), `rcs_samples_per_sp` and `rcs_buffer_size_per_sp` tune the RCS solve.
- Set `scene.timeframe_interval_s` to give timeframes a duration: object velocities are then derived from the trajectories and drive the Doppler shifts. Without it, objects are at rest.
- TR 38.901 specifies these targets for 0.5-52.6 GHz. Outside that range (e.g. 77-81 GHz) the simulation runs, but a warning is emitted and `tr38901_frequency_in_range` is false in the sample metadata.

## HDF5 Output

Outputs are written to:

```text
output/scene_TIMESTAMP.h5
output/radiomap_TIMESTAMP.h5
```

The writer validates every generated file by default.

Main layout:

```text
scenarios/<scenario_name>/<sample_id>/
  metadata/
  parameters/
    f_vector
    subband_f_vectors/
    frequency_bands/
    sionna_params/
    channel_params/
    antenna_params/
    material_params/
    device_params/
    link_params/
    trajectory_params/
    radiomap_params/
  timeframes/tf<n>/
    h/rx<i>_tx<j>
    tau/rx<i>_tx<j>
    a/rx<i>_tx<j>
    positions/devices/<entity_id>
    positions/objects/<object_id>
    orientations/devices/<entity_id>
    orientations/objects/<object_id>
    velocities/objects/<object_id>        (when timeframe_interval_s is set)
    path_is_sensing/rx<i>_tx<j>           (with sensing targets, next to tau/a)
    timestamps/rx<i>_tx<j>
```

Device indexing is UEs first, then BSs. UE monostatic links and ordered UE-BS bistatic links are generated. BS monostatic links are not generated by default.

`h` stores the selected channel representation. `tau` stores physical path delays when available. `a` stores physical path coefficients when available.

## Backend API

The GUI and CLI both build a `SimulationRequest` and call one of:

```python
from isac_6d_sampler.sim.dry_run import DryRunSimulator
from isac_6d_sampler.sim.sionna_backend import SionnaSimulator

result = DryRunSimulator().simulate(request)
result = SionnaSimulator().simulate(request)
```

Use `ReferenceH5Writer` to persist the result:

```python
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer, default_output_path

path = default_output_path(request)
ReferenceH5Writer().write(path, request, result)
```

The backend validates requests before importing the heavy Sionna/Mitsuba runtime, so invalid scene definitions fail early.

## Notes For Git

Keep in git:

- `src/`
- `tests/`
- `docs/`
- `scenarios/Outdoor6D_w_car.xml`
- `scenarios/meshes/*.ply`
- `pyproject.toml`
- `uv.lock`
- `README.md`

Do not commit generated HDF5 files, `.venv`, caches, `*.egg-info`, or raw measurement `.mat` files.
