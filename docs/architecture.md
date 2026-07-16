# Architecture

The package is organized around the reference HDF5 database format, not around GUI widgets or
notebook-era scripts.

## Data Contract

`core.model` defines the persistent request model:

- `BaseStation`: fixed scene entity, default 10 x 10 panel.
- `UserEquipment`: static or trajectory-driven entity, default 1 x 1 panel.
- `DynamicObject`: movable scene object, e.g. `CAR_obj`.
- `RadiomapConfig`: xy-parallel UE grid with configurable spacing and height.
- `FrequencyBand`: one or more sampled subbands, default 77-81 GHz.
- `SionnaConfig`: ray-tracing controls such as transmit power, `samples_per_src`, max depth,
  LOS, diffuse, specular, refraction, synthetic array, `batch_timeframes` and `max_timeframes`.
- `ChannelMode`: frequency-domain, CIR path, binned PDP, IFFT PDP and coherent-per-bin modes.

Antenna panel spacing is stored in meters in the request model and HDF5 metadata. The default BS
spacing is half a wavelength at the default 79 GHz center frequency. The Sionna backend converts
those meter values to the wavelength multiples required by `sionna.rt.PlanarArray` using the mean
simulated frequency for the request.

`core.config_io` serializes the full `SimulationRequest` to JSON. This is the reproducibility path
for large scene definitions: the GUI can build the request, the CLI can rerun it, and tests can assert
that the same request writes the same HDF5 hierarchy.

`core.frequencies` owns the shared band specification parser used by the CLI and GUI. The syntax is
`START_GHZ:STOP_GHZ:POINTS[:NAME]`, and multiple bands may be supplied as repeated CLI flags or as a
semicolon-separated GUI field.

`core.trajectory_specs` owns GUI-friendly trajectory point parsing and construction. The GUI stores
the resulting `TrajectorySpec` directly in the request model, so saved JSON requests preserve static,
linear and polyline UE/object paths.

`core.antenna_patterns` defines the allowed pattern registry. It exposes Sionna built-ins and a
journal-derived `isac_horn_77_81` pattern based on the 77-81 GHz horn beamwidths used in the
measurement campaign. The Sionna backend registers package-local patterns before constructing
`PlanarArray` instances, and the HDF5 writer records the selected pattern source, peak gain and
beamwidth metadata for each device. `AntennaPanel.pattern` is the canonical Sionna pattern name;
legacy JSON `element_diagram` values are accepted as pattern names and then kept synchronized so the
simulation and exported `ant_type` metadata describe the same element pattern.

`core.scenarios` discovers Sionna XML files and their named PLY shapes. It also extracts PLY vertex
bounds for scenario-viewer footprints without requiring the GUI to load the full Sionna/Mitsuba
scene.

`core.estimates` computes request-size summaries without materializing the full simulation plan or
importing Sionna. The validator uses it for `max_timeframes`; the CLI exposes it through
`--plan-only`; and the GUI logs the same estimate before starting a worker. The estimate covers
active devices, radiomap grid dimensions, object states, timeframes, links per timeframe, channel
dataset count, frequency sample count and requested ray/path controls.

## HDF5 Layout

`io.reference_h5.ReferenceH5Writer` writes the reference-style hierarchy, and
`io.schema.validate_reference_h5` checks generated files for the required spine, frequency/subband
consistency and per-timeframe link datasets:
the writer runs this validation by default after each write, so CLI and GUI generation paths fail
fast on schema regressions.

```text
scenarios/<scenario_name>/<sample_id>/
  metadata/
  parameters/
    f_vector
    subband_f_vectors/subband_f_vector_<n>
    frequency_bands/band<n>
    vna_params/
    sionna_params/
    material_params/
    channel_params/
    device_params/device<n>
    link_params/rx<i>_tx<j>
    antenna_params/bs<n>, ue<n>
    object_params/object<n>
    trajectory_params/ues/ue<n>, objects/object<n>
    radiomap_params/
  timeframes/tf<n>/
    parameters
    h/rx<i>_tx<j>
    tau/rx<i>_tx<j>
    positions/devices/<entity_id>
    positions/objects/<object_id>
    orientations/devices/<entity_id>
    orientations/objects/<object_id>
    timestamps/rx<i>_tx<j>
```

Entity indices follow the observed reference convention: UEs first, then BSs. With one UE and one
BS, `rx0_tx0` is UE monostatic, `rx1_tx1` is BS monostatic, and `rx0_tx1`/`rx1_tx0` are the two
ordered bi-static directions.

The additive `device_params` group maps each device index to the entity ID, entity type and antenna
group. `link_params` maps every channel dataset name to `rx_id`, `tx_id`, device groups, entity
types, indices and direction metadata, so downstream consumers do not need to infer link semantics
only from the tensor key. The `h`, `tau` and `timestamps` datasets also carry direct rx/tx link
attributes and JSON axis labels for local inspection without joining through `link_params`.

The `positions` and `orientations` groups are additive metadata beyond the original reference sample.
They make per-timeframe device state, object state and radiomap state directly inspectable while
preserving the reference `h/rx*_tx*` channel tensor layout. Fixed BS positions are repeated in each
timeframe so downstream readers can resolve all active device coordinates from the timeframe group.
Timeframe group attributes identify whether a frame is a normal scene frame or a radiomap sample;
radiomap frames include grid x/y indices, linear grid index and object trajectory state index.

`antenna_params/*/ant_orientation` follows the measured reference file convention and is stored as
yaw/pitch degrees. `ant_orientation_rad` and the per-timeframe `orientations` groups preserve the
full yaw/pitch/roll radians used by Sionna/Mitsuba.

The optional `tau/rx*_tx*` datasets store per-path propagation delays in seconds when the backend
provides path-domain data. They mirror the antenna dimensions of `h` and use a path-count last axis,
so raw CIR path coefficients can be interpreted without losing their delay coordinates.

The `frequency_bands`, `sionna_params`, `trajectory_params` and `radiomap_params` groups are also
additive reproducibility metadata. They expose the full simulation request in structured HDF5 form,
so downstream tools do not need to parse the `metadata/request_json` attribute to recover band
definitions, runtime controls, trajectory control points or radiomap grid coordinates.
`antenna_params/*` records panel spacing values in meters and labels them with spacing unit
attributes.
`material_params` records the calibrated material catalog used by the Sionna backend, keyed by scene
object prefix such as `CAR`, `CONCRETE` or `BRICKS`. Dynamic object parameter groups also record the
matched material prefix and material name when the target scene object maps to that catalog.

The `channel_params` group records how to interpret the last axis of each `h/rx*_tx*` tensor:
frequency coordinates for frequency-domain samples, delay-bin coordinates for gridded/coherent/PDP
modes, or path indices for raw CIR path coefficients. It also records the value quantity:
frequency-domain and coherent/IFFT delay modes are complex responses, `cir_paths` stores complex
path coefficients, and `pdp_binned` stores linear power accumulated from `abs(path_coefficient)^2`.
IFFT delay modes additionally record their renderer: exact non-uniform frequency response or the
faster oversampled delay-grid approximation.
Stored Sionna and dry-run channel coefficients are scaled by `sqrt(Ptx_W)` from
`sionna.tx_power_dbm`, so downstream `abs(h)^2` and `abs(a)^2` power traces scale linearly with
configured transmit power.
Request validation allows multi-band sweeps for `frequency_domain` and `cir_paths`. Delay-bin and
IFFT modes are restricted to a single contiguous uniformly sampled band so the exported
`delay_bin_s` coordinates remain physically meaningful.

## Simulation Boundary

`sim.sionna_backend.SionnaSimulator` owns all Sionna imports and scene mutation. This keeps the rest
of the project importable and testable without loading TensorFlow, Mitsuba or a GPU context.
Simulator entry points call the shared request validator after applying scene defaults and before
building execution plans; the Sionna backend does this before importing Mitsuba/Sionna so invalid or
oversized requests fail without touching the heavy runtime.
Before importing Sionna RT, it applies the request's GPU preference to Mitsuba: CUDA polarized mono
is requested first for GPU runs, while CPU runs force the LLVM polarized mono variant. The active
variant is recorded in sample metadata.

`sim.planner` builds the shared execution plan used by dry-run and Sionna backends: UEs are indexed
first, BSs second, every device gets a monostatic link, and each UE-BS pair gets both ordered
bi-static links. The Sionna backend batches links by compatible `(rx_panel, tx_panel, los)` groups.
Each batch adds all required receivers and transmitters, calls `PathSolver` once, and slices the
resulting Sionna tensor back into the requested ordered links. This avoids one ray-tracing solve per
link when several devices share the same panel configuration.

Radiomap planning derives the configured UE template as the runtime device `ue_radiomap` without
rewriting the source `SceneDesign`. With static objects, one timeframe is created for each xy grid
coordinate. With object trajectories, the planner creates the full xy grid for each object trajectory
sample and records the corresponding object positions and orientations in every timeframe.

When `batch_timeframes` is greater than one, the backend also groups consecutive timeframes whose
dynamic objects have identical positions and orientations. Static device states are de-duplicated
inside the grouped Sionna call, while moving UE or radiomap positions remain separate receiver or
transmitter entries. Timeframe batching stops before any object-state change so scene geometry is
never shared across incompatible object poses.

Sionna flattens planar arrays internally, so the backend reshapes results back to
`rx_rows, rx_cols, tx_rows, tx_cols, 1, samples` before HDF5 export.

`sim.dry_run.DryRunSimulator` produces deterministic complex frequency responses. It is not a
physics simulator; it exists to validate GUI flows, request serialization and HDF5 structure quickly.

## GUI Boundary

`gui.app` is a PySide6 controller over the same request model. GUI-only helpers are split out:
`gui.worker` owns the background simulation/write thread, `gui.scene_3d` owns the OpenGL scene
viewer, and `gui.channel_visualizer` owns the single-plot channel inspector. The controller
currently supports:

- scenario selection from `scenarios`;
- GPU-accelerated 3D mesh visualization from calibrated PLY geometry;
- object mesh selection from named PLY shapes in the selected scenario XML;
- adding BS, UE and object entities;
- selected-entity removal;
- marker-click entity selection and transform-gizmo drag positioning in a 3D scene view;
- numeric x/y/z placement controls for selected entities;
- selected-entity editing for panel rows/columns, panel spacing, element pattern, polarization,
  object mesh and BS/UE/object orientation;
- static, linear and polyline trajectory controls for UEs and objects;
- trajectory anchoring for created/applied UEs and objects so the first control point matches the marker;
- trajectory-apply synchronization that moves the marker to the edited first control point;
- immediate drag-position synchronization that translates UE/object trajectory control points;
- trajectory endpoint position/orientation gizmos for moving UEs and objects;
- radiomap x/y bounds, independent x/y spacing, height, draggable rectangle corners and live capped lattice preview;
- explicit radiomap UE template application so ordinary entity selection does not rewrite loaded
  radiomap panel, pattern, polarization or orientation settings;
- temporary simulation playback in the 3D view while trajectory timeframes are computed;
- antenna diagram overlays for BS and UE patterns;
- realtime channel visualization for computed or saved HDF5 channels;
- frequency band and channel mode controls;
- Sionna built-in and journal-derived antenna pattern selection;
- Sionna runtime controls for source samples, max paths, depth, seed, LOS, specular, diffuse,
  refraction, synthetic arrays, merge-shapes, GPU preference, transmit power, timeframe batching
  and timeframe guardrails, including zero-depth runs and unlimited path-count preservation;
- pre-run request-size summary in the execution log;
- output-directory selection for automatic HDF5 writes;
- single in-flight simulation worker, progress bar and verbose execution log;
- automatic reference-format output naming.

## CLI Boundary

`cli.main` is the scriptable entry point over the same request model. It can write templates, load
saved GUI JSON requests, run dry-run or Sionna backends, override frequency bands and Sionna runtime
settings, choose antenna patterns and panel spacing, enable radiomap mode, set radiomap
bounds/spacing/height, and add dynamic scene objects with static or trajectory-driven positions. The
CLI also exposes `--max-timeframes` so oversized scene and radiomap expansions fail during request
validation rather than after allocating a full plan. `--plan-only` validates the request and prints
the shared request-size estimate without simulating or writing an HDF5 file.

GUI simulations and saves are built from a snapshot of the current `SceneDesign`, so edits made in
the controller after pressing `Simulate` cannot mutate the worker's in-flight request. Loading a
request whose scenario path is outside the initially discovered selector entries adds that path as a
custom selector option, preserving the loaded scenario during subsequent save or simulate actions.
