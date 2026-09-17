from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CalibratedMaterialSpec:
    object_prefix: str
    name: str
    relative_permittivity: float
    conductivity: float
    scattering_coefficient: float
    scattering_pattern: str = "directive"
    alpha_r: float = 10.0
    source: str = "isac_journal_77ghz"
    itu_type: str | None = None
    valid_min_ghz: float | None = None
    valid_max_ghz: float | None = None
    clamp_frequency: bool = False
    rms_roughness_m: float = 0.003
    diffuse_reference_frequency_hz: float = 77e9


SCATTERING_COEFFICIENT_SCALE = 0.35
DIELECTRIC_CONDUCTIVITY_MULTIPLIER = 1.0
METAL_OBJECT_PREFIXES = frozenset({"METAL", "CAR"})
CONCRETE_LOW_BAND_SCATTERING_COEFFICIENT = 0.98
CONCRETE_LOW_BAND_CONDUCTIVITY_MULTIPLIER = 40.0
CONCRETE_77GHZ_PRESERVE_MIN_HZ = 70e9
CONCRETE_77GHZ_PRESERVE_MAX_HZ = 90e9


# The original constants were calibrated for the 77 GHz Outdoor6D setup. For the
# lower HDF5 measurement bands we use the ITU-R P.2040-3 frequency laws exposed
# by Sionna where available, while preserving the previous diffuse scattering
# settings. Ground-like entries are clamped above 10 GHz because Sionna's ITU
# dry-ground models are only tabulated for 1-10 GHz.
CALIBRATED_MATERIALS: tuple[CalibratedMaterialSpec, ...] = (
    CalibratedMaterialSpec("BRICKS", "my_bricks", 3.91, 0.047, 0.70, "directive", 10.0, "itu-r-p2040-3", "brick", 1.0, 40.0, False, 0.004),
    CalibratedMaterialSpec("CONCRETE", "my_concrete", 5.31, 1.12, 0.65, "backscattering", 1.0, "itu-r-p2040-3", "concrete", 1.0, 100.0, False, 0.003),
    CalibratedMaterialSpec("GLASS", "my_glass", 5.79, 0.56, 0.70, "directive", 50.0, "itu-r-p2040-3", "glass", 0.1, 100.0, False, 0.0005),
    CalibratedMaterialSpec("GRASS", "my_grass", 1.60, 0.066, 0.70, "directive", 15.0, "itu-r-p2040-3_clamped", "very_dry_ground", 1.0, 10.0, True, 0.015),
    CalibratedMaterialSpec("GRAVEL", "my_gravel", 5.24, 4.00, 0.75, "directive", 5.0, "itu-r-p2040-3_clamped", "medium_dry_ground", 1.0, 10.0, True, 0.012),
    CalibratedMaterialSpec("GROUND", "my_ground", 5.24, 4.00, 0.65, "backscattering", 1.0, "itu-r-p2040-3_clamped", "medium_dry_ground", 1.0, 10.0, True, 0.010),
    CalibratedMaterialSpec("METAL", "my_metal", 1.00, 10e9, 0.80, "backscattering", 3.0, "itu-r-p2040-3", "metal", 1.0, 100.0, False, 0.001),
    CalibratedMaterialSpec("PLASTIC", "my_plastic", 4.25, 0.433, 0.70, "directive", 20.0, "itu-r-p2040-3", "plasterboard", 1.0, 100.0, False, 0.0015),
    CalibratedMaterialSpec("CAR", "my_car_metal", 1.00, 10e9, 0.75, "backscattering", 4.0, "itu-r-p2040-3", "metal", 1.0, 100.0, False, 0.001),
)


def calibrated_material_specs() -> tuple[CalibratedMaterialSpec, ...]:
    return CALIBRATED_MATERIALS


def material_spec_for_object_name(object_name: str) -> CalibratedMaterialSpec | None:
    prefix = object_name.upper().split("_")[0]
    for spec in CALIBRATED_MATERIALS:
        if spec.object_prefix == prefix:
            return spec
    return None


def assign_calibrated_materials(scene, scene_name: str, frequency_hz: float | None = None) -> None:
    """Assign frequency-responsive material callbacks to known 6D objects.

    Permittivity and conductivity follow ITU-R P.2040-3 through Sionna's own
    material coefficient table. Scattering settings are kept from the calibrated
    77 GHz Outdoor6D setup because those are roughness/diffuse-tuning parameters
    rather than bulk dielectric constants.
    """
    try:
        from sionna.rt import RadioMaterial
    except ModuleNotFoundError as exc:
        raise RuntimeError("Sionna is required for material assignment") from exc

    representative_frequency_hz = float(frequency_hz or getattr(scene, "frequency", spec_reference_frequency_hz()))

    def material(spec: CalibratedMaterialSpec):
        callback = _itu_frequency_update_callback(spec) if spec.itu_type else None
        relative_permittivity, conductivity = material_properties_at_frequency(spec, representative_frequency_hz)
        return RadioMaterial(
            spec.name,
            relative_permittivity=relative_permittivity,
            conductivity=conductivity,
            scattering_coefficient=tuned_diffuse_scattering_coefficient_at_frequency(spec, representative_frequency_hz),
            scattering_pattern=spec.scattering_pattern,
            alpha_r=spec.alpha_r,
            frequency_update_callback=callback,
        )

    materials = {spec.object_prefix: material(spec) for spec in CALIBRATED_MATERIALS}
    for mat in materials.values():
        if mat.name not in scene.radio_materials:
            scene.add(mat)

    for obj in scene.objects.values():
        spec = material_spec_for_object_name(obj.name)
        if spec is not None:
            obj.radio_material = materials[spec.object_prefix]


def material_properties_at_frequency(spec: CalibratedMaterialSpec, frequency_hz: float) -> tuple[float, float]:
    """Evaluate the material model as ordinary Python floats for reports/tests."""
    if not spec.itu_type:
        return float(spec.relative_permittivity), _tuned_conductivity(spec, float(spec.conductivity), frequency_hz)
    try:
        a, b, c, d = _itu_material_coefficients(spec.itu_type, frequency_hz, spec.clamp_frequency)
    except ModuleNotFoundError:
        return float(spec.relative_permittivity), _tuned_conductivity(spec, float(spec.conductivity), frequency_hz)
    f_ghz = _frequency_ghz_for_spec(spec.itu_type, frequency_hz, spec.clamp_frequency)
    conductivity = float(c * f_ghz**d)
    return float(a * f_ghz**b), _tuned_conductivity(spec, conductivity, frequency_hz)


def spec_reference_frequency_hz() -> float:
    return 77e9


def diffuse_scattering_coefficient_at_frequency(spec: CalibratedMaterialSpec, frequency_hz: float) -> float:
    """Scale the 77 GHz calibrated diffuse coefficient by electrical roughness.

    The model follows the common Rayleigh roughness dependence: diffuse power
    grows with the square of 4*pi*sigma_h/lambda and saturates for very rough
    surfaces. The 77 GHz calibrated scattering coefficient remains the reference
    value; lower frequencies receive less diffuse scattering when the same
    physical RMS roughness is electrically smoother.
    """
    reference = _roughness_scattering_factor(spec.rms_roughness_m, spec.diffuse_reference_frequency_hz)
    current = _roughness_scattering_factor(spec.rms_roughness_m, frequency_hz)
    if reference <= 1e-12:
        return float(max(0.0, min(1.0, spec.scattering_coefficient)))
    value = float(spec.scattering_coefficient) * current / reference
    return float(max(0.0, min(1.0, value)))


def tuned_diffuse_scattering_coefficient_at_frequency(spec: CalibratedMaterialSpec, frequency_hz: float) -> float:
    """Return the simulation scattering coefficient after multipath-power tuning.

    The 77 GHz Outdoor6D values generate too much non-LoS power for the measured
    UPV bands. Concrete needs special treatment because the problematic peak is
    a specular reflection. A high scattering coefficient diverts power away from
    the coherent/specular term in Sionna's rough-surface model.
    """
    if spec.object_prefix == "CONCRETE" and not _is_77ghz_band(frequency_hz):
        return CONCRETE_LOW_BAND_SCATTERING_COEFFICIENT
    value = diffuse_scattering_coefficient_at_frequency(spec, frequency_hz) * SCATTERING_COEFFICIENT_SCALE
    return float(max(0.0, min(1.0, value)))


def _is_77ghz_band(frequency_hz: float) -> bool:
    value = float(frequency_hz)
    return CONCRETE_77GHZ_PRESERVE_MIN_HZ <= value <= CONCRETE_77GHZ_PRESERVE_MAX_HZ


def _roughness_scattering_factor(rms_roughness_m: float, frequency_hz: float) -> float:
    wavelength_m = 299_792_458.0 / max(float(frequency_hz), 1.0)
    roughness = 4.0 * math.pi * max(float(rms_roughness_m), 0.0) / wavelength_m
    return 1.0 - math.exp(-(roughness * roughness))


def diffuse_alpha_r_at_frequency(spec: CalibratedMaterialSpec, frequency_hz: float) -> float:
    # Keep the calibrated lobe-shape parameter fixed. It controls angular shape,
    # while scattering_coefficient controls frequency-dependent scattered power.
    return float(spec.alpha_r)


def _itu_frequency_update_callback(spec: CalibratedMaterialSpec):
    def callback(frequency_hz):
        import drjit as dr

        f_ghz = frequency_hz / 1e9
        if spec.clamp_frequency and spec.valid_min_ghz is not None and spec.valid_max_ghz is not None:
            f_ghz = dr.minimum(dr.maximum(f_ghz, spec.valid_min_ghz), spec.valid_max_ghz)
        a, b, c, d = _itu_material_coefficients_for_dr_frequency(spec.itu_type, f_ghz)
        relative_permittivity = a * dr.power(f_ghz, b)
        conductivity = c * dr.power(f_ghz, d)
        if spec.object_prefix not in METAL_OBJECT_PREFIXES:
            conductivity = conductivity * DIELECTRIC_CONDUCTIVITY_MULTIPLIER
        if spec.object_prefix == "CONCRETE":
            is_low_band = (frequency_hz < CONCRETE_77GHZ_PRESERVE_MIN_HZ) | (frequency_hz > CONCRETE_77GHZ_PRESERVE_MAX_HZ)
            conductivity = dr.select(is_low_band, conductivity * CONCRETE_LOW_BAND_CONDUCTIVITY_MULTIPLIER, conductivity)
        return relative_permittivity, conductivity

    return callback


def _tuned_conductivity(spec: CalibratedMaterialSpec, conductivity: float, frequency_hz: float) -> float:
    if spec.object_prefix in METAL_OBJECT_PREFIXES:
        return float(conductivity)
    value = float(conductivity) * DIELECTRIC_CONDUCTIVITY_MULTIPLIER
    if spec.object_prefix == "CONCRETE" and not _is_77ghz_band(frequency_hz):
        value *= CONCRETE_LOW_BAND_CONDUCTIVITY_MULTIPLIER
    return value


def _frequency_ghz_for_spec(itu_type: str, frequency_hz: float, clamp: bool) -> float:
    f_ghz = float(frequency_hz) / 1e9
    if clamp:
        ranges = _itu_material_ranges(itu_type)
        low = min(item[0] for item in ranges)
        high = max(item[1] for item in ranges)
        f_ghz = min(max(f_ghz, low), high)
    return f_ghz


def _itu_material_coefficients(itu_type: str, frequency_hz: float, clamp: bool) -> tuple[float, float, float, float]:
    f_ghz = _frequency_ghz_for_spec(itu_type, frequency_hz, clamp)
    return _itu_material_coefficients_for_type(itu_type, f_ghz, f_ghz)


def _itu_material_coefficients_for_type(
    itu_type: str | None,
    valid_min_ghz: float | None,
    valid_max_ghz: float | None,
) -> tuple[float, float, float, float]:
    if itu_type is None:
        raise ValueError("ITU material type is required")
    from sionna.rt.radio_materials.itu_material import ITU_MATERIALS_PROPERTIES

    props = ITU_MATERIALS_PROPERTIES[itu_type]
    if valid_min_ghz is not None and valid_max_ghz is not None:
        midpoint = 0.5 * (float(valid_min_ghz) + float(valid_max_ghz))
        for (low, high), coeffs in props.items():
            if low <= midpoint <= high:
                return tuple(float(value) for value in coeffs)
    # Use the lowest available range as a deterministic fallback.
    first_key = sorted(props)[0]
    return tuple(float(value) for value in props[first_key])


def _itu_material_coefficients_for_dr_frequency(itu_type: str | None, f_ghz):
    if itu_type is None:
        raise ValueError("ITU material type is required")
    import drjit as dr
    from sionna.rt.radio_materials.itu_material import ITU_MATERIALS_PROPERTIES

    props = ITU_MATERIALS_PROPERTIES[itu_type]
    first_key = sorted(props)[0]
    a, b, c, d = (float(value) for value in props[first_key])
    for (low, high), coeffs in sorted(props.items()):
        in_range = (f_ghz >= float(low)) & (f_ghz <= float(high))
        ca, cb, cc, cd = (float(value) for value in coeffs)
        a = dr.select(in_range, ca, a)
        b = dr.select(in_range, cb, b)
        c = dr.select(in_range, cc, c)
        d = dr.select(in_range, cd, d)
    return a, b, c, d


def _itu_material_ranges(itu_type: str) -> tuple[tuple[float, float], ...]:
    from sionna.rt.radio_materials.itu_material import ITU_MATERIALS_PROPERTIES

    return tuple((float(low), float(high)) for low, high in ITU_MATERIALS_PROPERTIES[itu_type])
