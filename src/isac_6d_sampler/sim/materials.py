from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibratedMaterialSpec:
    object_prefix: str
    name: str
    relative_permittivity: float
    conductivity: float
    scattering_coefficient: float
    scattering_pattern: str = "directive"
    alpha_r: float = 10.0
    source: str = "isac_journal"


CALIBRATED_MATERIALS: tuple[CalibratedMaterialSpec, ...] = (
    CalibratedMaterialSpec("BRICKS", "my_bricks", 3.91, 0.047, 0.70, "directive", 10.0),
    CalibratedMaterialSpec("CONCRETE", "my_concrete", 5.31, 1.12, 0.65, "backscattering", 1.0),
    CalibratedMaterialSpec("GLASS", "my_glass", 5.79, 0.56, 0.70, "directive", 50.0),
    CalibratedMaterialSpec("GRASS", "my_grass", 1.60, 0.066, 0.70, "directive", 15.0),
    CalibratedMaterialSpec("GRAVEL", "my_gravel", 5.24, 4.00, 0.75, "directive", 5.0),
    CalibratedMaterialSpec("GROUND", "my_ground", 5.24, 4.00, 0.65, "backscattering", 1.0),
    CalibratedMaterialSpec("METAL", "my_metal", 1.00, 10e9, 0.80, "backscattering", 3.0),
    CalibratedMaterialSpec("PLASTIC", "my_plastic", 4.25, 0.433, 0.70, "directive", 20.0),
    CalibratedMaterialSpec("CAR", "my_car_metal", 1.00, 10e9, 0.75, "backscattering", 4.0),
)


def calibrated_material_specs() -> tuple[CalibratedMaterialSpec, ...]:
    return CALIBRATED_MATERIALS


def material_spec_for_object_name(object_name: str) -> CalibratedMaterialSpec | None:
    prefix = object_name.upper().split("_")[0]
    for spec in CALIBRATED_MATERIALS:
        if spec.object_prefix == prefix:
            return spec
    return None


def assign_calibrated_materials(scene, scene_name: str) -> None:
    """Assign calibrated material callbacks to known 6D scene objects.

    The object naming is normalized because current scene files mix names such as
    ``BRICKS_obj`` and ``CONCRETE_obj`` while older journal code expected upper-case
    variants.
    """
    try:
        from sionna.rt import RadioMaterial
    except ModuleNotFoundError as exc:
        raise RuntimeError("Sionna is required for material assignment") from exc

    def material(spec: CalibratedMaterialSpec):
        return RadioMaterial(
            spec.name,
            relative_permittivity=spec.relative_permittivity,
            conductivity=spec.conductivity,
            scattering_coefficient=spec.scattering_coefficient,
            scattering_pattern=spec.scattering_pattern,
            alpha_r=spec.alpha_r,
        )

    materials = {spec.object_prefix: material(spec) for spec in CALIBRATED_MATERIALS}
    for mat in materials.values():
        if mat.name not in scene.radio_materials:
            scene.add(mat)

    for obj in scene.objects.values():
        spec = material_spec_for_object_name(obj.name)
        if spec is not None:
            obj.radio_material = materials[spec.object_prefix]
