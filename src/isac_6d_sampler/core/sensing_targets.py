from __future__ import annotations

from dataclasses import dataclass

from .model import DynamicObject, SensingTargetOptions, Vector3

# 3GPP TR 38.901 clause 7.9 sensing targets are specified for 0.5-52.6 GHz.
# Sionna does not enforce this range, so the backend only flags it.
TR38901_VALID_FREQUENCY_RANGE_HZ = (0.5e9, 52.6e9)


@dataclass(frozen=True, slots=True)
class SensingTargetType:
    name: str
    object_type: str
    dimensions_m: Vector3
    description: str


# Default (length, width, height) in the target LCS: length along x, the direction
# the front faces (azimuth 0), width along y. They follow TR 38.901 clause 7.9.1,
# and the backend always passes them to Sionna RT, which makes this table authoritative.
#
# AGV: Table 7.9.1-4 gives "0.5m x 1.0m x 0.5m" without saying which side is the
# length, and clause 7.9.2.1 states "The front of the AGV is the short edge of AGV in
# horizontal direction". With the front facing +x, the 0.5 m side must therefore run
# along y. Sionna RT 2.2 defaults to (0.5, 1.0, 0.5), which puts the front on the long
# side and rotates the body by 90 degrees relative to its scattering lobes.
SENSING_TARGET_TYPES: tuple[SensingTargetType, ...] = (
    SensingTargetType("HUMAN_3GPP", "human", (0.5, 0.5, 1.75), "Adult pedestrian, single scattering point"),
    SensingTargetType("CAR_3GPP", "vehicle-multi-sp", (5.0, 2.0, 1.6), "Vehicle, five scattering points"),
    SensingTargetType("CAR_SP_3GPP", "vehicle-single-sp", (5.0, 2.0, 1.6), "Vehicle, single scattering point"),
    SensingTargetType("AGV_3GPP", "agv-multi-sp", (1.0, 0.5, 0.5), "AGV, multiple scattering points"),
    SensingTargetType("AGV_SP_3GPP", "agv-single-sp", (1.0, 0.5, 0.5), "AGV, single scattering point"),
    SensingTargetType("UAV_SMALL_3GPP", "uav-small-size", (0.3, 0.4, 0.2), "Small UAV, single scattering point"),
    SensingTargetType("UAV_LARGE_3GPP", "uav-large-size", (1.6, 1.5, 0.7), "Large UAV, single scattering point"),
)
SENSING_TARGET_NAMES = tuple(target.name for target in SENSING_TARGET_TYPES)
_BY_NAME = {target.name: target for target in SENSING_TARGET_TYPES}


def sensing_target_type(object_name: str) -> SensingTargetType | None:
    return _BY_NAME.get(str(object_name))


def is_sensing_target(obj: DynamicObject | str) -> bool:
    name = obj if isinstance(obj, str) else obj.object_name
    return str(name) in _BY_NAME


def sensing_options(obj: DynamicObject) -> SensingTargetOptions:
    return obj.sensing if obj.sensing is not None else SensingTargetOptions()


def sensing_target_dimensions(obj: DynamicObject) -> Vector3 | None:
    """Cuboid dimensions of a catalog target, or ``None`` when a mesh defines its shape."""
    target = sensing_target_type(obj.object_name)
    if target is None:
        return None
    options = sensing_options(obj)
    if options.mesh:
        return None
    return options.dimensions or target.dimensions_m


def frequency_in_tr38901_range(frequency_hz: float) -> bool:
    low, high = TR38901_VALID_FREQUENCY_RANGE_HZ
    return low <= float(frequency_hz) <= high
