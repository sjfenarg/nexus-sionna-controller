from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AntennaPatternSpec:
    name: str
    label: str
    source: str
    max_gain_db: float | None = None
    h_3db_beamwidth_deg: float | None = None
    v_3db_beamwidth_deg: float | None = None


BUILTIN_PATTERN_NAMES = ("iso", "dipole", "hw_dipole", "tr38901")
ISAC_HORN_PATTERN = "isac_horn_77_81"

_PATTERNS = {
    "iso": AntennaPatternSpec(name="iso", label="Isotropic", source="sionna"),
    "dipole": AntennaPatternSpec(name="dipole", label="Short dipole", source="sionna"),
    "hw_dipole": AntennaPatternSpec(name="hw_dipole", label="Half-wave dipole", source="sionna"),
    "tr38901": AntennaPatternSpec(name="tr38901", label="3GPP TR 38.901", source="sionna"),
    ISAC_HORN_PATTERN: AntennaPatternSpec(
        name=ISAC_HORN_PATTERN,
        label="ISAC 77-81 GHz horn",
        source="isac_journal",
        max_gain_db=10.3,
        h_3db_beamwidth_deg=56.0,
        v_3db_beamwidth_deg=28.0,
    ),
}


def available_antenna_patterns() -> tuple[AntennaPatternSpec, ...]:
    return tuple(_PATTERNS[name] for name in _PATTERNS)


def available_pattern_names() -> tuple[str, ...]:
    return tuple(_PATTERNS)


def antenna_pattern_spec(name: str) -> AntennaPatternSpec:
    try:
        return _PATTERNS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported antenna pattern: {name}") from exc


def sionna_pattern_name(name: str) -> str:
    antenna_pattern_spec(name)
    return name


def register_sionna_antenna_patterns() -> None:
    """Register package-local antenna patterns with Sionna RT.

    Sionna's built-in registry already contains the standard names. This function adds the
    journal-derived 77-81 GHz horn approximation when Sionna RT is loaded.
    """
    from sionna.rt import PolarizedAntennaPattern, register_antenna_pattern

    try:
        register_antenna_pattern(ISAC_HORN_PATTERN, _create_isac_horn_factory(PolarizedAntennaPattern))
    except ValueError as exc:
        if "already" not in str(exc).lower() and "registered" not in str(exc).lower():
            raise


def _create_isac_horn_factory(polarized_pattern_cls):
    def factory(*, polarization, polarization_model="tr38901_2"):
        return polarized_pattern_cls(
            v_pattern=_v_isac_horn_pattern,
            polarization=polarization,
            polarization_model=polarization_model,
        )

    return factory


def _v_isac_horn_pattern(theta, phi):
    import drjit as dr
    import mitsuba as mi

    spec = antenna_pattern_spec(ISAC_HORN_PATTERN)
    theta_3db = float(spec.v_3db_beamwidth_deg) / 180.0 * dr.pi
    phi_3db = float(spec.h_3db_beamwidth_deg) / 180.0 * dr.pi
    max_gain_db = float(spec.max_gain_db)
    front_back_cap_db = 32.0

    phi = phi + dr.pi
    phi -= dr.floor(phi / (2.0 * dr.pi)) * 2.0 * dr.pi
    phi -= dr.pi

    theta_offset = theta - dr.pi / 2.0
    a_v = dr.min([12.0 * (theta_offset / theta_3db) ** 2, front_back_cap_db])
    a_h = dr.min([12.0 * (phi / phi_3db) ** 2, front_back_cap_db])
    attenuation_db = dr.min([a_v + a_h, front_back_cap_db])
    gain_db = max_gain_db - attenuation_db
    c_theta = dr.sqrt(dr.power(10.0, gain_db / 10.0))
    return mi.Complex2f(c_theta, 0.0)
