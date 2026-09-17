from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


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
POWERLOG_PATTERN = "powerLog"
POWERLOG_FREQUENCY_LABEL = "6GHz"
POWERLOG_FREQUENCY_LABELS_HZ = {
    "700MHz": 0.7e9,
    "3GHz": 3.0e9,
    "6GHz": 6.0e9,
    "9GHz": 9.0e9,
    "12GHz": 12.0e9,
    "15GHz": 15.0e9,
}
QOM_OMNI_PATTERN = "qom_st_2_18_omni"
QOM_OMNI_FREQUENCY_LABEL = "18GHz"
QOM_OMNI_FREQUENCY_LABELS_HZ = {
    "2GHz": 2.0e9,
    "2.1GHz": 2.1e9,
    "4GHz": 4.0e9,
    "8GHz": 8.0e9,
    "12GHz": 12.0e9,
    "18GHz": 18.0e9,
}
MEASURED_PATTERN_FOURIER_TERMS = 32
POWERLOG_BACK_LOBE_BOOST_DB = 18.0
POWERLOG_BACK_LOBE_WIDTH_EXPONENT = 0.35

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
    POWERLOG_PATTERN: AntennaPatternSpec(
        name=POWERLOG_PATTERN,
        label="powerLog measured antenna, 6 GHz",
        source="docs/powerLog_antenna.csv",
        max_gain_db=12.897,
    ),
    QOM_OMNI_PATTERN: AntennaPatternSpec(
        name=QOM_OMNI_PATTERN,
        label="QOM-ST-2-18-S-SG-R measured omni, 18 GHz",
        source="docs/QOM_ST_2_18_omni.csv",
        max_gain_db=-6.315,
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


def measured_frequency_pattern_name(pattern: str, frequency_label: str) -> str:
    safe_label = frequency_label.replace(".", "p")
    return f"{pattern}_{safe_label}"


def resolve_pattern_for_frequency(pattern: str, frequency_hz: float) -> str:
    if pattern == POWERLOG_PATTERN:
        label = _nearest_frequency_label(float(frequency_hz), POWERLOG_FREQUENCY_LABELS_HZ)
        return measured_frequency_pattern_name(POWERLOG_PATTERN, label)
    if pattern == QOM_OMNI_PATTERN:
        label = _nearest_frequency_label(float(frequency_hz), QOM_OMNI_FREQUENCY_LABELS_HZ)
        return measured_frequency_pattern_name(QOM_OMNI_PATTERN, label)
    return pattern


def _nearest_frequency_label(frequency_hz: float, labels_hz: dict[str, float]) -> str:
    return min(labels_hz, key=lambda label: abs(float(labels_hz[label]) - frequency_hz))


def register_sionna_antenna_patterns() -> None:
    """Register package-local antenna patterns with Sionna RT.

    Sionna's built-in registry already contains the standard names. This function adds the
    package-local antenna patterns when Sionna RT is loaded.
    """
    from sionna.rt import PolarizedAntennaPattern, register_antenna_pattern

    factories = [
        (ISAC_HORN_PATTERN, _create_isac_horn_factory(PolarizedAntennaPattern)),
        (POWERLOG_PATTERN, _create_powerlog_factory(PolarizedAntennaPattern)),
        (QOM_OMNI_PATTERN, _create_qom_omni_factory(PolarizedAntennaPattern)),
    ]
    factories.extend(
        (measured_frequency_pattern_name(POWERLOG_PATTERN, label), _create_powerlog_factory(PolarizedAntennaPattern, label))
        for label in POWERLOG_FREQUENCY_LABELS_HZ
    )
    factories.extend(
        (measured_frequency_pattern_name(QOM_OMNI_PATTERN, label), _create_qom_omni_factory(PolarizedAntennaPattern, label))
        for label in QOM_OMNI_FREQUENCY_LABELS_HZ
    )
    for name, factory in factories:
        try:
            register_antenna_pattern(name, factory)
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


def powerlog_normalized_gain_db_from_local_dirs(
    local_dirs: np.ndarray,
    frequency_label: str = POWERLOG_FREQUENCY_LABEL,
) -> np.ndarray:
    """Return the powerLog gain pattern normalized to its boresight maximum.

    The CSV provides measured H-plane and V-plane cuts. For 3D use we combine the
    two cuts with the common separable approximation: G(phi, theta)=Gmax +
    (H(phi)-Hmax) + (V(theta)-Vmax). The Sionna element pattern uses the same
    reconstruction below.
    """
    x = np.clip(local_dirs[..., 0], -1.0, 1.0)
    y = local_dirs[..., 1]
    z = local_dirs[..., 2]
    horizontal_angle = np.arctan2(y, x)
    vertical_angle = np.arctan2(z, np.maximum(np.sqrt(x * x + y * y), 1e-12))
    h_db, v_db, max_gain_db = _powerlog_numpy_cuts(frequency_label)
    gain_db = h_db(horizontal_angle) + v_db(vertical_angle) - 2.0 * max_gain_db
    return _apply_powerlog_back_lobe_boost_np(gain_db, horizontal_angle)


def _create_powerlog_factory(polarized_pattern_cls, frequency_label: str = POWERLOG_FREQUENCY_LABEL):
    def factory(*, polarization, polarization_model="tr38901_2"):
        return polarized_pattern_cls(
            v_pattern=lambda theta, phi: _v_powerlog_pattern(theta, phi, frequency_label),
            polarization=polarization,
            polarization_model=polarization_model,
        )

    return factory


def _v_powerlog_pattern(theta, phi, frequency_label: str = POWERLOG_FREQUENCY_LABEL):
    import drjit as dr
    import mitsuba as mi

    h_coeffs, v_coeffs, max_gain_db = _powerlog_fourier_coefficients(frequency_label)
    horizontal_angle = _wrap_radians_dr(phi, dr)
    vertical_angle = _wrap_radians_dr(theta - dr.pi / 2.0, dr)
    horizontal_db = _fourier_series_dr(horizontal_angle, h_coeffs, dr)
    vertical_db = _fourier_series_dr(vertical_angle, v_coeffs, dr)
    gain_db = horizontal_db + vertical_db - max_gain_db
    gain_db = _apply_powerlog_back_lobe_boost_dr(gain_db, horizontal_angle, max_gain_db, dr)
    gain_db = dr.max([gain_db, max_gain_db - 45.0])
    c_theta = dr.sqrt(dr.power(10.0, gain_db / 10.0))
    return mi.Complex2f(c_theta, 0.0)


def _apply_powerlog_back_lobe_boost_np(gain_db: np.ndarray, horizontal_angle: np.ndarray) -> np.ndarray:
    rear_weight = np.clip(-np.cos(horizontal_angle), 0.0, 1.0) ** POWERLOG_BACK_LOBE_WIDTH_EXPONENT
    return np.minimum(gain_db + POWERLOG_BACK_LOBE_BOOST_DB * rear_weight, 0.0)


def _apply_powerlog_back_lobe_boost_dr(gain_db, horizontal_angle, max_gain_db: float, dr):
    rear_weight = dr.power(dr.max([-dr.cos(horizontal_angle), 0.0]), POWERLOG_BACK_LOBE_WIDTH_EXPONENT)
    boosted = gain_db + POWERLOG_BACK_LOBE_BOOST_DB * rear_weight
    return dr.min([boosted, float(max_gain_db)])



def qom_omni_normalized_gain_db_from_local_dirs(
    local_dirs: np.ndarray,
    frequency_label: str = QOM_OMNI_FREQUENCY_LABEL,
) -> np.ndarray:
    """Return the QOM omni gain pattern normalized to its maximum.

    The CSV contains slant azimuth and two orthogonal slant elevation cuts. For
    3D use, we use the azimuth cut as H(phi) and the average of both elevation
    cuts as V(theta), then apply the same separable cut reconstruction used for
    the powerLog pattern.
    """
    x = np.clip(local_dirs[..., 0], -1.0, 1.0)
    y = local_dirs[..., 1]
    z = local_dirs[..., 2]
    horizontal_angle = np.arctan2(y, x)
    vertical_angle = np.pi / 2.0 - np.arctan2(z, np.maximum(np.sqrt(x * x + y * y), 1e-12))
    h_db, v_db, max_gain_db = _qom_omni_numpy_cuts(frequency_label)
    return h_db(horizontal_angle) + v_db(vertical_angle) - 2.0 * max_gain_db


def _create_qom_omni_factory(polarized_pattern_cls, frequency_label: str = QOM_OMNI_FREQUENCY_LABEL):
    def factory(*, polarization, polarization_model="tr38901_2"):
        return polarized_pattern_cls(
            v_pattern=lambda theta, phi: _v_qom_omni_pattern(theta, phi, frequency_label),
            polarization=polarization,
            polarization_model=polarization_model,
        )

    return factory


def _v_qom_omni_pattern(theta, phi, frequency_label: str = QOM_OMNI_FREQUENCY_LABEL):
    import drjit as dr
    import mitsuba as mi

    h_coeffs, v_coeffs, max_gain_db = _qom_omni_fourier_coefficients(frequency_label)
    horizontal_angle = _wrap_radians_dr(phi, dr)
    vertical_angle = _wrap_radians_dr(theta, dr)
    horizontal_db = _fourier_series_dr(horizontal_angle, h_coeffs, dr)
    vertical_db = _fourier_series_dr(vertical_angle, v_coeffs, dr)
    gain_db = horizontal_db + vertical_db - max_gain_db
    gain_db = dr.max([gain_db, max_gain_db - 45.0])
    c_theta = dr.sqrt(dr.power(10.0, gain_db / 10.0))
    return mi.Complex2f(c_theta, 0.0)


@lru_cache(maxsize=None)
def _qom_omni_fourier_coefficients(frequency_label: str = QOM_OMNI_FREQUENCY_LABEL) -> tuple[tuple[float, tuple[float, ...], tuple[float, ...]], tuple[float, tuple[float, ...], tuple[float, ...]], float]:
    data = _read_qom_omni_csv()
    horizontal = data[f"{frequency_label}_SlantAzimuth_dB"]
    vertical = 0.5 * (
        data[f"{frequency_label}_SlantElevation0_dB"]
        + data[f"{frequency_label}_SlantElevation90_dB"]
    )
    max_gain_db = float(max(np.max(horizontal), np.max(vertical)))
    angles_rad = np.deg2rad(data["angle_deg"])
    return _fit_fourier(angles_rad, horizontal), _fit_fourier(angles_rad, vertical), max_gain_db


@lru_cache(maxsize=None)
def _qom_omni_numpy_cuts(frequency_label: str = QOM_OMNI_FREQUENCY_LABEL):
    h_coeffs, v_coeffs, max_gain_db = _qom_omni_fourier_coefficients(frequency_label)

    def h(angle_rad):
        return _fourier_series_np(angle_rad, h_coeffs)

    def v(angle_rad):
        return _fourier_series_np(angle_rad, v_coeffs)

    return h, v, max_gain_db


@lru_cache(maxsize=1)
def _read_qom_omni_csv() -> dict[str, np.ndarray]:
    return _read_measured_pattern_csv("QOM_ST_2_18_omni.csv")


def _wrap_radians_dr(angle, dr):
    angle = angle + dr.pi
    angle -= dr.floor(angle / (2.0 * dr.pi)) * 2.0 * dr.pi
    return angle - dr.pi


def _fourier_series_dr(angle, coeffs, dr):
    a0, cos_coeffs, sin_coeffs = coeffs
    value = float(a0)
    for harmonic, (a_k, b_k) in enumerate(zip(cos_coeffs, sin_coeffs), start=1):
        value += float(a_k) * dr.cos(harmonic * angle) + float(b_k) * dr.sin(harmonic * angle)
    return value


@lru_cache(maxsize=None)
def _powerlog_fourier_coefficients(frequency_label: str = POWERLOG_FREQUENCY_LABEL) -> tuple[tuple[float, tuple[float, ...], tuple[float, ...]], tuple[float, tuple[float, ...], tuple[float, ...]], float]:
    data = _read_powerlog_csv()
    h = data[f"{frequency_label}_H_plane_dB"]
    v = data[f"{frequency_label}_V_plane_dB"]
    max_gain_db = float(max(np.max(h), np.max(v)))
    angles_rad = np.deg2rad(data["angle_deg"])
    return _fit_fourier(angles_rad, h), _fit_fourier(angles_rad, v), max_gain_db


@lru_cache(maxsize=None)
def _powerlog_numpy_cuts(frequency_label: str = POWERLOG_FREQUENCY_LABEL):
    h_coeffs, v_coeffs, max_gain_db = _powerlog_fourier_coefficients(frequency_label)

    def h(angle_rad):
        return _fourier_series_np(angle_rad, h_coeffs)

    def v(angle_rad):
        return _fourier_series_np(angle_rad, v_coeffs)

    return h, v, max_gain_db


def _fit_fourier(angles_rad: np.ndarray, values_db: np.ndarray) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    cols = [np.ones_like(angles_rad)]
    for harmonic in range(1, MEASURED_PATTERN_FOURIER_TERMS + 1):
        cols.append(np.cos(harmonic * angles_rad))
        cols.append(np.sin(harmonic * angles_rad))
    matrix = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(matrix, values_db, rcond=None)
    return (
        float(coeffs[0]),
        tuple(float(coeffs[1 + 2 * i]) for i in range(MEASURED_PATTERN_FOURIER_TERMS)),
        tuple(float(coeffs[2 + 2 * i]) for i in range(MEASURED_PATTERN_FOURIER_TERMS)),
    )


def _fourier_series_np(angle_rad: np.ndarray, coeffs: tuple[float, tuple[float, ...], tuple[float, ...]]) -> np.ndarray:
    a0, cos_coeffs, sin_coeffs = coeffs
    angle = (np.asarray(angle_rad, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
    value = np.full_like(angle, float(a0), dtype=np.float64)
    for harmonic, (a_k, b_k) in enumerate(zip(cos_coeffs, sin_coeffs), start=1):
        value += float(a_k) * np.cos(harmonic * angle) + float(b_k) * np.sin(harmonic * angle)
    return value


@lru_cache(maxsize=1)
def _read_powerlog_csv() -> dict[str, np.ndarray]:
    return _read_measured_pattern_csv("powerLog_antenna.csv")


def _read_measured_pattern_csv(filename: str) -> dict[str, np.ndarray]:
    path = Path(__file__).resolve().parents[3] / "docs" / filename
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty measured antenna CSV: {path}")
    columns = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}
