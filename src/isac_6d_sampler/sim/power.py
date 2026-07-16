from __future__ import annotations


def dbm_to_watt(power_dbm: float) -> float:
    return float(10.0 ** ((float(power_dbm) - 30.0) / 10.0))


def field_amplitude_from_dbm(power_dbm: float) -> float:
    return dbm_to_watt(power_dbm) ** 0.5
