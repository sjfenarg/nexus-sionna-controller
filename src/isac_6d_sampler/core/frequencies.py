from __future__ import annotations

from .model import FrequencyBand


def parse_band_spec(raw: str) -> FrequencyBand:
    """Parse ``START_GHZ:STOP_GHZ:POINTS[:NAME]`` into a frequency band."""
    parts = raw.strip().split(":")
    if len(parts) not in {3, 4}:
        raise ValueError("Band must be START_GHZ:STOP_GHZ:POINTS[:NAME]")
    try:
        start_ghz = float(parts[0])
        stop_ghz = float(parts[1])
        points = int(parts[2])
    except ValueError as exc:
        raise ValueError("Band start/stop must be numbers and points must be an integer") from exc
    if stop_ghz <= start_ghz:
        raise ValueError("Band stop frequency must be greater than start frequency")
    if points < 2:
        raise ValueError("Band points must be at least 2")
    name = parts[3].strip() if len(parts) == 4 and parts[3].strip() else f"{start_ghz:g}-{stop_ghz:g}GHz"
    return FrequencyBand(name=name, start_hz=start_ghz * 1e9, stop_hz=stop_ghz * 1e9, points=points)


def parse_band_specs(raw_values: list[str] | str) -> list[FrequencyBand]:
    """Parse repeated or semicolon/comma-separated band specifications."""
    if isinstance(raw_values, str):
        candidates = _split_specs(raw_values)
    else:
        candidates = []
        for raw in raw_values:
            candidates.extend(_split_specs(raw))
    if not candidates:
        raise ValueError("At least one band specification is required")
    return [parse_band_spec(raw) for raw in candidates]


def format_band_specs(bands: list[FrequencyBand]) -> str:
    return "; ".join(
        f"{band.start_hz / 1e9:g}:{band.stop_hz / 1e9:g}:{band.points}:{band.name}"
        for band in bands
    )


def _split_specs(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(",", ";").split(";") if part.strip()]
