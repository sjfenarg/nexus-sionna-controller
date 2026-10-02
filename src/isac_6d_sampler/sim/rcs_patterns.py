"""Monostatic RCS patterns of TR38901 and MSC scattering points.

The GUI evaluates the same RCS callable as ``RCSSolver``. TR38901's random
``sigma_S`` draw is disabled; MSC uses seeded prototype kernels at the carrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from isac_6d_sampler.core.model import Vector3


@dataclass(frozen=True, slots=True)
class ScatteringPointPattern:
    lcs_position: Vector3
    """Position of the scattering point relative to the target center, in the target LCS [m]."""
    rcs_dbsm: np.ndarray
    """Monostatic RCS [dBsm] seen from each requested direction."""


def monostatic_scattering_patterns(
    object_type: str,
    model_type: int,
    dimensions: Vector3,
    directions: np.ndarray,
    *,
    sensing_model: str = "tr38901",
    frequency_hz: float = 12.5e9,
    parameter_seed: int = 0,
) -> tuple[ScatteringPointPattern, ...]:
    """Monostatic RCS of every scattering point at the preview carrier.

    ``directions`` are unit vectors in the target LCS pointing from the target
    towards the observer, i.e. a co-located radar illuminates the target along
    ``-direction`` and receives the echo scattered along ``+direction``.
    """
    directions = np.ascontiguousarray(directions, dtype=np.float64).reshape(-1, 3)
    key = (
        str(object_type),
        int(model_type),
        tuple(round(float(value), 9) for value in dimensions),
        directions.tobytes(),
        sensing_model,
        float(frequency_hz),
        parameter_seed,
    )
    return _cached_patterns(key)


@lru_cache(maxsize=64)
def _cached_patterns(key) -> tuple[ScatteringPointPattern, ...]:
    object_type, model_type, dimensions, direction_bytes, sensing_model, frequency_hz, parameter_seed = key
    directions = np.frombuffer(direction_bytes, dtype=np.float64).reshape(-1, 3)
    _ensure_mitsuba_variant()

    import drjit as dr
    import mitsuba as mi
    from sionna.rt.rcs import TR38901SensingTarget

    length, width, height = dimensions
    target_class = TR38901SensingTarget
    model_options = {"model_type": model_type}
    if sensing_model == "msc":
        from sionna.rt.rcs import MSCSensingTarget
        target_class = MSCSensingTarget
        model_options = {"frequency_hz": frequency_hz, "parameter_seed": parameter_seed}
    target = target_class(
        name="rcs_pattern_probe",
        object_type=object_type,
        **model_options,
        length=length,
        width=width,
        height=height,
    )
    spst = target.scattering_model.spst
    lcs_positions = np.asarray(spst.lcs_positions.numpy(), dtype=np.float64).reshape(3, -1).T
    num_points = lcs_positions.shape[0]
    num_dirs = directions.shape[0]

    tiled = np.tile(directions, (num_points, 1))
    k_s = mi.Vector3f(tiled[:, 0], tiled[:, 1], tiled[:, 2])
    sigma = spst.eval_rcs(
        k_i=-k_s,
        k_s=k_s,
        # Evaluated in the target LCS: the GUI rotates the lobes with the target.
        st_orientations=mi.Point3f(0.0, 0.0, 0.0),
        st_indices=dr.zeros(mi.UInt, num_points * num_dirs),
        spst_indices=mi.UInt(np.repeat(np.arange(num_points, dtype=np.uint32), num_dirs)),
    )
    sigma = np.asarray(sigma.numpy(), dtype=np.float64).reshape(num_points, num_dirs)
    rcs_dbsm = 10.0 * np.log10(np.maximum(sigma, 1e-30))
    return tuple(
        ScatteringPointPattern(
            lcs_position=tuple(float(value) for value in lcs_positions[idx]),
            rcs_dbsm=rcs_dbsm[idx],
        )
        for idx in range(num_points)
    )


def _ensure_mitsuba_variant() -> None:
    import mitsuba as mi

    if mi.variant() is None:
        from isac_6d_sampler.sim.sionna_backend import _configure_mitsuba_variant

        _configure_mitsuba_variant(use_gpu=True)
