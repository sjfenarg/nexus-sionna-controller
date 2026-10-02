"""Seeded placeholder parameters, deliberately unrelated to fitted measurements."""

from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class MSCCenterParameters:
    """Paper Section III kernel parameters (angles in radians, amplitudes in m)."""

    w0: float = 1.0
    wr_side: float = 0.2
    wr_back: float = 0.1
    n: float = 4.0
    kappa_side: float = 3.0
    kappa_back: float = 2.0
    psi_side: float = math.pi / 2
    psi_back: float = math.pi
    psi_threshold: float = 1.3
    transition_width: float = 0.3
    beamwidth_phi: float = math.pi / 3
    beamwidth_theta: float = math.pi / 3
    reference_frequency_hz: float = 12.5e9
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self):
        for field in fields(self):
            if field.name != "normal" and not math.isfinite(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be finite")
        for name in ("w0", "wr_side", "wr_back", "n"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("kappa_side", "kappa_back", "transition_width", "reference_frequency_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("beamwidth_phi", "beamwidth_theta"):
            if not 0 < getattr(self, name) <= math.pi:
                raise ValueError(f"{name} must be in (0, pi]")
        if len(self.normal) != 3 or not all(math.isfinite(x) for x in self.normal):
            raise ValueError("normal must contain three finite components")
        if math.sqrt(sum(x*x for x in self.normal)) == 0:
            raise ValueError("normal must be nonzero")


@dataclass(frozen=True)
class MSCPolarizationParameters:
    """Whole-target lognormal power-ratio laws; HH/VV CPR has a positive sign."""

    xpr_mean_db: float = 12.0
    xpr_std_db: float = 3.0
    cpr_mean_db: float = -2.0
    cpr_std_db: float = 2.0

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, f.name)) for f in fields(self)):
            raise ValueError("polarization parameters must be finite")
        if self.xpr_std_db < 0 or self.cpr_std_db < 0:
            raise ValueError("ratio standard deviations must be non-negative")


def placeholder_parameters(count, seed, normals):
    """Fixed random prototypes: repeated solves do not regenerate the kernels."""
    rng = np.random.default_rng(seed)
    centers = tuple(
        MSCCenterParameters(
            w0=float(rng.uniform(0.5, 4.0)),
            wr_side=float(rng.uniform(0.05, 0.5)),
            wr_back=float(rng.uniform(0.02, 0.3)),
            n=float(rng.uniform(2.0, 8.0)),
            kappa_side=float(rng.uniform(1.5, 5.0)),
            kappa_back=float(rng.uniform(1.5, 4.0)),
            beamwidth_phi=float(rng.uniform(0.4, 1.5)),
            beamwidth_theta=float(rng.uniform(0.4, 1.5)),
            normal=tuple(normals[i]),
        ) for i in range(count)
    )
    laws = {
        mode: MSCPolarizationParameters(
            xpr_mean_db=float(rng.uniform(6, 20)),
            xpr_std_db=float(rng.uniform(1, 5)),
            cpr_mean_db=float(rng.uniform(-5, 3)),
            cpr_std_db=float(rng.uniform(1, 3)),
        ) for mode in ("monostatic", "bistatic")
    }
    return centers, laws
