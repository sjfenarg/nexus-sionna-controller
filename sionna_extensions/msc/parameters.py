"""Seeded placeholder parameters, deliberately unrelated to fitted measurements."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class MSCCenterParameters:
    """Reflecting-surface kernel parameters (angles in radians, amplitudes in m)."""

    w0: float = 1.0
    wr_side: float = 0.2
    wr_back: float = 0.1
    n: float = 2.0  # Legacy serialized field; only the fixed exponent is supported.
    kappa_side: float = 20.0
    kappa_back: float = 15.0
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
        if self.kappa_side != 20.0 or self.kappa_back != 15.0:
            raise ValueError("Trained kernel requires kappa_side=20 and kappa_back=15")
        if self.psi_threshold != 1.3 or self.transition_width != 0.3:
            raise ValueError("Trained kernel requires visibility constants 1.3 and 0.3")
        if self.n != 2.0:
            raise ValueError("The reflecting-surface kernel requires n=2")
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


    @classmethod
    def from_component(cls, component, reference_frequency_hz):
        """Convert a training-engine component, without loading a model file.

        mu_theta_0 is ELEVATION in degrees: 0 is horizontal, +90 is +Z.
        Sionna's zenith convention is not used to interpret this parameter.
        The resulting normal is in the target frame; use zero point Euler
        orientations so the training frame constructions remain target-fixed.
        """
        azimuth = math.radians(float(component["mu_phi_0"]))
        elevation = math.radians(float(component["mu_theta_0"]))
        return cls(
            w0=float(component["w"]),
            wr_side=float(component.get("w_side", 0.0)),
            wr_back=float(component.get("w_back", 0.0)),
            psi_side=math.radians(float(component.get("psi_side", 90.0))),
            psi_back=math.radians(float(component.get("psi_back", 180.0))),
            beamwidth_phi=math.radians(float(component["mu_beam_phi_0"])),
            beamwidth_theta=math.radians(float(component["mu_beam_theta_0"])),
            reference_frequency_hz=reference_frequency_hz,
            normal=(math.cos(elevation)*math.cos(azimuth),
                    math.cos(elevation)*math.sin(azimuth), math.sin(elevation)),
        )


def placeholder_parameters(count, seed, normals):
    """Fixed random prototypes: repeated solves do not regenerate the kernels."""
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = tuple(
        MSCCenterParameters(
            w0=float(rng.uniform(0.5, 4.0)),
            wr_side=float(rng.uniform(0.05, 0.5)),
            wr_back=float(rng.uniform(0.02, 0.3)),
            n=2.0,
            kappa_side=20.0,
            kappa_back=15.0,
            beamwidth_phi=float(rng.uniform(0.4, 1.5)),
            beamwidth_theta=float(rng.uniform(0.4, 1.5)),
            normal=tuple(normals[i]),
        ) for i in range(count)
    )
    return centers
