"""VV center RCS: Section III, equations (8)-(16) of the MSC manuscript."""

import math
import drjit as dr
import mitsuba as mi

from .parameters import MSCCenterParameters

C = 299_792_458.0


def sinc(x):
    """Unnormalized sin(x)/x with a finite value and derivative at the origin."""
    small = dr.abs(x) < 1e-4
    safe = dr.select(small, 1.0, x)
    return dr.select(small, 1.0 - dr.square(x)/6.0, dr.sin(safe)/safe)


class MSCRCS:
    """Sionna RCS callable returning non-negative VV power in square meters.

    Inputs are propagation directions in the point's LCS. Frequency is bound
    by MSCScatteringModel.set_frequency() before each RCSSolver solve. Path
    phase is supplied by the solver, never added to this power callable.
    """

    def __init__(self, parameters=None, frequency_hz=12.5e9):
        self.parameters = parameters or MSCCenterParameters()
        self.frequency_hz = frequency_hz

    @property
    def frequency_hz(self):
        return self._frequency_hz

    @frequency_hz.setter
    def frequency_hz(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        self._frequency_hz = value
        self._frequency = mi.Float(value)
        dr.make_opaque(self._frequency)

    def amplitude(self, k_i, k_s):
        p = self.parameters
        k_i, k_s = dr.normalize(mi.Vector3f(k_i)), dr.normalize(mi.Vector3f(k_s))
        normal = dr.normalize(mi.Vector3f(*p.normal))
        reflected = k_i - 2.0*dr.dot(k_i, normal)*normal
        # atan2 remains accurate around the specular maximum.
        psi = dr.atan2(dr.norm(dr.cross(reflected, k_s)), dr.dot(reflected, k_s))
        window = (1.0 + math.exp(-p.psi_threshold/p.transition_width)) / (
            1.0 + dr.exp((psi - p.psi_threshold)/p.transition_width))
        # Literal manuscript equation: the smooth window attenuates the rear lobe.
        spec = p.w0 * dr.square(dr.cos(psi)) * window
        side = p.wr_side * (
            dr.square(sinc(p.kappa_side*(psi - p.psi_side)))
            + dr.square(sinc(p.kappa_side*(psi - (2.0*math.pi - p.psi_side)))))
        back = p.wr_back * dr.square(sinc(p.kappa_back*(psi - p.psi_back)))
        # Project the configured shaping axis onto the specular transverse plane.
        axis = mi.Vector3f(*p.shaping_axis)
        transverse = axis - dr.dot(axis, reflected)*reflected
        # At alignment use a nonparallel Cartesian axis for a stable frame.
        fallback = dr.select(dr.abs(reflected.z) < 0.9,
                             mi.Vector3f(0, 0, 1), mi.Vector3f(0, 1, 0))
        transverse = dr.select(dr.squared_norm(transverse) > 1e-12, transverse,
                               fallback - dr.dot(fallback, reflected)*reflected)
        e_phi = dr.normalize(transverse)
        e_theta = dr.cross(reflected, e_phi)
        x, y, z = dr.dot(k_s, e_phi), dr.dot(k_s, e_theta), dr.dot(k_s, reflected)
        delta_phi = dr.atan2(x, z)
        delta_theta = dr.atan2(y, dr.sqrt(dr.square(x) + dr.square(z)))
        wavelength = C/self._frequency
        wavelength_ref = C/p.reference_frequency_hz
        l_phi = wavelength_ref/(2.0*math.sin(p.beamwidth_phi/2.0))
        l_theta = wavelength_ref/(2.0*math.sin(p.beamwidth_theta/2.0))
        shape = dr.abs(sinc(l_phi*dr.sin(delta_phi)/wavelength)) * dr.abs(
            sinc(l_theta*dr.sin(delta_theta)/wavelength))
        return (spec + side + back)*shape

    def __call__(self, k_i, k_s, seed=0):
        return dr.square(self.amplitude(k_i, k_s))
