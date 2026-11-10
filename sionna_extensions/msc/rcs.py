"""Signed reflecting_surface_per_lobe kernel, matching the trained engine."""

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
        view_tx = -k_i
        reflected = k_i - 2.0*dr.dot(k_i, normal)*normal

        def angle(a, b):
            return dr.acos(dr.clip(dr.dot(a, b), -1.0, 1.0))

        def visibility(x):
            return (1.0 + math.exp(-1.3/0.3))/(1.0 + dr.exp((x-1.3)/0.3))

        def look_at(direction, specular=False):
            # Exact training convention, evaluated in target/center coordinates.
            raw = mi.Vector3f(0, 0, 1) - direction*direction.z
            length = dr.norm(raw)
            fallback = mi.Vector3f(0, 1, 0)
            if specular and p.pole_azimuth_rad is not None:
                # Training retains azimuth through float32 cos(+/-pi/2).
                # Define the same aperture axes explicitly, without relying on
                # that residual. Axis signs cancel in the even sinc factors.
                fallback = mi.Vector3f(math.cos(p.pole_azimuth_rad),
                                       math.sin(p.pole_azimuth_rad), 0)
            b = dr.select(length < 1e-9, fallback, raw/dr.maximum(length, 1e-12))
            return dr.cross(b, direction), b

        # Duff tangent frame used by the trained engine for lobe directions.
        sign = dr.select(normal.z >= 0, 1.0, -1.0)
        a = -1.0/(sign + normal.z)
        b = normal.x*normal.y*a
        tangent = mi.Vector3f(1.0 + sign*normal.x*normal.x*a, sign*b, -sign*normal.x)
        side1 = math.cos(p.psi_side)*normal + math.sin(p.psi_side)*tangent
        side2 = math.cos(p.psi_side)*normal - math.sin(p.psi_side)*tangent
        back_angle = p.psi_back if p.psi_back <= math.pi else 2*math.pi-p.psi_back
        back = math.cos(back_angle)*normal + math.sin(back_angle)*tangent
        colat = angle(view_tx, normal)
        k_spec = dr.square(dr.cos(colat))*visibility(colat)
        scale_phi = 0.44294647068945243/max(math.sin(p.beamwidth_phi/2), 1.1920928955078125e-7)*self._frequency/p.reference_frequency_hz
        scale_theta = 0.44294647068945243/max(math.sin(p.beamwidth_theta/2), 1.1920928955078125e-7)*self._frequency/p.reference_frequency_hz

        def shape(direction, specular=False):
            t, b = look_at(direction, specular)
            v = k_s + view_tx if specular else k_s
            # torch.sinc is normalized and SIGNED, unlike the geometric sinc.
            return sinc(math.pi*dr.dot(v, t)*scale_phi)*sinc(math.pi*dr.dot(v, b)*scale_theta)

        result = p.w0*k_spec*visibility(angle(k_s, reflected))*shape(normal, True)
        for direction, weight, width in ((side1, p.wr_side, 20.0),
                                          (side2, p.wr_side, 20.0),
                                          (back, p.wr_back, 15.0)):
            kernel = dr.square(sinc(width*angle(view_tx, direction)))
            result += weight*kernel*visibility(angle(k_s, direction))*shape(direction)
        return result

    def __call__(self, k_i, k_s, seed=0):
        return dr.square(self.amplitude(k_i, k_s))
