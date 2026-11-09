"""Deterministic VV-only adapter for Sionna's Jones-matrix contract."""

import drjit as dr
import mitsuba as mi


class MSCCPM:
    """Unit VV response in the scattering-point basis; all other entries zero."""

    def __call__(self, k_i, k_s, seed=0):
        count = max(dr.width(k_i), dr.width(k_s))
        zero = dr.zeros(mi.Float, int(count))
        return mi.Matrix2f(1.0 + zero, zero, zero, zero), dr.zeros(mi.Matrix2f, int(count))
