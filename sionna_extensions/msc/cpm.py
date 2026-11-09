"""Deterministic VV-only adapter for Sionna's Jones-matrix contract."""

import drjit as dr
import mitsuba as mi


class MSCCPM:
    """Signed VV adapter restoring the kernel phase; all other entries zero."""

    def __init__(self, rcs=None):
        self.rcs = rcs

    def __call__(self, k_i, k_s, seed=0):
        count = max(dr.width(k_i), dr.width(k_s))
        zero = dr.zeros(mi.Float, int(count))
        sign = (dr.select(self.rcs.amplitude(k_i, k_s) < 0, -1.0, 1.0)
                if self.rcs is not None else 1.0)
        return mi.Matrix2f(sign + zero, zero, zero, zero), dr.zeros(mi.Matrix2f, int(count))
