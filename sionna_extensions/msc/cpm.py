"""VV-referenced whole-target CPM, manuscript equations (17)-(18)."""

import math
import drjit as dr
import mitsuba as mi

from ..tr38901.random_draws import gaussian, phase
from .parameters import MSCPolarizationParameters


class MSCCPM:
    """CPM with unit VV, independent cross-ratio draws, and HH/VV CPR.

    Random draws depend on realization seed and observation mode, not center
    index or point orientation. All points of a target share this callable.
    Parameters are placeholders; band-specific fitted laws can replace them.
    """

    def __init__(self, laws=None, target_type="vehicle", realization_seed=0,
                 random_phases=False, random_xpr=False, random_cpr=False):
        self.laws = laws or {m: MSCPolarizationParameters()
                             for m in ("monostatic", "bistatic")}
        if set(self.laws) != {"monostatic", "bistatic"}:
            raise ValueError("laws must define monostatic and bistatic parameters")
        self.target_type = target_type
        self.realization_seed = int(realization_seed)
        self.random_phases = random_phases
        self.random_xpr = random_xpr
        self.random_cpr = random_cpr

    def __call__(self, k_i, k_s, seed=0):
        count = max(dr.width(k_i), dr.width(k_s))
        zero = dr.zeros(mi.Float, int(count))
        mono = dr.dot(-dr.normalize(k_i), dr.normalize(k_s)) >= 1.0 - 1e-5
        real, imag = dr.zeros(mi.Matrix2f, int(count)), dr.zeros(mi.Matrix2f, int(count))
        for mode_index, mode in enumerate(("monostatic", "bistatic")):
            law = self.laws[mode]
            key = mi.UInt((int(seed) + self.realization_seed + mode_index*104729) & 0xffffffff)
            xpr_vh = gaussian(key, 101, law.xpr_mean_db, law.xpr_std_db) if self.random_xpr else law.xpr_mean_db
            xpr_hv = gaussian(key, 102, law.xpr_mean_db, law.xpr_std_db) if self.random_xpr else law.xpr_mean_db
            cpr = gaussian(key, 103, law.cpr_mean_db, law.cpr_std_db) if self.random_cpr else law.cpr_mean_db
            vh = dr.exp(-math.log(10.0)/20.0*xpr_vh) + zero
            hv = dr.exp(-math.log(10.0)/20.0*xpr_hv) + zero
            hh = dr.exp(math.log(10.0)/20.0*cpr) + zero
            phi_vh = phase(key, 104) if self.random_phases else zero
            phi_hv = phi_vh if mode == "monostatic" and self.target_type in ("vehicle", "agv") else (
                phase(key, 105) if self.random_phases else zero)
            phi_hh = phase(key, 106) if self.random_phases else zero
            candidate_real = mi.Matrix2f(1.0 + zero, vh*dr.cos(phi_vh), hv*dr.cos(phi_hv), hh*dr.cos(phi_hh))
            candidate_imag = mi.Matrix2f(zero, vh*dr.sin(phi_vh), hv*dr.sin(phi_hv), hh*dr.sin(phi_hh))
            active = mono if mode == "monostatic" else ~mono
            real = dr.select(active, candidate_real, real)
            imag = dr.select(active, candidate_imag, imag)
        return real, imag
