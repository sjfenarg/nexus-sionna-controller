"""MSC composition through Sionna's existing ScatteringModel contract."""

from dataclasses import asdict
import mitsuba as mi

from ..scattering_model import ScatteringModel
from .cpm import MSCCPM
from .rcs import MSCRCS


class MSCScatteringModel(ScatteringModel):
    def __init__(self, object_type, lcs_positions, lcs_orientations, parameters,
                 polarization_laws, parameter_seed=0, frequency_hz=12.5e9,
                 random_phases=False, random_xpr=False, random_cpr=False):
        self.object_type = object_type
        self.parameter_seed = parameter_seed
        self.parameters = tuple(parameters)
        self.polarization_laws = polarization_laws
        self._rcs = tuple(MSCRCS(p, frequency_hz) for p in self.parameters)
        self._cpm = MSCCPM(polarization_laws, object_type.split("-")[0], parameter_seed,
                          random_phases, random_xpr, random_cpr)
        super().__init__(mi.Point3f(lcs_positions), mi.Point3f(lcs_orientations), list(self._rcs), self._cpm)

    def set_frequency(self, frequency_hz):
        """Called by the installed RCSSolver hook once per scene solve."""
        for rcs in self._rcs:
            rcs.frequency_hz = frequency_hz

    @property
    def frequency_hz(self):
        return self._rcs[0].frequency_hz

    @property
    def random_phases(self):
        return self._cpm.random_phases

    @random_phases.setter
    def random_phases(self, value):
        self._cpm.random_phases = value

    @property
    def random_xpr(self):
        return self._cpm.random_xpr

    @random_xpr.setter
    def random_xpr(self, value):
        self._cpm.random_xpr = value

    @property
    def random_cpr(self):
        return self._cpm.random_cpr

    @random_cpr.setter
    def random_cpr(self, value):
        self._cpm.random_cpr = value

    def parameter_dict(self):
        """Serializable record of every prototype draw for reproducibility."""
        return {"parameter_seed": self.parameter_seed, "calibrated": False,
                "centers": [asdict(p) for p in self.parameters],
                "polarization": {mode: asdict(law) for mode, law in self.polarization_laws.items()}}
