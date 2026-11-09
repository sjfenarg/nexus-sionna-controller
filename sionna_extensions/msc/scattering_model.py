"""MSC composition through Sionna's existing ScatteringModel contract."""

from dataclasses import asdict
import mitsuba as mi

from ..scattering_model import ScatteringModel
from .cpm import MSCCPM
from .rcs import MSCRCS


class MSCScatteringModel(ScatteringModel):
    def __init__(self, object_type, lcs_positions, lcs_orientations, parameters,
                 parameter_seed=0, frequency_hz=12.5e9):
        self.object_type = object_type
        self.parameter_seed = parameter_seed
        self.parameters = tuple(parameters)
        self._rcs = tuple(MSCRCS(p, frequency_hz) for p in self.parameters)
        self._cpm = tuple(MSCCPM(rcs) for rcs in self._rcs)
        super().__init__(mi.Point3f(lcs_positions), mi.Point3f(lcs_orientations), list(self._rcs), list(self._cpm))

    def set_frequency(self, frequency_hz):
        """Called by the installed RCSSolver hook once per scene solve."""
        for rcs in self._rcs:
            rcs.frequency_hz = frequency_hz

    @property
    def frequency_hz(self):
        return self._rcs[0].frequency_hz

    def parameter_dict(self):
        """Serializable record of every prototype draw for reproducibility."""
        return {"parameter_seed": self.parameter_seed, "calibrated": False,
                "shape_mode": "reflecting_surface_per_lobe",
                "center_positions_m": self.spst.lcs_positions.numpy().T.tolist(),
                "center_orientations_rad": self.spst.lcs_orientations.numpy().T.tolist(),
                "centers": [asdict(p) for p in self.parameters],
                "polarization": "VV only (scattering-point basis)"}
