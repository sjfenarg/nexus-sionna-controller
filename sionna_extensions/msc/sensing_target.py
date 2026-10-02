"""MSC target geometry follows the installed TR38901 target templates."""

import math
import numpy as np
import mitsuba as mi

from sionna.rt.utils.meshes import clone_mesh
from ..tr38901 import TR38901SensingTarget
from ..tr38901.parameters import get_target_parameters
from .parameters import placeholder_parameters
from .scattering_model import MSCScatteringModel


class MSCSensingTarget(TR38901SensingTarget):
    """Human, AGV and car MSC prototypes with one or five centers.

    Vehicle/AGV templates preserve TR38901 size, positions and point frames.
    A five-point human is an additional cuboid template; TR38901 has only one.
    Caller-supplied center parameters and polarization laws replace prototypes.
    """

    def __init__(self, name, object_type="vehicle-multi-sp", parameter_seed=0,
                 frequency_hz=12.5e9, parameters=None, polarization_laws=None,
                 random_phases=False, random_xpr=False, random_cpr=False, **geometry):
        allowed = {"human", "human-single-sp", "human-multi-sp", "vehicle-single-sp",
                   "vehicle-multi-sp", "agv-single-sp", "agv-multi-sp"}
        if object_type not in allowed:
            raise ValueError(f"Unsupported MSC object_type: {object_type}")
        if isinstance(parameter_seed, bool) or not isinstance(parameter_seed, int) or parameter_seed < 0:
            raise ValueError("parameter_seed must be a non-negative integer")
        template = "human" if object_type.startswith("human") else object_type
        super().__init__(name, template, model_type=2, **geometry)
        source = self.scattering_model.spst
        normals = [(1.0, 0.0, 0.0)]
        positions, orientations = source.lcs_positions, source.lcs_orientations
        if object_type == "human-multi-sp":
            positions = mi.Point3f([self._length/2, -self._length/2, 0, 0, 0],
                                  [0, 0, self._width/2, -self._width/2, 0],
                                  [0, 0, 0, 0, self._height/2])
            orientations = mi.Point3f([0, math.pi, math.pi/2, -math.pi/2, 0], [0]*5, [0]*5)
            normals = [(1., 0., 0.)]*4 + [(0., 0., 1.)]
        elif object_type.endswith("multi-sp"):
            normals = [(math.sin(math.radians(lobe.theta_center)), 0.,
                        math.cos(math.radians(lobe.theta_center)))
                       for lobe in get_target_parameters(template, 2).lobes]
        count = np.asarray(positions.numpy()).shape[-1]
        defaults, laws = placeholder_parameters(count, parameter_seed, normals)
        parameters = tuple(parameters) if parameters is not None else defaults
        if len(parameters) != count:
            raise ValueError(f"Expected {count} center parameter sets")
        self._object_type = object_type
        self._scattering_model = MSCScatteringModel(
            object_type, positions, orientations, parameters, polarization_laws or laws,
            parameter_seed, frequency_hz, random_phases, random_xpr, random_cpr)

    def clone(self, name=None, as_mesh=False, props=None):
        mesh = clone_mesh(self.mi_mesh, name=name, props=props)
        if as_mesh:
            mesh.set_bsdf(self.radio_material)
            return mesh
        clone = MSCSensingTarget(mesh.id(), self.object_type, mi_mesh=mesh,
                                 parameter_seed=self.scattering_model.parameter_seed,
                                 color=self.radio_material.color, display_opacity=self.display_opacity)
        clone._length, clone._width, clone._height = self._length, self._width, self._height
        clone._scattering_model = self._scattering_model
        self._carry_over_pose(clone)
        return clone
