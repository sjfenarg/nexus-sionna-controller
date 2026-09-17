from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

Vector3 = tuple[float, float, float]
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
DEFAULT_CENTER_FREQUENCY_HZ = 79e9
DEFAULT_HALF_WAVELENGTH_SPACING_M = SPEED_OF_LIGHT_M_PER_S / (2.0 * DEFAULT_CENTER_FREQUENCY_HZ)
DYNAMIC_SCENE_OBJECT_NAMES = ("CAR_obj",)


class ChannelMode(str, Enum):
    """Channel rendering modes exposed by the controller."""

    FREQUENCY_DOMAIN = "frequency_domain"
    CIR_PATHS = "cir_paths"
    PDP_BINNED = "pdp_binned"
    PDP_IFFT_GRIDDED = "pdp_ifft_gridded"
    PDP_IFFT_EXACT = "pdp_ifft_exact"
    COHERENT_PER_BIN = "coherent_per_bin"


@dataclass(slots=True)
class FrequencyBand:
    name: str = "77-81GHz"
    start_hz: float = 77e9
    stop_hz: float = 81e9
    points: int = 1024

    def vector(self) -> np.ndarray:
        if self.points < 2:
            raise ValueError("Frequency bands require at least two points")
        if self.stop_hz <= self.start_hz:
            raise ValueError("Frequency band stop_hz must be larger than start_hz")
        return np.linspace(self.start_hz, self.stop_hz, self.points, dtype=np.float64)


@dataclass(slots=True)
class AntennaPanel:
    rows: int = 1
    cols: int = 1
    pattern: str = "iso"
    polarization: Literal["V", "H", "VH", "cross"] = "V"
    vertical_spacing_m: float = 0.0
    horizontal_spacing_m: float = 0.0
    element_diagram: str = "iso"
    orientation_rad: Vector3 = (0.0, 0.0, 0.0)
    v_pol_vector: Vector3 = (0.0, 0.0, 1.0)
    h_pol_vector: Vector3 | None = None

    @classmethod
    def default_bs(cls) -> "AntennaPanel":
        return cls(
            rows=10,
            cols=10,
            vertical_spacing_m=DEFAULT_HALF_WAVELENGTH_SPACING_M,
            horizontal_spacing_m=DEFAULT_HALF_WAVELENGTH_SPACING_M,
            element_diagram="iso",
        )

    @classmethod
    def default_ue(cls) -> "AntennaPanel":
        return cls(rows=1, cols=1, element_diagram="iso")

    @property
    def element_count(self) -> int:
        return int(self.rows) * int(self.cols)


@dataclass(slots=True)
class TrajectorySpec:
    kind: Literal["static", "linear", "polyline", "curve"] = "static"
    points: list[Vector3] = field(default_factory=lambda: [(0.0, 0.0, 0.0)])
    bezier_handles: list[tuple[Vector3, Vector3]] = field(default_factory=list)
    orientation_rad_points: list[Vector3] = field(default_factory=list)
    samples: int = 1
    start_static_fraction: float = 0.0
    end_static_fraction: float = 0.0
    easing: Literal["linear", "smoothstep"] = "linear"

    @classmethod
    def static(cls, position: Vector3) -> "TrajectorySpec":
        return cls(kind="static", points=[position], samples=1)

    @classmethod
    def linear(cls, start: Vector3, stop: Vector3, samples: int) -> "TrajectorySpec":
        return cls(kind="linear", points=[start, stop], samples=samples)


@dataclass(slots=True)
class BaseStation:
    id: str
    position: Vector3 = (0.0, 0.0, 2.0)
    orientation_rad: Vector3 = (0.0, 0.0, 0.0)
    panel: AntennaPanel = field(default_factory=AntennaPanel.default_bs)


@dataclass(slots=True)
class UserEquipment:
    id: str
    position: Vector3 = (0.0, 0.0, 1.5)
    orientation_rad: Vector3 = (0.0, 0.0, 0.0)
    panel: AntennaPanel = field(default_factory=AntennaPanel.default_ue)
    trajectory: TrajectorySpec = field(default_factory=lambda: TrajectorySpec.static((0.0, 0.0, 1.5)))


@dataclass(slots=True)
class DynamicObject:
    id: str
    object_name: str = "CAR_obj"
    position: Vector3 = (0.0, 0.0, 0.0)
    orientation_rad: Vector3 = (0.0, 0.0, 0.0)
    trajectory: TrajectorySpec = field(default_factory=lambda: TrajectorySpec.static((0.0, 0.0, 0.0)))


@dataclass(slots=True)
class RadiomapConfig:
    enabled: bool = False
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    x_spacing: float = 1.0
    y_spacing: float = 1.0
    height: float = 1.5
    ue_template: UserEquipment = field(default_factory=lambda: UserEquipment(id="rm_ue"))


@dataclass(slots=True)
class SionnaConfig:
    tx_power_dbm: float = 44.0
    samples_per_src: int = 500_000
    max_depth: int = 3
    max_num_paths_per_src: int | None = 100_000
    seed: int | None = None
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = False
    synthetic_array: bool = False
    merge_shapes: bool = False
    use_gpu: bool = True
    batch_timeframes: int = 1
    max_timeframes: int = 100_000
    ue_ue_links: bool = False


@dataclass(slots=True)
class SceneDesign:
    name: str = "Outdoor6D_w_car"
    scenario_path: Path = Path("scenarios/Outdoor6D_w_car.xml")
    description: str = "Calibrated Outdoor 6D digital twin"
    base_stations: list[BaseStation] = field(default_factory=list)
    user_equipments: list[UserEquipment] = field(default_factory=list)
    objects: list[DynamicObject] = field(default_factory=list)
    radiomap: RadiomapConfig = field(default_factory=RadiomapConfig)

    def ensure_defaults(self) -> None:
        if not self.base_stations:
            self.base_stations.append(
                BaseStation(id="bs0", position=(52.353, -19.516, 19.203))
            )
        if not self.user_equipments and not self.radiomap.enabled:
            self.user_equipments.append(UserEquipment(id="ue0", position=(0.0, 0.0, 1.5)))


@dataclass(slots=True)
class SimulationRequest:
    scene: SceneDesign = field(default_factory=SceneDesign)
    bands: list[FrequencyBand] = field(default_factory=lambda: [FrequencyBand()])
    channel_mode: ChannelMode = ChannelMode.FREQUENCY_DOMAIN
    sionna: SionnaConfig = field(default_factory=SionnaConfig)
    output_dir: Path = Path("output")
    sample_id: str = "s000"
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            return value

        return convert(self)
