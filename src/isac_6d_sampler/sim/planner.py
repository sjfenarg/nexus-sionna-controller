from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isac_6d_sampler.core.model import BaseStation, SceneDesign, UserEquipment, Vector3
from isac_6d_sampler.core.trajectories import (
    radiomap_grid_shape,
    sample_orientations,
    sample_radiomap_grid,
    sample_trajectory,
)


PlannedDevice = UserEquipment | BaseStation


@dataclass(frozen=True, slots=True)
class LinkPlan:
    rx_index: int
    tx_index: int
    rx: PlannedDevice
    tx: PlannedDevice

    @property
    def dataset_name(self) -> str:
        return f"rx{self.rx_index}_tx{self.tx_index}"

    @property
    def is_monostatic(self) -> bool:
        return self.rx_index == self.tx_index


@dataclass(frozen=True, slots=True)
class TimeframePlan:
    name: str
    device_positions: dict[str, Vector3]
    object_positions: dict[str, Vector3]
    device_orientations: dict[str, Vector3]
    object_orientations: dict[str, Vector3]
    links: tuple[LinkPlan, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimulationPlan:
    devices: tuple[PlannedDevice, ...]
    timeframes: tuple[TimeframePlan, ...]


def build_simulation_plan(scene: SceneDesign) -> SimulationPlan:
    """Create the ordered device/link/timeframe plan used by all backends.

    Device indexing follows the reference HDF5 convention used by this package:
    all UEs first, then all BSs. Links include each device's monostatic channel
    and both ordered UE-BS directions. Monostatic links are generated for UEs,
    which covers radar-like user devices without producing massive BS self-links
    for large arrays.
    """
    scene.ensure_defaults()
    user_equipments = planned_user_equipments(scene)
    devices: tuple[PlannedDevice, ...] = (*user_equipments, *scene.base_stations)
    links = _link_plan(devices)
    frames = _timeframes(scene, user_equipments, links)
    return SimulationPlan(devices=devices, timeframes=tuple(frames))


def position_for(device: PlannedDevice, timeframe: TimeframePlan) -> Vector3:
    return timeframe.device_positions.get(device.id, device.position)


def orientation_for(device: PlannedDevice, timeframe: TimeframePlan) -> Vector3:
    return timeframe.device_orientations.get(device.id, device.orientation_rad)


def planned_user_equipments(scene: SceneDesign) -> tuple[UserEquipment, ...]:
    if not scene.radiomap.enabled:
        return tuple(scene.user_equipments)
    template = scene.radiomap.ue_template
    if template.id == "rm_ue":
        return (
            UserEquipment(
                id="ue_radiomap",
                position=template.position,
                orientation_rad=template.orientation_rad,
                panel=template.panel,
                trajectory=template.trajectory,
            ),
        )
    return (template,)


def _link_plan(devices: tuple[PlannedDevice, ...]) -> tuple[LinkPlan, ...]:
    links: list[LinkPlan] = []
    for rx_index, rx in enumerate(devices):
        for tx_index, tx in enumerate(devices):
            if (rx_index == tx_index and isinstance(rx, UserEquipment)) or _is_ue_bs_pair(rx, tx):
                links.append(LinkPlan(rx_index=rx_index, tx_index=tx_index, rx=rx, tx=tx))
    return tuple(links)


def _timeframes(
    scene: SceneDesign,
    user_equipments: tuple[UserEquipment, ...],
    links: tuple[LinkPlan, ...],
) -> list[TimeframePlan]:
    base_station_positions = {entity.id: entity.position for entity in scene.base_stations}
    if scene.radiomap.enabled:
        radiomap_positions = sample_radiomap_grid(scene.radiomap)
        x_points, _ = radiomap_grid_shape(scene.radiomap)
        ue = user_equipments[0]
        sampled_objects = {
            obj.id: sample_trajectory(obj.trajectory)
            for obj in scene.objects
        }
        sampled_object_orientations = {
            obj.id: sample_orientations(obj.trajectory, obj.orientation_rad)
            for obj in scene.objects
        }
        object_frame_count = max((values.shape[0] for values in sampled_objects.values()), default=1)
        frames = []
        for object_idx in range(object_frame_count):
            object_positions = {
                entity_id: tuple(values[min(object_idx, len(values) - 1)])
                for entity_id, values in sampled_objects.items()
            }
            object_orientations = {
                entity_id: tuple(values[min(object_idx, len(values) - 1)])
                for entity_id, values in sampled_object_orientations.items()
            }
            for grid_idx, radiomap_pos in enumerate(radiomap_positions):
                frames.append(
                    TimeframePlan(
                        name=f"tf{len(frames):03d}",
                        device_positions={ue.id: tuple(radiomap_pos), **base_station_positions},
                        object_positions=dict(object_positions),
                        device_orientations={
                            ue.id: ue.orientation_rad,
                            **{entity.id: entity.orientation_rad for entity in scene.base_stations},
                        },
                        object_orientations=dict(object_orientations),
                        links=links,
                        metadata={
                            "frame_kind": "radiomap",
                            "radiomap_grid_index": grid_idx,
                            "radiomap_x_index": grid_idx % x_points,
                            "radiomap_y_index": grid_idx // x_points,
                            "radiomap_total_grid_points": int(radiomap_positions.shape[0]),
                            "radiomap_object_state_index": object_idx,
                            "radiomap_object_state_count": object_frame_count,
                        },
                    )
                )
        return frames

    sampled_devices: dict[str, np.ndarray] = {}
    sampled_device_orientations: dict[str, np.ndarray] = {}
    sampled_objects: dict[str, np.ndarray] = {}
    sampled_object_orientations: dict[str, np.ndarray] = {}
    max_frames = 1
    for ue in user_equipments:
        sampled_devices[ue.id] = sample_trajectory(ue.trajectory)
        sampled_device_orientations[ue.id] = sample_orientations(ue.trajectory, ue.orientation_rad)
        max_frames = max(max_frames, sampled_devices[ue.id].shape[0])
    for obj in scene.objects:
        sampled_objects[obj.id] = sample_trajectory(obj.trajectory)
        sampled_object_orientations[obj.id] = sample_orientations(obj.trajectory, obj.orientation_rad)
        max_frames = max(max_frames, sampled_objects[obj.id].shape[0])

    frames = []
    for idx in range(max_frames):
        device_positions = {
            entity_id: tuple(values[min(idx, len(values) - 1)])
            for entity_id, values in sampled_devices.items()
        }
        device_positions.update(base_station_positions)
        object_positions = {
            entity_id: tuple(values[min(idx, len(values) - 1)])
            for entity_id, values in sampled_objects.items()
        }
        device_orientations = {
            entity_id: tuple(values[min(idx, len(values) - 1)])
            for entity_id, values in sampled_device_orientations.items()
        }
        device_orientations.update({entity.id: entity.orientation_rad for entity in scene.base_stations})
        object_orientations = {
            entity_id: tuple(values[min(idx, len(values) - 1)])
            for entity_id, values in sampled_object_orientations.items()
        }
        frames.append(
            TimeframePlan(
                name=f"tf{idx:03d}",
                device_positions=device_positions,
                object_positions=object_positions,
                device_orientations=device_orientations,
                object_orientations=object_orientations,
                links=links,
                metadata={
                    "frame_kind": "scene",
                    "scene_frame_index": idx,
                    "scene_frame_count": max_frames,
                },
            )
        )
    return frames


def _is_ue_bs_pair(rx: PlannedDevice, tx: PlannedDevice) -> bool:
    return rx.__class__ is not tx.__class__
