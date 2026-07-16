from __future__ import annotations

from dataclasses import dataclass
import math

from .model import SimulationRequest, TrajectorySpec
from .trajectories import radiomap_grid_shape


@dataclass(frozen=True, slots=True)
class RequestSizeEstimate:
    """Cheap request-size estimate for operator feedback and guardrails."""

    scene_name: str
    radiomap_enabled: bool
    bs_count: int
    ue_count: int
    active_ue_count: int
    object_count: int
    object_state_count: int
    device_count: int
    links_per_timeframe: int
    timeframe_count: int
    channel_dataset_count: int
    frequency_sample_count: int
    band_count: int
    timeframe_batches: int
    batch_timeframes: int
    max_timeframes: int
    samples_per_src: int
    max_num_paths_per_src: int | None
    radiomap_x_points: int | None = None
    radiomap_y_points: int | None = None


def estimate_request_size(request: SimulationRequest) -> RequestSizeEstimate:
    """Estimate run size without building a full simulation plan or importing Sionna."""

    scene = request.scene
    bs_count = len(scene.base_stations)
    ue_count = len(scene.user_equipments)
    object_count = len(scene.objects)
    frequency_sample_count = sum(int(band.points) for band in request.bands)

    if scene.radiomap.enabled:
        radiomap_x_points, radiomap_y_points = radiomap_grid_shape(scene.radiomap)
        object_state_count = max((_trajectory_sample_count(obj.trajectory) for obj in scene.objects), default=1)
        timeframe_count = radiomap_x_points * radiomap_y_points * object_state_count
        active_ue_count = 1
    else:
        radiomap_x_points = None
        radiomap_y_points = None
        active_ue_count = ue_count
        object_state_count = max((_trajectory_sample_count(obj.trajectory) for obj in scene.objects), default=1)
        timeframe_count = max(
            [1]
            + [_trajectory_sample_count(ue.trajectory) for ue in scene.user_equipments]
            + [_trajectory_sample_count(obj.trajectory) for obj in scene.objects]
        )

    device_count = active_ue_count + bs_count
    links_per_timeframe = active_ue_count + (2 * active_ue_count * bs_count)
    channel_dataset_count = timeframe_count * links_per_timeframe
    batch_timeframes = max(1, int(request.sionna.batch_timeframes))
    timeframe_batches = math.ceil(timeframe_count / batch_timeframes)

    return RequestSizeEstimate(
        scene_name=scene.name,
        radiomap_enabled=scene.radiomap.enabled,
        bs_count=bs_count,
        ue_count=ue_count,
        active_ue_count=active_ue_count,
        object_count=object_count,
        object_state_count=object_state_count,
        device_count=device_count,
        links_per_timeframe=links_per_timeframe,
        timeframe_count=timeframe_count,
        channel_dataset_count=channel_dataset_count,
        frequency_sample_count=frequency_sample_count,
        band_count=len(request.bands),
        timeframe_batches=timeframe_batches,
        batch_timeframes=batch_timeframes,
        max_timeframes=request.sionna.max_timeframes,
        samples_per_src=request.sionna.samples_per_src,
        max_num_paths_per_src=request.sionna.max_num_paths_per_src,
        radiomap_x_points=radiomap_x_points,
        radiomap_y_points=radiomap_y_points,
    )


def format_request_estimate(estimate: RequestSizeEstimate) -> str:
    """Return a compact multiline summary suitable for CLI output and GUI logs."""

    lines = [
        f"scene: {estimate.scene_name}",
        f"radiomap: {'on' if estimate.radiomap_enabled else 'off'}",
        f"devices: {estimate.device_count} active ({estimate.active_ue_count} UE, {estimate.bs_count} BS)",
        f"objects: {estimate.object_count} configured, {estimate.object_state_count} state(s)",
        f"timeframes: {estimate.timeframe_count} (max {estimate.max_timeframes})",
        f"links/timeframe: {estimate.links_per_timeframe}",
        f"channel datasets: {estimate.channel_dataset_count}",
        f"frequency samples: {estimate.frequency_sample_count} across {estimate.band_count} band(s)",
        f"timeframe batches: {estimate.timeframe_batches} at batch_timeframes={estimate.batch_timeframes}",
        (
            "Sionna rays: "
            f"samples_per_src={estimate.samples_per_src}, "
            f"max_num_paths_per_src={estimate.max_num_paths_per_src if estimate.max_num_paths_per_src is not None else 'unlimited'}"
        ),
    ]
    if estimate.radiomap_enabled:
        lines.insert(
            2,
            f"radiomap grid: {estimate.radiomap_x_points} x {estimate.radiomap_y_points}",
        )
    return "\n".join(lines)


def _trajectory_sample_count(trajectory: TrajectorySpec) -> int:
    if trajectory.kind == "static" or len(trajectory.points) <= 1:
        return 1
    return max(1, int(trajectory.samples))
