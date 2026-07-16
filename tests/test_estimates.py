from isac_6d_sampler.core.estimates import estimate_request_size, format_request_estimate
from isac_6d_sampler.core.model import BaseStation, DynamicObject, RadiomapConfig, SceneDesign, TrajectorySpec


def test_estimate_request_size_counts_multi_bs_scene_links_and_batches():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0"), BaseStation(id="bs1")],
        objects=[
            DynamicObject(
                id="car0",
                trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 5),
            )
        ],
    )
    scene.ensure_defaults()
    request = scene_request(scene)
    request.sionna.batch_timeframes = 2

    estimate = estimate_request_size(request)

    assert estimate.device_count == 3
    assert estimate.active_ue_count == 1
    assert estimate.links_per_timeframe == 5
    assert estimate.timeframe_count == 5
    assert estimate.channel_dataset_count == 25
    assert estimate.timeframe_batches == 3


def test_estimate_request_size_counts_radiomap_grid_and_object_states():
    scene = SceneDesign(
        base_stations=[BaseStation(id="bs0")],
        objects=[
            DynamicObject(
                id="car0",
                trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2),
            )
        ],
        radiomap=RadiomapConfig(
            enabled=True,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.0,
            x_spacing=1.0,
            y_spacing=1.0,
        ),
    )
    request = scene_request(scene)

    estimate = estimate_request_size(request)

    assert estimate.radiomap_enabled
    assert estimate.radiomap_x_points == 2
    assert estimate.radiomap_y_points == 1
    assert estimate.object_state_count == 2
    assert estimate.timeframe_count == 4
    assert estimate.links_per_timeframe == 3
    assert "radiomap grid: 2 x 1" in format_request_estimate(estimate)


def scene_request(scene):
    from isac_6d_sampler.core.model import SimulationRequest

    return SimulationRequest(scene=scene)
