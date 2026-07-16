import pytest

from isac_6d_sampler.core.model import (
    BaseStation,
    ChannelMode,
    DynamicObject,
    FrequencyBand,
    RadiomapConfig,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.core.validation import validate_request


def test_validate_request_accepts_defaulted_scene():
    request = SimulationRequest()
    request.scene.ensure_defaults()

    validate_request(request)


def test_validate_request_accepts_multiband_frequency_domain():
    request = SimulationRequest()
    request.scene.ensure_defaults()
    request.channel_mode = ChannelMode.FREQUENCY_DOMAIN
    request.bands = [
        FrequencyBand(name="low", start_hz=77e9, stop_hz=78e9, points=4),
        FrequencyBand(name="high", start_hz=80e9, stop_hz=81e9, points=4),
    ]

    validate_request(request)


@pytest.mark.parametrize(
    "channel_mode",
    [
        ChannelMode.COHERENT_PER_BIN,
        ChannelMode.PDP_BINNED,
        ChannelMode.PDP_IFFT_GRIDDED,
        ChannelMode.PDP_IFFT_EXACT,
    ],
)
def test_validate_request_rejects_multiband_delay_bin_modes(channel_mode):
    request = SimulationRequest()
    request.scene.ensure_defaults()
    request.channel_mode = channel_mode
    request.bands = [
        FrequencyBand(name="low", start_hz=77e9, stop_hz=78e9, points=4),
        FrequencyBand(name="high", start_hz=80e9, stop_hz=81e9, points=4),
    ]

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    message = str(exc_info.value)
    assert f"{channel_mode.value} requires one contiguous uniformly sampled frequency band" in message
    assert "frequency_domain or cir_paths" in message


def test_validate_request_reports_duplicate_device_and_invalid_panel():
    request = SimulationRequest()
    request.scene.base_stations = [BaseStation(id="node0")]
    request.scene.user_equipments = [UserEquipment(id="node0")]
    request.scene.user_equipments[0].panel.rows = 0

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    message = str(exc_info.value)
    assert "Duplicate device id: node0" in message
    assert "user_equipments[0].panel.rows must be at least 1" in message


def test_validate_request_rejects_pattern_and_element_diagram_mismatch():
    request = SimulationRequest()
    request.scene.ensure_defaults()
    request.scene.base_stations[0].panel.pattern = "iso"
    request.scene.base_stations[0].panel.element_diagram = "dipole"

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    assert "base_stations[0].panel.element_diagram must match base_stations[0].panel.pattern" in str(exc_info.value)


def test_validate_request_checks_radiomap_bounds():
    request = SimulationRequest()
    request.scene.base_stations = [BaseStation(id="bs0")]
    request.scene.radiomap = RadiomapConfig(enabled=True, x_min=2.0, x_max=1.0)

    with pytest.raises(ValueError, match="radiomap"):
        validate_request(request)


def test_validate_request_rejects_radiomap_default_active_ue_id_collision():
    request = SimulationRequest()
    request.scene.base_stations = [BaseStation(id="ue_radiomap")]
    request.scene.user_equipments = []
    request.scene.radiomap = RadiomapConfig(enabled=True)

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    assert "Duplicate device id: ue_radiomap" in str(exc_info.value)


def test_validate_request_rejects_custom_radiomap_template_id_collision():
    request = SimulationRequest()
    request.scene.base_stations = [BaseStation(id="bs0")]
    request.scene.user_equipments = []
    request.scene.radiomap = RadiomapConfig(enabled=True, ue_template=UserEquipment(id="bs0"))

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    assert "Duplicate device id: bs0" in str(exc_info.value)


def test_validate_request_checks_sionna_runtime_bounds():
    request = SimulationRequest()
    request.scene.ensure_defaults()
    request.sionna.samples_per_src = 0
    request.sionna.tx_power_dbm = float("inf")
    request.sionna.batch_timeframes = 0
    request.sionna.max_timeframes = 0

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    message = str(exc_info.value)
    assert "sionna.samples_per_src must be at least 1" in message
    assert "sionna.tx_power_dbm must be finite" in message
    assert "sionna.batch_timeframes must be at least 1" in message
    assert "sionna.max_timeframes must be at least 1" in message


def test_validate_request_rejects_oversized_radiomap_without_materializing_grid():
    request = SimulationRequest()
    request.scene.base_stations = [BaseStation(id="bs0")]
    request.scene.user_equipments = []
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=0.0,
        x_max=1_000_000.0,
        y_min=0.0,
        y_max=0.0,
        x_spacing=1.0,
        y_spacing=1.0,
        height=1.5,
    )
    request.sionna.max_timeframes = 10

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    message = str(exc_info.value)
    assert "Estimated timeframe count 1000001 exceeds sionna.max_timeframes 10" in message


def test_validate_request_counts_object_trajectory_states_in_radiomap(tmp_path):
    request = SimulationRequest()
    request.scene.scenario_path = tmp_path / "missing.xml"
    request.scene.base_stations = [BaseStation(id="bs0")]
    request.scene.user_equipments = []
    request.scene.objects = [
        DynamicObject(
            id="car0",
            trajectory=TrajectorySpec.linear((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 3),
        )
    ]
    request.scene.radiomap = RadiomapConfig(
        enabled=True,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        x_spacing=1.0,
        y_spacing=1.0,
        height=1.5,
    )
    request.sionna.max_timeframes = 11

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    assert "Estimated timeframe count 12 exceeds sionna.max_timeframes 11" in str(exc_info.value)


def test_validate_request_rejects_unknown_trajectory_kind():
    request = SimulationRequest()
    request.scene.ensure_defaults()
    request.scene.user_equipments[0].trajectory = TrajectorySpec(
        kind="teleport",
        points=[(0.0, 0.0, 1.5), (1.0, 0.0, 1.5)],
        samples=2,
    )

    with pytest.raises(ValueError, match="kind must be one of"):
        validate_request(request)


def test_validate_request_rejects_malformed_linear_trajectory():
    request = SimulationRequest()
    request.scene.ensure_defaults()
    request.scene.user_equipments[0].trajectory = TrajectorySpec(
        kind="linear",
        points=[(0.0, 0.0, 1.5)],
        samples=0,
        start_static_fraction=0.6,
        end_static_fraction=0.5,
        easing="cubic",
    )

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    message = str(exc_info.value)
    assert "samples must be at least 1" in message
    assert "easing must be one of" in message
    assert "static fractions must sum to less than 1" in message
    assert "at least two points for linear" in message


def test_validate_request_checks_dynamic_object_name_against_existing_scenario(tmp_path):
    scenario = tmp_path / "scene.xml"
    mesh = tmp_path / "car.ply"
    mesh.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    scenario.write_text(
        """
        <scene>
          <shape type="ply" id="CAR_obj" name="CAR_obj">
            <string name="filename" value="car.ply"/>
          </shape>
        </scene>
        """,
        encoding="utf-8",
    )
    request = SimulationRequest()
    request.scene.scenario_path = scenario
    request.scene.base_stations = [BaseStation(id="bs0")]
    request.scene.user_equipments = [UserEquipment(id="ue0")]
    request.scene.objects = [DynamicObject(id="car0", object_name="MISSING_obj")]

    with pytest.raises(ValueError) as exc_info:
        validate_request(request)

    assert "MISSING_obj" in str(exc_info.value)
    assert "CAR_obj" in str(exc_info.value)


def test_validate_request_accepts_dynamic_object_name_from_existing_scenario(tmp_path):
    scenario = tmp_path / "scene.xml"
    (tmp_path / "car.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    scenario.write_text(
        """
        <scene>
          <shape type="ply" id="CAR_obj" name="CAR_obj">
            <string name="filename" value="car.ply"/>
          </shape>
        </scene>
        """,
        encoding="utf-8",
    )
    request = SimulationRequest()
    request.scene.scenario_path = scenario
    request.scene.base_stations = [BaseStation(id="bs0")]
    request.scene.user_equipments = [UserEquipment(id="ue0")]
    request.scene.objects = [DynamicObject(id="car0", object_name="CAR_obj")]

    validate_request(request)
