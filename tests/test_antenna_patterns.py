import pytest
import numpy as np
import sys
import types

from isac_6d_sampler.core.antenna_patterns import (
    ISAC_HORN_PATTERN,
    QOM_OMNI_FREQUENCY_LABELS_HZ,
    QOM_OMNI_PATTERN,
    POWERLOG_BACK_LOBE_BOOST_DB,
    POWERLOG_BACK_LOBE_WIDTH_EXPONENT,
    _apply_powerlog_back_lobe_boost_np,
    antenna_pattern_spec,
    available_pattern_names,
    measured_frequency_pattern_name,
    qom_omni_normalized_gain_db_from_local_dirs,
    register_sionna_antenna_patterns,
)
from isac_6d_sampler.core.config_io import request_from_dict


def test_available_patterns_include_journal_horn():
    assert ISAC_HORN_PATTERN in available_pattern_names()

    spec = antenna_pattern_spec(ISAC_HORN_PATTERN)

    assert spec.source == "isac_journal"
    assert spec.max_gain_db == 10.3
    assert spec.h_3db_beamwidth_deg == 56.0
    assert spec.v_3db_beamwidth_deg == 28.0


def test_config_rejects_unknown_antenna_pattern():
    with pytest.raises(ValueError, match="Unsupported antenna pattern"):
        request_from_dict(
            {
                "scene": {
                    "base_stations": [
                        {
                            "id": "bs0",
                            "panel": {"pattern": "not_a_pattern"},
                        }
                    ],
                }
            }
        )


def test_config_uses_element_diagram_as_legacy_pattern_name():
    request = request_from_dict(
        {
            "scene": {
                "base_stations": [
                    {
                        "id": "bs0",
                        "panel": {"element_diagram": ISAC_HORN_PATTERN},
                    }
                ],
            }
        }
    )

    panel = request.scene.base_stations[0].panel
    assert panel.pattern == ISAC_HORN_PATTERN
    assert panel.element_diagram == ISAC_HORN_PATTERN


def test_powerlog_back_lobe_boost_only_lifts_rear_hemisphere():
    gains = np.asarray([-20.0, -20.0, -20.0, -20.0])
    shoulder_angle = 3.0 * np.pi / 4.0
    angles = np.asarray([0.0, np.pi / 2.0, shoulder_angle, np.pi])

    boosted = _apply_powerlog_back_lobe_boost_np(gains, angles)

    assert boosted[0] == gains[0]
    assert np.isclose(boosted[1], gains[1])
    expected_shoulder = gains[2] + POWERLOG_BACK_LOBE_BOOST_DB * ((-np.cos(shoulder_angle)) ** POWERLOG_BACK_LOBE_WIDTH_EXPONENT)
    assert np.isclose(boosted[2], expected_shoulder)
    assert np.isclose(boosted[3], min(gains[3] + POWERLOG_BACK_LOBE_BOOST_DB, 0.0))


def test_qom_omni_main_lobe_is_horizontal_not_vertical():
    directions = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    gain = qom_omni_normalized_gain_db_from_local_dirs(directions, "18GHz")

    assert gain[0] > gain[1]
    assert gain[0] > gain[2]


def test_register_sionna_antenna_patterns_registers_horn_element_pattern(monkeypatch):
    calls = []
    fake_rt = types.ModuleType("sionna.rt")
    fake_rt.PolarizedAntennaPattern = object
    fake_rt.register_antenna_pattern = lambda name, factory: calls.append((name, factory))
    fake_sionna = types.ModuleType("sionna")
    fake_sionna.rt = fake_rt
    monkeypatch.setitem(sys.modules, "sionna", fake_sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", fake_rt)

    register_sionna_antenna_patterns()

    assert calls[0][0] == ISAC_HORN_PATTERN
    assert callable(calls[0][1])


def test_register_sionna_antenna_patterns_registers_qom_omni_frequencies(monkeypatch):
    calls = []
    fake_rt = types.ModuleType("sionna.rt")
    fake_rt.PolarizedAntennaPattern = object
    fake_rt.register_antenna_pattern = lambda name, factory: calls.append((name, factory))
    fake_sionna = types.ModuleType("sionna")
    fake_sionna.rt = fake_rt
    monkeypatch.setitem(sys.modules, "sionna", fake_sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", fake_rt)

    register_sionna_antenna_patterns()

    names = {name for name, _ in calls}
    assert QOM_OMNI_PATTERN in names
    assert {
        measured_frequency_pattern_name(QOM_OMNI_PATTERN, label)
        for label in QOM_OMNI_FREQUENCY_LABELS_HZ
    }.issubset(names)
