import pytest
import sys
import types

from isac_6d_sampler.core.antenna_patterns import (
    ISAC_HORN_PATTERN,
    antenna_pattern_spec,
    available_pattern_names,
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
