import pytest

from isac_6d_sampler.core.frequencies import format_band_specs, parse_band_spec, parse_band_specs


def test_parse_band_spec_uses_ghz_units():
    band = parse_band_spec("77:81:1024:e-band")

    assert band.name == "e-band"
    assert band.start_hz == 77e9
    assert band.stop_hz == 81e9
    assert band.points == 1024


def test_parse_band_specs_accepts_semicolon_and_comma_separators():
    bands = parse_band_specs("77:78:4:low; 80:81:5:high, 90:91:6")

    assert [band.points for band in bands] == [4, 5, 6]
    assert bands[-1].name == "90-91GHz"


def test_format_band_specs_round_trips():
    bands = parse_band_specs("77:78:4:low;80:81:5:high")

    assert [b.to_dict() if hasattr(b, "to_dict") else b.name for b in parse_band_specs(format_band_specs(bands))]
    assert [band.name for band in parse_band_specs(format_band_specs(bands))] == ["low", "high"]


def test_parse_band_spec_rejects_invalid_order():
    with pytest.raises(ValueError, match="stop frequency"):
        parse_band_spec("81:77:10")
