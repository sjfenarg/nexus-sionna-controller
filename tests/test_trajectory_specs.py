import pytest

from isac_6d_sampler.core.trajectory_specs import (
    anchor_trajectory,
    build_trajectory_spec,
    format_point_list,
    parse_point_list,
    translate_trajectory,
)


def test_parse_point_list_accepts_semicolon_triples():
    assert parse_point_list("0,0,1.5; 2,3,1.5") == [(0.0, 0.0, 1.5), (2.0, 3.0, 1.5)]


def test_format_point_list_is_gui_friendly():
    assert format_point_list([(0.0, 0.0, 1.5), (2.0, 3.0, 1.5)]) == "0,0,1.5; 2,3,1.5"


def test_build_linear_trajectory_uses_first_two_points():
    spec = build_trajectory_spec("linear", "0,0,0; 1,0,0; 2,0,0", samples=5)

    assert spec.kind == "linear"
    assert spec.points == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert spec.samples == 5


def test_build_polyline_requires_two_points():
    with pytest.raises(ValueError, match="Polyline"):
        build_trajectory_spec("polyline", "0,0,0", samples=5)


def test_translate_trajectory_moves_static_control_point():
    spec = build_trajectory_spec("static", "0,0,1.5", samples=1)

    moved = translate_trajectory(spec, (2.0, -1.0, 0.0))

    assert moved.kind == "static"
    assert moved.points == [(2.0, -1.0, 1.5)]
    assert moved.samples == 1


def test_translate_trajectory_preserves_motion_metadata():
    spec = build_trajectory_spec(
        "polyline",
        "0,0,1; 1,0,1; 1,1,1",
        samples=7,
        easing="smoothstep",
        start_static_fraction=0.1,
        end_static_fraction=0.2,
    )

    moved = translate_trajectory(spec, (10.0, 20.0, 1.0))

    assert moved.kind == "polyline"
    assert moved.points == [(10.0, 20.0, 2.0), (11.0, 20.0, 2.0), (11.0, 21.0, 2.0)]
    assert moved.samples == 7
    assert moved.easing == "smoothstep"
    assert moved.start_static_fraction == 0.1
    assert moved.end_static_fraction == 0.2


def test_anchor_trajectory_moves_first_point_to_anchor():
    spec = build_trajectory_spec("linear", "0,0,1.5; 5,0,1.5", samples=4)

    anchored = anchor_trajectory(spec, (10.0, -2.0, 3.0))

    assert anchored.points == [(10.0, -2.0, 3.0), (15.0, -2.0, 3.0)]
    assert anchored.samples == 4
