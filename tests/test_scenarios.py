import struct

import numpy as np

from isac_6d_sampler.core.scenarios import discover_scenarios, load_scenario_asset, read_ply_bounds


def test_discover_scenarios_extracts_named_ply_shapes(tmp_path):
    scenarios = tmp_path / "scenarios"
    mesh_dir = scenarios / "meshes"
    mesh_dir.mkdir(parents=True)
    _write_ascii_ply(mesh_dir / "CAR_obj.ply", [(0.0, 0.0, 0.0), (2.0, 1.0, 0.5)])
    _write_ascii_ply(mesh_dir / "WALL_obj.ply", [(-1.0, -3.0, 0.0), (0.0, 4.0, 2.0)])
    (scenarios / "scene.xml").write_text(
        """
        <scene>
          <shape type="ply" id="CAR_obj" name="CAR_obj">
            <string name="filename" value="meshes/CAR_obj.ply"/>
          </shape>
          <shape type="ply" id="WALL_obj">
            <string name="filename" value="meshes/WALL_obj.ply"/>
          </shape>
          <shape type="sphere" id="ignored"/>
        </scene>
        """,
        encoding="utf-8",
    )

    assets = discover_scenarios(scenarios)

    assert len(assets) == 1
    assert assets[0].name == "scene"
    assert assets[0].mesh_count == 2
    assert assets[0].object_names == ("CAR_obj", "WALL_obj")
    assert assets[0].meshes[0].path == mesh_dir / "CAR_obj.ply"
    assert assets[0].meshes[0].bounds.xy_min == (0.0, 0.0)
    assert assets[0].meshes[0].bounds.xy_max == (2.0, 1.0)
    assert assets[0].bounds.xy_min == (-1.0, -3.0)
    assert assets[0].bounds.xy_max == (2.0, 4.0)

    direct = load_scenario_asset(scenarios / "scene.xml")
    assert direct.object_names == ("CAR_obj", "WALL_obj")


def test_read_ply_bounds_supports_binary_little_endian(tmp_path):
    ply_path = tmp_path / "mesh.ply"
    vertices = [(-2.0, 1.0, 0.5), (3.0, 4.0, 2.5), (0.0, -1.0, -0.5)]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with ply_path.open("wb") as fh:
        fh.write(header)
        for vertex in vertices:
            fh.write(struct.pack("<fff", *vertex))

    bounds = read_ply_bounds(ply_path)

    np.testing.assert_allclose(bounds.min_xyz, [-2.0, -1.0, -0.5])
    np.testing.assert_allclose(bounds.max_xyz, [3.0, 4.0, 2.5])


def _write_ascii_ply(path, vertices):
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
