"""Classify radiomap positions covered by structural scene surfaces."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import struct
from xml.etree import ElementTree

import numpy as np

from .model import RadiomapConfig
from .trajectories import radiomap_grid_shape, sample_radiomap_grid


_PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}
_SIX_D_BUILDING_MATERIALS = {"bricks_obj", "glass_obj", "metal_obj", "plastic_obj", "concrete_obj"}
_HORIZONTAL_NORMAL_THRESHOLD = 0.7


def radiomap_inside_building_mask(scenario_path: Path, config: RadiomapConfig) -> np.ndarray:
    """Return a grid-order mask using the overhead-surface rule shown in the preview.

    A point is marked when a near-horizontal structural triangle lies above its
    configured radiomap height. Ground, vegetation, roads and dynamic objects do
    not contribute. The mask follows the same order as ``sample_radiomap_grid``.
    """
    positions = sample_radiomap_grid(config)
    mask = np.zeros(len(positions), dtype=bool)
    scenario_path = Path(scenario_path)
    if not scenario_path.is_file():
        return mask

    x_points, y_points = radiomap_grid_shape(config)
    angle = np.deg2rad(config.rotation_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    center = np.asarray([(config.x_min + config.x_max) / 2, (config.y_min + config.y_max) / 2])
    # Transform world-space triangles back to the radiomap's local grid axes.
    inverse_rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    local_positions = center + (positions[:, :2] - center) @ inverse_rotation
    xs = local_positions[:x_points, 0]
    ys = local_positions[::x_points, 1]
    grid = mask.reshape(y_points, x_points)

    for mesh_path in _building_mesh_paths(scenario_path):
        vertices, faces = _read_ply_mesh(str(mesh_path.resolve()), mesh_path.stat().st_mtime_ns)
        if not len(faces):
            continue
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_lengths = np.linalg.norm(normals, axis=1)
        overhead = (
            (np.abs(normals[:, 2]) / np.maximum(normal_lengths, 1e-12) > _HORIZONTAL_NORMAL_THRESHOLD)
            & (triangles[:, :, 2].max(axis=1) > config.height)
        )
        for world_triangle in triangles[overhead]:
            triangle = world_triangle.copy()
            triangle[:, :2] = center + (world_triangle[:, :2] - center) @ inverse_rotation
            _mark_covered_grid_points(grid, xs, ys, triangle, config.height)
    return mask


def _building_mesh_paths(scenario_path: Path) -> tuple[Path, ...]:
    root = ElementTree.parse(scenario_path).getroot()
    paths = []
    for shape in root.findall(".//shape"):
        if shape.get("type") != "ply":
            continue
        filename = shape.find("./string[@name='filename']")
        if filename is None or not filename.get("value"):
            continue
        name = (shape.get("name") or shape.get("id") or "").lower()
        stem = Path(filename.get("value")).stem.lower()
        structural_name = any(word in name or word in stem for word in ("building", "roof", "wall"))
        structural_name |= name in _SIX_D_BUILDING_MATERIALS or stem in _SIX_D_BUILDING_MATERIALS
        if not structural_name:
            continue
        if shape.find("./transform[@name='to_world']") is not None:
            raise ValueError(f"Building occupancy does not support transformed PLY shapes: {name}")
        path = Path(filename.get("value"))
        if not path.is_absolute():
            path = scenario_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Building mesh not found: {path}")
        paths.append(path)
    return tuple(paths)


def _mark_covered_grid_points(
    grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, triangle: np.ndarray, height: float,
) -> None:
    x_indices = np.flatnonzero((xs >= triangle[:, 0].min()) & (xs <= triangle[:, 0].max()))
    y_indices = np.flatnonzero((ys >= triangle[:, 1].min()) & (ys <= triangle[:, 1].max()))
    if not len(x_indices) or not len(y_indices):
        return
    a, b, c = triangle
    u, v = b[:2] - a[:2], c[:2] - a[:2]
    determinant = u[0] * v[1] - u[1] * v[0]
    if abs(determinant) < 1e-12:
        return
    dx = xs[x_indices][None, :] - a[0]
    dy = ys[y_indices][:, None] - a[1]
    first = (dx * v[1] - dy * v[0]) / determinant
    second = (u[0] * dy - u[1] * dx) / determinant
    roof_height = a[2] + first * (b[2] - a[2]) + second * (c[2] - a[2])
    grid[np.ix_(y_indices, x_indices)] |= (
        (first >= -1e-7) & (second >= -1e-7) & (first + second <= 1 + 1e-7)
        & (roof_height > height)
    )


@lru_cache(maxsize=12)
def _read_ply_mesh(path_text: str, _mtime_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Read PLY positions and triangle faces without loading GUI dependencies."""
    with Path(path_text).open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path_text}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        fmt = next((line.split()[1] for line in header if line.startswith("format ")), None)
        vertex_count = face_count = 0
        properties = []
        count_type, index_type = "uchar", "int"
        element = ""
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    vertex_count = int(parts[2])
                elif element == "face":
                    face_count = int(parts[2])
            elif len(parts) >= 3 and parts[0] == "property" and element == "vertex":
                properties.append((parts[2], parts[1]))
            elif len(parts) >= 5 and parts[:2] == ["property", "list"] and element == "face":
                count_type, index_type = parts[2:4]
        xyz_indices = [next(i for i, (name, _) in enumerate(properties) if name == axis) for axis in "xyz"]
        if fmt == "ascii":
            vertices = np.asarray([
                [float(values[i]) for i in xyz_indices]
                for values in (stream.readline().split() for _ in range(vertex_count))
            ], dtype=np.float64)
            polygons = []
            for _ in range(face_count):
                values = [int(value) for value in stream.readline().split()]
                polygons.append(values[1:1 + values[0]])
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            vertex_format = struct.Struct(endian + "".join(_PLY_TYPES[typ] for _, typ in properties))
            vertices = np.empty((vertex_count, 3), dtype=np.float64)
            for i in range(vertex_count):
                values = vertex_format.unpack(stream.read(vertex_format.size))
                vertices[i] = [values[index] for index in xyz_indices]
            count_format = struct.Struct(endian + _PLY_TYPES[count_type])
            index_format = _PLY_TYPES[index_type]
            index_size = struct.calcsize(endian + index_format)
            polygons = []
            for _ in range(face_count):
                count = count_format.unpack(stream.read(count_format.size))[0]
                polygons.append(struct.unpack(endian + str(count) + index_format, stream.read(count * index_size)))
        else:
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path_text}")
    faces = [(face[0], face[i], face[i + 1]) for face in polygons for i in range(1, len(face) - 1)]
    return vertices, np.asarray(faces, dtype=np.int32).reshape(-1, 3)
