from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from xml.etree import ElementTree

import numpy as np


Vector2 = tuple[float, float]


@dataclass(frozen=True, slots=True)
class MeshBounds:
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @property
    def xy_min(self) -> Vector2:
        return (self.min_xyz[0], self.min_xyz[1])

    @property
    def xy_max(self) -> Vector2:
        return (self.max_xyz[0], self.max_xyz[1])

    @property
    def width(self) -> float:
        return self.max_xyz[0] - self.min_xyz[0]

    @property
    def depth(self) -> float:
        return self.max_xyz[1] - self.min_xyz[1]


@dataclass(frozen=True, slots=True)
class MeshAsset:
    name: str
    path: Path
    bounds: MeshBounds | None = None


@dataclass(frozen=True, slots=True)
class ScenarioAsset:
    name: str
    path: Path
    meshes: tuple[MeshAsset, ...]
    object_meshes: tuple[MeshAsset, ...] = ()

    @property
    def mesh_count(self) -> int:
        return len(self.meshes)

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(mesh.name for mesh in self.meshes)

    @property
    def bounds(self) -> MeshBounds | None:
        bounded = [mesh.bounds for mesh in self.meshes if mesh.bounds is not None]
        if not bounded:
            return None
        mins = np.min(np.asarray([bounds.min_xyz for bounds in bounded], dtype=np.float64), axis=0)
        maxs = np.max(np.asarray([bounds.max_xyz for bounds in bounded], dtype=np.float64), axis=0)
        return MeshBounds(min_xyz=_float3(mins), max_xyz=_float3(maxs))


def discover_scenarios(root: Path) -> list[ScenarioAsset]:
    """Find Sionna/Mitsuba XML scene files below a scenario folder."""
    root = Path(root)
    assets: list[ScenarioAsset] = []
    for xml_path in sorted(root.rglob("*.xml")):
        assets.append(
            ScenarioAsset(
                name=xml_path.stem,
                path=xml_path,
                meshes=_parse_mesh_assets(xml_path),
                object_meshes=discover_object_meshes(root / "objects"),
            )
        )
    return assets


def load_scenario_asset(xml_path: Path) -> ScenarioAsset:
    xml_path = Path(xml_path)
    return ScenarioAsset(
        name=xml_path.stem,
        path=xml_path,
        meshes=_parse_mesh_assets(xml_path),
        object_meshes=discover_object_meshes(xml_path.parent / "objects"),
    )


def discover_object_meshes(root: Path) -> tuple[MeshAsset, ...]:
    root = Path(root)
    if not root.exists():
        return ()
    return tuple(
        MeshAsset(name=path.stem, path=path, bounds=read_obj_bounds(path))
        for path in sorted(root.glob("*.obj"))
    )


def _parse_mesh_assets(xml_path: Path) -> tuple[MeshAsset, ...]:
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError:
        return ()

    meshes: list[MeshAsset] = []
    for shape in root.findall(".//shape"):
        if shape.attrib.get("type") != "ply":
            continue
        name = shape.attrib.get("name") or shape.attrib.get("id")
        filename_element = shape.find("./string[@name='filename']")
        if not name or filename_element is None:
            continue
        filename = filename_element.attrib.get("value")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = xml_path.parent / mesh_path
        meshes.append(MeshAsset(name=name, path=mesh_path, bounds=read_ply_bounds(mesh_path)))
    return tuple(meshes)


def read_ply_bounds(path: Path) -> MeshBounds | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            header_bytes, vertex_count, fmt, properties = _read_ply_header(fh)
            if vertex_count <= 0:
                return None
            if fmt == "ascii":
                return _read_ascii_vertex_bounds(fh, vertex_count, properties)
            if fmt in {"binary_little_endian", "binary_big_endian"}:
                return _read_binary_vertex_bounds(fh, vertex_count, properties, fmt)
            return None
    except (OSError, UnicodeDecodeError, ValueError, struct.error):
        return None


def read_obj_bounds(path: Path) -> MeshBounds | None:
    path = Path(path)
    if not path.exists():
        return None
    points = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except (OSError, ValueError):
        return None
    if not points:
        return None
    return _bounds_from_points(np.asarray(points, dtype=np.float64))


def _read_ply_header(fh) -> tuple[bytes, int, str, list[tuple[str, str]]]:
    header_lines: list[str] = []
    header_bytes = bytearray()
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PLY header missing end_header")
        header_bytes.extend(line)
        decoded = line.decode("ascii").strip()
        header_lines.append(decoded)
        if decoded == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Not a PLY file")

    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property" and len(parts) >= 3:
            if parts[1] == "list":
                raise ValueError("List vertex properties are not supported")
            properties.append((parts[2], parts[1]))
    return bytes(header_bytes), vertex_count, fmt, properties


def _read_ascii_vertex_bounds(fh, vertex_count: int, properties: list[tuple[str, str]]) -> MeshBounds | None:
    indices = _xyz_property_indices(properties)
    points = []
    for _ in range(vertex_count):
        line = fh.readline().decode("ascii")
        values = line.split()
        points.append([float(values[index]) for index in indices])
    return _bounds_from_points(np.asarray(points, dtype=np.float64))


def _read_binary_vertex_bounds(
    fh,
    vertex_count: int,
    properties: list[tuple[str, str]],
    fmt: str,
) -> MeshBounds | None:
    endian = "<" if fmt == "binary_little_endian" else ">"
    property_formats = [_struct_format(prop_type) for _, prop_type in properties]
    row_format = endian + "".join(property_formats)
    row_size = struct.calcsize(row_format)
    indices = _xyz_property_indices(properties)
    points = np.empty((vertex_count, 3), dtype=np.float64)
    for row_idx in range(vertex_count):
        row = fh.read(row_size)
        if len(row) != row_size:
            raise ValueError("Unexpected end of binary PLY vertices")
        values = struct.unpack(row_format, row)
        points[row_idx] = [values[index] for index in indices]
    return _bounds_from_points(points)


def _xyz_property_indices(properties: list[tuple[str, str]]) -> tuple[int, int, int]:
    names = [name for name, _ in properties]
    try:
        return (names.index("x"), names.index("y"), names.index("z"))
    except ValueError as exc:
        raise ValueError("PLY vertex properties must include x, y and z") from exc


def _struct_format(ply_type: str) -> str:
    mapping = {
        "char": "b",
        "uchar": "B",
        "int8": "b",
        "uint8": "B",
        "short": "h",
        "ushort": "H",
        "int16": "h",
        "uint16": "H",
        "int": "i",
        "uint": "I",
        "int32": "i",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    try:
        return mapping[ply_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY scalar property type: {ply_type}") from exc


def _bounds_from_points(points: np.ndarray) -> MeshBounds | None:
    if points.size == 0:
        return None
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return MeshBounds(min_xyz=_float3(mins), max_xyz=_float3(maxs))


def _float3(values) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))
