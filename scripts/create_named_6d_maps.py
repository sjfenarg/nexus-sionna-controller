from __future__ import annotations

import json
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDOOR_XML = ROOT / "scenarios" / "Outdoor6D_w_car.xml"
TRANSFORM_PATH = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment" / "manual_surface_outdoor6d_to_scenebaker_transform.json"
DETAILED_DIR = ROOT / "scenarios" / "Detailed 6D Map"
DETAILED_XML = DETAILED_DIR / "Detailed 6D Map.xml"
SIMPLIFIED_SOURCE_XML = (
    ROOT
    / "scenarios"
    / "scenebaker_upv_vera_catastro_subset"
    / "upv_vera_catastro_subset_no_trees_no_lidar.xml"
)
SIMPLIFIED_XML = SIMPLIFIED_SOURCE_XML.parent / "Simplified 6D Map.xml"


def read_ply(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        vertex_count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        face_count = int(next(line.split()[-1] for line in header if line.startswith("element face ")))
        vertex_props = 0
        in_vertex = False
        for line in header:
            if line.startswith("element vertex "):
                in_vertex = True
                continue
            if line.startswith("element face "):
                in_vertex = False
                continue
            if in_vertex and line.startswith("property "):
                vertex_props += 1
        vertices = np.frombuffer(f.read(vertex_count * vertex_props * 4), dtype="<f4").reshape(vertex_count, vertex_props)[:, :3].copy()
        faces = []
        for _ in range(face_count):
            count = struct.unpack("<B", f.read(1))[0]
            faces.append(list(struct.unpack("<" + "i" * count, f.read(4 * count))))
    return vertices, faces


def write_ply(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
    with path.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(np.asarray(vertices, dtype="<f4").tobytes())
        for face in faces:
            f.write(struct.pack("<B", len(face)))
            f.write(struct.pack("<" + "i" * len(face), *face))


def transform_vertices(vertices: np.ndarray, transform: dict) -> np.ndarray:
    out = vertices.astype(np.float64).copy()
    center = np.array([transform["rotation_center_x"], transform["rotation_center_y"]], dtype=np.float64)
    xy = out[:, :2]
    xy = center + (xy - center) * float(transform["scale_xy"])
    angle = np.deg2rad(float(transform["rotation_deg"]))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)
    xy = (xy - center) @ rot.T + center
    xy += np.array([transform["dx_m"], transform["dy_m"]], dtype=np.float64)
    out[:, :2] = xy
    return out.astype(np.float32)


def create_detailed_map() -> None:
    transform = json.loads(TRANSFORM_PATH.read_text(encoding="utf-8"))
    meshes_dir = DETAILED_DIR / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(OUTDOOR_XML).getroot()
    for shape in list(root.findall("shape")):
        string = shape.find("string[@name='filename']")
        if string is None:
            continue
        source_name = Path(string.attrib["value"]).name
        if source_name.lower() == "car_obj.ply":
            root.remove(shape)
            continue
        source_path = OUTDOOR_XML.parent / string.attrib["value"]
        vertices, faces = read_ply(source_path)
        out_name = source_name
        write_ply(meshes_dir / out_name, transform_vertices(vertices, transform), faces)
        string.set("value", f"meshes/{out_name}")

    for bsdf in list(root.findall("bsdf")):
        if bsdf.attrib.get("id") == "mat-CAR":
            root.remove(bsdf)

    ET.indent(root, space="    ")
    ET.ElementTree(root).write(DETAILED_XML, encoding="utf-8", xml_declaration=True)


def create_simplified_map() -> None:
    shutil.copy2(SIMPLIFIED_SOURCE_XML, SIMPLIFIED_XML)


def main() -> None:
    create_detailed_map()
    create_simplified_map()
    print(DETAILED_XML)
    print(SIMPLIFIED_XML)


if __name__ == "__main__":
    main()
