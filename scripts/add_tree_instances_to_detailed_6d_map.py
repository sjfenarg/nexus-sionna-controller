from __future__ import annotations

import json
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_JSON = ROOT / "scenarios" / "tree_placement" / "tree_v10_final_to_detailed_6d_map_transform.json"
SCENARIO_XML = ROOT / "scenarios" / "Detailed 6D Map" / "Detailed 6D Map.xml"
BACKUP_XML = ROOT / "scenarios" / "Detailed 6D Map" / "Detailed 6D Map.before_tree_instances.xml"
MESH_DIR = ROOT / "scenarios" / "Detailed 6D Map" / "meshes"
TREE_OBJ = ROOT / "docs" / "Tree_V10_Final.obj"
TREE_COUNT = 4
OFFSET_X_M = 2.0
OFFSET_Y_M = 1.5


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idxs = []
                for part in line.split()[1:]:
                    token = part.split("/")[0]
                    if not token:
                        continue
                    idx = int(token)
                    idxs.append(idx - 1 if idx > 0 else len(vertices) + idx)
                if len(idxs) == 3:
                    faces.append(idxs)
                elif len(idxs) > 3:
                    first = idxs[0]
                    for i in range(1, len(idxs) - 1):
                        faces.append([first, idxs[i], idxs[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def obj_to_local_scene_xyz(vertices: np.ndarray) -> np.ndarray:
    xyz = np.column_stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]]).astype(np.float32)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    base_center = np.array([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2]], dtype=np.float32)
    return xyz - base_center


def transform_vertices(local_xyz: np.ndarray, dx: float, dy: float, dz: float, yaw_deg: float, scale: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c = np.float32(np.cos(angle))
    s = np.float32(np.sin(angle))
    out = np.empty_like(local_xyz, dtype=np.float32)
    sx = local_xyz[:, 0] * np.float32(scale)
    sy = local_xyz[:, 1] * np.float32(scale)
    out[:, 0] = c * sx - s * sy + np.float32(dx)
    out[:, 1] = s * sx + c * sy + np.float32(dy)
    out[:, 2] = local_xyz[:, 2] * np.float32(scale) + np.float32(dz)
    return out


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
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
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(np.asarray(vertices, dtype="<f4").tobytes(order="C"))
        prefix = np.full((len(faces), 1), 3, dtype=np.uint8)
        # Write face records as uchar count + three int32 indices.
        for face in np.asarray(faces, dtype="<i4"):
            f.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))


def ensure_tree_material(root: ET.Element) -> None:
    if root.find("bsdf[@id='mat-TREE']") is not None:
        return
    mat = ET.Element("bsdf", {"type": "radio-material", "id": "mat-TREE"})
    ET.SubElement(mat, "float", {"name": "relative_permittivity", "value": "1.5"})
    ET.SubElement(mat, "float", {"name": "conductivity", "value": "0.01"})
    ET.SubElement(mat, "float", {"name": "thickness", "value": "0.5"})
    ET.SubElement(mat, "float", {"name": "scattering_coefficient", "value": "0.2"})
    ET.SubElement(mat, "float", {"name": "xpd_coefficient", "value": "0.1"})
    ET.SubElement(mat, "rgb", {"name": "color", "value": "0.007000 0.558000 0.005000"})
    insert_at = 0
    for i, child in enumerate(list(root)):
        if child.tag in {"integrator", "bsdf"}:
            insert_at = i + 1
    root.insert(insert_at, mat)


def patch_xml(mesh_names: list[str]) -> None:
    if not BACKUP_XML.exists():
        shutil.copy2(SCENARIO_XML, BACKUP_XML)
    tree = ET.parse(SCENARIO_XML)
    root = tree.getroot()
    ensure_tree_material(root)
    for shape in list(root.findall("shape")):
        sid = shape.attrib.get("id", "")
        if sid.startswith("TREE_V10_Final_"):
            root.remove(shape)
    for i, mesh_name in enumerate(mesh_names):
        shape = ET.Element("shape", {"type": "ply", "id": f"TREE_V10_Final_{i:02d}", "name": f"TREE_V10_Final_{i:02d}"})
        ET.SubElement(shape, "string", {"name": "filename", "value": f"meshes/{mesh_name}"})
        ET.SubElement(shape, "boolean", {"name": "face_normals", "value": "false"})
        ET.SubElement(shape, "ref", {"id": "mat-TREE", "name": "bsdf"})
        root.append(shape)
    ET.indent(tree, space="    ")
    tree.write(SCENARIO_XML, encoding="utf-8", xml_declaration=True)


def main() -> None:
    transform = json.loads(TRANSFORM_JSON.read_text(encoding="utf-8"))
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {TREE_OBJ}")
    vertices, faces = load_obj(TREE_OBJ)
    print(f"Loaded {len(vertices)} vertices, {len(faces)} triangular faces")
    local_xyz = obj_to_local_scene_xyz(vertices)
    mesh_names = []
    placements = []
    for i in range(TREE_COUNT):
        dx = float(transform["dx_m"]) + i * OFFSET_X_M
        dy = float(transform["dy_m"]) + i * OFFSET_Y_M
        dz = float(transform["dz_m"])
        yaw = float(transform["yaw_deg"])
        scale = float(transform["scale"])
        out_vertices = transform_vertices(local_xyz, dx, dy, dz, yaw, scale)
        mesh_name = f"TREE_V10_Final_{i:02d}.ply"
        mesh_path = MESH_DIR / mesh_name
        print(f"Writing {mesh_path}")
        write_binary_ply(mesh_path, out_vertices, faces)
        mesh_names.append(mesh_name)
        placements.append({"id": f"TREE_V10_Final_{i:02d}", "x_m": dx, "y_m": dy, "z_m": dz, "yaw_deg": yaw, "scale": scale, "height_m": scale * 19.753479000000002})
    patch_xml(mesh_names)
    placement_path = ROOT / "scenarios" / "tree_placement" / "tree_v10_final_instances_in_detailed_6d_map.json"
    placement_path.write_text(json.dumps({"scenario": str(SCENARIO_XML.relative_to(ROOT)), "instances": placements}, indent=2), encoding="utf-8")
    print(f"Patched {SCENARIO_XML}")
    print(f"Saved {placement_path}")


if __name__ == "__main__":
    main()
