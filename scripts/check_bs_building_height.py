from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_XML = ROOT / "scenarios" / "Outdoor6D_w_car.xml"
TRANSFORM = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment" / "manual_surface_outdoor6d_to_scenebaker_transform.json"
H5_PATH = ROOT / "meas" / "nexus_trials.h5"
OUT_CSV = ROOT / "scenarios" / "outdoor6d_scenebaker_alignment" / "bs0_nearby_height_summary.csv"

OSM_BOUNDS = {
    "minlat": 39.4774000,
    "minlon": -0.3442000,
    "maxlat": 39.4842000,
    "maxlon": -0.3348000,
}


def latlon_to_scene_xy(lat: float, lon: float) -> np.ndarray:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    lon0 = (OSM_BOUNDS["minlon"] + OSM_BOUNDS["maxlon"]) / 2.0
    meters_per_lat = 111_320.0
    meters_per_lon = meters_per_lat * np.cos(np.deg2rad(lat0))
    return np.array([(lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat], dtype=np.float64)


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


def transform_xy(xy: np.ndarray, transform: dict) -> np.ndarray:
    center = np.array([transform["rotation_center_x"], transform["rotation_center_y"]], dtype=np.float64)
    scaled = center + (xy - center) * float(transform["scale_xy"])
    angle = np.deg2rad(float(transform["rotation_deg"]))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)
    return (scaled - center) @ rot.T + center + np.array([transform["dx_m"], transform["dy_m"]], dtype=np.float64)


def scenario_meshes() -> list[tuple[str, Path]]:
    root = ET.parse(SCENARIO_XML).getroot()
    meshes = []
    for shape in root.findall("shape"):
        string = shape.find("string[@name='filename']")
        if string is None:
            continue
        rel = string.attrib["value"]
        meshes.append((Path(rel).name, SCENARIO_XML.parent / rel))
    return meshes


def main() -> None:
    transform = json.loads(TRANSFORM.read_text(encoding="utf-8"))
    with h5py.File(H5_PATH, "r") as f:
        bs_lat, bs_lon = f["scenarios/nexus_trials/s001/parameters/antenna_params/bs0/ant_coord"][...]
    bs_xy = latlon_to_scene_xy(float(bs_lat), float(bs_lon))

    rows = []
    for name, path in scenario_meshes():
        vertices, faces = read_ply(path)
        local_z_scaled = vertices[:, 2].astype(np.float64) * float(transform["scale_xy"])
        distances = []
        z_values = []
        for face in faces:
            face_idx = np.asarray(face, dtype=np.int64)
            xy = transform_xy(vertices[face_idx, :2].astype(np.float64), transform)
            d = float(np.linalg.norm(xy.mean(axis=0) - bs_xy))
            if d <= 8.0:
                distances.append(d)
                z_values.extend(local_z_scaled[face_idx].tolist())
        if z_values:
            arr = np.asarray(z_values)
            rows.append(
                {
                    "mesh": name,
                    "faces_within_8m": len(distances),
                    "nearest_face_center_m": min(distances),
                    "z_min_m": float(arr.min()),
                    "z_median_m": float(np.median(arr)),
                    "z_p95_m": float(np.percentile(arr, 95)),
                    "z_max_m": float(arr.max()),
                }
            )

    rows.sort(key=lambda row: row["nearest_face_center_m"])
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write("mesh,faces_within_8m,nearest_face_center_m,z_min_m,z_median_m,z_p95_m,z_max_m\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in row.keys()) + "\n")

    print(f"bs0 XY: x={bs_xy[0]:.3f}, y={bs_xy[1]:.3f}")
    print(f"Saved {OUT_CSV}")
    for row in rows:
        print(
            f"{row['mesh']}: nearest={row['nearest_face_center_m']:.2f} m, "
            f"z median={row['z_median_m']:.2f} m, p95={row['z_p95_m']:.2f} m, max={row['z_max_m']:.2f} m"
        )


if __name__ == "__main__":
    main()
