from __future__ import annotations

import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCENARIO = ROOT / "scenarios" / "scenebaker_upv_vera_lidar_trees"
OUTPUT_SCENARIO = ROOT / "scenarios" / "scenebaker_upv_vera_catastro_subset"

OSM_BOUNDS = {
    "minlat": 39.4774000,
    "minlon": -0.3442000,
    "maxlat": 39.4842000,
    "maxlon": -0.3348000,
}

REQUESTED_POLYGON_LATLON = [
    (39.479673, -0.339566),  # top left
    (39.479160, -0.337873),  # top right
    (39.478200, -0.338270),  # bottom right
    (39.478740, -0.339993),  # bottom left
]

MESH_SPECS = [
    ("map_osm_areas_park.ply", "mat-park", False),
    ("map_osm_areas_pedestrian.ply", "mat-road", False),
    ("map_osm_paths_cycleway.ply", "mat-road", False),
    ("map_osm_paths_footway.ply", "mat-road", False),
    ("map_osm_paths_steps.ply", "mat-road", False),
    ("map_osm_roads_pedestrian.ply", "mat-road", False),
    ("map_osm_roads_primary.ply", "mat-road", False),
    ("map_osm_roads_residential.ply", "mat-road", False),
    ("map_osm_roads_secondary.ply", "mat-road", False),
    ("map_osm_roads_service.ply", "mat-road", False),
    ("map_osm_roads_tertiary.ply", "mat-road", False),
    ("map_osm_roads_track.ply", "mat-road", False),
    ("map_osm_roads_unclassified.ply", "mat-road", False),
    ("catastro_buildingparts-wall.ply", "mat-building_wall", True),
    ("catastro_buildingparts-roof.ply", "mat-building_roof", True),
    ("osm_fallback_buildings-wall.ply", "mat-building_wall", True),
    ("osm_fallback_buildings-roof.ply", "mat-building_roof", True),
]


def latlon_to_scene_xy(lat: float, lon: float) -> tuple[float, float]:
    lat0 = (OSM_BOUNDS["minlat"] + OSM_BOUNDS["maxlat"]) / 2.0
    lon0 = (OSM_BOUNDS["minlon"] + OSM_BOUNDS["maxlon"]) / 2.0
    meters_per_lat = 111_320.0
    meters_per_lon = meters_per_lat * math.cos(math.radians(lat0))
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


def points_in_poly(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        inside ^= crosses
        j = i
    return inside


def read_ply(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"{path} is not binary_little_endian PLY")
        vertex_count = int(next(line.split()[-1] for line in header_lines if line.startswith("element vertex ")))
        face_count = int(next(line.split()[-1] for line in header_lines if line.startswith("element face ")))

        vertices = np.frombuffer(f.read(vertex_count * 12), dtype="<f4").reshape(vertex_count, 3).copy()
        faces: list[list[int]] = []
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


def crop_mesh(vertices: np.ndarray, faces: list[list[int]], polygon_xy: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    if not faces:
        return vertices[:0], []
    centroids = np.array([vertices[face, :2].mean(axis=0) for face in faces])
    keep = points_in_poly(centroids, polygon_xy)
    kept_faces = [face for face, should_keep in zip(faces, keep) if should_keep]
    if not kept_faces:
        return vertices[:0], []
    used = sorted({index for face in kept_faces for index in face})
    remap = {old: new for new, old in enumerate(used)}
    return vertices[used], [[remap[index] for index in face] for face in kept_faces]


def make_ground(path: Path, polygon_xy: np.ndarray) -> None:
    margin = 12.0
    min_xy = polygon_xy.min(axis=0) - margin
    max_xy = polygon_xy.max(axis=0) + margin
    vertices = np.array(
        [
            [min_xy[0], min_xy[1], -0.8],
            [max_xy[0], min_xy[1], -0.8],
            [max_xy[0], max_xy[1], -0.8],
            [min_xy[0], max_xy[1], -0.8],
        ],
        dtype=np.float32,
    )
    write_ply(path, vertices, [[0, 1, 2], [0, 2, 3]])


def add_materials(scene: ET.Element) -> None:
    ET.SubElement(scene, "integrator", {"type": "path"})
    materials = """
    <bsdf type="itu-radio-material" id="mat-building_wall"><string name="type" value="concrete" /><float name="thickness" value="0.2" /><float name="scattering_coefficient" value="0.1" /><float name="xpd_coefficient" value="0.0" /></bsdf>
    <bsdf type="itu-radio-material" id="mat-building_roof"><string name="type" value="concrete" /><float name="thickness" value="0.15" /><float name="scattering_coefficient" value="0.1" /><float name="xpd_coefficient" value="0.0" /></bsdf>
    <bsdf type="radio-material" id="mat-road"><float name="relative_permittivity" value="3.18" /><float name="conductivity" value="0.003" /><float name="thickness" value="0.05" /><float name="scattering_coefficient" value="0.2" /><float name="xpd_coefficient" value="0.0" /><rgb name="color" value="0.086500 0.090842 0.088656" /></bsdf>
    <bsdf type="radio-material" id="mat-park"><float name="relative_permittivity" value="2.0" /><float name="conductivity" value="0.02" /><float name="thickness" value="0.1" /><float name="scattering_coefficient" value="0.3" /><float name="xpd_coefficient" value="0.0" /><rgb name="color" value="0.090000 0.520000 0.120000" /></bsdf>
    <bsdf type="itu-radio-material" id="mat-ground"><string name="type" value="medium_dry_ground" /><float name="thickness" value="0.1" /><float name="scattering_coefficient" value="0.3" /><float name="xpd_coefficient" value="0.0" /></bsdf>
    """
    for material in ET.fromstring(f"<root>{materials}</root>"):
        scene.append(material)


def add_shape(scene: ET.Element, shape_id: str, filename: str, material_id: str, face_normals: bool) -> None:
    shape = ET.SubElement(scene, "shape", {"type": "ply", "id": shape_id})
    ET.SubElement(shape, "string", {"name": "filename", "value": f"meshes/{filename}"})
    ET.SubElement(shape, "boolean", {"name": "face_normals", "value": str(face_normals).lower()})
    ET.SubElement(shape, "ref", {"id": material_id, "name": "bsdf"})


def make_plots(records: list[dict], polygon_xy: np.ndarray, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    colors = {
        "mat-building_wall": "#8c564b",
        "mat-building_roof": "#d62728",
        "mat-road": "#4c4c4c",
        "mat-park": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for record in records:
        vertices = record["vertices"]
        faces = record["faces"]
        if len(vertices) == 0:
            continue
        for face in faces:
            xy = vertices[face, :2]
            ax.fill(xy[:, 0], xy[:, 1], color=colors.get(record["material"], "#999999"), alpha=0.55, linewidth=0)
    closed = np.vstack([polygon_xy, polygon_xy[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="#1f77b4", linewidth=2.0, label="requested crop")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Scene X [m]")
    ax.set_ylabel("Scene Y [m]")
    ax.set_title("UPV Vera Catastro subset, plan view")
    ax.legend(loc="upper right")
    fig.savefig(plots_dir / "upv_vera_catastro_subset_plan.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax3 = fig.add_subplot(111, projection="3d")
    for record in records:
        vertices = record["vertices"]
        faces = record["faces"]
        if len(vertices) == 0:
            continue
        centroids = np.array([vertices[face].mean(axis=0) for face in faces])
        ax3.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            s=2,
            color=colors.get(record["material"], "#999999"),
            alpha=0.7,
        )
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")
    ax3.set_title("UPV Vera Catastro subset, face centroids")
    fig.savefig(plots_dir / "upv_vera_catastro_subset_3d_centroids.png", dpi=220)
    plt.close(fig)


def main() -> None:
    meshes_out = OUTPUT_SCENARIO / "meshes"
    if OUTPUT_SCENARIO.exists():
        shutil.rmtree(OUTPUT_SCENARIO)
    meshes_out.mkdir(parents=True)

    polygon_xy = np.array([latlon_to_scene_xy(lat, lon) for lat, lon in REQUESTED_POLYGON_LATLON], dtype=np.float64)
    make_ground(meshes_out / "Plane.ply", polygon_xy)

    scene = ET.Element("scene", {"version": "3.0.0"})
    add_materials(scene)
    add_shape(scene, "py_ground", "Plane.ply", "mat-ground", False)

    records = []
    summary = []
    for filename, material, face_normals in MESH_SPECS:
        src = SOURCE_SCENARIO / "meshes" / filename
        vertices, faces = read_ply(src)
        cropped_vertices, cropped_faces = crop_mesh(vertices, faces, polygon_xy)
        if not cropped_faces:
            continue
        write_ply(meshes_out / filename, cropped_vertices, cropped_faces)
        shape_id = Path(filename).stem.replace("-", "_")
        add_shape(scene, shape_id, filename, material, face_normals)
        records.append({"name": filename, "material": material, "vertices": cropped_vertices, "faces": cropped_faces})
        bounds = np.c_[cropped_vertices.min(axis=0), cropped_vertices.max(axis=0)]
        summary.append((filename, len(cropped_vertices), len(cropped_faces), *bounds.flatten(order="F")))

    ET.indent(scene, space="    ")
    ET.ElementTree(scene).write(
        OUTPUT_SCENARIO / "upv_vera_catastro_subset_no_trees_no_lidar.xml",
        encoding="utf-8",
        xml_declaration=True,
    )

    with (OUTPUT_SCENARIO / "mesh_summary.csv").open("w", encoding="utf-8") as f:
        f.write("mesh,vertices,faces,x_min,y_min,z_min,x_max,y_max,z_max\n")
        for row in summary:
            f.write(",".join(map(str, row)) + "\n")

    with (OUTPUT_SCENARIO / "crop_polygon_scene_xy.csv").open("w", encoding="utf-8") as f:
        f.write("label,lat,lon,x_scene_m,y_scene_m\n")
        labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
        for label, (lat, lon), (x, y) in zip(labels, REQUESTED_POLYGON_LATLON, polygon_xy):
            f.write(f"{label},{lat},{lon},{x:.3f},{y:.3f}\n")

    make_plots(records, polygon_xy, OUTPUT_SCENARIO)


if __name__ == "__main__":
    main()
