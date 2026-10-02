from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import struct

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QMatrix4x4, QVector3D
import pyqtgraph.opengl as gl

from isac_6d_sampler.core.antenna_patterns import (
    POWERLOG_PATTERN,
    QOM_OMNI_PATTERN,
    antenna_pattern_spec,
    powerlog_normalized_gain_db_from_local_dirs,
    qom_omni_normalized_gain_db_from_local_dirs,
)
from isac_6d_sampler.core.model import DYNAMIC_SCENE_OBJECT_NAMES, RadiomapConfig, SceneDesign, TrajectorySpec
from isac_6d_sampler.core.scenarios import (
    MeshBounds,
    ScenarioAsset,
    _struct_format,
    _xyz_property_indices,
    object_mesh_path,
    read_obj_bounds,
)
from isac_6d_sampler.core.sensing_targets import (
    is_sensing_target,
    sensing_options,
    sensing_target_dimensions,
    sensing_target_type,
)
from isac_6d_sampler.core.trajectories import normalized_bezier_handles, sample_radiomap_preview_grid, sample_trajectory


_AXES = {
    "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
}

_AXIS_COLORS = {
    "x": (1.0, 0.05, 0.04, 1.0),
    "y": (0.05, 0.75, 0.18, 1.0),
    "z": (0.08, 0.25, 1.0, 1.0),
}

_PLANE_HANDLES = {
    "xy": ("x", "y", "z", (1.0, 0.9, 0.08, 0.9)),
    "xz": ("x", "z", "y", (1.0, 0.35, 0.9, 0.9)),
    "yz": ("y", "z", "x", (0.1, 0.9, 0.95, 0.9)),
}

_ROTATION_DRAG_SENSITIVITY = 0.15
_ANTENNA_DIAGRAM_RADIUS_M = 2.0
_ANTENNA_DIAGRAM_DB_FLOOR = -30.0


class Scene3DView(gl.GLViewWidget):
    """GPU-backed 3D scene preview for scenario meshes, entities and trajectories."""

    def __init__(
        self,
        on_selected=None,
        on_moved=None,
        on_transform_started=None,
        on_trajectory_point_moved=None,
        on_trajectory_handle_moved=None,
        on_trajectory_endpoint_transformed=None,
        on_trajectory_transform_started=None,
        on_radiomap_bounds_changed=None,
        on_radiomap_transform_started=None,
        parent=None,
    ):
        super().__init__(parent)
        self._asset: ScenarioAsset | None = None
        self._design: SceneDesign | None = None
        self._radiomap: RadiomapConfig | None = None
        self._selected_id: str | None = None
        self._on_selected = on_selected
        self._on_moved = on_moved
        self._on_transform_started = on_transform_started
        self._on_trajectory_point_moved = on_trajectory_point_moved
        self._on_trajectory_handle_moved = on_trajectory_handle_moved
        self._on_trajectory_endpoint_transformed = on_trajectory_endpoint_transformed
        self._on_trajectory_transform_started = on_trajectory_transform_started
        self._on_radiomap_bounds_changed = on_radiomap_bounds_changed
        self._on_radiomap_transform_started = on_radiomap_transform_started
        self._items: list[object] = []
        self._dynamic_items: list[object] = []
        self._antenna_diagram_items: list[object] = []
        self._rcs_preview_frequency_hz = 12.5e9
        self._path_items: list[object] = []
        self._path_overlay_entries: list[dict] = []
        self._path_overlay_visible = False
        self._last_camera_bounds: MeshBounds | None = None
        self._drag_state: dict | None = None
        self._orthographic = False
        self.setObjectName("scene_3d_view")
        self.setMinimumHeight(520)
        self.setBackgroundColor((248, 250, 252))

    def set_scene(
        self,
        asset: ScenarioAsset | None,
        design: SceneDesign,
        radiomap: RadiomapConfig | None,
        selected_id: str | None,
        *,
        reset_camera: bool = True,
    ) -> None:
        self._asset = asset
        self._design = design
        self._radiomap = radiomap
        self._selected_id = selected_id
        self._rebuild_scene(reset_camera=reset_camera)

    def set_selected_entity(self, entity_id: str | None) -> None:
        self._selected_id = entity_id
        self._rebuild_dynamic_items()

    def set_axis_view(self, axis: str, sign: int) -> None:
        elevation, azimuth = _axis_view_angles(axis, sign)
        self._orthographic = True
        self.setCameraPosition(elevation=elevation, azimuth=azimuth)

    def show_antenna_diagrams(self, frequency_hz: float | None = None) -> None:
        if frequency_hz is not None:
            self._rcs_preview_frequency_hz = float(frequency_hz)
        self.hide_antenna_diagrams()
        if self._design is None:
            return
        for entity, color_bias in (
            *[(entity, (1.0, 0.22, 0.04)) for entity in self._design.base_stations],
            *[(entity, (0.05, 0.38, 1.0)) for entity in self._visible_user_equipments()],
        ):
            item = _antenna_diagram_item(
                origin=np.asarray(entity.position, dtype=np.float64),
                orientation_rad=entity.orientation_rad,
                pattern_name=entity.panel.pattern,
                color_bias=color_bias,
            )
            self.addItem(item)
            self._items.append(item)
            self._antenna_diagram_items.append(item)
        self._add_scattering_point_diagrams()

    def _add_scattering_point_diagrams(self) -> None:
        """Monostatic RCS lobe of every scattering point of the sensing targets."""
        for obj in self._design.objects:
            if not is_sensing_target(obj):
                continue
            dimensions = self._sensing_target_lcs_dimensions(obj)
            if dimensions is None:
                continue
            try:
                meshes = _scattering_point_lobe_meshes(
                    obj, dimensions, frequency_hz=self._rcs_preview_frequency_hz)
            except Exception as exc:  # noqa: BLE001 - a missing Sionna/GPU must not break the view
                print(f"Scattering-point patterns unavailable for {obj.id}: {exc}")
                continue
            for vertices, faces, face_colors in meshes:
                item = _lobe_mesh_item(vertices, faces, face_colors)
                self.addItem(item)
                self._items.append(item)
                self._antenna_diagram_items.append(item)

    def _sensing_target_lcs_dimensions(self, obj) -> tuple[float, float, float] | None:
        dimensions = sensing_target_dimensions(obj)
        if dimensions is not None or self._asset is None:
            return dimensions
        # Mesh-shaped target: Sionna places the scattering points from the mesh AABB.
        path = object_mesh_path(Path(self._asset.path).parent / "objects", sensing_options(obj).mesh)
        bounds = read_obj_bounds(path) if path is not None else None
        if bounds is None:
            return None
        extents = np.asarray(bounds.max_xyz, dtype=np.float64) - np.asarray(bounds.min_xyz, dtype=np.float64)
        return tuple(float(value) for value in extents)

    def hide_antenna_diagrams(self) -> None:
        for item in list(self._antenna_diagram_items):
            self.removeItem(item)
            if item in self._items:
                self._items.remove(item)
        self._antenna_diagram_items.clear()

    def antenna_diagrams_visible(self) -> bool:
        return bool(self._antenna_diagram_items)

    def show_simulation_paths(
        self,
        result,
        *,
        max_paths: int = 36,
        min_delay_separation_ns: float = 2.0,
        min_range_m: float | None = None,
        max_range_m: float | None = None,
        tx_id: str | None = None,
        rx_id: str | None = None,
    ) -> tuple[int, dict]:
        self.hide_simulation_paths()
        if result is None or not getattr(result, "timeframes", None):
            return 0, {"reason": "no result/timeframes"}
        timeframe = result.timeframes[0]
        device_positions = timeframe.metadata.get("device_positions", {})
        selected, stats = _select_important_bs_to_ue_paths(
            timeframe.links,
            device_positions,
            max_paths=max_paths,
            min_delay_separation_s=float(min_delay_separation_ns) * 1e-9,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            tx_id=tx_id,
            rx_id=rx_id,
        )
        self._path_overlay_entries = list(selected)
        self._path_overlay_visible = bool(selected)
        self._add_path_overlay_items()
        return len(selected), stats

    def hide_simulation_paths(self) -> None:
        for item in list(self._path_items):
            self._safe_remove_item(item)
            if item in self._items:
                self._items.remove(item)
        self._path_items.clear()
        self._path_overlay_visible = False
        self._path_overlay_entries.clear()

    def simulation_paths_visible(self) -> bool:
        return self._path_overlay_visible and bool(self._path_overlay_entries)

    def _add_path_overlay_items(self) -> None:
        for item in list(self._path_items):
            self._safe_remove_item(item)
            if item in self._items:
                self._items.remove(item)
        self._path_items.clear()
        for entry in self._path_overlay_entries:
            item = gl.GLLinePlotItem(
                pos=entry["points"].astype(np.float32),
                color=entry["color"],
                width=entry["width"],
                antialias=True,
                mode="line_strip",
            )
            self.addItem(item)
            self._items.append(item)
            self._path_items.append(item)

    def _safe_remove_item(self, item) -> None:
        try:
            self.removeItem(item)
        except ValueError:
            pass

    def restore_perspective(self) -> None:
        if self._orthographic:
            self._orthographic = False
            self.update()

    def projectionMatrix(self, region, viewport):
        if not self._orthographic:
            return super().projectionMatrix(region, viewport)
        x0, y0, width, height = viewport
        distance = float(self.opts["distance"])
        fov = float(self.opts["fov"])
        near_clip = distance * 0.001
        far_clip = distance * 1000.0
        half_width = distance * np.tan(0.5 * np.deg2rad(fov))
        half_height = half_width * height / width
        left = half_width * ((region[0] - x0) * (2.0 / width) - 1.0)
        right = half_width * ((region[0] + region[2] - x0) * (2.0 / width) - 1.0)
        bottom = half_height * ((region[1] - y0) * (2.0 / height) - 1.0)
        top = half_height * ((region[1] + region[3] - y0) * (2.0 / height) - 1.0)
        matrix = QMatrix4x4()
        matrix.ortho(float(left), float(right), float(bottom), float(top), near_clip, far_clip)
        return matrix

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            state = self._begin_radiomap_corner_drag(event.position().x(), event.position().y())
            if state is not None:
                self._drag_state = state
                if self._on_radiomap_transform_started is not None:
                    self._on_radiomap_transform_started()
                event.accept()
                return
            if self._selected_trajectory_kind() == "curve":
                state = self._begin_trajectory_handle_drag(event.position().x(), event.position().y())
                if state is not None:
                    self._drag_state = state
                    if self._on_trajectory_transform_started is not None:
                        self._on_trajectory_transform_started(state["entity_id"])
                    event.accept()
                    return
                state = self._begin_trajectory_point_drag(event.position().x(), event.position().y())
                if state is not None:
                    self._drag_state = state
                    if self._on_trajectory_transform_started is not None:
                        self._on_trajectory_transform_started(state["entity_id"])
                    event.accept()
                    return
            state = self._begin_gizmo_drag(event.position().x(), event.position().y())
            if state is not None:
                self._drag_state = state
                if state.get("target") == "trajectory_endpoint" and self._on_trajectory_transform_started is not None:
                    self._on_trajectory_transform_started(state["entity_id"])
                elif self._on_transform_started is not None and self._selected_id is not None:
                    self._on_transform_started(self._selected_id)
                event.accept()
                return
            if self._selected_trajectory_kind() != "curve":
                state = self._begin_trajectory_handle_drag(event.position().x(), event.position().y())
                if state is not None:
                    self._drag_state = state
                    if self._on_trajectory_transform_started is not None:
                        self._on_trajectory_transform_started(state["entity_id"])
                    event.accept()
                    return
                state = self._begin_trajectory_point_drag(event.position().x(), event.position().y())
                if state is not None:
                    self._drag_state = state
                    if self._on_trajectory_transform_started is not None:
                        self._on_trajectory_transform_started(state["entity_id"])
                    event.accept()
                    return
            selected_id = self._pick_entity(event.position().x(), event.position().y())
            if selected_id is not None:
                if self._on_selected is not None:
                    self._on_selected(selected_id)
                self._selected_id = selected_id
                self._rebuild_dynamic_items()
                event.accept()
                return
            if self._selected_id is not None:
                self._selected_id = None
                self._drag_state = None
                if self._on_selected is not None:
                    self._on_selected(None)
                self._rebuild_dynamic_items()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_state is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self._drag_state.get("target") == "radiomap_corner":
                bounds = self._drag_radiomap_corner(
                    event.position().x(),
                    event.position().y(),
                    self._drag_state,
                )
                if bounds is not None and self._on_radiomap_bounds_changed is not None:
                    self._on_radiomap_bounds_changed(*bounds)
                event.accept()
                return
            if self._drag_state.get("target") == "trajectory_endpoint":
                position = self._drag_position(event.position().x(), event.position().y(), self._drag_state)
                orientation = self._drag_orientation(event.position().x(), event.position().y(), self._drag_state)
                if position is not None or orientation is not None:
                    entity = self._selected_entity()
                    if entity is not None and hasattr(entity, "trajectory"):
                        point_index = self._drag_state["point_index"]
                        if position is None:
                            position = np.asarray(entity.trajectory.points[point_index], dtype=np.float64)
                        if orientation is None:
                            orientation = np.asarray(self._trajectory_endpoint_orientation(entity), dtype=np.float64)
                        if self._on_trajectory_endpoint_transformed is not None:
                            self._on_trajectory_endpoint_transformed(
                                self._drag_state["entity_id"],
                                point_index,
                                position,
                                orientation,
                            )
                        else:
                            entity.trajectory.points[point_index] = tuple(float(value) for value in position)
                    self._rebuild_dynamic_items()
                event.accept()
                return
            if self._drag_state.get("target") == "trajectory_handle":
                position = self._drag_trajectory_xy(
                    event.position().x(),
                    event.position().y(),
                    self._drag_state,
                )
                if position is not None:
                    if self._on_trajectory_handle_moved is not None:
                        self._on_trajectory_handle_moved(
                            self._drag_state["entity_id"],
                            self._drag_state["point_index"],
                            self._drag_state["handle_side"],
                            position,
                        )
                    else:
                        entity = self._selected_entity()
                        if entity is not None and hasattr(entity, "trajectory"):
                            handles = normalized_bezier_handles(entity.trajectory)
                            handle_in, handle_out = handles[self._drag_state["point_index"]]
                            if self._drag_state["handle_side"] == "in":
                                handles[self._drag_state["point_index"]] = (tuple(float(value) for value in position), handle_out)
                            else:
                                handles[self._drag_state["point_index"]] = (handle_in, tuple(float(value) for value in position))
                            entity.trajectory.bezier_handles = handles
                    self._rebuild_dynamic_items()
                event.accept()
                return
            if self._drag_state["kind"] == "trajectory_point":
                position = self._drag_trajectory_xy(
                    event.position().x(),
                    event.position().y(),
                    self._drag_state,
                )
                if position is not None:
                    if self._on_trajectory_point_moved is not None:
                        self._on_trajectory_point_moved(
                            self._drag_state["entity_id"],
                            self._drag_state["point_index"],
                            position,
                        )
                    else:
                        entity = self._selected_entity()
                        if entity is not None and hasattr(entity, "trajectory"):
                            entity.trajectory.points[self._drag_state["point_index"]] = tuple(
                                float(value) for value in position
                            )
                    self._rebuild_dynamic_items()
                event.accept()
                return
            position = self._drag_position(event.position().x(), event.position().y(), self._drag_state)
            orientation = self._drag_orientation(event.position().x(), event.position().y(), self._drag_state)
            if position is not None or orientation is not None:
                entity = self._selected_entity()
                if entity is None:
                    return
                if position is None:
                    position = np.asarray(entity.position, dtype=np.float64)
                if orientation is None:
                    orientation = np.asarray(entity.orientation_rad, dtype=np.float64)
                if self._on_moved is not None and self._selected_id is not None:
                    self._on_moved(self._selected_id, position, orientation)
                else:
                    entity.position = tuple(float(value) for value in position)
                    entity.orientation_rad = tuple(float(value) for value in orientation)
                self._rebuild_dynamic_items()
            event.accept()
            return
        if event.buttons():
            self.restore_perspective()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_state = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.restore_perspective()
        super().wheelEvent(event)

    def _rebuild_scene(self, *, reset_camera: bool = True) -> None:
        self._clear_items()
        if self._design is None:
            return

        bounds = self._scene_bounds()
        self._add_grid(bounds)
        self._add_axes(bounds)
        self._add_meshes()
        self._add_dynamic_items()
        if self._path_overlay_visible and self._path_overlay_entries:
            self._add_path_overlay_items()

        if reset_camera and bounds is not None:
            self._fit_camera(bounds)

    def _clear_items(self) -> None:
        for item in list(self._items):
            self._safe_remove_item(item)
        self._items.clear()
        self._dynamic_items.clear()
        self._antenna_diagram_items.clear()
        self._path_items.clear()

    def _add_item(self, item) -> None:
        self.addItem(item)
        self._items.append(item)

    def _add_dynamic_item(self, item) -> None:
        self.addItem(item)
        self._items.append(item)
        self._dynamic_items.append(item)

    def _rebuild_dynamic_items(self) -> None:
        for item in list(self._dynamic_items):
            self._safe_remove_item(item)
            if item in self._items:
                self._items.remove(item)
        self._dynamic_items.clear()
        if self._design is None:
            return
        self._add_dynamic_items()

    def _add_dynamic_items(self) -> None:
        self._add_dynamic_object_meshes()
        self._add_sensing_target_meshes()
        self._add_radiomap()
        self._add_trajectories()
        self._add_trajectory_handles()
        self._add_entities()
        self._add_pointing_vectors()
        self._add_gizmo()

    def _fit_camera(self, bounds: MeshBounds) -> None:
        mins = np.asarray(bounds.min_xyz, dtype=np.float64)
        maxs = np.asarray(bounds.max_xyz, dtype=np.float64)
        center = (mins + maxs) / 2.0
        radius = max(float(np.linalg.norm(maxs - mins) / 2.0), 1.0)
        self.opts["center"] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.setCameraPosition(distance=radius * 2.7, elevation=27.0, azimuth=-45.0)
        self._last_camera_bounds = bounds

    def _scene_bounds(self) -> MeshBounds | None:
        points = []
        if self._asset:
            for mesh in self._visible_meshes():
                if mesh.bounds is not None:
                    points.extend([mesh.bounds.min_xyz, mesh.bounds.max_xyz])
        if self._design is not None:
            for entity in [*self._design.base_stations, *self._visible_user_equipments(), *self._design.objects]:
                points.append(entity.position)
                if hasattr(entity, "trajectory"):
                    points.extend(entity.trajectory.points)
                    if entity.trajectory.kind == "curve":
                        for handle_in, handle_out in normalized_bezier_handles(entity.trajectory):
                            points.extend([handle_in, handle_out])
        if self._radiomap and self._radiomap.enabled:
            z = self._radiomap.height
            points.extend(
                [
                    (self._radiomap.x_min, self._radiomap.y_min, z),
                    (self._radiomap.x_max, self._radiomap.y_max, z),
                ]
            )
        if not points:
            return None
        values = np.asarray(points, dtype=np.float64)
        return MeshBounds(
            min_xyz=tuple(np.min(values, axis=0).tolist()),
            max_xyz=tuple(np.max(values, axis=0).tolist()),
        )

    def _add_grid(self, bounds: MeshBounds | None) -> None:
        grid = gl.GLGridItem()
        if bounds is None:
            grid.setSize(x=20.0, y=20.0)
            grid.setSpacing(x=1.0, y=1.0)
        else:
            mins = np.asarray(bounds.min_xyz, dtype=np.float64)
            maxs = np.asarray(bounds.max_xyz, dtype=np.float64)
            span = np.maximum(maxs - mins, 1.0)
            center = (mins + maxs) / 2.0
            spacing = max(1.0, round(max(span[0], span[1]) / 20.0))
            grid.setSize(x=float(span[0] * 1.1), y=float(span[1] * 1.1))
            grid.setSpacing(x=float(spacing), y=float(spacing))
            grid.translate(float(center[0]), float(center[1]), float(mins[2]))
        self._add_item(grid)

    def _add_axes(self, bounds: MeshBounds | None) -> None:
        if bounds is None:
            origin = np.zeros(3, dtype=np.float32)
            length = 10.0
        else:
            mins = np.asarray(bounds.min_xyz, dtype=np.float32)
            maxs = np.asarray(bounds.max_xyz, dtype=np.float32)
            origin = mins
            length = max(float(np.linalg.norm(maxs - mins) * 0.12), 3.0)
        axes = [
            (np.asarray([origin, origin + [length, 0.0, 0.0]], dtype=np.float32), (1.0, 0.1, 0.1, 1.0)),
            (np.asarray([origin, origin + [0.0, length, 0.0]], dtype=np.float32), (0.1, 0.75, 0.2, 1.0)),
            (np.asarray([origin, origin + [0.0, 0.0, length]], dtype=np.float32), (0.1, 0.25, 1.0, 1.0)),
        ]
        for positions, color in axes:
            self._add_item(gl.GLLinePlotItem(pos=positions, color=color, width=3.0, antialias=True, mode="lines"))

    def _add_meshes(self) -> None:
        if self._asset is None:
            return
        for mesh in self._static_meshes():
            vertices, faces = _load_ply_mesh(str(mesh.path), _mtime(mesh.path))
            if vertices.size == 0 or faces.size == 0:
                continue
            mesh_data = gl.MeshData(vertexes=vertices.astype(np.float32), faces=faces.astype(np.int32))
            item = gl.GLMeshItem(
                meshdata=mesh_data,
                color=_mesh_color(mesh.name),
                smooth=False,
                drawEdges=True,
                drawFaces=True,
                shader="balloon",
                glOptions="opaque",
            )
            self._add_item(item)

    def _add_dynamic_object_meshes(self) -> None:
        if self._asset is None:
            return
        for mesh in self._dynamic_meshes():
            vertices, faces = _load_mesh_preview(str(mesh.path), _mtime(mesh.path))
            if vertices.size == 0 or faces.size == 0:
                continue
            for rendered_vertices in self._mesh_render_vertices(mesh, vertices):
                mesh_data = gl.MeshData(vertexes=rendered_vertices.astype(np.float32), faces=faces.astype(np.int32))
                item = gl.GLMeshItem(
                    meshdata=mesh_data,
                    color=_mesh_color(mesh.name),
                    smooth=False,
                    drawEdges=True,
                    drawFaces=True,
                    shader="balloon",
                    glOptions="opaque",
                )
                self._add_dynamic_item(item)

    def _add_sensing_target_meshes(self) -> None:
        """Draw sensing targets as translucent cuboids (or their optional mesh)."""
        if self._design is None:
            return
        for obj in self._design.objects:
            if not is_sensing_target(obj):
                continue
            vertices, faces = self._sensing_target_geometry(obj)
            if vertices.size == 0 or faces.size == 0:
                continue
            rendered = _transform_mesh_vertices(
                vertices,
                (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0,
                np.asarray(obj.position, dtype=np.float64),
                obj.orientation_rad,
            )
            mesh_data = gl.MeshData(vertexes=rendered.astype(np.float32), faces=faces.astype(np.int32))
            item = gl.GLMeshItem(
                meshdata=mesh_data,
                color=SENSING_TARGET_COLOR,
                smooth=False,
                drawEdges=True,
                drawFaces=True,
                shader="balloon",
                glOptions="translucent",
            )
            self._add_dynamic_item(item)

    def _sensing_target_geometry(self, obj) -> tuple[np.ndarray, np.ndarray]:
        dimensions = sensing_target_dimensions(obj)
        if dimensions is not None:
            return _cuboid_mesh(dimensions)
        if self._asset is None:
            return np.empty((0, 3)), np.empty((0, 3))
        path = object_mesh_path(Path(self._asset.path).parent / "objects", sensing_options(obj).mesh)
        if path is None:
            return np.empty((0, 3)), np.empty((0, 3))
        return _load_mesh_preview(str(path), _mtime(path))

    def _mesh_render_vertices(self, mesh, vertices: np.ndarray) -> tuple[np.ndarray, ...]:
        if self._design is None:
            return (vertices,)
        objects = [obj for obj in self._design.objects if obj.object_name == mesh.name]
        if not objects:
            return ()
        center = _mesh_anchor(mesh, vertices)
        return tuple(
            _transform_mesh_vertices(
                vertices,
                center,
                np.asarray(obj.position, dtype=np.float64),
                obj.orientation_rad,
            )
            for obj in objects
        )

    def _visible_meshes(self):
        return (*self._static_meshes(), *self._dynamic_meshes())

    def _static_meshes(self):
        if self._asset is None:
            return ()
        return tuple(mesh for mesh in self._asset.meshes if mesh.name not in DYNAMIC_SCENE_OBJECT_NAMES)

    def _dynamic_meshes(self):
        if self._asset is None:
            return ()
        active_dynamic_objects = {
            obj.object_name
            for obj in self._design.objects
        } if self._design is not None else set()
        xml_meshes = tuple(
            mesh
            for mesh in self._asset.meshes
            if mesh.name in DYNAMIC_SCENE_OBJECT_NAMES and mesh.name in active_dynamic_objects
        )
        external_meshes = tuple(
            mesh
            for mesh in getattr(self._asset, "object_meshes", ())
            if mesh.name in active_dynamic_objects
        )
        return (*xml_meshes, *external_meshes)

    def _add_entities(self) -> None:
        if self._design is None:
            return
        entities = [
            *[(entity, (1.0, 0.1, 0.08, 1.0), 13.0) for entity in self._design.base_stations],
            *[(entity, (0.05, 0.25, 1.0, 1.0), 10.0) for entity in self._visible_user_equipments()],
            *[(entity, (1.0, 0.55, 0.05, 1.0), 10.0) for entity in self._design.objects],
        ]
        for entity, color, size in entities:
            selected = entity.id == self._selected_id
            marker = gl.GLScatterPlotItem(
                pos=np.asarray([entity.position], dtype=np.float32),
                color=(1.0, 1.0, 0.05, 1.0) if selected else color,
                size=size * (1.45 if selected else 1.0),
                pxMode=True,
            )
            self._add_dynamic_item(marker)

    def _add_gizmo(self) -> None:
        entity = self._selected_entity()
        if entity is None:
            return
        origin = np.asarray(entity.position, dtype=np.float32)
        self._add_transform_gizmo(origin, entity.orientation_rad, self._gizmo_length(), alpha=1.0)

        endpoint = self._trajectory_endpoint_pose(entity)
        if endpoint is not None:
            endpoint_position, endpoint_orientation = endpoint
            self._add_transform_gizmo(
                endpoint_position.astype(np.float32),
                endpoint_orientation,
                self._gizmo_length() * 0.72,
                alpha=0.82,
            )

    def _add_transform_gizmo(self, origin: np.ndarray, orientation_rad, size: float, *, alpha: float) -> None:
        axes = _local_axes(orientation_rad)
        self._add_plane_handles(origin, axes, size)
        for axis_name, axis_vector in axes.items():
            axis = axis_vector.astype(np.float32)
            end = origin + axis * size
            self._add_dynamic_item(
                gl.GLLinePlotItem(
                    pos=np.asarray([origin, end], dtype=np.float32),
                    color=(*_AXIS_COLORS[axis_name][:3], alpha),
                    width=4.0,
                    antialias=True,
                    mode="lines",
                )
            )
            ring = _rotation_ring(origin, axis, size * 0.82)
            self._add_dynamic_item(
                gl.GLLinePlotItem(
                    pos=ring,
                    color=(*_AXIS_COLORS[axis_name][:3], 0.72 * alpha),
                    width=2.0,
                    antialias=True,
                    mode="line_strip",
                )
            )

    def _add_plane_handles(self, origin: np.ndarray, axes: dict[str, np.ndarray], size: float) -> None:
        for plane_name, (first_axis, second_axis, _normal_axis, color) in _PLANE_HANDLES.items():
            points = _plane_handle_points(
                origin.astype(np.float64),
                axes[first_axis],
                axes[second_axis],
                size,
            )
            self._add_dynamic_item(
                gl.GLLinePlotItem(
                    pos=points.astype(np.float32),
                    color=color,
                    width=2.0,
                    antialias=True,
                    mode="line_strip",
                )
            )

    def _gizmo_length(self) -> float:
        return _screen_pixels_to_world(
            distance=float(self.opts.get("distance", 40.0)),
            fov_degrees=float(self.opts.get("fov", 60.0)),
            viewport_height=max(float(self.height()), 1.0),
            pixels=92.0,
        )

    def _add_pointing_vectors(self) -> None:
        if self._design is None:
            return
        length = self._pointing_length()
        for entity in [*self._design.base_stations, *self._visible_user_equipments(), *self._design.objects]:
            origin = np.asarray(entity.position, dtype=np.float32)
            axes = _local_axes(entity.orientation_rad)
            forward = axes["x"].astype(np.float32)
            end = origin + forward * length
            selected = entity.id == self._selected_id
            color = (1.0, 0.82, 0.05, 1.0) if selected else (0.12, 0.12, 0.12, 0.86)
            self._add_dynamic_item(
                gl.GLLinePlotItem(
                    pos=np.asarray([origin, end], dtype=np.float32),
                    color=color,
                    width=4.0 if selected else 2.0,
                    antialias=True,
                    mode="lines",
                )
            )
            arrowhead = _arrowhead_segments(
                end.astype(np.float64),
                forward.astype(np.float64),
                axes["y"],
                axes["z"],
                length,
            )
            self._add_dynamic_item(
                gl.GLLinePlotItem(
                    pos=arrowhead.astype(np.float32),
                    color=color,
                    width=3.0 if selected else 1.8,
                    antialias=True,
                    mode="lines",
                )
            )

    def _pointing_length(self) -> float:
        bounds = self._scene_bounds()
        if bounds is None:
            return 4.0
        mins = np.asarray(bounds.min_xyz, dtype=np.float64)
        maxs = np.asarray(bounds.max_xyz, dtype=np.float64)
        return max(float(np.linalg.norm(maxs - mins) * 0.035), 1.0)

    def _add_trajectories(self) -> None:
        if self._design is None:
            return
        for entity in [*self._visible_user_equipments(), *self._design.objects]:
            if len(entity.trajectory.points) < 2:
                continue
            positions = _trajectory_preview_positions(entity.trajectory)
            item = gl.GLLinePlotItem(
                pos=positions.astype(np.float32),
                color=(0.0, 0.7, 0.8, 1.0),
                width=3.0,
                antialias=True,
                mode="line_strip",
            )
            self._add_dynamic_item(item)
            if entity.id == self._selected_id and entity.trajectory.kind == "curve":
                self._add_curve_tangent_pointing(positions)

    def _add_trajectory_handles(self) -> None:
        entity = self._selected_entity()
        if entity is None or not hasattr(entity, "trajectory"):
            return
        trajectory = entity.trajectory
        if trajectory.kind not in {"linear", "polyline", "curve"} or len(trajectory.points) < 2:
            return
        editable_points = np.asarray(trajectory.points[1:], dtype=np.float32)
        self._add_dynamic_item(
            gl.GLScatterPlotItem(
                pos=editable_points,
                color=(1.0, 0.05, 0.78, 1.0),
                size=14.0,
                pxMode=True,
            )
        )
        if trajectory.kind == "curve":
            self._add_curve_control_handles(trajectory)
        endpoint = np.asarray([trajectory.points[-1]], dtype=np.float32)
        orientation = self._trajectory_endpoint_orientation(entity)
        axes = _local_axes(orientation)
        forward = axes["x"].astype(np.float32)
        origin = endpoint[0]
        length = self._pointing_length() * 0.82
        end = origin + forward * length
        color = (1.0, 0.05, 0.78, 1.0)
        self._add_dynamic_item(
            gl.GLLinePlotItem(
                pos=np.asarray([origin, end], dtype=np.float32),
                color=color,
                width=3.0,
                antialias=True,
                mode="lines",
            )
        )
        self._add_dynamic_item(
            gl.GLLinePlotItem(
                pos=_arrowhead_segments(
                    end.astype(np.float64),
                    forward.astype(np.float64),
                    axes["y"],
                    axes["z"],
                    length,
                ).astype(np.float32),
                color=color,
                width=2.2,
                antialias=True,
                mode="lines",
            )
        )

    def _add_curve_control_handles(self, trajectory) -> None:
        handles = normalized_bezier_handles(trajectory)
        handle_points = []
        connector_segments = []
        last_index = len(trajectory.points) - 1
        for index, (point, (handle_in, handle_out)) in enumerate(zip(trajectory.points, handles, strict=True)):
            point_array = np.asarray(point, dtype=np.float64)
            if index > 0:
                handle_in_array = np.asarray(handle_in, dtype=np.float64)
                handle_points.append(handle_in_array)
                connector_segments.extend([point_array, handle_in_array])
            if index < last_index:
                handle_out_array = np.asarray(handle_out, dtype=np.float64)
                handle_points.append(handle_out_array)
                connector_segments.extend([point_array, handle_out_array])
        if connector_segments:
            self._add_dynamic_item(
                gl.GLLinePlotItem(
                    pos=np.asarray(connector_segments, dtype=np.float32),
                    color=(0.45, 0.1, 0.75, 0.76),
                    width=1.8,
                    antialias=True,
                    mode="lines",
                )
            )
        if handle_points:
            self._add_dynamic_item(
                gl.GLScatterPlotItem(
                    pos=np.asarray(handle_points, dtype=np.float32),
                    color=(0.55, 0.0, 1.0, 1.0),
                    size=10.0,
                    pxMode=True,
                )
            )

    def _add_curve_tangent_pointing(self, positions: np.ndarray) -> None:
        segments = _curve_tangent_arrow_segments(positions, self._pointing_length() * 0.42)
        if len(segments) == 0:
            return
        self._add_dynamic_item(
            gl.GLLinePlotItem(
                pos=segments.astype(np.float32),
                color=(1.0, 0.72, 0.0, 0.95),
                width=2.4,
                antialias=True,
                mode="lines",
            )
        )

    def _add_radiomap(self) -> None:
        if not self._radiomap or not self._radiomap.enabled:
            return
        config = self._radiomap
        z = config.height
        corners = np.asarray(
            [
                (config.x_min, config.y_min, z),
                (config.x_max, config.y_min, z),
                (config.x_max, config.y_max, z),
                (config.x_min, config.y_max, z),
                (config.x_min, config.y_min, z),
            ],
            dtype=np.float32,
        )
        surface_vertices = corners[:4]
        surface_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        surface_data = gl.MeshData(vertexes=surface_vertices, faces=surface_faces)
        self._add_dynamic_item(
            gl.GLMeshItem(
                meshdata=surface_data,
                color=(0.75, 0.1, 0.75, 0.16),
                smooth=False,
                drawFaces=True,
                drawEdges=False,
                glOptions="translucent",
            )
        )
        self._add_dynamic_item(
            gl.GLLinePlotItem(
                pos=corners,
                color=(0.75, 0.1, 0.75, 1.0),
                width=2.0,
                antialias=True,
                mode="line_strip",
            )
        )
        handles = np.asarray(
            [
                (config.x_min, config.y_min, z),
                (config.x_max, config.y_max, z),
            ],
            dtype=np.float32,
        )
        self._add_dynamic_item(
            gl.GLScatterPlotItem(
                pos=handles,
                color=(1.0, 0.1, 0.85, 1.0),
                size=14.0,
                pxMode=True,
            )
        )
        try:
            preview, _total = sample_radiomap_preview_grid(config)
        except ValueError:
            return
        if len(preview):
            self._add_dynamic_item(
                gl.GLScatterPlotItem(
                    pos=np.asarray(preview, dtype=np.float32),
                    color=(0.75, 0.1, 0.75, 0.9),
                    size=4.0,
                    pxMode=True,
                )
            )

    def _selected_entity(self):
        if self._design is None or self._selected_id is None:
            return None
        for entity in [*self._design.base_stations, *self._visible_user_equipments(), *self._design.objects]:
            if entity.id == self._selected_id:
                return entity
        return None

    def _selected_trajectory_kind(self) -> str | None:
        entity = self._selected_entity()
        if entity is None or not hasattr(entity, "trajectory"):
            return None
        return entity.trajectory.kind

    def _pick_entity(self, x: float, y: float) -> str | None:
        if self._design is None:
            return None
        best_id = None
        best_distance = None
        marker_radius_px = 16.0
        for entity in [*self._design.base_stations, *self._visible_user_equipments(), *self._design.objects]:
            screen = self._project_to_screen(entity.position)
            if screen is None:
                continue
            distance = _screen_distance((x, y), screen)
            if distance <= marker_radius_px and (best_distance is None or distance < best_distance):
                best_id = entity.id
                best_distance = distance
        return best_id

    def _visible_user_equipments(self):
        if self._design is None:
            return []
        if self._radiomap is not None and self._radiomap.enabled:
            return []
        return self._design.user_equipments

    def _begin_radiomap_corner_drag(self, x: float, y: float) -> dict | None:
        config = self._radiomap
        if config is None or not config.enabled:
            return None
        corners = {
            "min": np.asarray([config.x_min, config.y_min, config.height], dtype=np.float64),
            "max": np.asarray([config.x_max, config.y_max, config.height], dtype=np.float64),
        }
        best_name = None
        best_distance = None
        for name, point in corners.items():
            screen = self._project_to_screen(point)
            if screen is None:
                continue
            distance = _screen_distance((x, y), screen)
            if distance <= 18.0 and (best_distance is None or distance < best_distance):
                best_name = name
                best_distance = distance
        if best_name is None:
            return None
        start = corners[best_name]
        screen_origin = self._project_to_screen(start)
        screen_x = self._project_to_screen(start + _AXES["x"])
        screen_y = self._project_to_screen(start + _AXES["y"])
        return {
            "target": "radiomap_corner",
            "corner": best_name,
            "mouse_start": np.asarray([x, y], dtype=np.float64),
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "x_min": float(config.x_min),
            "x_max": float(config.x_max),
            "y_min": float(config.y_min),
            "y_max": float(config.y_max),
            "height": float(config.height),
            "world_per_pixel": _radiomap_corner_world_per_pixel(config),
            "screen_x_axis": _screen_unit_vector(screen_origin, screen_x),
            "screen_y_axis": _screen_unit_vector(screen_origin, screen_y),
        }

    def _drag_radiomap_corner(self, x: float, y: float, state: dict) -> tuple[float, float, float, float] | None:
        mouse_delta = np.asarray([x, y], dtype=np.float64) - state["mouse_start"]
        x_axis = state.get("screen_x_axis")
        y_axis = state.get("screen_y_axis")
        if x_axis is None or y_axis is None:
            x_pixels, y_pixels = float(mouse_delta[0]), float(-mouse_delta[1])
        else:
            x_pixels, y_pixels = _screen_plane_components(mouse_delta, x_axis, y_axis)
        world_per_pixel = float(state["world_per_pixel"])
        point_x = float(state["start_x"]) + x_pixels * world_per_pixel
        point_y = float(state["start_y"]) + y_pixels * world_per_pixel
        if state["corner"] == "min":
            return (point_x, float(state["x_max"]), point_y, float(state["y_max"]))
        return (float(state["x_min"]), point_x, float(state["y_min"]), point_y)

    def _begin_gizmo_drag(self, x: float, y: float) -> dict | None:
        entity = self._selected_entity()
        if entity is None:
            return None
        endpoint = self._trajectory_endpoint_pose(entity)
        if endpoint is not None:
            endpoint_position, endpoint_orientation = endpoint
            handle = self._pick_transform_handle(
                x,
                y,
                endpoint_position,
                endpoint_orientation,
                self._gizmo_length() * 0.72,
            )
            if handle is not None:
                return self._transform_drag_state(
                    handle,
                    endpoint_position,
                    endpoint_orientation,
                    x,
                    y,
                    target="trajectory_endpoint",
                    entity_id=entity.id,
                    point_index=len(entity.trajectory.points) - 1,
                )
        handle = self._pick_gizmo_handle(x, y)
        if handle is None:
            return None
        start = np.asarray(entity.position, dtype=np.float64)
        start_orientation = np.asarray(entity.orientation_rad, dtype=np.float64)
        return self._transform_drag_state(
            handle,
            start,
            start_orientation,
            x,
            y,
            target="entity",
        )

    def _transform_drag_state(
        self,
        handle: dict,
        start: np.ndarray,
        start_orientation: np.ndarray,
        x: float,
        y: float,
        *,
        target: str,
        entity_id: str | None = None,
        point_index: int | None = None,
    ) -> dict | None:
        axes = _local_axes(start_orientation)
        axis = axes[handle["axis"]]
        if handle["kind"] in {"translate_axis", "translate_plane"}:
            anchor = start
        else:
            anchor = start
        screen_origin = self._project_to_screen(start)
        screen_x = self._project_to_screen(start + axes["x"])
        screen_y = self._project_to_screen(start + axes["y"])
        screen_z = self._project_to_screen(start + axes["z"])
        screen_axes = {
            "x": _screen_unit_vector(screen_origin, screen_x),
            "y": _screen_unit_vector(screen_origin, screen_y),
            "z": _screen_unit_vector(screen_origin, screen_z),
        }
        world_per_pixel = _screen_pixels_to_world(
            distance=float(self.opts.get("distance", 40.0)),
            fov_degrees=float(self.opts.get("fov", 60.0)),
            viewport_height=max(float(self.height()), 1.0),
            pixels=1.0,
        )
        if anchor is None:
            return None
        return {
            "kind": handle["kind"],
            "target": target,
            "entity_id": entity_id,
            "point_index": point_index,
            "axis_name": handle["axis"],
            "axis": axis,
            "start": start,
            "start_orientation": start_orientation,
            "anchor": anchor,
            "mouse_start": np.asarray([x, y], dtype=np.float64),
            "screen_center": np.asarray(screen_origin, dtype=np.float64) if screen_origin is not None else None,
            "screen_angle_start": _screen_angle(screen_origin, (x, y)) if screen_origin is not None else None,
            "screen_axes": screen_axes,
            "world_per_pixel": world_per_pixel,
            "plane_axes": handle.get("plane_axes"),
            "local_axes": axes,
        }

    def _begin_trajectory_point_drag(self, x: float, y: float) -> dict | None:
        entity = self._selected_entity()
        if entity is None or not hasattr(entity, "trajectory"):
            return None
        trajectory = entity.trajectory
        if trajectory.kind not in {"linear", "polyline", "curve"} or len(trajectory.points) < 2:
            return None
        point_index = self._pick_trajectory_anchor(x, y, trajectory)
        if point_index is None:
            return None
        start = np.asarray(trajectory.points[point_index], dtype=np.float64)
        drag_state = self._trajectory_xy_drag_state(trajectory, start, x, y)
        return {
            "kind": "trajectory_point",
            "target": "trajectory_point",
            "entity_id": entity.id,
            "point_index": point_index,
            "start": start,
            **drag_state,
        }

    def _trajectory_endpoint_pose(self, entity) -> tuple[np.ndarray, tuple[float, float, float]] | None:
        if not hasattr(entity, "trajectory"):
            return None
        trajectory = entity.trajectory
        if trajectory.kind == "curve":
            return None
        if trajectory.kind not in {"linear", "polyline"} or len(trajectory.points) < 2:
            return None
        return (
            np.asarray(trajectory.points[-1], dtype=np.float64),
            self._trajectory_endpoint_orientation(entity),
        )

    def _trajectory_endpoint_orientation(self, entity) -> tuple[float, float, float]:
        trajectory = entity.trajectory
        if len(trajectory.orientation_rad_points) >= len(trajectory.points):
            return tuple(float(value) for value in trajectory.orientation_rad_points[-1])
        if len(trajectory.orientation_rad_points) == 1:
            return tuple(float(value) for value in trajectory.orientation_rad_points[0])
        return tuple(float(value) for value in entity.orientation_rad)

    def _begin_trajectory_handle_drag(self, x: float, y: float) -> dict | None:
        entity = self._selected_entity()
        if entity is None or not hasattr(entity, "trajectory"):
            return None
        trajectory = entity.trajectory
        if trajectory.kind != "curve" or len(trajectory.points) < 2:
            return None
        picked = self._pick_trajectory_handle(x, y, trajectory)
        if picked is None:
            return None
        point_index, handle_side, start = picked
        return {
            "kind": "trajectory_handle",
            "target": "trajectory_handle",
            "entity_id": entity.id,
            "point_index": point_index,
            "handle_side": handle_side,
            "start": start,
            **self._trajectory_xy_drag_state(trajectory, start, x, y),
        }

    def _pick_trajectory_anchor(self, x: float, y: float, trajectory) -> int | None:
        best_index = None
        best_distance = None
        for point_index, point in enumerate(trajectory.points[1:], start=1):
            screen = self._project_to_screen(point)
            if screen is None:
                continue
            distance = _screen_distance((x, y), screen)
            if distance <= 16.0 and (best_distance is None or distance < best_distance):
                best_index = point_index
                best_distance = distance
        return best_index

    def _pick_trajectory_handle(self, x: float, y: float, trajectory) -> tuple[int, str, np.ndarray] | None:
        best = None
        best_distance = None
        for point_index, (handle_in, handle_out) in enumerate(normalized_bezier_handles(trajectory)):
            for side, point in (("in", handle_in), ("out", handle_out)):
                if (point_index == 0 and side == "in") or (point_index == len(trajectory.points) - 1 and side == "out"):
                    continue
                screen = self._project_to_screen(point)
                if screen is None:
                    continue
                distance = _screen_distance((x, y), screen)
                if distance <= 14.0 and (best_distance is None or distance < best_distance):
                    best = (point_index, side, np.asarray(point, dtype=np.float64))
                    best_distance = distance
        return best

    def _trajectory_xy_drag_state(self, trajectory, start: np.ndarray, x: float, y: float) -> dict:
        screen_origin = self._project_to_screen(start)
        screen_x = self._project_to_screen(start + _AXES["x"])
        screen_y = self._project_to_screen(start + _AXES["y"])
        return {
            "mouse_start": np.asarray([x, y], dtype=np.float64),
            "screen_x_axis": _screen_unit_vector(screen_origin, screen_x),
            "screen_y_axis": _screen_unit_vector(screen_origin, screen_y),
            "world_per_pixel": _trajectory_world_per_pixel(trajectory),
            "plane_z": float(trajectory.points[0][2]),
        }

    def _drag_trajectory_xy(self, x: float, y: float, state: dict) -> np.ndarray | None:
        mouse_delta = np.asarray([x, y], dtype=np.float64) - state["mouse_start"]
        x_axis = state.get("screen_x_axis")
        y_axis = state.get("screen_y_axis")
        if x_axis is None or y_axis is None:
            x_pixels, y_pixels = float(mouse_delta[0]), float(-mouse_delta[1])
        else:
            x_pixels, y_pixels = _screen_plane_components(mouse_delta, x_axis, y_axis)
        point = np.asarray(state["start"], dtype=np.float64).copy()
        point[0] += x_pixels * float(state["world_per_pixel"])
        point[1] += y_pixels * float(state["world_per_pixel"])
        point[2] = float(state["plane_z"])
        return point

    def _camera_plane_drag_basis(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        camera = self.cameraPosition()
        center = self.opts.get("center")
        if camera is None or center is None:
            return None
        camera_position = np.asarray([camera.x(), camera.y(), camera.z()], dtype=np.float64)
        center_position = np.asarray([center.x(), center.y(), center.z()], dtype=np.float64)
        view_direction = _normalized(center_position - camera_position)
        if float(np.linalg.norm(view_direction)) <= 1e-12:
            view_direction = _normalized(np.asarray(point, dtype=np.float64) - camera_position)
        world_up = _AXES["z"]
        if abs(float(np.dot(view_direction, world_up))) > 0.95:
            world_up = _AXES["y"]
        right = _normalized(np.cross(view_direction, world_up))
        up = _normalized(np.cross(right, view_direction))
        if float(np.linalg.norm(right)) <= 1e-12 or float(np.linalg.norm(up)) <= 1e-12:
            return None
        distance = float(self.opts.get("distance", 40.0)) if self._orthographic else float(
            np.linalg.norm(camera_position - point)
        )
        world_per_pixel = _screen_pixels_to_world(
            distance=max(distance, 1e-3),
            fov_degrees=float(self.opts.get("fov", 60.0)),
            viewport_height=max(float(self.height()), 1.0),
            pixels=1.0,
        )
        return right, up, world_per_pixel

    def _drag_position(self, x: float, y: float, state: dict) -> np.ndarray | None:
        if state["kind"] not in {"translate_axis", "translate_plane"}:
            return None
        mouse_delta = np.asarray([x, y], dtype=np.float64) - state["mouse_start"]
        world_per_pixel = float(state["world_per_pixel"])
        if state["kind"] == "translate_axis":
            screen_axis = state["screen_axes"].get(state["axis_name"])
            if screen_axis is None:
                return None
            axis = state["axis"]
            delta = axis * float(np.dot(mouse_delta, screen_axis)) * world_per_pixel
        else:
            plane_axes = state.get("plane_axes")
            if plane_axes is None:
                return None
            first_name, second_name = plane_axes
            first_screen = state["screen_axes"].get(first_name)
            second_screen = state["screen_axes"].get(second_name)
            if first_screen is None or second_screen is None:
                return None
            first_axis = state["local_axes"][first_name]
            second_axis = state["local_axes"][second_name]
            first_pixels, second_pixels = _screen_plane_components(mouse_delta, first_screen, second_screen)
            delta = (first_axis * first_pixels + second_axis * second_pixels) * world_per_pixel
        return state["start"] + delta

    def _drag_orientation(self, x: float, y: float, state: dict) -> np.ndarray | None:
        if state["kind"] != "rotate":
            return None
        screen_center = state.get("screen_center")
        screen_angle_start = state.get("screen_angle_start")
        if screen_center is None or screen_angle_start is None:
            return None
        screen_angle = _screen_angle(screen_center, (x, y))
        if screen_angle is None:
            return None
        delta = _wrap_angle(screen_angle - screen_angle_start) * _ROTATION_DRAG_SENSITIVITY
        orientation = np.asarray(state["start_orientation"], dtype=np.float64)
        # The model stores yaw, pitch, roll. Map local Z/Y/X ring edits to those controls.
        orientation_index = {"z": 0, "y": 1, "x": 2}[state["axis_name"]]
        orientation[orientation_index] += delta
        return orientation

    def _pick_gizmo_handle(self, x: float, y: float) -> dict | None:
        entity = self._selected_entity()
        if entity is None:
            return None
        origin = np.asarray(entity.position, dtype=np.float64)
        return self._pick_transform_handle(x, y, origin, entity.orientation_rad, self._gizmo_length())

    def _pick_transform_handle(self, x: float, y: float, origin: np.ndarray, orientation_rad, size: float) -> dict | None:
        axes = _local_axes(orientation_rad)
        best = None
        best_distance = None
        axis_threshold_px = 10.0
        ring_threshold_px = 8.0
        for plane_name, (first_axis, second_axis, normal_axis, _color) in _PLANE_HANDLES.items():
            polygon = [
                self._project_to_screen(point)
                for point in _plane_handle_points(origin, axes[first_axis], axes[second_axis], size)
            ]
            polygon = [point for point in polygon if point is not None]
            if len(polygon) >= 4 and _screen_point_in_polygon((x, y), polygon[:-1]):
                return {
                    "kind": "translate_plane",
                    "axis": normal_axis,
                    "plane": plane_name,
                    "plane_axes": (first_axis, second_axis),
                }
        for axis_name, axis in axes.items():
            axis_start = self._project_to_screen(origin)
            axis_end = self._project_to_screen(origin + axis * size)
            if axis_start is not None and axis_end is not None:
                distance = _screen_distance_to_segment((x, y), axis_start, axis_end)
                if distance <= axis_threshold_px and (best_distance is None or distance < best_distance):
                    best = {"kind": "translate_axis", "axis": axis_name}
                    best_distance = distance
            ring_points = [
                self._project_to_screen(point)
                for point in _rotation_ring(origin, axis, size * 0.82, samples=72)
            ]
            ring_points = [point for point in ring_points if point is not None]
            if len(ring_points) >= 2:
                distance = _screen_distance_to_polyline((x, y), ring_points)
                if distance <= ring_threshold_px and (best_distance is None or distance < best_distance):
                    best = {"kind": "rotate", "axis": axis_name}
                    best_distance = distance
        return best

    def _project_to_screen(self, point) -> tuple[float, float] | None:
        viewport = self.getViewport()
        viewport_rect = _viewport_rect(viewport)
        region = (0, 0, self.width(), self.height())
        projection = self.projectionMatrix(region, viewport)
        model_view = self.viewMatrix()
        projected = QVector3D(float(point[0]), float(point[1]), float(point[2])).project(
            model_view,
            projection,
            viewport_rect,
        )
        if not np.isfinite([projected.x(), projected.y(), projected.z()]).all():
            return None
        return (float(projected.x()), float(self.height() - projected.y()))

    def _mouse_ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray] | None:
        viewport = self.getViewport()
        viewport_rect = _viewport_rect(viewport)
        region = (0, 0, self.width(), self.height())
        projection = self.projectionMatrix(region, viewport)
        model_view = self.viewMatrix()
        gl_y = self.height() - y
        near = QVector3D(float(x), float(gl_y), 0.0).unproject(model_view, projection, viewport_rect)
        far = QVector3D(float(x), float(gl_y), 1.0).unproject(model_view, projection, viewport_rect)
        origin = np.asarray([near.x(), near.y(), near.z()], dtype=np.float64)
        far_point = np.asarray([far.x(), far.y(), far.z()], dtype=np.float64)
        direction = far_point - origin
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return None
        return origin, direction / norm


@lru_cache(maxsize=64)
def _load_mesh_preview(
    path_text: str,
    _mtime_value: float,
    max_faces: int = 120_000,
) -> tuple[np.ndarray, np.ndarray]:
    suffix = Path(path_text).suffix.lower()
    if suffix == ".ply":
        return _load_ply_mesh(path_text, _mtime_value, max_faces=max_faces)
    if suffix == ".obj":
        return _load_obj_mesh(path_text, _mtime_value, max_faces=max_faces)
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)


@lru_cache(maxsize=64)
def _load_ply_mesh(
    path_text: str,
    _mtime_value: float,
    max_faces: int = 120_000,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_text)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    try:
        with path.open("rb") as fh:
            header = _read_mesh_ply_header(fh)
            vertex_count = header["vertex_count"]
            if vertex_count <= 0:
                return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
            if header["format"] == "ascii":
                vertices = _read_ascii_points(fh, vertex_count, header["vertex_properties"])
                faces = _read_ascii_faces(fh, header["face_count"])
            elif header["format"] in {"binary_little_endian", "binary_big_endian"}:
                vertices = _read_binary_points(
                    fh,
                    vertex_count,
                    header["vertex_properties"],
                    header["format"],
                )
                faces = _read_binary_faces(fh, header)
            else:
                return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    except (OSError, UnicodeDecodeError, ValueError, struct.error):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    triangles = _triangulate_faces(faces)
    if len(triangles) > max_faces:
        indices = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
        triangles = triangles[indices]
    return vertices.astype(np.float32), triangles.astype(np.int32)


@lru_cache(maxsize=64)
def _load_obj_mesh(
    path_text: str,
    _mtime_value: float,
    max_faces: int = 120_000,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_text)
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    face = [_obj_vertex_index(part, len(vertices)) for part in parts]
                    face = [index for index in face if index is not None]
                    if len(face) >= 3:
                        faces.append(face)
    except (OSError, ValueError):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    if not vertices or not faces:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    triangles = _triangulate_faces(faces)
    if len(triangles) > max_faces:
        indices = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
        triangles = triangles[indices]
    return np.asarray(vertices, dtype=np.float32), triangles.astype(np.int32)


def _obj_vertex_index(token: str, vertex_count: int) -> int | None:
    if not token:
        return None
    value = token.split("/", maxsplit=1)[0]
    if not value:
        return None
    index = int(value)
    if index > 0:
        return index - 1
    if index < 0:
        return vertex_count + index
    return None


def _read_mesh_ply_header(fh) -> dict:
    header_lines: list[str] = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PLY header missing end_header")
        decoded = line.decode("ascii").strip()
        header_lines.append(decoded)
        if decoded == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Not a PLY file")

    fmt = ""
    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    face_count_type = "uchar"
    face_index_type = "int"
    element = ""
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            element = parts[1]
            if element == "vertex":
                vertex_count = int(parts[2])
            elif element == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and element == "vertex" and len(parts) >= 3:
            if parts[1] == "list":
                raise ValueError("List vertex properties are not supported")
            vertex_properties.append((parts[2], parts[1]))
        elif parts[0] == "property" and element == "face" and len(parts) >= 5 and parts[1] == "list":
            face_count_type = parts[2]
            face_index_type = parts[3]

    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "vertex_properties": vertex_properties,
        "face_count_type": face_count_type,
        "face_index_type": face_index_type,
    }


def _read_ascii_points(fh, vertex_count: int, properties: list[tuple[str, str]]) -> np.ndarray:
    indices = _xyz_property_indices(properties)
    points = np.empty((vertex_count, 3), dtype=np.float64)
    for row_idx in range(vertex_count):
        values = fh.readline().decode("ascii").split()
        points[row_idx] = [float(values[index]) for index in indices]
    return points


def _read_binary_points(
    fh,
    vertex_count: int,
    properties: list[tuple[str, str]],
    fmt: str,
) -> np.ndarray:
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
    return points


def _read_ascii_faces(fh, face_count: int) -> tuple[tuple[int, ...], ...]:
    faces = []
    for _ in range(face_count):
        values = fh.readline().decode("ascii").split()
        if not values:
            continue
        count = int(values[0])
        faces.append(tuple(int(value) for value in values[1 : 1 + count]))
    return tuple(faces)


def _read_binary_faces(fh, header: dict) -> tuple[tuple[int, ...], ...]:
    endian = "<" if header["format"] == "binary_little_endian" else ">"
    count_format = endian + _struct_format(header["face_count_type"])
    index_format = endian + _struct_format(header["face_index_type"])
    count_size = struct.calcsize(count_format)
    index_size = struct.calcsize(index_format)
    faces = []
    for _ in range(header["face_count"]):
        count_bytes = fh.read(count_size)
        if len(count_bytes) != count_size:
            raise ValueError("Unexpected end of binary PLY face counts")
        count = struct.unpack(count_format, count_bytes)[0]
        indices = []
        for _index in range(count):
            index_bytes = fh.read(index_size)
            if len(index_bytes) != index_size:
                raise ValueError("Unexpected end of binary PLY face indices")
            indices.append(int(struct.unpack(index_format, index_bytes)[0]))
        faces.append(tuple(indices))
    return tuple(faces)


def _triangulate_faces(faces: tuple[tuple[int, ...], ...]) -> np.ndarray:
    triangles = []
    for face in faces:
        if len(face) < 3:
            continue
        first = face[0]
        for index in range(1, len(face) - 1):
            triangles.append((first, face[index], face[index + 1]))
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _intersect_ray_plane(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    denominator = float(np.dot(ray_direction, plane_normal))
    if abs(denominator) <= 1e-9:
        return None
    distance = float(np.dot(plane_point - ray_origin, plane_normal) / denominator)
    return ray_origin + ray_direction * distance


def _closest_point_on_axis_to_ray(
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> np.ndarray | None:
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    axis_dot_ray = float(np.dot(axis_direction, ray_direction))
    denominator = 1.0 - axis_dot_ray * axis_dot_ray
    if abs(denominator) <= 1e-9:
        return axis_point
    delta = axis_point - ray_origin
    axis_distance = float((-np.dot(delta, axis_direction) + np.dot(delta, ray_direction) * axis_dot_ray) / denominator)
    return axis_point + axis_direction * axis_distance


def _point_to_ray_distance(point: np.ndarray, ray_origin: np.ndarray, ray_direction: np.ndarray) -> float:
    offset = point - ray_origin
    projected = ray_origin + ray_direction * max(float(np.dot(offset, ray_direction)), 0.0)
    return float(np.linalg.norm(point - projected))


def _screen_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


def _screen_distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    stop: tuple[float, float],
) -> float:
    point_array = np.asarray(point, dtype=np.float64)
    start_array = np.asarray(start, dtype=np.float64)
    stop_array = np.asarray(stop, dtype=np.float64)
    segment = stop_array - start_array
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared <= 1e-12:
        return float(np.linalg.norm(point_array - start_array))
    t = float(np.clip(np.dot(point_array - start_array, segment) / segment_length_squared, 0.0, 1.0))
    closest = start_array + segment * t
    return float(np.linalg.norm(point_array - closest))


def _screen_distance_to_polyline(
    point: tuple[float, float],
    points: list[tuple[float, float]],
) -> float:
    if len(points) < 2:
        return float("inf")
    return min(
        _screen_distance_to_segment(point, start, stop)
        for start, stop in zip(points, points[1:])
    )


def _screen_unit_vector(
    start: tuple[float, float] | None,
    stop: tuple[float, float] | None,
) -> np.ndarray | None:
    if start is None or stop is None:
        return None
    vector = np.asarray(stop, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def _screen_plane_components(
    mouse_delta: np.ndarray,
    first_screen_axis: np.ndarray,
    second_screen_axis: np.ndarray,
) -> tuple[float, float]:
    basis = np.column_stack([first_screen_axis, second_screen_axis])
    if abs(float(np.linalg.det(basis))) <= 1e-6:
        return (
            float(np.dot(mouse_delta, first_screen_axis)),
            float(np.dot(mouse_delta, second_screen_axis)),
        )
    result = np.linalg.solve(basis, mouse_delta)
    return (float(result[0]), float(result[1]))


def _screen_angle(
    center: tuple[float, float] | np.ndarray | None,
    point: tuple[float, float],
) -> float | None:
    if center is None:
        return None
    vector = np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if float(np.linalg.norm(vector)) <= 1e-6:
        return None
    return float(np.arctan2(vector[1], vector[0]))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _screen_point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    inside = False
    x, y = point
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _local_axes(orientation_rad) -> dict[str, np.ndarray]:
    yaw, pitch, roll = (float(value) for value in orientation_rad)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rotation = rz @ ry @ rx
    return {
        "x": _normalized(rotation @ _AXES["x"]),
        "y": _normalized(rotation @ _AXES["y"]),
        "z": _normalized(rotation @ _AXES["z"]),
    }


SENSING_TARGET_COLOR = (0.2, 0.4, 1.0, 0.5)


def _cuboid_mesh(dimensions) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned cuboid centered at the origin, as (vertices, triangle faces)."""
    half = np.asarray(dimensions, dtype=np.float64) / 2.0
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # -x
            [4, 6, 7], [4, 7, 5],  # +x
            [0, 4, 5], [0, 5, 1],  # -y
            [2, 3, 7], [2, 7, 6],  # +y
            [0, 2, 6], [0, 6, 4],  # -z
            [1, 5, 7], [1, 7, 3],  # +z
        ],
        dtype=np.int32,
    )
    return signs * half, faces


def _mesh_anchor(mesh, vertices: np.ndarray) -> np.ndarray:
    if getattr(mesh, "bounds", None) is not None:
        mins = np.asarray(mesh.bounds.min_xyz, dtype=np.float64)
        maxs = np.asarray(mesh.bounds.max_xyz, dtype=np.float64)
        return (mins + maxs) / 2.0
    return (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0


def _transform_mesh_vertices(
    vertices: np.ndarray,
    anchor: np.ndarray,
    position: np.ndarray,
    orientation_rad,
) -> np.ndarray:
    axes = _local_axes(orientation_rad)
    rotation = np.column_stack([axes["x"], axes["y"], axes["z"]])
    local_vertices = np.asarray(vertices, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    return local_vertices @ rotation.T + np.asarray(position, dtype=np.float64)


def _pointing_vector(orientation_rad) -> np.ndarray:
    """Return the local +X boresight direction used as the GUI pointing vector."""
    return _local_axes(orientation_rad)["x"]


def _plane_handle_points(
    origin: np.ndarray,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    size: float,
) -> np.ndarray:
    inner = size * 0.24
    outer = size * 0.48
    points = [
        origin + first_axis * inner + second_axis * inner,
        origin + first_axis * outer + second_axis * inner,
        origin + first_axis * outer + second_axis * outer,
        origin + first_axis * inner + second_axis * outer,
        origin + first_axis * inner + second_axis * inner,
    ]
    return np.asarray(points, dtype=np.float64)


def _arrowhead_segments(
    tip: np.ndarray,
    forward: np.ndarray,
    side: np.ndarray,
    up: np.ndarray,
    length: float,
) -> np.ndarray:
    forward = _normalized(forward)
    side = _normalized(side)
    up = _normalized(up)
    head_length = length * 0.28
    head_width = length * 0.12
    base = tip - forward * head_length
    points = [
        tip,
        base + side * head_width,
        tip,
        base - side * head_width,
        tip,
        base + up * head_width,
        tip,
        base - up * head_width,
    ]
    return np.asarray(points, dtype=np.float64)


def _rotation_ring(origin: np.ndarray, normal: np.ndarray, radius: float, samples: int = 96) -> np.ndarray:
    normal = _normalized(np.asarray(normal, dtype=np.float64))
    reference = _AXES["z"] if abs(float(np.dot(normal, _AXES["z"]))) < 0.9 else _AXES["x"]
    tangent = _normalized(np.cross(normal, reference))
    bitangent = _normalized(np.cross(normal, tangent))
    angles = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    points = [
        origin + radius * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
        for angle in angles
    ]
    return np.asarray(points, dtype=np.float32)


def _signed_angle(start_vector: np.ndarray, current_vector: np.ndarray, axis: np.ndarray) -> float:
    start = _normalized(start_vector)
    current = _normalized(current_vector)
    axis = _normalized(axis)
    sine = float(np.dot(axis, np.cross(start, current)))
    cosine = float(np.clip(np.dot(start, current), -1.0, 1.0))
    return float(np.arctan2(sine, cosine))


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _screen_pixels_to_world(
    *,
    distance: float,
    fov_degrees: float,
    viewport_height: float,
    pixels: float,
) -> float:
    visible_height = distance * 2.0 * np.tan(0.5 * np.deg2rad(fov_degrees))
    return float(visible_height * pixels / max(viewport_height, 1.0))


def _radiomap_corner_world_per_pixel(config: RadiomapConfig) -> float:
    span = max(abs(float(config.x_max) - float(config.x_min)), abs(float(config.y_max) - float(config.y_min)), 10.0)
    return span / 500.0


def _trajectory_preview_positions(trajectory: TrajectorySpec) -> np.ndarray:
    if trajectory.kind != "curve":
        return np.asarray(trajectory.points, dtype=np.float64)
    preview = TrajectorySpec(
        kind="curve",
        points=list(trajectory.points),
        bezier_handles=normalized_bezier_handles(trajectory),
        samples=max(96, int(trajectory.samples)),
        start_static_fraction=0.0,
        end_static_fraction=0.0,
        easing="linear",
    )
    return sample_trajectory(preview)


def _trajectory_world_per_pixel(trajectory) -> float:
    points = list(trajectory.points)
    if trajectory.kind == "curve":
        for handle_in, handle_out in normalized_bezier_handles(trajectory):
            points.extend([handle_in, handle_out])
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return 0.01
    span = np.ptp(values[:, :2], axis=0)
    return max(float(np.max(span)), 5.0) / 420.0


def _curve_tangent_arrow_segments(positions: np.ndarray, length: float, max_arrows: int = 12) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return np.empty((0, 3), dtype=np.float64)
    count = min(max_arrows, max(1, len(positions) // 8))
    indices = np.linspace(0, len(positions) - 1, count, dtype=np.int64)
    tangents = np.zeros_like(positions)
    tangents[0] = positions[1] - positions[0]
    tangents[-1] = positions[-1] - positions[-2]
    if len(positions) > 2:
        tangents[1:-1] = positions[2:] - positions[:-2]
    tangents[:, 2] = 0.0
    segments = []
    for index in indices:
        tangent = _normalized(tangents[index])
        if float(np.linalg.norm(tangent)) <= 1e-12:
            continue
        origin = positions[index]
        tip = origin + tangent * length
        side = _normalized(np.cross(_AXES["z"], tangent))
        if float(np.linalg.norm(side)) <= 1e-12:
            side = _AXES["y"]
        segments.extend([origin, tip])
        segments.extend(_arrowhead_segments(tip, tangent, side, _AXES["z"], length * 0.55))
    return np.asarray(segments, dtype=np.float64)


def _axis_view_angles(axis: str, sign: int) -> tuple[float, float]:
    normalized_axis = axis.lower()
    normalized_sign = 1 if sign >= 0 else -1
    if normalized_axis == "z":
        return (90.0 * normalized_sign, -90.0)
    if normalized_axis == "x":
        return (0.0, 0.0 if normalized_sign > 0 else 180.0)
    if normalized_axis == "y":
        return (0.0, 90.0 if normalized_sign > 0 else -90.0)
    raise ValueError(f"Unsupported axis view: {axis}")


def _antenna_diagram_item(
    *,
    origin: np.ndarray,
    orientation_rad,
    pattern_name: str,
    color_bias: tuple[float, float, float],
) -> gl.GLMeshItem:
    vertices, faces, face_colors = _antenna_diagram_mesh(
        origin=origin,
        orientation_rad=orientation_rad,
        pattern_name=pattern_name,
        color_bias=color_bias,
    )
    return _lobe_mesh_item(vertices, faces, face_colors)


def _lobe_mesh_item(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray) -> gl.GLMeshItem:
    mesh_data = gl.MeshData(
        vertexes=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        faceColors=face_colors.astype(np.float32),
    )
    return gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=False,
        drawFaces=True,
        drawEdges=False,
        shader="balloon",
        glOptions="translucent",
    )


def _select_important_bs_to_ue_paths(
    links,
    device_positions: dict,
    *,
    max_paths: int,
    min_delay_separation_s: float,
    min_range_m: float | None = None,
    max_range_m: float | None = None,
    tx_id: str | None = None,
    rx_id: str | None = None,
) -> tuple[list[dict], dict]:
    stats = {
        "links_total": len(links),
        "bs_to_ue_links": 0,
        "links_with_positions": 0,
        "links_with_path_metadata": 0,
        "candidate_paths": 0,
        "selected_paths": 0,
        "range_filter_min_m": None if min_range_m is None else float(min_range_m),
        "range_filter_max_m": None if max_range_m is None else float(max_range_m),
        "tx_filter": tx_id or "all",
        "rx_filter": rx_id or "all",
        "paths_after_range_filter": 0,
        "paths_rejected_geometry_mismatch": 0,
    }
    candidates = []
    for link in links:
        if not (str(link.tx_id).startswith("bs") and str(link.rx_id).startswith("ue")):
            continue
        if tx_id is not None and str(link.tx_id) != str(tx_id):
            continue
        if rx_id is not None and str(link.rx_id) != str(rx_id):
            continue
        stats["bs_to_ue_links"] += 1
        tx_pos = device_positions.get(link.tx_id)
        rx_pos = device_positions.get(link.rx_id)
        if tx_pos is None or rx_pos is None:
            continue
        stats["links_with_positions"] += 1
        delays = link.metadata.get("path_delays_s")
        coeffs = link.metadata.get("path_coefficients")
        if delays is None or coeffs is None:
            continue
        stats["links_with_path_metadata"] += 1
        vertices = link.metadata.get("path_vertices")
        path_rows = _flatten_path_metrics(delays, coeffs)
        stats["candidate_paths"] += len(path_rows)
        if not path_rows:
            continue
        best_by_delay_bin = {}
        for path_index, delay_s, power, rx_ant_index, tx_ant_index in path_rows:
            if not np.isfinite(delay_s) or delay_s < 0.0 or not np.isfinite(power) or power <= 0.0:
                continue
            range_m = delay_s * 299_792_458.0
            if min_range_m is not None and range_m < float(min_range_m):
                continue
            if max_range_m is not None and range_m > float(max_range_m):
                continue
            stats["paths_after_range_filter"] += 1
            bin_key = int(round(delay_s / max(min_delay_separation_s, 1e-12)))
            previous = best_by_delay_bin.get(bin_key)
            if previous is None or power > previous["power"]:
                best_by_delay_bin[bin_key] = {
                    "link": link,
                    "path_index": int(path_index),
                    "delay_s": float(delay_s),
                    "power": float(power),
                    "range_m": float(range_m),
                    "tx_pos": np.asarray(tx_pos, dtype=np.float64),
                    "rx_pos": np.asarray(rx_pos, dtype=np.float64),
                    "vertices": vertices,
                    "rx_ant_index": int(rx_ant_index),
                    "tx_ant_index": int(tx_ant_index),
                }
        candidates.extend(best_by_delay_bin.values())
    candidates.sort(key=lambda item: item["power"], reverse=True)
    selected: list[dict] = []
    used_delays_by_link: dict[tuple[str, str], list[float]] = {}
    for candidate in candidates:
        link_key = (str(candidate["link"].tx_id), str(candidate["link"].rx_id))
        used_delays = used_delays_by_link.setdefault(link_key, [])
        if any(abs(candidate["delay_s"] - value) < min_delay_separation_s for value in used_delays):
            continue
        points = _path_polyline_points(candidate)
        if points is None or len(points) < 2:
            continue
        geometry_range_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        expected_range_m = float(candidate["range_m"])
        tolerance_m = max(0.25, 0.01 * expected_range_m)
        if abs(geometry_range_m - expected_range_m) > tolerance_m:
            stats["paths_rejected_geometry_mismatch"] += 1
            continue
        used_delays.append(candidate["delay_s"])
        selected.append({**candidate, "points": points, "geometry_range_m": geometry_range_m})
        if len(selected) >= max_paths:
            break
    if not selected:
        return [], stats
    max_power = max(item["power"] for item in selected)
    min_power = min(item["power"] for item in selected)
    span_db = max(1.0, 10.0 * np.log10(max(max_power, 1e-30) / max(min_power, 1e-30)))
    out = []
    for item in selected:
        rel_db = 10.0 * np.log10(max(item["power"], 1e-30) / max(max_power, 1e-30))
        normalized = float(np.clip(1.0 + rel_db / max(span_db, 1.0), 0.0, 1.0))
        out.append(
            {
                "points": item["points"],
                "color": _path_color(normalized),
                "width": 1.5 + 4.5 * normalized,
                "tx_id": str(item["link"].tx_id),
                "rx_id": str(item["link"].rx_id),
                "range_m": float(item["range_m"]),
                "geometry_range_m": float(item["geometry_range_m"]),
                "delay_ns": float(item["delay_s"] * 1e9),
                "power": float(item["power"]),
            }
        )
    stats["selected_paths"] = len(out)
    stats["selected_range_min_m"] = min(item["range_m"] for item in selected)
    stats["selected_range_max_m"] = max(item["range_m"] for item in selected)
    return out, stats


def _flatten_path_metrics(delays, coeffs) -> list[tuple[int, float, float, int, int]]:
    delays_arr = np.asarray(delays, dtype=np.float64)
    coeffs_arr = np.asarray(coeffs)
    if delays_arr.size == 0 or coeffs_arr.size == 0:
        return []
    path_count = delays_arr.shape[-1]
    delays_flat = delays_arr.reshape((-1, path_count))
    if coeffs_arr.shape[-1] != path_count:
        return []
    coeffs_flat = coeffs_arr.reshape((-1, path_count))
    if coeffs_flat.shape[0] != delays_flat.shape[0]:
        return []
    prefix_shape = delays_arr.shape[:-1]
    if len(prefix_shape) < 4:
        return []
    rx_cols = int(prefix_shape[1])
    tx_cols = int(prefix_shape[3])
    out = []
    for idx in range(path_count):
        delay_column = delays_flat[:, idx]
        coeff_column = coeffs_flat[:, idx]
        coeff_power = np.abs(coeff_column) ** 2
        valid = np.isfinite(delay_column) & (delay_column >= 0.0) & np.isfinite(coeff_power) & (coeff_power > 0.0)
        if not np.any(valid):
            continue
        valid_rows = np.flatnonzero(valid)
        row_index = int(valid_rows[np.argmax(coeff_power[valid_rows])])
        antenna_index = np.unravel_index(row_index, prefix_shape)
        rx_ant_index = int(antenna_index[0]) * rx_cols + int(antenna_index[1])
        tx_ant_index = int(antenna_index[2]) * tx_cols + int(antenna_index[3])
        out.append(
            (
                idx,
                float(delay_column[row_index]),
                float(coeff_power[row_index]),
                rx_ant_index,
                tx_ant_index,
            )
        )
    return out


def _path_polyline_points(candidate: dict) -> np.ndarray | None:
    tx_pos = np.asarray(candidate["tx_pos"], dtype=np.float64)
    rx_pos = np.asarray(candidate["rx_pos"], dtype=np.float64)
    points = [tx_pos]
    vertices = candidate.get("vertices")
    path_index = int(candidate.get("path_index", 0))
    if vertices is not None:
        arr = np.asarray(vertices, dtype=np.float64)
        rx_ant_index = int(candidate.get("rx_ant_index", 0))
        tx_ant_index = int(candidate.get("tx_ant_index", 0))
        if (
            arr.ndim == 5
            and arr.shape[-1] == 3
            and 0 <= rx_ant_index < arr.shape[0]
            and 0 <= tx_ant_index < arr.shape[1]
            and 0 <= path_index < arr.shape[2]
        ):
            interactions = arr[rx_ant_index, tx_ant_index, path_index]
            finite = np.all(np.isfinite(interactions), axis=1)
            interactions = interactions[finite]
            if interactions.size:
                points.extend(interactions)
    points.append(rx_pos)
    polyline = np.asarray(points, dtype=np.float64)
    if len(polyline) < 2 or not np.all(np.isfinite(polyline)):
        return None
    return polyline


def _path_color(normalized_power: float) -> tuple[float, float, float, float]:
    value = float(np.clip(normalized_power, 0.0, 1.0))
    low = np.asarray([0.05, 0.35, 1.0], dtype=np.float64)
    high = np.asarray([1.0, 0.08, 0.02], dtype=np.float64)
    rgb = low * (1.0 - value) + high * value
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.35 + 0.55 * value)


_SCATTERING_LOBE_COLOR = (0.15, 0.95, 0.35)


def _scattering_point_lobe_meshes(
    obj,
    dimensions,
    *,
    alpha_samples: int = 25,
    beta_samples: int = 48,
    frequency_hz: float = 12.5e9,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One lobe mesh per scattering point of a sensing target.

    Each lobe is the monostatic RCS seen from every direction, normalized to the
    strongest value over all points of the target so that their relative
    strengths stay visible.
    """
    from isac_6d_sampler.sim.rcs_patterns import monostatic_scattering_patterns

    directions, _ = _diagram_local_directions(max(int(alpha_samples), 5), max(int(beta_samples), 8))
    patterns = monostatic_scattering_patterns(
        sensing_target_type(obj.object_name).object_type,
        sensing_options(obj).model_type,
        dimensions,
        directions,
        sensing_model=sensing_target_type(obj.object_name).model,
        parameter_seed=sensing_options(obj).msc_parameter_seed,
        frequency_hz=frequency_hz,
    )
    peak_dbsm = max(float(np.max(pattern.rcs_dbsm)) for pattern in patterns)
    radius_m = max(1.0, 0.6 * max(dimensions))
    axes = _local_axes(obj.orientation_rad)
    rotation = np.column_stack([axes["x"], axes["y"], axes["z"]])
    center = np.asarray(obj.position, dtype=np.float64)
    return [
        _antenna_diagram_mesh(
            origin=center + rotation @ np.asarray(pattern.lcs_position, dtype=np.float64),
            orientation_rad=obj.orientation_rad,
            gain_db=pattern.rcs_dbsm - peak_dbsm,
            radius_m=radius_m,
            color_bias=_SCATTERING_LOBE_COLOR,
            alpha_samples=alpha_samples,
            beta_samples=beta_samples,
        )
        for pattern in patterns
    ]


def _diagram_local_directions(alpha_samples: int, beta_samples: int) -> tuple[np.ndarray, list[slice]]:
    """Unit directions of a lobe mesh: poles along local +/-x, rings around the x-axis.

    Returns all directions stacked, plus the slice of each pole/ring in that array.
    """
    alpha = np.linspace(0.0, np.pi, alpha_samples, dtype=np.float64)
    beta = np.linspace(0.0, 2.0 * np.pi, beta_samples, endpoint=False, dtype=np.float64)
    blocks = []
    slices = []
    offset = 0
    for a_idx, a_value in enumerate(alpha):
        if a_idx in {0, alpha_samples - 1}:
            block = np.asarray([[np.cos(a_value), 0.0, np.sin(a_value)]], dtype=np.float64)
        else:
            block = np.stack(
                [
                    np.full_like(beta, np.cos(a_value)),
                    np.sin(a_value) * np.cos(beta),
                    np.sin(a_value) * np.sin(beta),
                ],
                axis=-1,
            )
        blocks.append(block)
        slices.append(slice(offset, offset + block.shape[0]))
        offset += block.shape[0]
    return np.concatenate(blocks, axis=0), slices


def _antenna_diagram_mesh(
    *,
    origin: np.ndarray,
    orientation_rad,
    pattern_name: str = "iso",
    color_bias: tuple[float, float, float] = (0.2, 0.5, 1.0),
    alpha_samples: int = 25,
    beta_samples: int = 48,
    gain_db: np.ndarray | None = None,
    radius_m: float = _ANTENNA_DIAGRAM_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lobe mesh of a normalized pattern (0 dB maximum, clipped at the dB floor).

    The pattern is ``pattern_name`` unless ``gain_db`` gives the values directly,
    one per direction of ``_diagram_local_directions(alpha_samples, beta_samples)``.
    """
    alpha_samples = max(int(alpha_samples), 5)
    beta_samples = max(int(beta_samples), 8)
    local_dirs, slices = _diagram_local_directions(alpha_samples, beta_samples)
    if gain_db is None:
        gain_db = _antenna_normalized_gain_db(pattern_name, local_dirs)
    gain_db = np.clip(np.asarray(gain_db, dtype=np.float64), _ANTENNA_DIAGRAM_DB_FLOOR, 0.0)
    normalized_all = (gain_db - _ANTENNA_DIAGRAM_DB_FLOOR) / abs(_ANTENNA_DIAGRAM_DB_FLOOR)
    radius_all = float(radius_m) * np.maximum(normalized_all, 0.04)
    axes = _local_axes(orientation_rad)
    world_all = (
        local_dirs[:, 0, None] * axes["x"]
        + local_dirs[:, 1, None] * axes["y"]
        + local_dirs[:, 2, None] * axes["z"]
    )
    origin_arr = np.asarray(origin, dtype=np.float64)

    vertices: list[np.ndarray] = []
    normalized_values: list[float] = []
    ring_indices: list[list[int]] = []
    for block in slices:
        ring = []
        for idx in range(block.start, block.stop):
            vertices.append(origin_arr + world_all[idx] * radius_all[idx])
            normalized_values.append(float(normalized_all[idx]))
            ring.append(len(vertices) - 1)
        ring_indices.append(ring)

    faces = []
    colors = []
    for a_idx in range(alpha_samples - 1):
        current = ring_indices[a_idx]
        following = ring_indices[a_idx + 1]
        if len(current) == 1:
            pole = current[0]
            for b_idx in range(beta_samples):
                p1 = following[b_idx]
                p2 = following[(b_idx + 1) % beta_samples]
                _append_non_degenerate_antenna_face(faces, colors, vertices, normalized_values, (pole, p1, p2), color_bias)
        elif len(following) == 1:
            pole = following[0]
            for b_idx in range(beta_samples):
                p0 = current[b_idx]
                p1 = current[(b_idx + 1) % beta_samples]
                _append_non_degenerate_antenna_face(faces, colors, vertices, normalized_values, (p0, p1, pole), color_bias)
        else:
            for b_idx in range(beta_samples):
                p0 = current[b_idx]
                p1 = current[(b_idx + 1) % beta_samples]
                p2 = following[b_idx]
                p3 = following[(b_idx + 1) % beta_samples]
                _append_non_degenerate_antenna_face(faces, colors, vertices, normalized_values, (p0, p2, p1), color_bias)
                _append_non_degenerate_antenna_face(faces, colors, vertices, normalized_values, (p1, p2, p3), color_bias)

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32), np.asarray(colors, dtype=np.float32)


def _append_non_degenerate_antenna_face(
    faces: list[tuple[int, int, int]],
    colors: list[tuple[float, float, float, float]],
    vertices: list[np.ndarray],
    normalized_values: list[float],
    face: tuple[int, int, int],
    color_bias: tuple[float, float, float],
) -> None:
    p0, p1, p2 = (vertices[index] for index in face)
    area = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
    if area <= 1e-10:
        return
    faces.append(face)
    face_gain = float(np.mean([normalized_values[index] for index in face]))
    colors.append(_antenna_diagram_color(face_gain, color_bias))


def _antenna_normalized_gain_db(pattern_name: str, local_dirs: np.ndarray) -> np.ndarray:
    try:
        spec = antenna_pattern_spec(pattern_name)
    except ValueError:
        spec = antenna_pattern_spec("iso")
    x = np.clip(local_dirs[..., 0], -1.0, 1.0)
    y = local_dirs[..., 1]
    z = local_dirs[..., 2]
    if spec.name == "iso":
        return np.zeros_like(x, dtype=np.float64)
    if spec.name == POWERLOG_PATTERN:
        return powerlog_normalized_gain_db_from_local_dirs(local_dirs)
    if spec.name == QOM_OMNI_PATTERN:
        return qom_omni_normalized_gain_db_from_local_dirs(local_dirs)
    if spec.name in {"dipole", "hw_dipole"}:
        field = np.clip(0.5 * (x + 1.0), 10.0 ** (_ANTENNA_DIAGRAM_DB_FLOOR / 20.0), 1.0)
        if spec.name == "hw_dipole":
            field = np.sqrt(field)
        return 20.0 * np.log10(field)
    horizontal_bw = float(spec.h_3db_beamwidth_deg or 65.0)
    vertical_bw = float(spec.v_3db_beamwidth_deg or 65.0)
    phi_deg = np.rad2deg(np.arctan2(y, x))
    theta_deg = np.rad2deg(np.arctan2(z, np.maximum(np.sqrt(x * x + y * y), 1e-12)))
    horizontal_attenuation = 12.0 * (phi_deg / horizontal_bw) ** 2
    vertical_attenuation = 12.0 * (theta_deg / vertical_bw) ** 2
    attenuation = np.minimum(horizontal_attenuation + vertical_attenuation, abs(_ANTENNA_DIAGRAM_DB_FLOOR))
    return -attenuation


def _antenna_diagram_color(normalized_gain: float, color_bias: tuple[float, float, float]) -> tuple[float, float, float, float]:
    value = float(np.clip(normalized_gain, 0.0, 1.0))
    low = np.asarray([0.08, 0.20, 0.42], dtype=np.float64)
    high = np.asarray(color_bias, dtype=np.float64)
    rgb = np.clip(low * (1.0 - value) + high * value, 0.0, 1.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.36 + 0.42 * value)


def _viewport_rect(viewport) -> QRect:
    if isinstance(viewport, QRect):
        return viewport
    x, y, width, height = viewport
    return QRect(int(x), int(y), int(width), int(height))


def _mesh_color(name: str) -> tuple[float, float, float, float]:
    palette = {
        "ground": (0.72, 0.72, 0.68, 1.0),
        "grass": (0.38, 0.76, 0.42, 1.0),
        "gravel": (0.64, 0.61, 0.55, 1.0),
        "concrete": (0.70, 0.72, 0.76, 1.0),
        "bricks": (0.78, 0.42, 0.34, 1.0),
        "glass": (0.48, 0.78, 0.92, 0.74),
        "metal": (0.68, 0.70, 0.76, 1.0),
        "plastic": (0.88, 0.80, 0.32, 1.0),
        "car": (0.24, 0.42, 0.88, 1.0),
    }
    lowered = name.lower()
    for key, color in palette.items():
        if key in lowered:
            return color
    return (0.62, 0.66, 0.74, 1.0)
