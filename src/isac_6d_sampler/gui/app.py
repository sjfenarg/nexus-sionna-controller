from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

import numpy as np

from isac_6d_sampler.core.antenna_patterns import available_pattern_names
from isac_6d_sampler.core.config_io import read_request, write_request
from isac_6d_sampler.core.estimates import estimate_request_size, format_request_estimate
from isac_6d_sampler.core.frequencies import format_band_specs, parse_band_specs
from isac_6d_sampler.core.ids import unique_entity_id
from isac_6d_sampler.core.model import (
    BaseStation,
    ChannelMode,
    DEFAULT_HALF_WAVELENGTH_SPACING_M,
    DYNAMIC_SCENE_OBJECT_NAMES,
    DynamicObject,
    RadiomapConfig,
    SceneDesign,
    SimulationRequest,
    TrajectorySpec,
    UserEquipment,
)
from isac_6d_sampler.core.scenarios import discover_scenarios, load_scenario_asset
from isac_6d_sampler.core.trajectory_specs import (
    anchor_trajectory,
    build_trajectory_spec,
    format_bezier_handles,
    format_point_list,
    translate_trajectory,
)
from isac_6d_sampler.core.trajectories import default_bezier_handles, normalized_bezier_handles
from isac_6d_sampler.core.validation import validate_request


def main() -> int:
    try:
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QPushButton,
            QProgressBar,
            QScrollArea,
            QSpinBox,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        from isac_6d_sampler.gui.channel_visualizer import ChannelVisualizerWindow
        from isac_6d_sampler.gui.scene_3d import Scene3DView
        from isac_6d_sampler.gui.worker import SimulationWorker
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the GUI extra with: python -m pip install -e .[gui]") from exc

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ISAC 6D Sampler")
            self.scenario_root = Path("scenarios")
            self.output_dir = Path("output")
            self.output_dir_text = QLineEdit(str(self.output_dir))
            self.design = SceneDesign()
            self.design.ensure_defaults()
            self.undo_stack = []
            self.worker = None
            self.channel_visualizer = None
            self._simulation_preview_design = None
            self._last_simulation_result = None
            self._syncing_controls = False

            self.scenario_assets = discover_scenarios(self.scenario_root)
            self.scenario_combo = QComboBox()
            for asset in self.scenario_assets:
                self.scenario_combo.addItem(f"{asset.name} ({asset.mesh_count} meshes)", str(asset.path))
            self.object_name = QComboBox()
            self._refresh_object_choices()

            self.entity_list = QListWidget()
            self.view = Scene3DView(
                on_selected=self._select_entity_by_id,
                on_moved=self._transform_entity_from_view,
                on_transform_started=self._begin_view_transform,
                on_trajectory_point_moved=self._move_trajectory_point_from_view,
                on_trajectory_handle_moved=self._move_trajectory_handle_from_view,
                on_trajectory_endpoint_transformed=self._transform_trajectory_endpoint_from_view,
                on_trajectory_transform_started=self._begin_view_transform,
                on_radiomap_bounds_changed=self._set_radiomap_bounds_from_view,
            )
            undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
            undo_shortcut.activated.connect(self.undo)
            view_buttons = QHBoxLayout()
            for label, axis, sign in (
                ("XY +Z", "z", 1),
                ("XY -Z", "z", -1),
                ("YZ +X", "x", 1),
                ("YZ -X", "x", -1),
                ("XZ +Y", "y", 1),
                ("XZ -Y", "y", -1),
            ):
                button = QPushButton(label)
                button.setObjectName(f"view_{label.lower().replace(' ', '_').replace('+', 'pos').replace('-', 'neg')}")
                button.clicked.connect(lambda _checked=False, a=axis, s=sign: self.view.set_axis_view(a, s))
                view_buttons.addWidget(button)
            antenna_button = QPushButton("Antennas")
            antenna_button.setObjectName("view_antenna_diagrams")
            antenna_button.clicked.connect(self.show_antenna_diagrams)
            view_buttons.addWidget(antenna_button)
            self.paths_button = QPushButton("Paths")
            self.paths_button.setObjectName("view_simulation_paths")
            self.paths_button.clicked.connect(self.show_simulation_paths)
            self.path_min_range = _double_spin(0.0, 10_000.0, 0.0, decimals=2)
            self.path_min_range.setObjectName("path_min_range_m")
            self.path_max_range = _double_spin(0.0, 10_000.0, 150.0, decimals=2)
            self.path_max_range.setObjectName("path_max_range_m")
            self.path_link_combo = QComboBox()
            self.path_link_combo.setObjectName("path_link_combo")
            self.path_link_combo.addItem("All BS->UE", None)
            view_buttons.addWidget(self.paths_button)
            view_buttons.addWidget(QLabel("Path range m"))
            view_buttons.addWidget(self.path_min_range)
            view_buttons.addWidget(self.path_max_range)
            view_buttons.addWidget(QLabel("Link"))
            view_buttons.addWidget(self.path_link_combo)
            view_buttons.addStretch(1)

            self.bs_rows = _spin(1, 128, 10)
            self.bs_cols = _spin(1, 128, 10)
            self.bs_v_spacing = _double_spin(0.0, 100.0, DEFAULT_HALF_WAVELENGTH_SPACING_M, decimals=6)
            self.bs_h_spacing = _double_spin(0.0, 100.0, DEFAULT_HALF_WAVELENGTH_SPACING_M, decimals=6)
            self.ue_rows = _spin(1, 64, 1)
            self.ue_cols = _spin(1, 64, 1)
            self.ue_v_spacing = _double_spin(0.0, 100.0, 0.0, decimals=6)
            self.ue_h_spacing = _double_spin(0.0, 100.0, 0.0, decimals=6)
            self.bs_pattern = QComboBox()
            self.bs_pattern.addItems(list(available_pattern_names()))
            self.bs_polarization = QComboBox()
            self.bs_polarization.addItems(["V", "H", "VH", "cross"])
            self.ue_pattern = QComboBox()
            self.ue_pattern.addItems(list(available_pattern_names()))
            self.ue_polarization = QComboBox()
            self.ue_polarization.addItems(["V", "H", "VH", "cross"])
            self.bs_antenna_selected = QLabel("Selected BS: none")
            self.ue_antenna_selected = QLabel("Selected UE: none")
            self.pos_x = _double_spin(-10_000.0, 10_000.0, 0.0)
            self.pos_y = _double_spin(-10_000.0, 10_000.0, 0.0)
            self.pos_z = _double_spin(-10_000.0, 10_000.0, 1.5)
            self.yaw = _double_spin(-6.283, 6.283, 0.0)
            self.pitch = _double_spin(-6.283, 6.283, 0.0)
            self.roll = _double_spin(-6.283, 6.283, 0.0)
            self.samples = _spin(1, 10000, 8)
            self.trajectory_kind = QComboBox()
            self.trajectory_kind.addItems(["static", "linear", "polyline", "curve"])
            self.trajectory_points = QLineEdit("0,0,1.5; 5,0,1.5")
            self.trajectory_handles = QLineEdit("")
            self.trajectory_easing = QComboBox()
            self.trajectory_easing.addItems(["linear", "smoothstep"])
            self.start_static = _double_spin(0.0, 0.95, 0.0)
            self.end_static = _double_spin(0.0, 0.95, 0.0)
            self.add_curve_point = QPushButton("Add Middle")
            self.add_curve_point.setObjectName("trajectory_add_middle")
            self.add_curve_point.clicked.connect(self.add_curve_middle_point)
            self.remove_curve_point = QPushButton("Remove Middle")
            self.remove_curve_point.setObjectName("trajectory_remove_middle")
            self.remove_curve_point.clicked.connect(self.remove_curve_middle_point)
            self.samples_per_src = _spin(1, 10_000_000, 500_000)
            self.max_num_paths_per_src = _spin(1, 10_000_000, 100_000)
            self.max_num_paths_per_src.setObjectName("max_num_paths_per_src")
            self.max_paths_unlimited = QCheckBox()
            self.max_paths_unlimited.setObjectName("max_paths_unlimited")
            self.max_paths_unlimited.toggled.connect(self.max_num_paths_per_src.setDisabled)
            self.max_depth = _spin(0, 20, 3)
            self.max_depth.setObjectName("max_depth")
            self.batch_timeframes = _spin(1, 10_000, 1)
            self.max_timeframes = _spin(1, 1_000_000_000, 100_000)
            self.seed = _spin(-1, 2_147_483_647, -1)
            self.los = QCheckBox()
            self.los.setChecked(True)
            self.specular = QCheckBox()
            self.specular.setChecked(True)
            self.diffuse = QCheckBox()
            self.diffuse.setChecked(True)
            self.refraction = QCheckBox()
            self.refraction.setChecked(False)
            self.ue_ue_links = QCheckBox()
            self.ue_ue_links.setChecked(False)
            self.synthetic_array = QCheckBox()
            self.synthetic_array.setChecked(False)
            self.merge_shapes = QCheckBox()
            self.merge_shapes.setChecked(False)
            self.use_gpu = QCheckBox()
            self.use_gpu.setChecked(True)
            self.dry_run = QCheckBox()
            self.dry_run.setChecked(False)
            self.channel_mode = QComboBox()
            for mode in ChannelMode:
                self.channel_mode.addItem(mode.value, mode.value)
            self.channel_visualizer_enabled = QCheckBox()
            self.channel_visualizer_enabled.setObjectName("channel_visualizer_enabled")
            self.band_specs = QLineEdit("77:81:1024:77-81GHz")
            self.tx_power_dbm = _double_spin(-200.0, 100.0, 44.0, decimals=2)
            self.tx_power_dbm.setObjectName("tx_power_dbm")

            self.radiomap_enabled = QCheckBox()
            self.rm_x_min = _double_spin(-1000.0, 1000.0, -10.0)
            self.rm_x_max = _double_spin(-1000.0, 1000.0, 10.0)
            self.rm_y_min = _double_spin(-1000.0, 1000.0, -10.0)
            self.rm_y_max = _double_spin(-1000.0, 1000.0, 10.0)
            self.rm_x_spacing = _double_spin(0.05, 100.0, 1.0)
            self.rm_y_spacing = _double_spin(0.05, 100.0, 1.0)
            self.rm_height = _double_spin(-100.0, 100.0, 1.5)

            self.progress = QProgressBar()
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMinimumHeight(120)

            add_bs = QPushButton("Add BS")
            add_bs.clicked.connect(self.add_bs)
            add_ue = QPushButton("Add UE")
            add_ue.clicked.connect(self.add_ue)
            add_obj = QPushButton("Add Object")
            add_obj.clicked.connect(self.add_object)
            remove_entity = QPushButton("Remove")
            remove_entity.clicked.connect(self.remove_selected_entity)
            apply_entity = QPushButton("Apply Entity")
            apply_entity.clicked.connect(self.apply_entity)
            apply_trajectory = QPushButton("Apply Traj")
            apply_trajectory.clicked.connect(self.apply_trajectory)
            apply_radiomap_template = QPushButton("Apply RM Template")
            apply_radiomap_template.clicked.connect(self.apply_radiomap_template)
            load_config = QPushButton("Load")
            load_config.clicked.connect(self.load_config)
            save_config = QPushButton("Save")
            save_config.clicked.connect(self.save_config)
            browse_output = QPushButton("Output Dir")
            browse_output.clicked.connect(self.browse_output_dir)
            self.simulate_button = QPushButton("Simulate")
            self.simulate_button.clicked.connect(self.simulate)

            project_section = _section(
                "Project",
                [
                    ("Scenario", self.scenario_combo),
                    ("Output dir", self.output_dir_text),
                    ("Object mesh", self.object_name),
                ],
                object_name="config_section_project",
            )
            pose_section = _section(
                "Selected Entity Pose",
                [
                    ("Position x m", self.pos_x),
                    ("Position y m", self.pos_y),
                    ("Position z m", self.pos_z),
                    ("Yaw rad", self.yaw),
                    ("Pitch rad", self.pitch),
                    ("Roll rad", self.roll),
                ],
                object_name="config_section_pose",
            )
            bs_antenna_section = _section(
                "BS Antenna Panel",
                [
                    ("Selected", self.bs_antenna_selected),
                    ("BS rows", self.bs_rows),
                    ("BS cols", self.bs_cols),
                    ("BS v spacing", self.bs_v_spacing),
                    ("BS h spacing", self.bs_h_spacing),
                    ("BS pattern", self.bs_pattern),
                    ("BS polarization", self.bs_polarization),
                ],
                object_name="config_section_bs_antenna",
            )
            ue_antenna_section = _section(
                "UE Antenna Panel",
                [
                    ("Selected", self.ue_antenna_selected),
                    ("UE rows", self.ue_rows),
                    ("UE cols", self.ue_cols),
                    ("UE v spacing", self.ue_v_spacing),
                    ("UE h spacing", self.ue_h_spacing),
                    ("UE pattern", self.ue_pattern),
                    ("UE polarization", self.ue_polarization),
                ],
                object_name="config_section_ue_antenna",
            )
            trajectory_section = _section(
                "Trajectory",
                [
                    ("Kind", self.trajectory_kind),
                    ("Points", self.trajectory_points),
                    ("Bezier handles", self.trajectory_handles),
                    ("Curve points", _inline(self.add_curve_point, self.remove_curve_point)),
                    ("Easing", self.trajectory_easing),
                    ("Start static", self.start_static),
                    ("End static", self.end_static),
                ],
                object_name="config_section_trajectory",
            )
            channel_section = _section(
                "Channel Output",
                [
                    ("Mode", self.channel_mode),
                    ("Bands", self.band_specs),
                    ("TX power dBm", self.tx_power_dbm),
                    ("Dry run", self.dry_run),
                    ("Realtime visualizer", self.channel_visualizer_enabled),
                ],
                object_name="config_section_channel",
            )
            tracing_section = _section(
                "Sionna Tracing",
                [
                    ("Scene samples", self.samples),
                    ("Samples/src", self.samples_per_src),
                    ("Max paths/src", self.max_num_paths_per_src),
                    ("Max paths unlimited", self.max_paths_unlimited),
                    ("Max depth", self.max_depth),
                    ("Batch timeframes", self.batch_timeframes),
                    ("Max timeframes", self.max_timeframes),
                    ("Seed -1 auto", self.seed),
                    ("LOS", self.los),
                    ("Specular", self.specular),
                    ("Diffuse", self.diffuse),
                    ("Refraction", self.refraction),
                    ("UE-UE links", self.ue_ue_links),
                    ("Synthetic array", self.synthetic_array),
                    ("Merge shapes", self.merge_shapes),
                    ("Use GPU", self.use_gpu),
                ],
                object_name="config_section_tracing",
            )
            radiomap_section = _section(
                "Radiomap",
                [
                    ("Enabled", self.radiomap_enabled),
                    ("X min", self.rm_x_min),
                    ("X max", self.rm_x_max),
                    ("Y min", self.rm_y_min),
                    ("Y max", self.rm_y_max),
                    ("X spacing", self.rm_x_spacing),
                    ("Y spacing", self.rm_y_spacing),
                    ("Height", self.rm_height),
                ],
                object_name="config_section_radiomap",
            )

            actions = QGroupBox("Actions")
            actions.setObjectName("config_section_actions")
            buttons = QGridLayout(actions)
            for index, button in enumerate(
                [
                    add_bs,
                    add_ue,
                    add_obj,
                    remove_entity,
                    apply_entity,
                    apply_trajectory,
                    apply_radiomap_template,
                    load_config,
                    save_config,
                    browse_output,
                    self.simulate_button,
                ]
            ):
                buttons.addWidget(button, index // 3, index % 3)

            controls_widget = QWidget()
            controls = QVBoxLayout(controls_widget)
            controls.setContentsMargins(8, 8, 8, 8)
            controls.setSpacing(10)
            for section in (
                project_section,
                pose_section,
                bs_antenna_section,
                ue_antenna_section,
                trajectory_section,
                channel_section,
                tracing_section,
                radiomap_section,
                actions,
            ):
                controls.addWidget(section)
            controls.addStretch(1)

            controls_scroll = QScrollArea()
            controls_scroll.setObjectName("config_scroll")
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setWidget(controls_widget)
            controls_scroll.setMinimumWidth(380)

            left = QVBoxLayout()
            left.addWidget(controls_scroll, 3)
            left.addWidget(QLabel("Entities"))
            left.addWidget(self.entity_list, 1)

            right = QVBoxLayout()
            right.addLayout(view_buttons)
            right.addWidget(self.view, 8)
            right.addWidget(self.progress, 0)
            right.addWidget(self.log, 2)

            root = QHBoxLayout()
            left_widget = QWidget()
            left_widget.setLayout(left)
            root.addWidget(left_widget, 0)
            right_widget = QWidget()
            right_widget.setLayout(right)
            root.addWidget(right_widget, 1)
            central = QWidget()
            central.setLayout(root)
            self.setCentralWidget(central)
            self.entity_list.currentItemChanged.connect(self._on_selected_entity_changed)
            self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
            self.trajectory_kind.currentIndexChanged.connect(self._on_trajectory_kind_changed)
            self.samples.valueChanged.connect(self._on_scene_samples_changed)
            self._connect_selected_antenna_editing()
            self._connect_radiomap_preview_refresh()
            self.refresh()

        def show_antenna_diagrams(self):
            self.view.show_antenna_diagrams()
            QTimer.singleShot(10_000, self.view.hide_antenna_diagrams)

        def show_simulation_paths(self):
            if self.view.simulation_paths_visible():
                self.view.hide_simulation_paths()
                self.log.append("Hid simulation paths")
                return
            if self._last_simulation_result is None:
                self.log.append("Run a simulation first. Path cache is only available after simulation completes.")
                return
            min_range = self.path_min_range.value()
            max_range = self.path_max_range.value()
            if max_range <= min_range:
                self.log.append("Path max range must be larger than min range")
                return
            link_filter = self.path_link_combo.currentData()
            tx_filter = None
            rx_filter = None
            if isinstance(link_filter, tuple) and len(link_filter) == 2:
                tx_filter, rx_filter = link_filter
            count, stats = self.view.show_simulation_paths(
                self._last_simulation_result,
                min_range_m=min_range,
                max_range_m=max_range,
                tx_id=tx_filter,
                rx_id=rx_filter,
            )
            if count:
                rejected = int(stats.get("paths_rejected_geometry_mismatch", 0))
                self.log.append(
                    f"Showing {count} strongest separated BS->UE paths in requested range "
                    f"{min_range:.2f}-{max_range:.2f} m for {self.path_link_combo.currentText()}; "
                    f"actual delay ranges {stats['selected_range_min_m']:.3f}-{stats['selected_range_max_m']:.3f} m; "
                    f"rejected {rejected} inconsistent geometries"
                )
            else:
                self.log.append(f"No drawable BS->UE path cache found in the last simulation. Stats: {stats}")

        def add_bs(self):
            self._push_undo("add BS")
            orientation = (self.yaw.value(), self.pitch.value(), self.roll.value())
            bs = BaseStation(
                id=unique_entity_id("bs", self._existing_entity_ids()),
                position=self._position_from_controls(),
                orientation_rad=orientation,
            )
            bs.panel.rows = self.bs_rows.value()
            bs.panel.cols = self.bs_cols.value()
            self._apply_panel_controls(
                bs.panel,
                orientation,
                self.bs_v_spacing.value(),
                self.bs_h_spacing.value(),
                self.bs_pattern.currentText(),
                self.bs_polarization.currentText(),
            )
            self.design.base_stations.append(bs)
            self.refresh()

        def add_ue(self):
            self._push_undo("add UE")
            orientation = (self.yaw.value(), self.pitch.value(), self.roll.value())
            ue = UserEquipment(
                id=unique_entity_id("ue", self._existing_entity_ids()),
                position=self._position_from_controls(),
                orientation_rad=orientation,
            )
            ue.panel.rows = self.ue_rows.value()
            ue.panel.cols = self.ue_cols.value()
            self._apply_panel_controls(
                ue.panel,
                orientation,
                self.ue_v_spacing.value(),
                self.ue_h_spacing.value(),
                self.ue_pattern.currentText(),
                self.ue_polarization.currentText(),
            )
            ue.trajectory = anchor_trajectory(
                self._trajectory_from_controls(default_position=ue.position),
                ue.position,
            )
            ue.trajectory.orientation_rad_points = self._trajectory_orientation_points(ue, ue.trajectory)
            self._set_trajectory_controls(ue.trajectory)
            self.design.user_equipments.append(ue)
            self.refresh()

        def add_object(self):
            self._push_undo("add object")
            object_name = self.object_name.currentText() or "CAR_obj"
            prefix = object_name.split("_", maxsplit=1)[0].lower() or "object"
            orientation = (self.yaw.value(), self.pitch.value(), self.roll.value())
            position = (0.0, 0.0, 0.75)
            obj = DynamicObject(
                id=unique_entity_id(prefix, self._existing_entity_ids()),
                object_name=object_name,
                position=position,
                orientation_rad=orientation,
            )
            obj.trajectory = anchor_trajectory(
                self._trajectory_from_controls(default_position=obj.position),
                obj.position,
            )
            obj.trajectory.orientation_rad_points = self._trajectory_orientation_points(obj, obj.trajectory)
            self._set_trajectory_controls(obj.trajectory)
            self.design.objects.append(obj)
            self.refresh()

        def _existing_entity_ids(self) -> set[str]:
            return {entity.id for entity in [*self.design.base_stations, *self.design.user_equipments, *self.design.objects]}

        def _on_scenario_changed(self, *_):
            self._refresh_object_choices()
            self.refresh()

        def _connect_radiomap_preview_refresh(self):
            self.radiomap_enabled.toggled.connect(lambda *_: self.refresh(reset_camera=False))
            for control in (
                self.rm_x_min,
                self.rm_x_max,
                self.rm_y_min,
                self.rm_y_max,
                self.rm_x_spacing,
                self.rm_y_spacing,
                self.rm_height,
            ):
                control.valueChanged.connect(lambda *_: None if self._syncing_controls else self.refresh(reset_camera=False))

        def _connect_selected_antenna_editing(self):
            for control in (
                self.bs_rows,
                self.bs_cols,
                self.bs_v_spacing,
                self.bs_h_spacing,
                self.ue_rows,
                self.ue_cols,
                self.ue_v_spacing,
                self.ue_h_spacing,
            ):
                control.valueChanged.connect(self._on_selected_antenna_controls_changed)
            self.bs_pattern.currentIndexChanged.connect(self._on_selected_antenna_controls_changed)
            self.bs_polarization.currentIndexChanged.connect(self._on_selected_antenna_controls_changed)
            self.ue_pattern.currentIndexChanged.connect(self._on_selected_antenna_controls_changed)
            self.ue_polarization.currentIndexChanged.connect(self._on_selected_antenna_controls_changed)

        def _current_scenario_asset(self):
            current_path = self.scenario_combo.currentData()
            if not current_path:
                return None
            for asset in self.scenario_assets:
                if str(asset.path) == str(current_path):
                    return asset
            return None

        def _ensure_scenario_option(self, scenario_path: Path):
            scenario_path = Path(scenario_path)
            scenario_data = str(scenario_path)
            if self.scenario_combo.findData(scenario_data) >= 0:
                return
            label = f"{scenario_path.stem} (custom)"
            if scenario_path.exists():
                try:
                    asset = load_scenario_asset(scenario_path)
                    if all(str(existing.path) != scenario_data for existing in self.scenario_assets):
                        self.scenario_assets.append(asset)
                    label = f"{asset.name} ({asset.mesh_count} meshes)"
                except Exception:  # noqa: BLE001 - custom scenario can still be preserved without preview metadata
                    pass
            else:
                label = f"{scenario_path.stem} (missing)"
            self.scenario_combo.addItem(label, scenario_data)

        def _refresh_object_choices(self):
            current = self.object_name.currentText()
            self.object_name.clear()
            asset = self._current_scenario_asset()
            available = set(asset.object_names) if asset is not None else set(DYNAMIC_SCENE_OBJECT_NAMES)
            names = [name for name in DYNAMIC_SCENE_OBJECT_NAMES if not available or name in available]
            if asset is not None:
                names.extend(mesh.name for mesh in asset.object_meshes)
            names = list(dict.fromkeys(names))
            if not names:
                names = [DYNAMIC_SCENE_OBJECT_NAMES[0]]
            self.object_name.addItems(names)
            if current:
                index = self.object_name.findText(current)
                if index >= 0:
                    self.object_name.setCurrentIndex(index)

        def refresh(self, *, reset_camera: bool = True):
            selected_id = self._selected_entity_id()
            self.entity_list.clear()
            for entity in [*self.design.base_stations, *self.design.user_equipments, *self.design.objects]:
                item_widget = QListWidgetItem(f"{entity.id} @ {entity.position}")
                item_widget.setData(Qt.UserRole, entity.id)
                self.entity_list.addItem(item_widget)
                if entity.id == selected_id:
                    self.entity_list.setCurrentItem(item_widget)
            view_design = self._simulation_preview_design or self.design
            self.view.set_scene(
                self._current_scenario_asset(),
                view_design,
                self._radiomap_config_from_controls(enabled=self.radiomap_enabled.isChecked()),
                self._selected_entity_id(),
                reset_camera=reset_camera,
            )

        def _set_radiomap_bounds_from_view(self, x_min: float, x_max: float, y_min: float, y_max: float):
            x0, x1 = sorted((float(x_min), float(x_max)))
            y0, y1 = sorted((float(y_min), float(y_max)))
            was_syncing = self._syncing_controls
            self._syncing_controls = True
            try:
                self.rm_x_min.setValue(x0)
                self.rm_x_max.setValue(x1)
                self.rm_y_min.setValue(y0)
                self.rm_y_max.setValue(y1)
            finally:
                self._syncing_controls = was_syncing
            self.refresh(reset_camera=False)

        def _select_entity_by_id(self, entity_id: str | None):
            if entity_id is None:
                self.entity_list.clearSelection()
                return
            for row in range(self.entity_list.count()):
                item = self.entity_list.item(row)
                if item.data(Qt.UserRole) == entity_id:
                    self.entity_list.setCurrentItem(item)
                    return

        def _push_undo(self, label: str):
            selected_id = self._selected_entity_id()
            self.undo_stack.append((label, deepcopy(self.design), selected_id))
            if len(self.undo_stack) > 100:
                self.undo_stack.pop(0)

        def _begin_view_transform(self, entity_id: str):
            self._push_undo(f"transform {entity_id}")

        def _move_trajectory_point_from_view(self, entity_id: str, point_index: int, position):
            entity = self._find_entity_by_id(entity_id)
            if entity is None or not hasattr(entity, "trajectory"):
                return
            trajectory = entity.trajectory
            if point_index <= 0 or point_index >= len(trajectory.points):
                return
            points = list(trajectory.points)
            points[0] = entity.position
            old_point = points[point_index]
            points[point_index] = tuple(float(value) for value in position)
            bezier_handles = self._shift_curve_point_handles(
                trajectory,
                point_index,
                old_point,
                points[point_index],
                points,
            )
            entity.trajectory = TrajectorySpec(
                kind=trajectory.kind,
                points=points,
                bezier_handles=bezier_handles,
                orientation_rad_points=self._trajectory_orientation_points(
                    entity,
                    trajectory,
                    source_orientations=trajectory.orientation_rad_points,
                ),
                samples=self.samples.value(),
                start_static_fraction=trajectory.start_static_fraction,
                end_static_fraction=trajectory.end_static_fraction,
                easing=trajectory.easing,
            )
            self._set_trajectory_controls(entity.trajectory)

        def _move_trajectory_handle_from_view(self, entity_id: str, point_index: int, handle_side: str, position):
            entity = self._find_entity_by_id(entity_id)
            if entity is None or not hasattr(entity, "trajectory"):
                return
            trajectory = entity.trajectory
            if trajectory.kind != "curve" or point_index < 0 or point_index >= len(trajectory.points):
                return
            points = list(trajectory.points)
            handles = normalized_bezier_handles(trajectory)
            plane_z = float(points[0][2])
            handle_position = (float(position[0]), float(position[1]), plane_z)
            handle_in, handle_out = handles[point_index]
            if handle_side == "in":
                handles[point_index] = (handle_position, handle_out)
            elif handle_side == "out":
                handles[point_index] = (handle_in, handle_position)
            else:
                return
            entity.trajectory = TrajectorySpec(
                kind=trajectory.kind,
                points=points,
                bezier_handles=handles,
                orientation_rad_points=self._trajectory_orientation_points(
                    entity,
                    trajectory,
                    source_orientations=trajectory.orientation_rad_points,
                ),
                samples=self.samples.value(),
                start_static_fraction=trajectory.start_static_fraction,
                end_static_fraction=trajectory.end_static_fraction,
                easing=trajectory.easing,
            )
            self._set_trajectory_controls(entity.trajectory)

        def _transform_trajectory_endpoint_from_view(self, entity_id: str, point_index: int, position, orientation):
            entity = self._find_entity_by_id(entity_id)
            if entity is None or not hasattr(entity, "trajectory"):
                return
            trajectory = entity.trajectory
            if point_index <= 0 or point_index >= len(trajectory.points):
                return
            points = list(trajectory.points)
            points[0] = entity.position
            old_point = points[point_index]
            points[point_index] = tuple(float(value) for value in position)
            bezier_handles = self._shift_curve_point_handles(
                trajectory,
                point_index,
                old_point,
                points[point_index],
                points,
            )
            orientation_points = self._trajectory_orientation_points(
                entity,
                trajectory,
                source_orientations=trajectory.orientation_rad_points,
            )
            orientation_points[point_index] = tuple(float(value) for value in orientation)
            entity.trajectory = TrajectorySpec(
                kind=trajectory.kind,
                points=points,
                bezier_handles=bezier_handles,
                orientation_rad_points=orientation_points,
                samples=self.samples.value(),
                start_static_fraction=trajectory.start_static_fraction,
                end_static_fraction=trajectory.end_static_fraction,
                easing=trajectory.easing,
            )
            self._set_trajectory_controls(entity.trajectory)

        def undo(self):
            if not self.undo_stack:
                self.log.append("Nothing to undo")
                return
            label, design, selected_id = self.undo_stack.pop()
            self.design = deepcopy(design)
            self.design.ensure_defaults()
            self.refresh(reset_camera=False)
            if selected_id is not None:
                self._select_entity_by_id(selected_id)
            self.log.append(f"Undid {label}")

        def _transform_entity_from_view(self, entity_id: str, position, orientation):
            entity = self._find_entity_by_id(entity_id)
            if entity is None:
                return
            old_position = entity.position
            new_position = tuple(float(value) for value in position)
            new_orientation = tuple(float(value) for value in orientation)
            delta = (
                new_position[0] - old_position[0],
                new_position[1] - old_position[1],
                new_position[2] - old_position[2],
            )
            entity.position = new_position
            entity.orientation_rad = new_orientation
            if hasattr(entity, "trajectory") and any(abs(value) > 1e-12 for value in delta):
                entity.trajectory = translate_trajectory(entity.trajectory, delta)
            if hasattr(entity, "trajectory"):
                entity.trajectory.orientation_rad_points = self._trajectory_orientation_points(
                    entity,
                    entity.trajectory,
                    source_orientations=entity.trajectory.orientation_rad_points,
                )
            self._set_position_controls(entity.position)
            self.yaw.setValue(entity.orientation_rad[0])
            self.pitch.setValue(entity.orientation_rad[1])
            self.roll.setValue(entity.orientation_rad[2])
            if hasattr(entity, "trajectory"):
                self._set_trajectory_controls(entity.trajectory)
            for row in range(self.entity_list.count()):
                item = self.entity_list.item(row)
                if item.data(Qt.UserRole) == entity.id:
                    item.setText(f"{entity.id} @ {entity.position}")
                    break

        def _selected_entity_id(self):
            item = self.entity_list.currentItem()
            return item.data(Qt.UserRole) if item else None

        def _find_selected_entity(self):
            entity_id = self._selected_entity_id()
            if not entity_id:
                return None
            return self._find_entity_by_id(entity_id)

        def _find_entity_by_id(self, entity_id: str):
            for entity in [*self.design.base_stations, *self.design.user_equipments, *self.design.objects]:
                if entity.id == entity_id:
                    return entity
            return None

        def _on_selected_entity_changed(self, *_):
            entity = self._find_selected_entity()
            if entity is not None:
                self._set_entity_controls(entity)
            else:
                self._set_antenna_selection_labels(None)
            self.view.set_selected_entity(self._selected_entity_id())

        def _set_antenna_selection_labels(self, entity):
            if isinstance(entity, BaseStation):
                self.bs_antenna_selected.setText(f"Selected BS: {entity.id}")
                self.ue_antenna_selected.setText("Selected UE: none")
            elif isinstance(entity, UserEquipment):
                self.bs_antenna_selected.setText("Selected BS: none")
                self.ue_antenna_selected.setText(f"Selected UE: {entity.id}")
            else:
                self.bs_antenna_selected.setText("Selected BS: none")
                self.ue_antenna_selected.setText("Selected UE: none")

        def _set_entity_controls(self, entity):
            was_syncing = self._syncing_controls
            self._syncing_controls = True
            try:
                self._set_position_controls(entity.position)
                self.yaw.setValue(entity.orientation_rad[0])
                self.pitch.setValue(entity.orientation_rad[1])
                self.roll.setValue(entity.orientation_rad[2])
                self._set_antenna_selection_labels(entity)
                if isinstance(entity, (BaseStation, UserEquipment)):
                    if isinstance(entity, BaseStation):
                        self.bs_rows.setValue(entity.panel.rows)
                        self.bs_cols.setValue(entity.panel.cols)
                        self.bs_v_spacing.setValue(entity.panel.vertical_spacing_m)
                        self.bs_h_spacing.setValue(entity.panel.horizontal_spacing_m)
                        self._set_combo_value(self.bs_pattern, entity.panel.pattern)
                        self._set_combo_value(self.bs_polarization, entity.panel.polarization)
                    else:
                        self.ue_rows.setValue(entity.panel.rows)
                        self.ue_cols.setValue(entity.panel.cols)
                        self.ue_v_spacing.setValue(entity.panel.vertical_spacing_m)
                        self.ue_h_spacing.setValue(entity.panel.horizontal_spacing_m)
                        self._set_combo_value(self.ue_pattern, entity.panel.pattern)
                        self._set_combo_value(self.ue_polarization, entity.panel.polarization)
                if isinstance(entity, DynamicObject):
                    self._set_combo_value(self.object_name, entity.object_name)
            finally:
                self._syncing_controls = was_syncing
            if hasattr(entity, "trajectory"):
                self._set_trajectory_controls(entity.trajectory)

        def _set_combo_value(self, combo, value: str):
            index = combo.findText(value)
            if index < 0:
                combo.addItem(value)
                index = combo.findText(value)
            combo.setCurrentIndex(index)

        def _on_scene_samples_changed(self, *_):
            if self._syncing_controls:
                return
            self._sync_scene_trajectory_samples()

        def _on_selected_antenna_controls_changed(self, *_):
            if self._syncing_controls:
                return
            entity = self._find_selected_entity()
            if not isinstance(entity, (BaseStation, UserEquipment)):
                return
            self._push_undo(f"antenna {entity.id}")
            if isinstance(entity, BaseStation):
                entity.panel.rows = self.bs_rows.value()
                entity.panel.cols = self.bs_cols.value()
                vertical_spacing = self.bs_v_spacing.value()
                horizontal_spacing = self.bs_h_spacing.value()
                pattern = self.bs_pattern.currentText()
                polarization = self.bs_polarization.currentText()
            else:
                entity.panel.rows = self.ue_rows.value()
                entity.panel.cols = self.ue_cols.value()
                vertical_spacing = self.ue_v_spacing.value()
                horizontal_spacing = self.ue_h_spacing.value()
                pattern = self.ue_pattern.currentText()
                polarization = self.ue_polarization.currentText()
            self._apply_panel_controls(
                entity.panel,
                entity.orientation_rad,
                vertical_spacing,
                horizontal_spacing,
                pattern,
                polarization,
            )
            if self.view.antenna_diagrams_visible():
                self.view.show_antenna_diagrams()

        def _sync_scene_trajectory_samples(self):
            samples = self.samples.value()
            for entity in [*self.design.user_equipments, *self.design.objects]:
                entity.trajectory.samples = samples

        def _trajectory_orientation_points(
            self,
            entity,
            trajectory: TrajectorySpec,
            *,
            source_orientations=None,
        ) -> list[tuple[float, float, float]]:
            count = len(trajectory.points)
            if count <= 0:
                return []
            source = list(source_orientations or trajectory.orientation_rad_points)
            start = tuple(float(value) for value in entity.orientation_rad)
            if count == 1:
                return [start]
            orientations = [start]
            for index in range(1, count):
                if index < len(source):
                    orientations.append(tuple(float(value) for value in source[index]))
                elif source:
                    orientations.append(tuple(float(value) for value in source[-1]))
                else:
                    orientations.append(start)
            return orientations

        def _scene_trajectory_samples(self) -> int:
            values = [
                int(entity.trajectory.samples)
                for entity in [*self.design.user_equipments, *self.design.objects]
                if hasattr(entity, "trajectory")
            ]
            return max(values) if values else self.samples.value()

        def _on_trajectory_kind_changed(self, *_):
            if self._syncing_controls:
                return
            entity = self._find_selected_entity()
            if entity is None or not hasattr(entity, "trajectory"):
                return
            kind = self.trajectory_kind.currentText()
            if kind == entity.trajectory.kind:
                return
            try:
                self._push_undo(f"trajectory kind {entity.id}")
                entity.trajectory = self._trajectory_for_kind(entity, kind)
                self._set_trajectory_controls(entity.trajectory)
                self.refresh(reset_camera=False)
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR updating trajectory kind: {exc}")

        def _trajectory_for_kind(self, entity, kind: str) -> TrajectorySpec:
            start = entity.position
            old = entity.trajectory
            if kind == "static":
                return TrajectorySpec(
                    kind="static",
                    points=[start],
                    orientation_rad_points=[entity.orientation_rad],
                    samples=self.samples.value(),
                    start_static_fraction=old.start_static_fraction,
                    end_static_fraction=old.end_static_fraction,
                    easing=old.easing,
                )
            if kind == "linear":
                end = old.points[1] if len(old.points) >= 2 else (start[0] + 5.0, start[1], start[2])
                trajectory = TrajectorySpec(kind="linear", points=[start, end])
                return TrajectorySpec(
                    kind="linear",
                    points=[start, end],
                    orientation_rad_points=self._trajectory_orientation_points(
                        entity,
                        trajectory,
                        source_orientations=old.orientation_rad_points,
                    ),
                    samples=self.samples.value(),
                    start_static_fraction=old.start_static_fraction,
                    end_static_fraction=old.end_static_fraction,
                    easing=old.easing,
                )
            if kind == "polyline":
                points = list(old.points)
                if len(points) < 2:
                    points = [start, (start[0] + 5.0, start[1], start[2])]
                points[0] = start
                trajectory = TrajectorySpec(kind="polyline", points=points)
                return TrajectorySpec(
                    kind="polyline",
                    points=points,
                    orientation_rad_points=self._trajectory_orientation_points(
                        entity,
                        trajectory,
                        source_orientations=old.orientation_rad_points,
                    ),
                    samples=self.samples.value(),
                    start_static_fraction=old.start_static_fraction,
                    end_static_fraction=old.end_static_fraction,
                    easing=old.easing,
                )
            if kind == "curve":
                points = list(old.points)
                if len(points) < 2:
                    points = [start, (start[0] + 5.0, start[1], start[2])]
                points[0] = start
                points = self._curve_plane_points(points)
                old_curve = TrajectorySpec(kind="curve", points=points, bezier_handles=old.bezier_handles)
                trajectory = TrajectorySpec(kind="curve", points=points, bezier_handles=normalized_bezier_handles(old_curve))
                return TrajectorySpec(
                    kind="curve",
                    points=points,
                    bezier_handles=trajectory.bezier_handles,
                    orientation_rad_points=self._trajectory_orientation_points(
                        entity,
                        trajectory,
                        source_orientations=old.orientation_rad_points,
                    ),
                    samples=self.samples.value(),
                    start_static_fraction=old.start_static_fraction,
                    end_static_fraction=old.end_static_fraction,
                    easing=old.easing,
                )
            raise ValueError("Trajectory kind must be static, linear, polyline, or curve")

        def _position_from_controls(self):
            return (self.pos_x.value(), self.pos_y.value(), self.pos_z.value())

        def _set_position_controls(self, position):
            self.pos_x.setValue(position[0])
            self.pos_y.setValue(position[1])
            self.pos_z.setValue(position[2])

        def _apply_controls_to_entity(self, entity):
            orientation = (self.yaw.value(), self.pitch.value(), self.roll.value())
            new_position = self._position_from_controls()
            entity.position = new_position
            entity.orientation_rad = orientation
            if isinstance(entity, BaseStation):
                entity.panel.rows = self.bs_rows.value()
                entity.panel.cols = self.bs_cols.value()
                self._apply_panel_controls(
                    entity.panel,
                    orientation,
                    self.bs_v_spacing.value(),
                    self.bs_h_spacing.value(),
                    self.bs_pattern.currentText(),
                    self.bs_polarization.currentText(),
                )
            elif isinstance(entity, UserEquipment):
                source_orientations = entity.trajectory.orientation_rad_points
                entity.panel.rows = self.ue_rows.value()
                entity.panel.cols = self.ue_cols.value()
                self._apply_panel_controls(
                    entity.panel,
                    orientation,
                    self.ue_v_spacing.value(),
                    self.ue_h_spacing.value(),
                    self.ue_pattern.currentText(),
                    self.ue_polarization.currentText(),
                )
                entity.trajectory = anchor_trajectory(
                    self._trajectory_from_controls(default_position=entity.position),
                    entity.position,
                )
                entity.trajectory.orientation_rad_points = self._trajectory_orientation_points(
                    entity,
                    entity.trajectory,
                    source_orientations=source_orientations,
                )
                self._set_trajectory_controls(entity.trajectory)
            elif isinstance(entity, DynamicObject):
                source_orientations = entity.trajectory.orientation_rad_points
                entity.object_name = self.object_name.currentText() or entity.object_name
                entity.trajectory = anchor_trajectory(
                    self._trajectory_from_controls(default_position=entity.position),
                    entity.position,
                )
                entity.trajectory.orientation_rad_points = self._trajectory_orientation_points(
                    entity,
                    entity.trajectory,
                    source_orientations=source_orientations,
                )
                self._set_trajectory_controls(entity.trajectory)

        def _apply_panel_controls(self, panel, orientation, vertical_spacing, horizontal_spacing, pattern, polarization):
            panel.pattern = pattern
            panel.element_diagram = panel.pattern
            panel.polarization = polarization
            panel.orientation_rad = orientation
            panel.vertical_spacing_m = vertical_spacing
            panel.horizontal_spacing_m = horizontal_spacing

        def _trajectory_from_controls(self, default_position):
            points_text = self.trajectory_points.text().strip()
            if not points_text:
                x, y, z = default_position
                points_text = f"{x:g},{y:g},{z:g}"
            return build_trajectory_spec(
                self.trajectory_kind.currentText(),
                points_text,
                self.samples.value(),
                bezier_handles_text=self.trajectory_handles.text().strip(),
                easing=self.trajectory_easing.currentText(),
                start_static_fraction=self.start_static.value(),
                end_static_fraction=self.end_static.value(),
            )

        def _set_trajectory_controls(self, spec: TrajectorySpec):
            self._syncing_controls = True
            try:
                kind_index = self.trajectory_kind.findText(spec.kind)
                if kind_index >= 0:
                    self.trajectory_kind.setCurrentIndex(kind_index)
                easing_index = self.trajectory_easing.findText(spec.easing)
                if easing_index >= 0:
                    self.trajectory_easing.setCurrentIndex(easing_index)
                self.start_static.setValue(spec.start_static_fraction)
                self.end_static.setValue(spec.end_static_fraction)
                self.trajectory_points.setText(format_point_list(spec.points))
                self.trajectory_handles.setText(format_bezier_handles(normalized_bezier_handles(spec)) if spec.kind == "curve" else "")
                curve_enabled = spec.kind == "curve"
                self.trajectory_handles.setEnabled(curve_enabled)
                self.add_curve_point.setEnabled(curve_enabled)
                self.remove_curve_point.setEnabled(curve_enabled and len(spec.points) > 2)
            finally:
                self._syncing_controls = False

        def add_curve_middle_point(self):
            entity = self._find_selected_entity()
            if entity is None or not hasattr(entity, "trajectory"):
                self.log.append("ERROR adding curve point: select a UE or object")
                return
            try:
                self._push_undo(f"add curve point {entity.id}")
                trajectory = entity.trajectory if entity.trajectory.kind == "curve" else self._trajectory_for_kind(entity, "curve")
                points = list(trajectory.points)
                if len(points) < 2:
                    points = [entity.position, (entity.position[0] + 5.0, entity.position[1], entity.position[2])]
                insert_at = len(points) - 1
                previous_point = np.asarray(points[insert_at - 1], dtype=np.float64)
                next_point = np.asarray(points[insert_at], dtype=np.float64)
                middle = tuple(float(value) for value in ((previous_point + next_point) / 2.0))
                points.insert(insert_at, middle)
                points = self._curve_plane_points(points)
                entity.trajectory = TrajectorySpec(
                    kind="curve",
                    points=points,
                    bezier_handles=default_bezier_handles(points),
                    orientation_rad_points=self._trajectory_orientation_points(entity, TrajectorySpec(kind="curve", points=points)),
                    samples=self.samples.value(),
                    start_static_fraction=trajectory.start_static_fraction,
                    end_static_fraction=trajectory.end_static_fraction,
                    easing=trajectory.easing,
                )
                self._set_trajectory_controls(entity.trajectory)
                self.refresh(reset_camera=False)
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR adding curve point: {exc}")

        def remove_curve_middle_point(self):
            entity = self._find_selected_entity()
            if entity is None or not hasattr(entity, "trajectory") or entity.trajectory.kind != "curve":
                self.log.append("ERROR removing curve point: select a curve UE or object")
                return
            trajectory = entity.trajectory
            if len(trajectory.points) <= 2:
                return
            self._push_undo(f"remove curve point {entity.id}")
            points = list(trajectory.points)
            points.pop(-2)
            points = self._curve_plane_points(points)
            entity.trajectory = TrajectorySpec(
                kind="curve",
                points=points,
                bezier_handles=default_bezier_handles(points),
                orientation_rad_points=self._trajectory_orientation_points(entity, TrajectorySpec(kind="curve", points=points), source_orientations=trajectory.orientation_rad_points),
                samples=self.samples.value(),
                start_static_fraction=trajectory.start_static_fraction,
                end_static_fraction=trajectory.end_static_fraction,
                easing=trajectory.easing,
            )
            self._set_trajectory_controls(entity.trajectory)
            self.refresh(reset_camera=False)

        def _curve_plane_points(self, points):
            if not points:
                return []
            plane_z = float(points[0][2])
            return [(float(point[0]), float(point[1]), plane_z) for point in points]

        def _shift_curve_point_handles(self, trajectory, point_index, old_point, new_point, points):
            if trajectory.kind != "curve":
                return list(trajectory.bezier_handles)
            handles = normalized_bezier_handles(trajectory)
            delta = np.asarray(new_point, dtype=np.float64) - np.asarray(old_point, dtype=np.float64)
            plane_z = float(points[0][2])
            shifted = []
            for index, (handle_in, handle_out) in enumerate(handles):
                if index == point_index:
                    handle_in = tuple((np.asarray(handle_in, dtype=np.float64) + delta).tolist())
                    handle_out = tuple((np.asarray(handle_out, dtype=np.float64) + delta).tolist())
                shifted.append((
                    (float(handle_in[0]), float(handle_in[1]), plane_z),
                    (float(handle_out[0]), float(handle_out[1]), plane_z),
                ))
            return shifted

        def apply_trajectory(self):
            entity = self._find_selected_entity()
            if entity is None:
                self.log.append("ERROR applying trajectory: select a UE or object")
                return
            if not hasattr(entity, "trajectory"):
                self.log.append("ERROR applying trajectory: BSs are fixed in the scene")
                return
            try:
                self._push_undo(f"trajectory {entity.id}")
                source_orientations = entity.trajectory.orientation_rad_points
                entity.trajectory = anchor_trajectory(
                    self._trajectory_from_controls(default_position=entity.position),
                    entity.position,
                )
                entity.trajectory.orientation_rad_points = self._trajectory_orientation_points(
                    entity,
                    entity.trajectory,
                    source_orientations=source_orientations,
                )
                entity.trajectory.samples = self.samples.value()
                self._set_trajectory_controls(entity.trajectory)
                self.refresh(reset_camera=False)
                self.log.append(f"Updated trajectory and position for {entity.id}")
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR applying trajectory: {exc}")

        def apply_entity(self):
            entity = self._find_selected_entity()
            if entity is None:
                self.log.append("ERROR applying entity settings: select a BS, UE or object")
                return
            try:
                self._push_undo(f"entity {entity.id}")
                self._apply_controls_to_entity(entity)
                self.refresh()
                self.log.append(f"Updated entity settings for {entity.id}")
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR applying entity settings: {exc}")

        def remove_selected_entity(self):
            entity = self._find_selected_entity()
            if entity is None:
                self.log.append("ERROR removing entity: select a BS, UE or object")
                return
            entity_id = entity.id
            self._push_undo(f"remove {entity_id}")
            self.design.base_stations = [
                base_station for base_station in self.design.base_stations if base_station.id != entity_id
            ]
            self.design.user_equipments = [
                user_equipment for user_equipment in self.design.user_equipments if user_equipment.id != entity_id
            ]
            self.design.objects = [scene_object for scene_object in self.design.objects if scene_object.id != entity_id]
            self.entity_list.clearSelection()
            self.refresh()
            self.log.append(f"Removed {entity_id}")

        def _request_from_controls(self) -> SimulationRequest:
            self._sync_scene_trajectory_samples()
            self.output_dir = Path(self.output_dir_text.text().strip() or "output")
            if self.scenario_combo.currentData():
                self.design.scenario_path = Path(self.scenario_combo.currentData())
                self.design.name = self.design.scenario_path.stem
            if self.radiomap_enabled.isChecked():
                if self.rm_x_max.value() < self.rm_x_min.value():
                    raise ValueError("Radiomap x max must be greater than or equal to x min")
                if self.rm_y_max.value() < self.rm_y_min.value():
                    raise ValueError("Radiomap y max must be greater than or equal to y min")
            self.design.radiomap = self._radiomap_config_from_controls(enabled=self.radiomap_enabled.isChecked())
            request = SimulationRequest(
                scene=deepcopy(self.design),
                output_dir=self.output_dir,
                dry_run=self.dry_run.isChecked(),
            )
            request.channel_mode = ChannelMode(self.channel_mode.currentData())
            request.bands = parse_band_specs(self.band_specs.text())
            request.sionna.samples_per_src = self.samples_per_src.value()
            request.sionna.tx_power_dbm = self.tx_power_dbm.value()
            request.sionna.max_num_paths_per_src = (
                None if self.max_paths_unlimited.isChecked() else self.max_num_paths_per_src.value()
            )
            request.sionna.max_depth = self.max_depth.value()
            request.sionna.batch_timeframes = self.batch_timeframes.value()
            request.sionna.max_timeframes = self.max_timeframes.value()
            request.sionna.seed = None if self.seed.value() < 0 else self.seed.value()
            request.sionna.los = self.los.isChecked()
            request.sionna.specular_reflection = self.specular.isChecked()
            request.sionna.diffuse_reflection = self.diffuse.isChecked()
            request.sionna.refraction = self.refraction.isChecked()
            request.sionna.ue_ue_links = self.ue_ue_links.isChecked()
            request.sionna.synthetic_array = self.synthetic_array.isChecked()
            request.sionna.merge_shapes = self.merge_shapes.isChecked()
            request.sionna.use_gpu = self.use_gpu.isChecked()
            validate_request(request)
            return request

        def _radiomap_config_from_controls(self, *, enabled: bool) -> RadiomapConfig:
            return RadiomapConfig(
                enabled=enabled,
                x_min=self.rm_x_min.value(),
                x_max=self.rm_x_max.value(),
                y_min=self.rm_y_min.value(),
                y_max=self.rm_y_max.value(),
                x_spacing=self.rm_x_spacing.value(),
                y_spacing=self.rm_y_spacing.value(),
                height=self.rm_height.value(),
                ue_template=self._radiomap_template_from_controls(apply_controls=False),
            )

        def _radiomap_template_from_controls(self, *, apply_controls: bool) -> UserEquipment:
            template = deepcopy(self.design.radiomap.ue_template)
            if not template.id:
                template.id = "rm_ue"
            template.position = (template.position[0], template.position[1], self.rm_height.value())
            if apply_controls:
                template.orientation_rad = (self.yaw.value(), self.pitch.value(), self.roll.value())
                template.panel.rows = self.ue_rows.value()
                template.panel.cols = self.ue_cols.value()
                template.panel.pattern = self.ue_pattern.currentText()
                template.panel.element_diagram = self.ue_pattern.currentText()
                template.panel.polarization = self.ue_polarization.currentText()
                template.panel.orientation_rad = template.orientation_rad
                template.panel.vertical_spacing_m = self.ue_v_spacing.value()
                template.panel.horizontal_spacing_m = self.ue_h_spacing.value()
            return template

        def apply_radiomap_template(self):
            try:
                self.design.radiomap.ue_template = self._radiomap_template_from_controls(apply_controls=True)
                self.log.append("Updated radiomap UE template")
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR applying radiomap template: {exc}")

        def _apply_request_to_controls(self, request: SimulationRequest):
            self.design = deepcopy(request.scene)
            self.design.ensure_defaults()
            self.output_dir = request.output_dir
            self.output_dir_text.setText(str(self.output_dir))
            self.dry_run.setChecked(request.dry_run)
            self.tx_power_dbm.setValue(request.sionna.tx_power_dbm)
            self.samples_per_src.setValue(request.sionna.samples_per_src)
            self.max_paths_unlimited.setChecked(request.sionna.max_num_paths_per_src is None)
            self.max_num_paths_per_src.setValue(request.sionna.max_num_paths_per_src or 100_000)
            self.max_num_paths_per_src.setDisabled(request.sionna.max_num_paths_per_src is None)
            self.max_depth.setValue(request.sionna.max_depth)
            self.batch_timeframes.setValue(request.sionna.batch_timeframes)
            self.max_timeframes.setValue(request.sionna.max_timeframes)
            self.seed.setValue(-1 if request.sionna.seed is None else request.sionna.seed)
            self.los.setChecked(request.sionna.los)
            self.specular.setChecked(request.sionna.specular_reflection)
            self.diffuse.setChecked(request.sionna.diffuse_reflection)
            self.refraction.setChecked(request.sionna.refraction)
            self.ue_ue_links.setChecked(request.sionna.ue_ue_links)
            self.synthetic_array.setChecked(request.sionna.synthetic_array)
            self.merge_shapes.setChecked(request.sionna.merge_shapes)
            self.use_gpu.setChecked(request.sionna.use_gpu)
            self.band_specs.setText(format_band_specs(request.bands))
            mode_index = self.channel_mode.findData(request.channel_mode.value)
            if mode_index >= 0:
                self.channel_mode.setCurrentIndex(mode_index)
            scenario_path = str(request.scene.scenario_path)
            self._ensure_scenario_option(request.scene.scenario_path)
            scenario_index = self.scenario_combo.findData(scenario_path)
            if scenario_index >= 0:
                self.scenario_combo.setCurrentIndex(scenario_index)
            self.radiomap_enabled.setChecked(request.scene.radiomap.enabled)
            self.rm_x_min.setValue(request.scene.radiomap.x_min)
            self.rm_x_max.setValue(request.scene.radiomap.x_max)
            self.rm_y_min.setValue(request.scene.radiomap.y_min)
            self.rm_y_max.setValue(request.scene.radiomap.y_max)
            self.rm_x_spacing.setValue(request.scene.radiomap.x_spacing)
            self.rm_y_spacing.setValue(request.scene.radiomap.y_spacing)
            self.rm_height.setValue(request.scene.radiomap.height)
            self._set_radiomap_template_controls(request.scene.radiomap.ue_template)
            self._syncing_controls = True
            try:
                self.samples.setValue(self._scene_trajectory_samples())
            finally:
                self._syncing_controls = False
            self._sync_scene_trajectory_samples()
            self.refresh()

        def _set_radiomap_template_controls(self, template: UserEquipment):
            was_syncing = self._syncing_controls
            self._syncing_controls = True
            try:
                self.ue_rows.setValue(template.panel.rows)
                self.ue_cols.setValue(template.panel.cols)
                self.ue_v_spacing.setValue(template.panel.vertical_spacing_m)
                self.ue_h_spacing.setValue(template.panel.horizontal_spacing_m)
                self._set_combo_value(self.ue_pattern, template.panel.pattern)
                self._set_combo_value(self.ue_polarization, template.panel.polarization)
                self.yaw.setValue(template.orientation_rad[0])
                self.pitch.setValue(template.orientation_rad[1])
                self.roll.setValue(template.orientation_rad[2])
            finally:
                self._syncing_controls = was_syncing

        def load_config(self):
            filename, _ = QFileDialog.getOpenFileName(self, "Load request", str(Path.cwd()), "JSON (*.json)")
            if not filename:
                return
            try:
                self._apply_request_to_controls(read_request(Path(filename)))
                self.log.append(f"Loaded {filename}")
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR loading config: {exc}")

        def save_config(self):
            filename, _ = QFileDialog.getSaveFileName(self, "Save request", str(Path.cwd() / "request.json"), "JSON (*.json)")
            if not filename:
                return
            try:
                request = self._request_from_controls()
                write_request(Path(filename), request)
                self.log.append(f"Saved {filename}")
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR saving config: {exc}")

        def browse_output_dir(self):
            directory = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir_text.text())
            if directory:
                self.output_dir_text.setText(directory)

        def simulate(self):
            if self._simulation_running():
                self.log.append("Simulation is already running")
                return
            try:
                request = self._request_from_controls()
            except Exception as exc:  # noqa: BLE001 - shown to GUI user
                self.log.append(f"ERROR preparing request: {exc}")
                return
            self.simulate_button.setEnabled(False)
            self.progress.setMaximum(1)
            self.progress.setValue(0)
            self.log.append("Planned run size:\n" + format_request_estimate(estimate_request_size(request)))
            if request.dry_run:
                self.log.append("Dry run enabled: scenario meshes are not ray traced")
            self.log.append("Starting simulation")
            self.view.hide_simulation_paths()
            self._last_simulation_result = None
            self.path_link_combo.clear()
            self.path_link_combo.addItem("All BS->UE", None)
            if self.channel_visualizer_enabled.isChecked():
                self.channel_visualizer = ChannelVisualizerWindow()
                self.channel_visualizer.show()
            self._simulation_preview_design = deepcopy(request.scene)
            self.worker = SimulationWorker(request)
            self.worker.progress.connect(self._on_progress)
            self.worker.timeframe_pose.connect(self._on_timeframe_pose)
            self.worker.result_ready.connect(self._on_result_ready)
            self.worker.finished_path.connect(self._on_finished)
            self.worker.failed.connect(self._on_failed)
            self.worker.finished.connect(self._on_worker_stopped)
            self.worker.start()

        def _simulation_running(self):
            return self.worker is not None and self.worker.isRunning()

        def _on_progress(self, current, total, message):
            self.progress.setMaximum(max(1, total))
            self.progress.setValue(min(current, self.progress.maximum()))
            self.log.append(message)

        def _on_timeframe_pose(self, _index, device_positions, device_orientations, object_positions, object_orientations):
            if self._simulation_preview_design is None:
                return
            for entity in [*self._simulation_preview_design.user_equipments, *self._simulation_preview_design.base_stations]:
                if entity.id in device_positions:
                    entity.position = tuple(float(value) for value in device_positions[entity.id])
                if entity.id in device_orientations:
                    entity.orientation_rad = tuple(float(value) for value in device_orientations[entity.id])
            for entity in self._simulation_preview_design.objects:
                if entity.id in object_positions:
                    entity.position = tuple(float(value) for value in object_positions[entity.id])
                if entity.id in object_orientations:
                    entity.orientation_rad = tuple(float(value) for value in object_orientations[entity.id])
            self.refresh(reset_camera=False)

        def _on_result_ready(self, request, result):
            self._last_simulation_result = result
            self._refresh_path_link_choices(result)
            if not self.channel_visualizer_enabled.isChecked():
                return
            if self.channel_visualizer is None:
                self.channel_visualizer = ChannelVisualizerWindow()
                self.channel_visualizer.show()
            self.channel_visualizer.set_result(request, result)
            self.log.append("Updated realtime channel visualizer")

        def _refresh_path_link_choices(self, result):
            current = self.path_link_combo.currentData()
            self.path_link_combo.blockSignals(True)
            self.path_link_combo.clear()
            self.path_link_combo.addItem("All BS->UE", None)
            pairs = []
            if result is not None and getattr(result, "timeframes", None):
                for link in result.timeframes[0].links:
                    tx_id = str(link.tx_id)
                    rx_id = str(link.rx_id)
                    if tx_id.startswith("bs") and rx_id.startswith("ue"):
                        pairs.append((tx_id, rx_id))
            for tx_id, rx_id in sorted(set(pairs), key=lambda item: (_natural_id_key(item[0]), _natural_id_key(item[1]))):
                self.path_link_combo.addItem(f"{tx_id} -> {rx_id}", (tx_id, rx_id))
            if current is not None:
                for idx in range(self.path_link_combo.count()):
                    if self.path_link_combo.itemData(idx) == current:
                        self.path_link_combo.setCurrentIndex(idx)
                        break
            self.path_link_combo.blockSignals(False)

        def _on_finished(self, path):
            self.progress.setValue(self.progress.maximum())
            if self.channel_visualizer_enabled.isChecked():
                try:
                    if self.channel_visualizer is None:
                        self.channel_visualizer = ChannelVisualizerWindow()
                        self.channel_visualizer.show()
                    self.channel_visualizer.set_h5_path(path)
                    self.log.append("Updated realtime channel visualizer from saved HDF5")
                except Exception as exc:  # noqa: BLE001 - shown to GUI user
                    self.log.append(f"ERROR updating channel visualizer: {exc}")
            self.log.append(f"Saved {path}")

        def _on_failed(self, message):
            self.log.append(f"ERROR: {message}")

        def _on_worker_stopped(self):
            self.simulate_button.setEnabled(True)
            self._simulation_preview_design = None
            self.refresh(reset_camera=False)
            self.worker = None

    def _natural_id_key(value):
        text = str(value)
        prefix = text.rstrip("0123456789")
        suffix = text[len(prefix):]
        return (prefix, int(suffix) if suffix.isdigit() else -1, text)

    def _section(title, rows, *, object_name):
        group = QGroupBox(title)
        group.setObjectName(object_name)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for label, widget in rows:
            form.addRow(label, widget)
        return group

    def _inline(*widgets):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        return container

    def _spin(minimum, maximum, value):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def _double_spin(minimum, maximum, value, decimals=3):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        return widget

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
