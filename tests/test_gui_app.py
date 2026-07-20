def test_gui_runtime_controls_and_loaded_custom_scenario_preservation(monkeypatch, tmp_path):
    from copy import deepcopy

    import pytest

    pytest.importorskip("PySide6", reason="PySide6 is required for GUI tests")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QGroupBox, QPushButton, QScrollArea, QSpinBox, QWidget

    from isac_6d_sampler.core.model import RadiomapConfig, SimulationRequest
    from isac_6d_sampler.gui import app as gui_app

    custom_scenario = tmp_path / "custom_scene.xml"
    custom_scenario.write_text("<scene />\n", encoding="utf-8")
    observed = {}

    def fake_exec(self):
        max_depth = max_paths = tx_power = unlimited = None
        main_window = None
        for window in self.topLevelWidgets():
            max_depth = max_depth or window.findChild(QSpinBox, "max_depth")
            max_paths = max_paths or window.findChild(QSpinBox, "max_num_paths_per_src")
            tx_power = tx_power or window.findChild(QDoubleSpinBox, "tx_power_dbm")
            unlimited = unlimited or window.findChild(QCheckBox, "max_paths_unlimited")
            if hasattr(window, "_apply_request_to_controls"):
                main_window = window

        assert max_depth is not None
        assert max_paths is not None
        assert tx_power is not None
        assert unlimited is not None
        assert main_window is not None
        assert main_window.findChild(QWidget, "scene_3d_view") is not None
        assert main_window.findChild(QScrollArea, "config_scroll") is not None
        observed["view_buttons"] = {
            button.objectName()
            for button in main_window.findChildren(QPushButton)
            if button.objectName().startswith("view_")
        }
        observed["config_sections"] = {
            section.objectName()
            for section in main_window.findChildren(QGroupBox)
            if section.objectName().startswith("config_section_")
        }
        observed["object_choices"] = [
            main_window.object_name.itemText(index)
            for index in range(main_window.object_name.count())
        ]

        observed["max_depth_minimum"] = max_depth.minimum()
        observed["tx_power_initial"] = tx_power.value()
        observed["dry_run_initial_checked"] = main_window.dry_run.isChecked()
        observed["max_paths_initial_enabled"] = max_paths.isEnabled()
        unlimited.setChecked(True)
        observed["max_paths_disabled_when_unlimited"] = not max_paths.isEnabled()
        main_window.show_antenna_diagrams()
        observed["antenna_diagrams_visible"] = len(main_window.view._antenna_diagram_items)
        main_window.view.hide_antenna_diagrams()
        observed["antenna_diagrams_hidden"] = len(main_window.view._antenna_diagram_items)

        request = SimulationRequest()
        request.scene.ensure_defaults()
        request.scene.scenario_path = custom_scenario
        request.scene.name = custom_scenario.stem
        request.scene.radiomap = RadiomapConfig(enabled=True)
        request.scene.radiomap.ue_template.panel.pattern = "dipole"
        request.scene.radiomap.ue_template.panel.element_diagram = "dipole"
        request.scene.radiomap.ue_template.panel.rows = 2
        request.scene.radiomap.ue_template.panel.cols = 3
        request.sionna.tx_power_dbm = 12.5
        main_window._apply_request_to_controls(request)
        main_window._select_entity_by_id("bs0")
        roundtrip = main_window._request_from_controls()
        observed["scenario_path"] = str(roundtrip.scene.scenario_path)
        observed["scenario_name"] = roundtrip.scene.name
        observed["tx_power_roundtrip"] = roundtrip.sionna.tx_power_dbm
        observed["radiomap_request_ue_count"] = len(roundtrip.scene.user_equipments)
        observed["radiomap_template_pattern"] = roundtrip.scene.radiomap.ue_template.panel.pattern
        observed["radiomap_template_shape"] = (
            roundtrip.scene.radiomap.ue_template.panel.rows,
            roundtrip.scene.radiomap.ue_template.panel.cols,
        )
        main_window._set_radiomap_bounds_from_view(4.0, -2.0, 3.0, -1.0)
        radiomap_bounds_roundtrip = main_window._request_from_controls()
        observed["dragged_radiomap_bounds"] = (
            radiomap_bounds_roundtrip.scene.radiomap.x_min,
            radiomap_bounds_roundtrip.scene.radiomap.x_max,
            radiomap_bounds_roundtrip.scene.radiomap.y_min,
            radiomap_bounds_roundtrip.scene.radiomap.y_max,
        )

        main_window._select_entity_by_id("ue0")
        main_window._set_combo_value(main_window.trajectory_kind, "linear")
        ue = main_window.design.user_equipments[0]
        observed["linear_trajectory_from_selection"] = (
            ue.trajectory.kind,
            ue.trajectory.points[0],
            ue.trajectory.points[1],
        )
        main_window._move_trajectory_point_from_view("ue0", 1, (2.0, 3.0, 4.0))
        observed["dragged_linear_endpoint"] = main_window.design.user_equipments[0].trajectory.points[1]
        main_window._transform_trajectory_endpoint_from_view(
            "ue0",
            1,
            (2.0, 3.0, 4.0),
            (0.1, 0.2, 0.3),
        )
        observed["dragged_endpoint_orientation"] = (
            main_window.design.user_equipments[0].trajectory.orientation_rad_points[0],
            main_window.design.user_equipments[0].trajectory.orientation_rad_points[1],
        )
        main_window._set_combo_value(main_window.trajectory_kind, "curve")
        main_window.add_curve_middle_point()
        main_window._move_trajectory_handle_from_view("ue0", 0, "out", (0.0, 4.0, 9.0))
        curve = main_window.design.user_equipments[0].trajectory
        observed["curve_trajectory_edit"] = (
            curve.kind,
            len(curve.points),
            curve.bezier_handles[0][1],
            {point[2] for point in curve.points},
        )
        main_window.add_object()
        observed["added_object_position"] = main_window.design.objects[-1].position
        main_window.samples.setValue(12)
        sampled_roundtrip = main_window._request_from_controls()
        observed["scene_samples_are_shared"] = (
            main_window.design.user_equipments[0].trajectory.samples,
            sampled_roundtrip.scene.objects[0].trajectory.samples,
            len(sampled_roundtrip.scene.user_equipments),
        )

        main_window._select_entity_by_id("bs0")
        observed["bs_antenna_selected_label"] = main_window.bs_antenna_selected.text()
        observed["ue_antenna_unselected_label"] = main_window.ue_antenna_selected.text()
        main_window.bs_rows.setValue(7)
        main_window.bs_cols.setValue(8)
        main_window._set_combo_value(main_window.bs_pattern, "dipole")
        edited_bs_roundtrip = main_window._request_from_controls()
        observed["selected_bs_antenna_edit"] = (
            edited_bs_roundtrip.scene.base_stations[0].panel.rows,
            edited_bs_roundtrip.scene.base_stations[0].panel.cols,
            edited_bs_roundtrip.scene.base_stations[0].panel.pattern,
        )

        main_window._select_entity_by_id("ue0")
        observed["ue_antenna_selected_label"] = main_window.ue_antenna_selected.text()
        observed["bs_antenna_unselected_label"] = main_window.bs_antenna_selected.text()
        main_window.ue_rows.setValue(4)
        main_window.ue_cols.setValue(5)
        main_window._set_combo_value(main_window.ue_pattern, "tr38901")
        observed["selected_ue_antenna_edit"] = (
            main_window.design.user_equipments[0].panel.rows,
            main_window.design.user_equipments[0].panel.cols,
            main_window.design.user_equipments[0].panel.pattern,
        )
        main_window.apply_radiomap_template()
        applied_roundtrip = main_window._request_from_controls()
        observed["applied_radiomap_template_pattern"] = applied_roundtrip.scene.radiomap.ue_template.panel.pattern
        observed["applied_radiomap_template_shape"] = (
            applied_roundtrip.scene.radiomap.ue_template.panel.rows,
            applied_roundtrip.scene.radiomap.ue_template.panel.cols,
        )

        main_window._select_entity_by_id("bs0")
        original_position = main_window.design.base_stations[0].position
        original_orientation = main_window.design.base_stations[0].orientation_rad
        main_window._begin_view_transform("bs0")
        main_window._transform_entity_from_view("bs0", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
        original_set_scene = main_window.view.set_scene
        observed["undo_reset_camera_values"] = []

        def record_set_scene(*args, **kwargs):
            observed["undo_reset_camera_values"].append(kwargs.get("reset_camera", True))
            return original_set_scene(*args, **kwargs)

        main_window.view.set_scene = record_set_scene
        main_window.undo()
        observed["undo_restored_bs"] = (
            main_window.design.base_stations[0].position == original_position
            and main_window.design.base_stations[0].orientation_rad == original_orientation
        )
        main_window.view.set_scene = original_set_scene

        preview_original_position = main_window.design.user_equipments[0].position
        main_window._simulation_preview_design = deepcopy(main_window.design)
        preview_set_scene = main_window.view.set_scene
        observed["simulation_preview_positions"] = []

        def record_preview_scene(_asset, design, *_args, **_kwargs):
            observed["simulation_preview_positions"].append(design.user_equipments[0].position)
            return preview_set_scene(_asset, design, *_args, **_kwargs)

        main_window.view.set_scene = record_preview_scene
        main_window._on_timeframe_pose(
            1,
            {"ue0": (9.0, 8.0, 7.0), "bs0": (1.0, 2.0, 3.0)},
            {"ue0": (0.4, 0.5, 0.6)},
            {},
            {},
        )
        observed["simulation_preview_design_position"] = main_window._simulation_preview_design.user_equipments[0].position
        observed["simulation_preview_actual_position"] = main_window.design.user_equipments[0].position
        observed["simulation_preview_orientation"] = main_window._simulation_preview_design.user_equipments[0].orientation_rad
        main_window._on_worker_stopped()
        observed["simulation_preview_cleared"] = main_window._simulation_preview_design is None
        observed["simulation_preview_actual_restored"] = main_window.design.user_equipments[0].position == preview_original_position

        for window in self.topLevelWidgets():
            window.close()
        return 0

    monkeypatch.setattr(QApplication, "exec", fake_exec)

    assert gui_app.main() == 0
    assert observed == {
        "max_depth_minimum": 0,
        "tx_power_initial": 44.0,
        "dry_run_initial_checked": False,
        "max_paths_initial_enabled": True,
        "max_paths_disabled_when_unlimited": True,
        "antenna_diagrams_visible": 2,
        "antenna_diagrams_hidden": 0,
        "object_choices": ["CAR_obj", "DRONE_obj"],
        "scenario_path": str(custom_scenario),
        "scenario_name": custom_scenario.stem,
        "tx_power_roundtrip": 12.5,
        "radiomap_request_ue_count": 1,
        "radiomap_template_pattern": "dipole",
        "radiomap_template_shape": (2, 3),
        "dragged_radiomap_bounds": (-2.0, 4.0, -1.0, 3.0),
        "linear_trajectory_from_selection": ("linear", (0.0, 0.0, 1.5), (5.0, 0.0, 1.5)),
        "dragged_linear_endpoint": (2.0, 3.0, 4.0),
        "dragged_endpoint_orientation": ((0.0, 0.0, 0.0), (0.1, 0.2, 0.3)),
        "curve_trajectory_edit": ("curve", 3, (0.0, 4.0, 1.5), {1.5}),
        "added_object_position": (0.0, 0.0, 0.75),
        "scene_samples_are_shared": (12, 12, 1),
        "bs_antenna_selected_label": "Selected BS: bs0",
        "ue_antenna_unselected_label": "Selected UE: none",
        "selected_bs_antenna_edit": (7, 8, "dipole"),
        "ue_antenna_selected_label": "Selected UE: ue0",
        "bs_antenna_unselected_label": "Selected BS: none",
        "selected_ue_antenna_edit": (4, 5, "tr38901"),
        "applied_radiomap_template_pattern": "tr38901",
        "applied_radiomap_template_shape": (4, 5),
        "undo_reset_camera_values": [False],
        "undo_restored_bs": True,
        "simulation_preview_positions": [(9.0, 8.0, 7.0), (0.0, 0.0, 1.5)],
        "simulation_preview_design_position": (9.0, 8.0, 7.0),
        "simulation_preview_actual_position": (0.0, 0.0, 1.5),
        "simulation_preview_orientation": (0.4, 0.5, 0.6),
        "simulation_preview_cleared": True,
        "simulation_preview_actual_restored": True,
        "view_buttons": {
            "view_xy_posz",
            "view_xy_negz",
            "view_yz_posx",
            "view_yz_negx",
            "view_xz_posy",
            "view_xz_negy",
            "view_antenna_diagrams",
        },
        "config_sections": {
            "config_section_actions",
            "config_section_bs_antenna",
            "config_section_ue_antenna",
            "config_section_channel",
            "config_section_pose",
            "config_section_project",
            "config_section_radiomap",
            "config_section_tracing",
            "config_section_trajectory",
        },
    }
