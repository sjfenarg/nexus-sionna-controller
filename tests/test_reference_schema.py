import h5py
import numpy as np
import pytest

from isac_6d_sampler.cli import main
from isac_6d_sampler.core.model import SimulationRequest
from isac_6d_sampler.io import reference_h5
from isac_6d_sampler.io.reference_h5 import ReferenceH5Writer
from isac_6d_sampler.io.schema import validate_reference_h5
from isac_6d_sampler.sim.dry_run import DryRunSimulator


def test_validate_reference_h5_accepts_generated_dry_run_output(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "generated.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    assert validate_reference_h5(out) == []


def test_validate_reference_h5_accepts_legacy_reference_without_link_metadata(tmp_path):
    out = tmp_path / "legacy_reference.h5"
    with h5py.File(out, "w") as h5:
        sample = h5.create_group("scenarios/legacy_scene/s000")
        sample.create_group("metadata")
        params = sample.create_group("parameters")
        params.create_dataset("f_vector", data=np.array([[77.0e9, 78.0e9]], dtype=np.float64))
        subbands = params.create_group("subband_f_vectors")
        subbands.create_dataset("subband_f_vector_0", data=np.array([[77.0e9, 78.0e9]], dtype=np.float64))
        vna = params.create_group("vna_params")
        vna.attrs["n_subbands"] = 1
        vna.create_dataset("subbands_f_start", data=np.array([77.0e9], dtype=np.float64))
        vna.create_dataset("subbands_f_stop", data=np.array([78.0e9], dtype=np.float64))
        vna.create_dataset("subbands_points", data=np.array([2], dtype=np.int64))
        params.create_group("antenna_params")

        tf = sample.create_group("timeframes/tf000")
        tf_params = tf.create_group("parameters")
        tf_params.attrs["n_channels"] = 1
        h_group = tf.create_group("h")
        h_group.create_dataset("rx0_tx0", data=np.zeros((1, 1, 2), dtype=np.complex64))
        timestamp_group = tf.create_group("timestamps")
        timestamp_group.create_dataset("rx0_tx0", data=np.array([["2026-01-01T00:00:00"]], dtype=object), dtype=h5py.string_dtype())

    assert validate_reference_h5(out) == []


def test_reference_writer_validates_output_by_default(monkeypatch, tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "generated.h5"
    monkeypatch.setattr(reference_h5, "validate_reference_h5", lambda _: ["synthetic schema failure"])

    with pytest.raises(ValueError, match="synthetic schema failure"):
        ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))


def test_reference_writer_can_skip_output_validation(monkeypatch, tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "generated.h5"
    monkeypatch.setattr(reference_h5, "validate_reference_h5", lambda _: ["synthetic schema failure"])

    ReferenceH5Writer(validate_output=False).write(out, request, DryRunSimulator().simulate(request))

    assert out.exists()


def test_validate_reference_h5_reports_missing_required_groups(tmp_path):
    out = tmp_path / "broken.h5"
    with h5py.File(out, "w") as h5:
        h5.create_group("scenarios/scene0/s000/parameters")

    errors = validate_reference_h5(out)

    assert "/scenarios/scene0/s000/metadata group is required" in errors
    assert "/scenarios/scene0/s000/timeframes group is required" in errors
    assert "/scenarios/scene0/s000/parameters/f_vector is required" in errors


def test_validate_reference_h5_reports_timestamp_shape_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_timestamp.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        timestamp_group = sample["timeframes/tf000/timestamps"]
        del timestamp_group["rx0_tx1"]
        timestamp_group.create_dataset("rx0_tx1", data=np.array(["bad-shape"], dtype=object), dtype=h5py.string_dtype())

    errors = validate_reference_h5(out)

    assert any("/timeframes/tf000/timestamps/rx0_tx1 shape" in error for error in errors)


def test_validate_reference_h5_reports_tau_shape_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_tau.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        tau_group = sample["timeframes/tf000/tau"]
        del tau_group["rx0_tx1"]
        tau_group.create_dataset("rx0_tx1", data=np.zeros((2,), dtype=np.float64))

    errors = validate_reference_h5(out)

    assert any("/timeframes/tf000/tau/rx0_tx1 leading shape" in error for error in errors)


def test_validate_reference_h5_reports_channel_count_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_channel_count.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        sample["timeframes/tf000/parameters"].attrs["n_channels"] = 999

    errors = validate_reference_h5(out)

    assert any("n_channels 999 does not match" in error for error in errors)


def test_validate_reference_h5_reports_link_params_count_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_link_count.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        sample["parameters/link_params"].attrs["n_links"] = 999

    errors = validate_reference_h5(out)

    assert any("link_params n_links 999 does not match" in error for error in errors)


def test_validate_reference_h5_reports_link_params_metadata_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_link_metadata.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        sample["timeframes/tf000/h/rx0_tx1"].attrs["rx_id"] = "wrong_ue"

    errors = validate_reference_h5(out)

    assert any("h/rx0_tx1 attr rx_id='wrong_ue' does not match" in error for error in errors)


def test_validate_reference_h5_reports_device_params_count_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_device_count.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        sample["parameters/device_params"].attrs["n_devices"] = 999

    errors = validate_reference_h5(out)

    assert any("device_params n_devices 999 does not match" in error for error in errors)


def test_validate_reference_h5_reports_link_device_reference_mismatch(tmp_path):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "bad_link_device_ref.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    with h5py.File(out, "r+") as h5:
        sample = h5[f"scenarios/{request.scene.name}/{request.sample_id}"]
        sample["parameters/link_params/rx0_tx1"].attrs["rx_device_group"] = "device1"

    errors = validate_reference_h5(out)

    assert any("link_params/rx0_tx1 attr rx_id='ue0' does not match" in error for error in errors)


def test_cli_validate_h5_accepts_generated_output(tmp_path, capsys):
    request = SimulationRequest(dry_run=True, output_dir=tmp_path)
    request.scene.ensure_defaults()
    out = tmp_path / "generated.h5"
    ReferenceH5Writer().write(out, request, DryRunSimulator().simulate(request))

    assert main(["--validate-h5", str(out)]) == 0

    captured = capsys.readouterr()
    assert "valid" in captured.out


def test_cli_validate_h5_reports_errors(tmp_path, capsys):
    out = tmp_path / "broken.h5"
    with h5py.File(out, "w") as h5:
        h5.create_group("metadata")

    assert main(["--validate-h5", str(out)]) == 1

    captured = capsys.readouterr()
    assert "missing /scenarios group" in captured.err
