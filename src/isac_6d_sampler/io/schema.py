from __future__ import annotations

from pathlib import Path

import h5py

_REQUIRED_SAMPLE_GROUPS = ("metadata", "parameters", "timeframes")
_REQUIRED_PARAMETER_GROUPS = ("antenna_params", "subband_f_vectors", "vna_params")


def validate_reference_h5(path: Path | str) -> list[str]:
    """Return schema/consistency errors for an ISAC reference-style HDF5 file."""
    errors: list[str] = []
    path = Path(path)
    if not path.exists():
        return [f"{path} does not exist"]

    try:
        with h5py.File(path, "r") as h5:
            _validate_root(h5, errors)
    except OSError as exc:
        return [f"{path} is not a readable HDF5 file: {exc}"]
    return errors


def _validate_root(h5: h5py.File, errors: list[str]) -> None:
    if "scenarios" not in h5:
        errors.append("missing /scenarios group")
        return
    scenarios = h5["scenarios"]
    if not isinstance(scenarios, h5py.Group) or not scenarios:
        errors.append("/scenarios must contain at least one scenario group")
        return

    for scenario_name, scenario in scenarios.items():
        if not isinstance(scenario, h5py.Group):
            errors.append(f"/scenarios/{scenario_name} must be a group")
            continue
        sample_names = [name for name in scenario.keys() if name != "metadata"]
        if not sample_names:
            errors.append(f"/scenarios/{scenario_name} contains no sample groups")
        for sample_name in sample_names:
            sample = scenario[sample_name]
            if isinstance(sample, h5py.Group):
                _validate_sample(f"/scenarios/{scenario_name}/{sample_name}", sample, errors)
            else:
                errors.append(f"/scenarios/{scenario_name}/{sample_name} must be a group")


def _validate_sample(path: str, sample: h5py.Group, errors: list[str]) -> None:
    _require_groups(path, sample, _REQUIRED_SAMPLE_GROUPS, errors)
    link_params = None
    device_params = None
    if "parameters" in sample and isinstance(sample["parameters"], h5py.Group):
        _validate_parameters(f"{path}/parameters", sample["parameters"], errors)
        if "device_params" in sample["parameters"] and isinstance(sample["parameters/device_params"], h5py.Group):
            device_params = sample["parameters/device_params"]
        if "link_params" in sample["parameters"] and isinstance(sample["parameters/link_params"], h5py.Group):
            link_params = sample["parameters/link_params"]
            _validate_link_device_refs(f"{path}/parameters/link_params", link_params, device_params, errors)
    if "timeframes" in sample and isinstance(sample["timeframes"], h5py.Group):
        _validate_timeframes(f"{path}/timeframes", sample["timeframes"], link_params, errors)


def _validate_parameters(path: str, params: h5py.Group, errors: list[str]) -> None:
    _require_groups(path, params, _REQUIRED_PARAMETER_GROUPS, errors)
    if "f_vector" not in params:
        errors.append(f"{path}/f_vector is required")
        return
    f_vector = params["f_vector"]
    if not isinstance(f_vector, h5py.Dataset):
        errors.append(f"{path}/f_vector must be a dataset")
        return
    if f_vector.ndim != 2 or f_vector.shape[0] != 1 or f_vector.shape[1] < 1:
        errors.append(f"{path}/f_vector must have shape (1, n_frequencies)")

    if "subband_f_vectors" in params and isinstance(params["subband_f_vectors"], h5py.Group):
        _validate_subbands(f"{path}/subband_f_vectors", params["subband_f_vectors"], f_vector.shape[1], errors)
    if "vna_params" in params and isinstance(params["vna_params"], h5py.Group):
        _validate_vna(f"{path}/vna_params", params["vna_params"], errors)
    if "channel_params" in params and isinstance(params["channel_params"], h5py.Group):
        _validate_channel_params(f"{path}/channel_params", params["channel_params"], errors)
    if "device_params" in params and isinstance(params["device_params"], h5py.Group):
        _validate_device_params(f"{path}/device_params", params["device_params"], errors)
    if "link_params" in params and isinstance(params["link_params"], h5py.Group):
        _validate_link_params(f"{path}/link_params", params["link_params"], errors)


def _validate_subbands(path: str, group: h5py.Group, expected_total: int, errors: list[str]) -> None:
    subbands = [obj for _, obj in sorted(group.items()) if isinstance(obj, h5py.Dataset)]
    if not subbands:
        errors.append(f"{path} must contain at least one subband_f_vector_* dataset")
        return
    total = 0
    for dataset in subbands:
        if dataset.ndim != 2 or dataset.shape[0] != 1 or dataset.shape[1] < 1:
            errors.append(f"{dataset.name} must have shape (1, n_frequencies)")
        else:
            total += dataset.shape[1]
    if total != expected_total:
        errors.append(f"{path} frequency count {total} does not match f_vector count {expected_total}")


def _validate_vna(path: str, group: h5py.Group, errors: list[str]) -> None:
    n_subbands = int(group.attrs.get("n_subbands", 0))
    if n_subbands < 1:
        errors.append(f"{path} must define positive n_subbands")
    for name in ("subbands_f_start", "subbands_f_stop", "subbands_points"):
        if name not in group:
            errors.append(f"{path}/{name} is required")
            continue
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            errors.append(f"{path}/{name} must be a dataset")
        elif n_subbands and dataset.shape != (n_subbands,):
            errors.append(f"{path}/{name} must have shape ({n_subbands},)")


def _validate_channel_params(path: str, group: h5py.Group, errors: list[str]) -> None:
    mode = group.attrs.get("mode")
    sample_axis = group.attrs.get("sample_axis")
    if not mode:
        errors.append(f"{path} must define mode")
    if not sample_axis:
        errors.append(f"{path} must define sample_axis")
    if sample_axis in {"frequency_hz", "delay_bin_s"} and "sample_coordinates" not in group:
        errors.append(f"{path}/sample_coordinates is required for sample_axis={sample_axis}")
    if sample_axis == "delay_bin_s" and "delay_bin_width_s" not in group.attrs:
        errors.append(f"{path} must define delay_bin_width_s for delay_bin_s output")


def _validate_link_params(path: str, group: h5py.Group, errors: list[str]) -> None:
    link_names = [name for name, obj in group.items() if isinstance(obj, h5py.Group)]
    n_links = int(group.attrs.get("n_links", 0))
    if n_links < 1:
        errors.append(f"{path} must define positive n_links")
    elif n_links != len(link_names):
        errors.append(f"{path} n_links {n_links} does not match link group count {len(link_names)}")
    for name in link_names:
        link = group[name]
        for attr in (
            "rx_id",
            "tx_id",
            "rx_index",
            "tx_index",
            "rx_device_group",
            "tx_device_group",
            "direction",
            "is_monostatic",
        ):
            if attr not in link.attrs:
                errors.append(f"{path}/{name} missing attr {attr}")


def _validate_device_params(path: str, group: h5py.Group, errors: list[str]) -> None:
    device_names = [name for name, obj in group.items() if isinstance(obj, h5py.Group)]
    n_devices = int(group.attrs.get("n_devices", 0))
    if n_devices < 1:
        errors.append(f"{path} must define positive n_devices")
    elif n_devices != len(device_names):
        errors.append(f"{path} n_devices {n_devices} does not match device group count {len(device_names)}")
    for name in device_names:
        device = group[name]
        for attr in ("index", "entity_id", "entity_type", "antenna_group"):
            if attr not in device.attrs:
                errors.append(f"{path}/{name} missing attr {attr}")


def _validate_link_device_refs(
    path: str,
    link_params: h5py.Group,
    device_params: h5py.Group | None,
    errors: list[str],
) -> None:
    if device_params is None:
        return
    for name, link in link_params.items():
        if not isinstance(link, h5py.Group):
            continue
        for side in ("rx", "tx"):
            group_attr = f"{side}_device_group"
            id_attr = f"{side}_id"
            if group_attr not in link.attrs or id_attr not in link.attrs:
                continue
            device_group_name = str(link.attrs[group_attr])
            if device_group_name not in device_params:
                errors.append(f"{path}/{name} references missing device_params/{device_group_name}")
                continue
            device = device_params[device_group_name]
            if not isinstance(device, h5py.Group):
                errors.append(f"{device.name} must be a group")
                continue
            if "entity_id" in device.attrs and device.attrs["entity_id"] != link.attrs[id_attr]:
                errors.append(
                    f"{path}/{name} attr {id_attr}={link.attrs[id_attr]!r} does not match "
                    f"{device.name} attr entity_id={device.attrs['entity_id']!r}"
                )


def _validate_timeframes(
    path: str,
    timeframes: h5py.Group,
    link_params: h5py.Group | None,
    errors: list[str],
) -> None:
    if not timeframes:
        errors.append(f"{path} must contain at least one timeframe")
        return
    for timeframe_name, timeframe in timeframes.items():
        if not isinstance(timeframe, h5py.Group):
            errors.append(f"{path}/{timeframe_name} must be a group")
            continue
        tf_path = f"{path}/{timeframe_name}"
        _require_groups(tf_path, timeframe, ("h", "parameters", "timestamps"), errors)
        n_channels = None
        if "parameters" in timeframe and isinstance(timeframe["parameters"], h5py.Group):
            n_channels = int(timeframe["parameters"].attrs.get("n_channels", 0))
            if n_channels < 1:
                errors.append(f"{tf_path}/parameters must define positive n_channels")
        if "h" in timeframe and "timestamps" in timeframe:
            tau_group = timeframe["tau"] if "tau" in timeframe and isinstance(timeframe["tau"], h5py.Group) else None
            _validate_link_datasets(
                tf_path,
                timeframe["h"],
                timeframe["timestamps"],
                tau_group,
                link_params,
                n_channels,
                errors,
            )


def _validate_link_datasets(
    path: str,
    h_group: h5py.Group,
    timestamp_group: h5py.Group,
    tau_group: h5py.Group | None,
    link_params: h5py.Group | None,
    expected_channel_count: int | None,
    errors: list[str],
) -> None:
    if not isinstance(h_group, h5py.Group) or not isinstance(timestamp_group, h5py.Group):
        return
    channel_names = [name for name, obj in h_group.items() if isinstance(obj, h5py.Dataset)]
    if not channel_names:
        errors.append(f"{path}/h must contain at least one channel dataset")
    if expected_channel_count is not None and expected_channel_count != len(channel_names):
        errors.append(
            f"{path}/parameters n_channels {expected_channel_count} does not match "
            f"{path}/h dataset count {len(channel_names)}"
        )
    for name, dataset in h_group.items():
        if not isinstance(dataset, h5py.Dataset):
            errors.append(f"{path}/h/{name} must be a dataset")
            continue
        if link_params is not None:
            _require_channel_dataset_attrs(path, "h", name, dataset, errors)
            _validate_channel_link_metadata(path, name, dataset, link_params, errors)
        if name not in timestamp_group:
            errors.append(f"{path}/timestamps/{name} missing for channel dataset")
        elif isinstance(timestamp_group[name], h5py.Dataset):
            timestamp = timestamp_group[name]
            if timestamp.shape != dataset.shape[:-1]:
                errors.append(
                    f"{path}/timestamps/{name} shape {timestamp.shape} must match "
                    f"{path}/h/{name} leading shape {dataset.shape[:-1]}"
                )
        else:
            errors.append(f"{path}/timestamps/{name} must be a dataset")

        if tau_group is not None and name in tau_group:
            tau = tau_group[name]
            if not isinstance(tau, h5py.Dataset):
                errors.append(f"{path}/tau/{name} must be a dataset")
            elif tau.ndim != dataset.ndim or tau.shape[:-1] != dataset.shape[:-1]:
                errors.append(
                    f"{path}/tau/{name} leading shape {tau.shape[:-1]} must match "
                    f"{path}/h/{name} leading shape {dataset.shape[:-1]}"
                )
            elif link_params is not None:
                _require_channel_dataset_attrs(path, "tau", name, tau, errors)


def _require_channel_dataset_attrs(
    path: str,
    group_name: str,
    name: str,
    dataset: h5py.Dataset,
    errors: list[str],
) -> None:
    for attr in ("rx_id", "tx_id", "rx_index", "tx_index", "dataset_role", "sample_axis"):
        if attr not in dataset.attrs:
            errors.append(f"{path}/{group_name}/{name} missing attr {attr}")


def _validate_channel_link_metadata(
    path: str,
    name: str,
    dataset: h5py.Dataset,
    link_params: h5py.Group,
    errors: list[str],
) -> None:
    if name not in link_params:
        errors.append(f"{path}/h/{name} missing matching parameters/link_params/{name}")
        return
    link = link_params[name]
    if not isinstance(link, h5py.Group):
        errors.append(f"{link.name} must be a group")
        return
    for attr in ("rx_id", "tx_id", "rx_index", "tx_index", "is_monostatic"):
        if attr in dataset.attrs and attr in link.attrs and dataset.attrs[attr] != link.attrs[attr]:
            errors.append(
                f"{path}/h/{name} attr {attr}={dataset.attrs[attr]!r} does not match "
                f"{link.name} attr {attr}={link.attrs[attr]!r}"
            )


def _require_groups(path: str, parent: h5py.Group, names: tuple[str, ...], errors: list[str]) -> None:
    for name in names:
        if name not in parent:
            errors.append(f"{path}/{name} group is required")
        elif not isinstance(parent[name], h5py.Group):
            errors.append(f"{path}/{name} must be a group")
