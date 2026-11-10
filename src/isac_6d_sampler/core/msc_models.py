"""Packaged, versioned trained MSC coefficients, independent of Sionna imports."""

import json
import math
from importlib.resources import files


def trained_msc_model(model_id: str) -> dict:
    if model_id != "agv_k5":
        raise ValueError(f"Unknown trained MSC model: {model_id}")
    model = json.loads(files("isac_6d_sampler").joinpath("data", "msc", f"{model_id}.json").read_text())
    if (model["target"] != "agv" or model["n_components"] != 5
            or len(model["components"]) != 5
            or model["shape_mode"] != "reflecting_surface_per_lobe"
            or model["polarization"] != "VV"):
        raise ValueError("Incompatible packaged AGV K5 model")
    return model


def trained_msc_dimensions(model_id: str) -> tuple[float, float, float]:
    centers = trained_msc_model(model_id)["components"]
    dimensions = tuple(2 * max(abs(c["mu"][axis]) for c in centers) for axis in range(3))
    if not all(math.isfinite(x) and x > 0 for x in dimensions):
        raise ValueError("Trained MSC centers must define a positive finite cuboid")
    return dimensions
