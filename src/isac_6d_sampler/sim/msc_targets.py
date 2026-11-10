"""Shared MSC construction for simulation and GUI RCS previews."""

from isac_6d_sampler.core.msc_models import trained_msc_model


def make_msc_target(*, name, object_type, frequency_hz, parameter_seed=0,
                    trained_model=None, **geometry):
    from sionna.rt.rcs import MSCSensingTarget

    options = dict(name=name, object_type=object_type, frequency_hz=frequency_hz, **geometry)
    if trained_model is None:
        return MSCSensingTarget(parameter_seed=parameter_seed, **options)
    model = trained_msc_model(trained_model)
    target = MSCSensingTarget.from_components(
        components=model["components"], reference_frequency_hz=model["reference_frequency_hz"],
        **options)
    target.scattering_model.model_provenance = {
        "model_id": trained_model,
        "source_run_id": model["source_run_id"],
        "source_model_sha256": model["source_model_sha256"],
        "source_engine_sha256": model["source_engine_sha256"],
        "polarization": "VV", "measurement_validated": False,
    }
    return target
