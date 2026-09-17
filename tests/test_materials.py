import sys
import types

from isac_6d_sampler.sim.materials import (
    assign_calibrated_materials,
    calibrated_material_specs,
    material_spec_for_object_name,
    material_properties_at_frequency,
    tuned_diffuse_scattering_coefficient_at_frequency,
)


def test_material_catalog_resolves_scene_object_prefixes():
    car = material_spec_for_object_name("CAR_obj")
    concrete = material_spec_for_object_name("CONCRETE_obj")

    assert car is not None
    assert car.name == "my_car_metal"
    assert concrete is not None
    assert concrete.object_prefix == "CONCRETE"
    assert material_spec_for_object_name("UNKNOWN_obj") is None


def test_concrete_is_absorbent_at_measurement_bands_and_preserves_77ghz():
    concrete = material_spec_for_object_name("CONCRETE_obj")
    assert concrete is not None

    _, low_band_conductivity = material_properties_at_frequency(concrete, 10e9)
    _, automotive_conductivity = material_properties_at_frequency(concrete, 77e9)

    assert tuned_diffuse_scattering_coefficient_at_frequency(concrete, 10e9) == 0.98
    assert tuned_diffuse_scattering_coefficient_at_frequency(concrete, 77e9) < 0.3
    assert low_band_conductivity > automotive_conductivity


def test_assign_calibrated_materials_adds_catalog_and_assigns_matching_objects(monkeypatch):
    fake_rt = types.ModuleType("sionna.rt")
    fake_rt.RadioMaterial = _FakeRadioMaterial
    fake_sionna = types.ModuleType("sionna")
    fake_sionna.rt = fake_rt
    monkeypatch.setitem(sys.modules, "sionna", fake_sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", fake_rt)

    scene = _FakeScene(objects={"CAR_obj": _FakeObject("CAR_obj"), "TREE_obj": _FakeObject("TREE_obj")})

    assign_calibrated_materials(scene, "Outdoor6D_w_car")

    assert len(scene.radio_materials) == len(calibrated_material_specs())
    assert scene.objects["CAR_obj"].radio_material.name == "my_car_metal"
    assert scene.objects["TREE_obj"].radio_material is None


class _FakeRadioMaterial:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


class _FakeObject:
    def __init__(self, name):
        self.name = name
        self.radio_material = None


class _FakeScene:
    def __init__(self, objects):
        self.objects = objects
        self.radio_materials = {}

    def add(self, material):
        self.radio_materials[material.name] = material
