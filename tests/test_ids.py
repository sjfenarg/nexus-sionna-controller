from isac_6d_sampler.core.ids import unique_entity_id


def test_unique_entity_id_fills_first_available_gap():
    assert unique_entity_id("bs", {"bs0", "bs2", "ue0"}) == "bs1"


def test_unique_entity_id_uses_shared_namespace():
    assert unique_entity_id("ue", {"bs0", "ue0", "ue1", "car0"}) == "ue2"


def test_unique_entity_id_normalizes_object_prefix():
    assert unique_entity_id("CAR obj", {"car_obj0"}) == "car_obj1"


def test_unique_entity_id_falls_back_for_empty_prefix():
    assert unique_entity_id("", {"entity0"}) == "entity1"
