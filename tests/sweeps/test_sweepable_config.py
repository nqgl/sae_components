from saeco.sweeps.sweepable_config import SweepableConfig, SweepVar, Swept


class LeafConfig(SweepableConfig):
    x: int


class RootConfig(SweepableConfig):
    leaf: LeafConfig
    y: int


class SingleValueConfig(SweepableConfig):
    x: int


def test_to_swept_nodes_is_cached_property() -> None:
    var = SweepVar(name="offset", values=[1, 2])
    cfg = RootConfig(leaf=LeafConfig(x=Swept(10, 20)), y=var + 5)

    nodes = cfg.sweep_info_tree

    assert cfg.sweep_info_tree is nodes
    assert nodes.swept_combinations_count_including_vars() == 4
    assert nodes.get_paths_to_sweep_expressions() == [["y"]]
    assert "to_swept_nodes" not in cfg.model_dump()


def test_select_instance_by_index_uses_cached_swept_nodes() -> None:
    cfg = RootConfig(leaf=LeafConfig(x=Swept(10, 20)), y=Swept(30, 40))

    assert cfg.sweep_info_tree is cfg.sweep_info_tree
    assert cfg.select_instance_by_index(0).is_concrete()


def test_to_swept_nodes_cache_is_cleared_on_model_copy_update() -> None:
    cfg = SingleValueConfig(x=Swept(10, 20))
    assert cfg.sweep_info_tree.swept_fields

    copied = cfg.model_copy(update={"x": 30})

    assert copied.sweep_info_tree.swept_fields == {}


def test_to_swept_nodes_cache_is_cleared_on_field_assignment() -> None:
    cfg = SingleValueConfig(x=Swept(10, 20))
    assert cfg.sweep_info_tree.swept_fields

    cfg.x = 30

    assert cfg.sweep_info_tree.swept_fields == {}
