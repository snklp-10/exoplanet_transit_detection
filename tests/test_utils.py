import numpy as np

from exotransit.utils import load_config, resolve, seed_everything


def test_config_loads_and_has_expected_keys():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["split"]["by"] == "kepid"
    assert abs(cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"] - 1.0) < 1e-9


def test_resolve_is_absolute():
    assert resolve("data/splits.csv").is_absolute()


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
