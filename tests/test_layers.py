import pytest

from lorascan.layers import load_locations, resolve_location
from lorascan.runs import CONFIGS

REG = load_locations(CONFIGS / "locations.yaml")


def test_proportions_partition_the_model():
    parts = [resolve_location(n, 28, 0, REG)["resolved_layers"] for n in ("early", "middle", "late")]
    assert sorted(sum(parts, [])) == list(range(28))


def test_late_is_the_end_of_the_stack():
    assert resolve_location("late", 28, 0, REG)["resolved_layers"][-1] == 27


def test_random_is_seeded_and_recorded():
    a = resolve_location("random", 28, 42, REG)
    assert a == resolve_location("random", 28, 42, REG)
    assert a["resolved_layers"] != resolve_location("random", 28, 43, REG)["resolved_layers"]
    assert a["seed"] == 42 and len(a["resolved_layers"]) == round(0.33 * 28)


def test_random_fixed_seed_ignores_run_seed():
    spec = {"type": "random", "fraction": 0.25, "seed": 7}
    assert resolve_location(spec, 24, 1)["resolved_layers"] == resolve_location(spec, 24, 2)["resolved_layers"]


def test_stride_and_explicit():
    assert resolve_location("every_other", 8, 0, REG)["resolved_layers"] == [0, 2, 4, 6]
    assert resolve_location({"type": "explicit", "layers": [3, 1]}, 8, 0)["resolved_layers"] == [1, 3]


def test_validation():
    with pytest.raises(KeyError):
        resolve_location("nope", 8, 0, REG)
    with pytest.raises(ValueError):
        resolve_location({"type": "explicit", "layers": [8]}, 8, 0)
