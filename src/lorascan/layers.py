"""Translate named LoRA locations ("late", "random", ...) into concrete layer indices.

A location *name* is an abstraction, not a formula: it is looked up in a registry
(configs/locations.yaml, optionally extended per model) and resolved against the
model's layer count. The fully resolved spec - including the actual layer indices
and, for random draws, the seed - is what gets stored and hashed with the run.
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml


def load_locations(path: str | Path) -> dict:
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("locations", {})


def _bounds(spec: dict, num_layers: int) -> tuple[int, int]:
    start, end = spec["start"], spec["end"]
    if not (0.0 <= start < end <= 1.0):
        raise ValueError(f"Need 0 <= start < end <= 1, got start={start}, end={end}")
    return round(start * num_layers), round(end * num_layers)


def resolve_location(location, num_layers: int, seed: int, registry: dict | None = None) -> dict:
    """Resolve a location (registry name or inline spec dict) to a fully explicit spec.

    Returns a dict with at least: name, type, resolved_layers (sorted list of ints).
    Random locations also record the seed actually used.
    """
    if isinstance(location, str):
        registry = registry or {}
        if location not in registry:
            raise KeyError(f"Unknown LoRA location {location!r}. Known: {sorted(registry)}")
        spec = {"name": location, **registry[location]}
    elif isinstance(location, dict):
        spec = {"name": "custom", **location}
    else:
        raise TypeError(f"location must be a name or a dict, got {type(location).__name__}")

    kind = spec.get("type")
    if kind == "proportion":
        lo, hi = _bounds(spec, num_layers)
        layers = list(range(lo, hi))
    elif kind == "stride":
        lo, hi = _bounds(spec, num_layers)
        layers = list(range(lo, hi, int(spec.get("step", 2))))
    elif kind == "explicit":
        layers = sorted({int(i) for i in spec["layers"]})
    elif kind == "random":
        fraction = spec["fraction"]
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        k = max(1, round(fraction * num_layers))
        used_seed = spec["seed"] if spec.get("seed") is not None else seed
        layers = sorted(random.Random(used_seed).sample(range(num_layers), k))
        spec["seed"] = used_seed
    else:
        raise ValueError(f"Unknown location type {kind!r} for {spec['name']!r}")

    spec.pop("seed_from", None)
    if not layers:
        raise ValueError(f"Location {spec['name']!r} resolves to no layers for a {num_layers}-layer model")
    bad = [i for i in layers if not 0 <= i < num_layers]
    if bad:
        raise ValueError(f"Location {spec['name']!r} has layers {bad} outside 0..{num_layers - 1}")
    spec["num_layers"] = num_layers
    spec["resolved_layers"] = layers
    return spec
