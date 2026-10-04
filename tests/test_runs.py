import copy

from lorascan import layers, runs

LOC = layers.resolve_location("late", 28, 42, layers.load_locations(runs.CONFIGS / "locations.yaml"))


def cfgs(name="qwen_gsm8k_grpo.yaml"):
    return runs.load_and_resolve(runs.CONFIGS / "experiments" / name)


def test_sweep_expands_one_experiment_per_value():
    names = [c["name"] for c in cfgs()]
    assert names == [f"qwen15_gsm8k_grpo_{x}" for x in ("early", "middle", "late", "random", "all")]


def test_resolution_inlines_components_and_defaults():
    c = cfgs()[0]
    assert c["model"]["num_layers"] == 28 and c["dataset"]["evaluator"] == "gsm8k"
    assert c["lora"]["rank"] == 1 and c["training"]["num_generations"] == 4
    assert c["training"]["per_device_train_batch_size"] == 8  # from defaults.yaml


def test_scale_sweep_changes_model():
    assert [c["model"]["name"] for c in cfgs("scale_gsm8k_grpo.yaml")] == \
        ["qwen25-0.5b", "qwen25-1.5b", "qwen25-3b", "gemma2-2b"]


def test_hash_is_deterministic_and_sensitive():
    c = cfgs()[2]
    h = runs.run_hash(c, 42, LOC, "abc")
    assert h == runs.run_hash(copy.deepcopy(c), 42, LOC, "abc")
    assert h != runs.run_hash(c, 43, LOC, "abc")
    assert h != runs.run_hash(c, 42, LOC, "def")
    other = {**LOC, "resolved_layers": LOC["resolved_layers"][:-1]}
    assert h != runs.run_hash(c, 42, other, "abc")
    c2 = copy.deepcopy(c); c2["lora"]["rank"] = 2
    assert h != runs.run_hash(c2, 42, LOC, "abc")


def test_hash_ignores_experiment_name():
    c = cfgs()[2]
    c2 = {**c, "name": "renamed"}
    assert runs.run_hash(c, 42, LOC, None) == runs.run_hash(c2, 42, LOC, None)


def test_run_lifecycle_and_collect(tmp_path):
    c = cfgs()[2]
    run = runs.Run(tmp_path, "abc123")
    run.start(c, 42, LOC, "v")
    assert not run.complete
    run.finish([{"item_id": "t-0", "correct": True}], {"eval": {"accuracy": 1.0, "n": 1}, "train": {"steps": 5}})
    assert run.complete and runs.load_predictions(tmp_path, "abc123")[0]["correct"]
    row = runs.collect_results(tmp_path)[0]
    assert row["location"] == "late" and row["accuracy"] == 1.0 and row["train_steps"] == 5


def test_registry(tmp_path):
    p = tmp_path / "reg.yaml"
    runs.register_run("exp", 42, "abc", p)
    runs.register_run("exp", 123, "def", p)
    assert runs._read_yaml(p) == {"exp": {42: "abc", 123: "def"}}
