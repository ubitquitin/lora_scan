"""Config resolution, run identity (hashing), and on-disk run records.

    experiment YAML --(load_and_resolve)--> resolved config(s)
    resolved config + seed + resolved layers + code version --(run_hash)--> run_id
    runs/<run_id>/{config.yaml, metadata.json, checkpoint/, predictions.jsonl, metrics.json}
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("LORASCAN_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"
REGISTRY = ROOT / "predictions" / "registry.yaml"

METHODS = {"none", "sft", "grpo"}


# ------------------------------------------------------------------ config resolution
def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(d: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def _slug(value) -> str:
    if isinstance(value, dict):
        value = value.get("name") or json.dumps(value, sort_keys=True)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


def expand_sweep(raw: dict) -> list[dict]:
    """`sweep: {dotted.key: [v1, v2]}` -> one experiment per combination (cartesian)."""
    base = {k: v for k, v in raw.items() if k != "sweep"}
    sweep = raw.get("sweep") or {}
    if not sweep:
        return [base]
    keys = list(sweep)
    variants = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        v = copy.deepcopy(base)
        for k, val in zip(keys, combo):
            set_path(v, k, val)
        v["name"] = "_".join([base["name"], *(_slug(x) for x in combo)])
        variants.append(v)
    return variants


def load_component(kind: str, name: str) -> dict:
    path = CONFIGS / kind / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No {kind[:-1]} config {name!r} (looked for {path})")
    return _read_yaml(path)


def resolve_experiment(raw: dict) -> dict:
    """Inline model/dataset/prompt references and merge defaults."""
    defaults = _read_yaml(CONFIGS / "defaults.yaml")
    cfg = copy.deepcopy(raw)
    for key in ("name", "method", "model", "dataset", "prompt"):
        if key not in cfg:
            raise ValueError(f"Experiment is missing required key {key!r}")
    if cfg["method"] not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {cfg['method']!r}")

    cfg["model"] = load_component("models", cfg["model"])
    cfg["dataset"] = load_component("datasets", cfg["dataset"])
    cfg["prompt"] = load_component("prompts", cfg["prompt"])
    cfg["lora"] = deep_merge(defaults["lora"], cfg.get("lora"))
    cfg["data"] = deep_merge(defaults["data"], cfg.get("data"))
    cfg["evaluation"] = deep_merge(defaults["evaluation"], cfg.get("evaluation"))
    cfg["reproducibility"] = deep_merge(defaults["reproducibility"], cfg.get("reproducibility"))
    cfg["training"] = deep_merge(defaults["training"].get(cfg["method"], {}), cfg.get("training"))
    cfg.setdefault("seeds", [42])
    return cfg


def load_and_resolve(experiment_path: str | Path) -> list[dict]:
    """Load an experiment file -> list of fully resolved experiments (sweep expanded)."""
    raw = _read_yaml(Path(experiment_path))
    return [resolve_experiment(v) for v in expand_sweep(raw)]


# ------------------------------------------------------------------ identity
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def code_version(root: Path = ROOT) -> str:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return run("rev-parse", "--short=12", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
    except Exception:
        return "nogit"


def run_hash(config: dict, seed: int, location: dict | None, version: str | None, length: int = 10) -> str:
    """Hash of everything that determines a run's outcome.

    Covers: resolved config (model, dataset, prompt, lora, training, ...), the seed, the
    *resolved layer indices* (so a random draw is part of identity) and the code version.
    The experiment name and seed list are excluded: identical configs dedupe across names.
    """
    cfg = {k: v for k, v in config.items() if k not in {"name", "seeds", "description"}}
    if cfg["method"] == "none":  # baseline has no adapter
        cfg.pop("lora", None), cfg.pop("training", None)
        location = None
    payload = {"config": cfg, "seed": seed, "location": location, "code_version": version}
    return hashlib.sha256(canonical(payload).encode()).hexdigest()[:length]


# ------------------------------------------------------------------ run records
class Run:
    def __init__(self, runs_dir: str | Path, run_id: str):
        self.id = run_id
        self.dir = Path(runs_dir) / run_id
        self.config_path = self.dir / "config.yaml"
        self.metadata_path = self.dir / "metadata.json"
        self.metrics_path = self.dir / "metrics.json"
        self.predictions_path = self.dir / "predictions.jsonl"
        self.checkpoint_dir = self.dir / "checkpoint"

    @property
    def complete(self) -> bool:
        return self.metrics_path.exists()

    def start(self, config: dict, seed: int, location: dict | None, version: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(
                {"run_id": self.id, "experiment_id": config["name"], "seed": seed,
                 "lora_location": location, "config": config},
                f, sort_keys=False)
        self._write_meta({
            "run_id": self.id, "experiment_id": config["name"], "seed": seed,
            "status": "running", "started_at": time.time(), "code_version": code_version(),
            "hash_code_version": version, "python": platform.python_version(), "host": platform.node(),
            "packages": _package_versions(),
        })

    def update_meta(self, **kw) -> None:
        meta = json.loads(self.metadata_path.read_text())
        meta.update(kw)
        self._write_meta(meta)

    def _write_meta(self, meta: dict) -> None:
        self.metadata_path.write_text(json.dumps(meta, indent=2, default=str))

    def finish(self, per_item: list[dict], metrics: dict) -> None:
        import math

        def sanitize_value(obj):
            """Convert inf/nan to None for JSON serialization."""
            if isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return None
            return obj

        def sanitize_dict(d):
            """Recursively sanitize a dict."""
            if isinstance(d, dict):
                return {k: sanitize_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [sanitize_dict(v) for v in d]
            else:
                return sanitize_value(d)

        with open(self.predictions_path, "w") as f:
            for row in per_item:
                f.write(json.dumps(sanitize_dict(row)) + "\n")
        self.metrics_path.write_text(json.dumps(sanitize_dict(metrics), indent=2))
        meta = json.loads(self.metadata_path.read_text())
        self.update_meta(status="complete", finished_at=time.time(),
                         duration_s=time.time() - meta["started_at"])

    def fail(self, error: str) -> None:
        if self.metadata_path.exists():
            self.update_meta(status="failed", error=error, finished_at=time.time())


def _package_versions() -> dict:
    from importlib import metadata
    out = {}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def register_run(experiment_id: str, seed: int, run_id: str, path: Path = REGISTRY) -> None:
    """Maintain predictions/registry.yaml: experiment_id -> seed -> run_id."""
    reg = _read_yaml(path) if path.exists() else {}
    reg.setdefault(experiment_id, {})[int(seed)] = run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(reg, f, sort_keys=True)


# ------------------------------------------------------------------ reading results back
def load_predictions(runs_dir: str | Path, run_id: str) -> list[dict]:
    with open(Path(runs_dir) / run_id / "predictions.jsonl") as f:
        return [json.loads(line) for line in f]


def collect_results(runs_dir: str | Path = RUNS) -> list[dict]:
    """One flat row per completed run (feed to pandas.DataFrame)."""
    rows = []
    for d in sorted(Path(runs_dir).glob("*/metrics.json")):
        run_dir = d.parent
        rec = _read_yaml(run_dir / "config.yaml")
        cfg, loc = rec["config"], rec.get("lora_location") or {}
        metrics = json.loads(d.read_text())
        rows.append({
            "run_id": rec["run_id"], "experiment_id": rec["experiment_id"], "seed": rec["seed"],
            "model": cfg["model"]["name"], "dataset": cfg["dataset"]["name"], "method": cfg["method"],
            "prompt": cfg["prompt"]["name"], "rank": cfg["lora"]["rank"] if cfg["method"] != "none" else None,
            "location": loc.get("name"), "n_lora_layers": len(loc.get("resolved_layers", [])) or None,
            **metrics.get("eval", {}),
            **{f"train_{k}": v for k, v in metrics.get("train", {}).items() if not isinstance(v, (list, dict))},
        })
    return rows
