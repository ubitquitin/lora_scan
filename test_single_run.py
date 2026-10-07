#!/usr/bin/env python
"""Run a single experiment to see the full error traceback."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))

from lorascan import runs, layers as layers_mod, train, evaluate

# Load the config
configs = runs.load_and_resolve("configs/experiments/sft_sweep_mmlu_gsm8k.yaml")
cfg = configs[0]  # First experiment
seed = cfg["seeds"][0]  # First seed

print(f"Running: {cfg['name']} with seed={seed}")
print(f"Dataset: {cfg['dataset']['name']}")
print(f"Model: {cfg['model']['name']}")
print(f"Prompt: {cfg['prompt']['name']}")
print()

# Verify GPU
gpu_info = train.verify_gpu()
print(f"GPU: {gpu_info['device_name']} ({gpu_info['memory_gb']}GB)")
print()

# Setup
train.seed_everything(seed)
base_registry = layers_mod.load_locations(runs.CONFIGS / "locations.yaml")
registry = {**base_registry, **cfg["model"].get("locations", {})}
location = layers_mod.resolve_location(cfg["lora"]["location"], cfg["model"]["num_layers"], seed, registry)

print(f"LoRA location: {location['name']} - layers {location['resolved_layers']}")
print()

# Load model and data
print("Loading model and tokenizer...")
model, tokenizer = train.load_model_and_tokenizer(cfg["model"])

print("Loading dataset...")
evaluator = evaluate.get_evaluator(cfg["dataset"]["evaluator"])
train_items = evaluate.load_items(cfg["dataset"], cfg["data"], "train", cfg["data"]["max_train_samples"])
eval_items = evaluate.load_items(cfg["dataset"], cfg["data"], "eval", cfg["data"]["max_eval_samples"])

print(f"Training samples: {len(train_items)}")
print(f"Eval samples: {len(eval_items)}")
print()

# Create a temporary output directory
output_dir = Path("./test_run_output")
output_dir.mkdir(exist_ok=True)

print("Starting training...")
print("=" * 70)

# This will show the full traceback when it fails
model, info = train.train(
    cfg, seed, location["resolved_layers"], model, tokenizer,
    train_items, evaluator, output_dir
)

print("Training completed!")
print(f"Info: {info}")
