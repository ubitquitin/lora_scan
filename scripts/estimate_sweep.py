#!/usr/bin/env python
"""Estimate sweep duration and show what will run.

Usage:
    python scripts/estimate_sweep.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml
    python scripts/estimate_sweep.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml --time-per-step 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lorascan import layers as layers_mod  # noqa: E402
from lorascan import runs  # noqa: E402


def estimate_time(max_steps, time_per_step=0.3):
    """Estimate training time given steps and avg time per step."""
    total_seconds = max_steps * time_per_step
    if total_seconds < 60:
        return f"{total_seconds:.0f}s"
    elif total_seconds < 3600:
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(total_seconds // 3600)
        mins = int((total_seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="Path to experiment YAML file")
    ap.add_argument("--time-per-step", type=float, default=0.3,
                    help="Estimated seconds per training step (default: 0.3s for small models)")
    a = ap.parse_args()

    # Load and resolve the experiment
    configs = list(runs.load_and_resolve(a.experiment))

    # Group by dataset and model
    by_dataset = {}
    for cfg in configs:
        dataset = cfg["dataset"]["name"]
        model = cfg["model"]["hf_id"].split("/")[-1]
        if dataset not in by_dataset:
            by_dataset[dataset] = {}
        if model not in by_dataset[dataset]:
            by_dataset[dataset][model] = []
        by_dataset[dataset][model].append(cfg)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Sweep Estimate: {a.experiment}")
    print(f"{'='*80}\n")

    total_runs = len(configs)
    total_time_seconds = 0

    for dataset, models in sorted(by_dataset.items()):
        print(f"Dataset: {dataset}")
        print(f"{'-'*80}")
        for model, model_configs in sorted(models.items()):
            n_variants = len(model_configs)
            max_steps = model_configs[0]["training"].get("max_steps", 500)
            est_time_per_run = max_steps * a.time_per_step
            est_time_total = est_time_per_run * n_variants

            total_time_seconds += est_time_total

            print(f"  {model:30s} | {n_variants:2d} runs × {max_steps:4d} steps | "
                  f"~{estimate_time(max_steps, a.time_per_step)} per run | "
                  f"~{estimate_time(est_time_total / n_variants / len(model_configs[0]['seeds']), a.time_per_step)} per seed | "
                  f"Total: ~{estimate_time(est_time_total, 1.0)}")

        print()

    # Overall summary
    print(f"{'='*80}")
    print(f"  Total Runs: {total_runs}")
    print(f"  Estimated Total Time: ~{estimate_time(total_time_seconds, 1.0)}")
    print(f"{'='*80}")
    print(f"\nNOTE: Time estimates assume {a.time_per_step}s per step.")
    print(f"      Adjust with --time-per-step for your GPU:")
    print(f"        - Fast GPU (A100/H100):     --time-per-step 0.1")
    print(f"        - Mid-range (RTX 3090/4090): --time-per-step 0.3 (default)")
    print(f"        - Slower GPU (RTX 3080):     --time-per-step 0.5")
    print(f"\n      Larger models will be slower. Run a single job first to calibrate.\n")

    # Show breakdown by config variant
    print(f"\n{'='*80}")
    print("  Configuration Breakdown")
    print(f"{'='*80}\n")

    variant_names = {}
    for cfg in configs:
        name = cfg["name"]
        if name not in variant_names:
            variant_names[name] = {
                "count": 0,
                "dataset": cfg["dataset"]["name"],
                "model": cfg["model"]["hf_id"].split("/")[-1],
                "seeds": len(cfg["seeds"]),
            }
        variant_names[name]["count"] += 1

    for name, info in sorted(variant_names.items()):
        print(f"  {name:40s} | {info['model']:20s} | {info['dataset']:10s} | "
              f"{info['count']} variants × {info['seeds']} seeds")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
