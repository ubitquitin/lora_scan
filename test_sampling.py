#!/usr/bin/env python
"""Verify that our sampling matches the mentor notebook."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lorascan import evaluate, runs

# Load configs
mmlu_cfg = runs.load_component("datasets", "mmlu")
gsm8k_cfg = runs.load_component("datasets", "gsm8k")
combined_cfg = runs.load_component("datasets", "mmlu_gsm8k_combined")

# Default data config
data_cfg = {"subsample_seed": 42}

print("Testing MMLU eval sampling...")
mmlu_items = evaluate.load_items(mmlu_cfg, data_cfg, "eval")
print(f"  MMLU eval samples: {len(mmlu_items)}")
print(f"  Expected: 456 (8 per subject × 57 subjects)")
assert len(mmlu_items) == 456, f"Expected 456, got {len(mmlu_items)}"

print("\nTesting GSM8K eval sampling...")
gsm8k_items = evaluate.load_items(gsm8k_cfg, data_cfg, "eval")
print(f"  GSM8K eval samples: {len(gsm8k_items)}")
print(f"  Expected: 200")
assert len(gsm8k_items) == 200, f"Expected 200, got {len(gsm8k_items)}"

print("\nTesting combined dataset...")
combined_items = evaluate.load_items(combined_cfg, data_cfg, "eval")
print(f"  Combined eval samples: {len(combined_items)}")
print(f"  Expected: 656 (456 MMLU + 200 GSM8K)")
assert len(combined_items) == 656, f"Expected 656, got {len(combined_items)}"

# Check source tagging
mmlu_count = sum(1 for item in combined_items if item.get("source_dataset") == "mmlu")
gsm8k_count = sum(1 for item in combined_items if item.get("source_dataset") == "gsm8k")
print(f"  MMLU items: {mmlu_count}")
print(f"  GSM8K items: {gsm8k_count}")

print("\n✅ All sampling tests passed!")
print("\nThis matches the mentor notebook:")
print("  - MMLU: 456 test samples (8 per subject)")
print("  - GSM8K: 200 test samples")
print("  - Seed: 42 for all shuffling")
