#!/usr/bin/env python3
"""Verify GSM8K formatting matches mentor's rank1 setup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lorascan import runs, evaluate
from transformers import AutoTokenizer

# Load configs
exp_cfg = runs.load_and_resolve("configs/experiments/mentor_rank1_gsm8k.yaml")[0]
model_cfg = exp_cfg["model"]
dataset_cfg = exp_cfg["dataset"]
prompt_cfg = exp_cfg["prompt"]
data_cfg = exp_cfg["data"]

print("="*70)
print("GSM8K Format Verification")
print("="*70)

# Load tokenizer only
print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_cfg['hf_id'])
print(f"   Model: {model_cfg['hf_id']}")
print(f"   EOS token: {repr(tokenizer.eos_token)}")
print(f"   EOS token ID: {tokenizer.eos_token_id}")

# Load one training example
print("\n2. Loading sample training data...")
train_items = evaluate.load_items(dataset_cfg, data_cfg, "train", limit=1)
item = train_items[0]

print(f"\n   Question: {item['question'][:80]}...")
print(f"   Gold answer: {item['gold']}")
print(f"   Target format: {item['target'][:100]}...")

# Render training text
print("\n3. Checking training text format...")
training_text = evaluate.render_training_text(tokenizer, item, prompt_cfg)

print(f"\n   Full training text:")
print(f"   {'='*70}")
print(training_text)
print(f"   {'='*70}")

# Check components
print("\n4. Verification checks:")
checks = [
    ("System prompt contains 'helpful assistant'", "helpful assistant" in training_text.lower()),
    ("Question includes 'step by step'", "step by step" in training_text.lower()),
    ("Question includes '\\boxed{}'", "\\boxed{}" in training_text),
    ("Target includes '\\boxed{...}'", f"\\boxed{{{item['gold']}}}" in training_text),
    ("Ends with EOS token", training_text.endswith(tokenizer.eos_token) or
                           training_text.endswith("<|im_end|>") or
                           training_text.endswith("</s>")),
]

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"   {status} {check_name}")

# Show expected format
print("\n5. Expected mentor format:")
print("   Prompt: '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n'")
print("           '<|im_start|>user\\n{question}\\n\\n'")
print("           'Please reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n'")
print("   Target: '<|im_start|>assistant\\n{cleaned_cot}\\n\\n\\\\boxed{N}<|im_end|>'")
print("   Where:")
print("   - cleaned_cot has <<calculator>> markup removed")
print("   - #### suffix removed")
print("   - <|im_end|> is the EOS token")

print("\n" + "="*70)
print("Verification complete!")
print("="*70)
