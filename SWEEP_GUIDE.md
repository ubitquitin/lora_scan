# SFT Sweep Guide: Multi-Model, Multi-Dataset Evaluation

This guide explains how to run comprehensive SFT sweeps across models and datasets with detailed progress tracking.

## Quick Start

```bash
# 1. Estimate what will run and how long it will take
python scripts/estimate_sweep.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml

# 2. Preview the exact jobs (dry run)
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml --dry-run

# 3. Run the full sweep
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml

# 4. Aggregate results
python scripts/aggregate.py
```

## What Gets Evaluated

The `sft_sweep_mmlu_gsm8k.yaml` sweep runs:

- **Models**: qwen25-0.5b, qwen25-1.5b, qwen25-3b, gemma2-2b (4 models)
- **Datasets**: GSM8K (math reasoning) and MMLU (multiple choice knowledge) (2 datasets)
- **Seeds**: 42, 123, 456 (3 seeds for statistical significance)
- **Total**: 4 models × 2 datasets × 3 seeds = **24 training runs**

Each run:
1. Loads the model and applies LoRA to late layers
2. Fine-tunes for 500 steps with progress reporting
3. Evaluates on the test set
4. Saves predictions, metrics, and adapter weights

## Progress Reporting

During each training run, you'll see real-time progress:

```
======================================================================
  Training started: 500 steps
======================================================================
[████████████████████░░░░░░░░░░░░░░░░░░░░] 50.0% | Step 250/500 | Loss: 0.8234 | Elapsed: 2m 15s | ETA: 2m 15s | 1.85 steps/s
```

This shows:
- **Progress bar**: Visual representation of completion
- **Step counter**: Current step / total steps
- **Loss**: Current training loss
- **Elapsed**: Time since training started
- **ETA**: Estimated time remaining
- **Throughput**: Steps per second

## Time Estimates

Estimated time per run (500 steps):
- **Fast GPU (A100/H100)**: ~1-2 minutes
- **Mid-range (RTX 3090/4090)**: ~2-5 minutes
- **Slower GPU (RTX 3080)**: ~4-8 minutes

**Total sweep time** (24 runs):
- Fast: ~30-60 minutes
- Mid-range: ~1-2 hours
- Slower: ~2-4 hours

Larger models (3b) will take longer than smaller ones (0.5b).

## Running Subsets

### Test on a single model first
```bash
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml \
  --only qwen25-0.5b \
  --seeds 42
```

### Run only GSM8K experiments
```bash
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml \
  --only gsm8k
```

### Run only MMLU experiments
```bash
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml \
  --only mmlu
```

### Run specific model + dataset
```bash
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml \
  --only qwen25-1.5b mmlu \
  --seeds 42
```

## Understanding Results

After running, results are stored in `runs/<run_id>/`:

```
runs/
  abc1234567/
    config.yaml           # Full resolved config
    metadata.json         # Training info (steps, loss, params)
    checkpoint/           # LoRA adapter weights
    predictions.jsonl     # Per-item predictions and correctness
    metrics.json          # Aggregate metrics (accuracy, etc.)
```

### Aggregate Results

```bash
python scripts/aggregate.py
```

Creates `results/runs.parquet` with all metrics across runs. You can then analyze:

```python
import pandas as pd

df = pd.read_parquet("results/runs.parquet")

# Compare models on GSM8K
gsm8k = df[df["dataset"] == "gsm8k"]
print(gsm8k.groupby("model")["accuracy"].agg(["mean", "std"]))

# Compare datasets for a specific model
qwen_1_5 = df[df["model"] == "Qwen/Qwen2.5-1.5B"]
print(qwen_1_5.groupby("dataset")["accuracy"].agg(["mean", "std"]))

# Best performing combinations
print(df.nlargest(5, "accuracy")[["model", "dataset", "accuracy", "seed"]])
```

## Resuming Failed Runs

The sweep automatically skips completed runs. If a run fails:

```bash
# Re-run only failed/incomplete runs
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml

# Force re-run everything (overwrites)
python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml --force
```

## Customizing the Sweep

Edit `configs/experiments/sft_sweep_mmlu_gsm8k.yaml`:

### Change training duration
```yaml
training:
  max_steps: 1000  # Longer training
```

### Add more models
```yaml
sweep:
  model: [qwen25-0.5b, qwen25-1.5b, qwen25-3b, qwen25-7b, gemma2-2b]
```

### Change LoRA location
```yaml
lora: {rank: 1, alpha: 32, dropout: 0.0, location: early}  # or middle, random, all
```

### Sweep LoRA rank
```yaml
sweep:
  model: [qwen25-0.5b, qwen25-1.5b]
  dataset: [gsm8k, mmlu]
  lora.rank: [1, 4, 8]  # Test different LoRA ranks
```

## Monitoring GPU Usage

While the sweep runs, monitor GPU in another terminal:

```bash
watch -n 1 nvidia-smi
```

Or use the provided script before starting:
```bash
python scripts/check_gpu.py
```

## Tips

1. **Start small**: Run a single job first to calibrate time estimates
   ```bash
   python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml \
     --only qwen25-0.5b gsm8k --seeds 42
   ```

2. **Monitor progress**: The progress bar updates every `logging_steps` (default: 10)

3. **Batch sizes**: If you get OOM errors, reduce batch size in `configs/defaults.yaml`:
   ```yaml
   training:
     sft:
       per_device_train_batch_size: 8  # Reduce from 16
       gradient_accumulation_steps: 2  # Increase to maintain effective batch size
   ```

4. **Eval batch size**: Also in `defaults.yaml`:
   ```yaml
   evaluation:
     batch_size: 32  # Reduce from 64 if OOM during evaluation
   ```

5. **Resume interrupted sweeps**: Just re-run the same command—completed runs are skipped

## Troubleshooting

### "No CUDA-capable GPU detected"
Ensure your environment is activated and PyTorch has CUDA support:
```bash
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory (OOM)
Reduce batch sizes in `configs/defaults.yaml` as shown above.

### Slow progress
- Check GPU utilization with `nvidia-smi`
- Larger models are naturally slower
- Consider using bf16 if your GPU supports it (automatic)

### Different results across seeds
This is expected! That's why we run multiple seeds. Aggregate results with:
```bash
python scripts/aggregate.py
```
Then analyze mean ± std across seeds.
