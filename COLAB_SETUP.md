# Running Sweeps on Google Colab

This guide helps you run long-running LoRA sweeps on Google Colab with automatic result persistence.

## Problem

- 72 runs × ~40 minutes each = ~48 hours total
- Colab disconnects after inactivity or when your computer sleeps
- When disconnected, all results are lost unless explicitly saved

## Solution Options

### Option 1: Persistent Colab with Auto-Save (Recommended)

**Use:** `run_sweep_colab_persistent.ipynb`

**Features:**
- ✅ Auto-saves results to Google Drive after each run
- ✅ Survives runtime disconnections
- ✅ Resume from where you left off
- ✅ Early stopping to reduce training time (~20-30% faster)
- ✅ Keep-alive script to prevent disconnections

**Time estimate:** Can complete in 2-3 Colab sessions (~12 hours each with T4 GPU)

### Option 2: Kaggle Notebooks (Alternative)

**Advantages:**
- Can run up to 11 hours unattended
- More reliable for unattended runs
- Similar setup as Colab

**Disadvantages:**
- Single 11-hour window (can't resume mid-run)
- Must save everything to output folder before time limit

**Setup:**
1. Create a Kaggle notebook
2. Copy cells from `run_sweep_colab_persistent.ipynb`
3. Change Drive mount to Kaggle output folder:
   ```python
   OUTPUT_DIR = '/kaggle/working/results'
   ```

### Option 3: Run in Batches

Instead of running all 72 experiments, run smaller batches:

```python
# Run only 0.5B model experiments (24 runs, ~16 hours)
!python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml --only "qwen25-0.5b"

# Run only 1.5B model experiments (24 runs, ~16 hours)  
!python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml --only "qwen25-1.5b"

# Run only GSM8K dataset experiments (36 runs, ~24 hours)
!python scripts/run.py configs/experiments/sft_sweep_mmlu_gsm8k.yaml --only "gsm8k"
```

## Early Stopping Configuration

The sweep now includes early stopping to reduce wasted computation:

```yaml
training:
  early_stopping_enabled: true
  early_stopping_patience: 3       # Stop if no improvement for 3 evaluations
  early_stopping_eval_steps: 50    # Evaluate every 50 steps
```

**Expected impact:** 
- Training will stop when loss plateaus (typically 300-350 steps vs. full 500)
- Saves ~20-30% time per run
- Total sweep time: ~33 hours instead of 48 hours

To disable early stopping, set `early_stopping_enabled: false` in the config.

## Tips for Long-Running Colab Sessions

1. **Keep browser tab active**: Colab is more likely to stay connected if the tab is visible
2. **Use keep-alive script**: Run the JavaScript keep-alive cell in the notebook
3. **Monitor progress**: Check Google Drive folder periodically to see completed runs
4. **Save frequently**: The auto-save script syncs after every run, so you can safely disconnect and resume

## Troubleshooting

### "Runtime disconnected"
- **Solution**: Just re-run the sweep cell. It will skip completed runs and continue from where it left off.

### "Out of memory" errors
- **Solution**: The config is tuned for 12GB GPU. If using T4 (15GB), you should be fine. If still getting OOM, reduce `per_device_train_batch_size` from 4 to 2.

### Results not saving to Drive
- **Solution**: Check that Drive is mounted (`/content/drive/MyDrive` should be accessible). Re-run the mount cell if needed.

### Want to start fresh
- **Solution**: Delete `/content/drive/MyDrive/lora_scan_results/runs/` folder and re-run the sweep.

## Comparison: Colab vs. Kaggle

| Feature | Colab (with auto-save) | Kaggle |
|---------|------------------------|---------|
| Max unattended time | ~2 hours (with keep-alive: ~6 hours) | 11 hours |
| Can resume after disconnect | ✅ Yes | ❌ No |
| GPU availability | T4, L4, A100 (Colab Pro) | T4, P100 |
| Storage persistence | Google Drive | Output folder |
| Best for | Multi-session sweeps | Single long runs |

## Recommended Approach

1. **First session** (~12 hours): Run the persistent Colab notebook with keep-alive script
2. **Check progress**: See how many runs completed in Google Drive
3. **Second session** (~12 hours): Re-run the same notebook cell - it will resume automatically
4. **Repeat** until all 72 runs complete

With early stopping enabled, you should complete the full sweep in 2-3 Colab sessions.
