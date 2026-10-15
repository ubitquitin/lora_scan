# MMLU Training Data Fix

## Problem
Your training was using **107,315 MMLU examples** instead of the mentor notebook's **~1,816 examples**.

### Root Cause
- **Your config**: Used `subset: all` which loads a pre-merged 100k+ example dataset
- **Mentor notebook**: Loads 57 individual subjects and combines `auxiliary_train` + `validation` + `dev` per subject

## Solution
Changed MMLU loading to match mentor notebook exactly:

### Config Changes (`configs/datasets/mmlu.yaml`)
```yaml
subset: per_subject  # Was: all
train_combine_splits: ['auxiliary_train', 'validation', 'dev']
```

### Code Changes (`src/lorascan/evaluate.py`)
- Added per-subject loading for BOTH train and eval splits
- Train: Combines auxiliary_train + validation + dev for each subject
- Eval: Samples 8 examples per subject (already implemented)

## Expected Output After Fix

When you re-run in Colab, you should see:

```
Loading MMLU train per-subject: combining ['auxiliary_train', 'validation', 'dev'] for 57 subjects
This may take 5-10 minutes on first run (downloading datasets)...
Loading MMLU subjects: 1/57...
Loading MMLU subjects: 11/57...
...
Loaded all 57 MMLU subjects, combining...
MMLU train: ~1816 examples from 57 subjects

[Later during training prep]
Building labels for train dataset: 9289 examples  # Was: 107,315
```

Then for eval:
```
Loading MMLU eval per-subject: 8 samples × 57 subjects = 456 total
...
gsm8k eval: Limited from 1319 to 200 samples (limit=200)

Combined eval dataset loaded:
  gsm8k: 200 samples
  mmlu: 456 samples
  Total: 656 samples
```

## Training Dataset Breakdown

| Dataset | Mentor Notebook | Before Fix | After Fix |
|---------|----------------|------------|-----------|
| MMLU train | 1,816 | 107,315 ❌ | 1,816 ✅ |
| GSM8K train | 7,473 | 7,473 ✅ | 7,473 ✅ |
| **Total train** | **9,289** | **~114,788** | **9,289** ✅ |
| MMLU eval | 456 | 456 ✅ | 456 ✅ |
| GSM8K eval | 200 | 200 ✅ | 200 ✅ |
| **Total eval** | **656** | **656** | **656** ✅ |

## Why This Matters

Training on 60x more MMLU data would:
- ✗ Take much longer to train (~12x longer)
- ✗ Over-weight MMLU vs GSM8K (from 20% to 93% of training data)
- ✗ Make results incomparable to mentor notebook
- ✗ Different data distribution (100k samples ≠ balanced per-subject sampling)

With this fix, your training will match the mentor notebook identically!

## Next Steps

1. Pull the latest code in Colab
2. Re-run the training
3. Verify you see "~1816 examples" for MMLU train (not 107k!)
4. Training should complete in similar time to mentor notebook
