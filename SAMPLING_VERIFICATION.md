# Sampling Verification: Matching Mentor Notebook

## Current Status

### ✅ What's Correct
1. **Seed**: Changed from 0 to 42 (matches mentor)
2. **MMLU eval**: Per-subject sampling (8 samples × 57 subjects = 456 total)
3. **GSM8K eval**: Limited to 200 samples
4. **Progress reporting**: Added to show loading progress

### ⚠️ Known Issues

#### Issue 1: Slow First Load (NOT A BUG)
The message "Generating dev split: 5/5" and apparent hanging is **NORMAL** on first run.

**Why it's slow:**
- Loading 57 separate MMLU subject datasets
- Each needs to be downloaded and cached (5-10 minutes on first run)
- Subsequent runs will be fast (uses cached data)

**What you should see with updated code:**
```
Loading MMLU with per-subject sampling: 8 samples × 57 subjects = 456 total
This may take 5-10 minutes on first run (downloading datasets)...
Loading MMLU subjects: 1/57...
Loading MMLU subjects: 11/57...
Loading MMLU subjects: 21/57...
...
Loaded all 57 MMLU subjects, combining...
```

#### Issue 2: MMLU Train Split (NEEDS VERIFICATION)

**Mentor notebook trains on:**
- `auxiliary_train` + `validation` + `dev` (combined per subject)
- Total: 1,816 examples

**Your config trains on:**
- `train_split: auxiliary_train` (might be just one split)
- Unknown if `subset: all` automatically includes validation+dev

**To verify:** After training starts, check the output for:
```
Combined train: X examples (MMLU: ??? + GSM8K: 7473)
```

**Expected:** MMLU should contribute ~1,816 examples (matching mentor)
**If less:** You're missing validation+dev splits

## Test Set Comparison

| Dataset | Mentor Notebook | Your Config (Updated) | Match? |
|---------|----------------|----------------------|---------|
| MMLU eval | 456 (8 per subject × 57) | 456 (8 per subject × 57) | ✅ YES |
| MMLU eval seed | 42 | 42 | ✅ YES |
| GSM8K eval | 200 | 200 | ✅ YES |
| GSM8K eval seed | 42 | 42 | ✅ YES |
| MMLU train | 1,816 (aux+val+dev) | ??? (needs verification) | ❓ UNKNOWN |
| GSM8K train | 7,473 (full) | 7,473 (full) | ✅ YES |

## Expected Results After Fix

Once loading completes, your results should closely match the mentor notebook's **base model** results:

**Mentor Base Model (Qwen 2.5-0.5B-Instruct):**
- MMLU: 0.4605 (210/456)
- GSM8K: 0.3250 (65/200)

**Your Previous Results (different test sets):**
- MMLU: 0.4337 (6090/14042) ← Used full 14K test set
- GSM8K: 0.3154 (416/1319) ← Used full 1.3K test set

**Your New Results (should match mentor):**
- MMLU: ~0.46 (?/456) ← Same 456-sample test set
- GSM8K: ~0.32 (?/200) ← Same 200-sample test set

## Next Steps

1. **Re-run in Colab** with the updated code
2. **Wait 5-10 minutes** for MMLU data loading (don't interrupt!)
3. **Watch for progress messages** showing subject loading
4. **Check training dataset size** when training starts:
   - Should see: "Combined train: ~9289 examples (MMLU: ~1816 + GSM8K: 7473)"
   - If MMLU shows less than 1,816, we need to fix train split loading
5. **Compare final eval results** to mentor baseline

## If Training Dataset Doesn't Match

If the training shows fewer than 1,816 MMLU examples, you'll need to either:
1. Modify `mmlu.yaml` to specify multiple train splits
2. Add code to load per-subject and combine splits (like mentor does)

Let me know what the training dataset size shows and I can help fix it!
