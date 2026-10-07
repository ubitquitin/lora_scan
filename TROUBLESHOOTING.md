# Troubleshooting Guide

## Common Issues and Solutions

### Failed Runs

**Issue:** Some runs fail with errors like `OverflowError: cannot convert float infinity to integer`

**What happens:**
- The run is marked as `"status": "failed"` in `metadata.json`
- The failed run is **NOT saved to Google Drive** (as of the updated notebook)
- The sweep continues to the next run

**To identify failed runs:**
```python
# Run the "Check Progress" cell in the notebook
# It will show:
#   ❌ FAILED RUNS (X):
#     • experiment_name
#       ID: run_id
#       Error: error message...
```

**To fix:**
1. Run the "Clean Up Failed/Stuck Runs" cell to delete failed runs from Drive
2. Re-run the sweep - it will retry the deleted runs

### Stuck "Running" Runs

**Issue:** After Colab disconnects, some runs show `"status": "running"` forever

**What causes this:**
- Colab disconnected mid-training
- The run never got to complete or fail
- Status remains "running" in metadata.json

**To identify stuck runs:**
```python
# Run the "Check Progress" cell
# It will show:
#   ⚠️  STUCK RUNS (X) - likely from disconnection:
#     • experiment_name (ID: run_id)
```

**To fix:**
1. Run the "Clean Up Failed/Stuck Runs" cell
2. Set `CLEAN_STUCK = True` (default)
3. These will be deleted and re-run next time

### Progress Tracking

**Current progress in your example:**
```
4 runs in Drive:
- 1 failed (seed 42, gsm8k_chat)
- 2 complete (seeds 123, 456, gsm8k_chat)
- 1 stuck running (seed 42, gsm8k_cot)

Status: 2/72 completed (2.8%)
```

**What to do:**
1. Delete the failed and stuck runs:
   ```python
   # Run cleanup cell with:
   CLEAN_FAILED = True
   CLEAN_STUCK = True
   ```

2. This leaves only the 2 successful runs in Drive

3. Re-run the sweep - it will:
   - Skip the 2 completed runs
   - Retry the 2 deleted runs
   - Continue with the remaining 68 runs

### Why Did a Run Fail?

Common causes:

1. **OverflowError: cannot convert float infinity to integer**
   - Usually happens with extreme loss values during training
   - Early stopping should help catch this earlier
   - Could be a data issue or learning rate too high

2. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size` from 4 to 2
   - Or reduce `max_length` from 1024 to 512

3. **Model download issues**
   - Transient network errors
   - Usually works on retry

### Monitoring Progress

**During a run:**
Watch for these messages in real-time:
```
[X/72] TODO run_id experiment_name seed=X
======================================================================
  Training started: 500 steps
======================================================================
  Early stopping enabled: patience=3, eval_steps=50
[████████████] 100.0% | Step 350/500 | Loss: 0.5695
======================================================================
  Training completed in 24m 13s
======================================================================

✓ SUCCESS - Syncing to Google Drive...
✓ Saved to Drive | Progress: 3 completed, 1 failed
```

**Failed run output:**
```
[X/72] TODO run_id experiment_name seed=X
...
Traceback (most recent call last):
  ...
OverflowError: cannot convert float infinity to integer

✗ FAILED - NOT saving to Drive
✗ Progress: 3 completed, 2 failed
```

**Between sessions:**
```python
# Run the "Check Progress" cell anytime
# Shows: completed/failed/stuck/remaining
```

### Best Practices

1. **Check progress before each session**
   ```python
   # Run "Check Progress" cell
   ```

2. **Clean up failed/stuck runs before resuming**
   ```python
   # Run "Clean Up Failed/Stuck Runs" cell
   ```

3. **Monitor the first few runs**
   - Make sure they're succeeding
   - Check that early stopping is working
   - Verify Drive syncs are happening

4. **Don't manually edit Drive files**
   - Let the notebook manage the sync
   - Manual changes may cause issues

5. **If a run keeps failing:**
   - Note the error from "Check Progress"
   - Check if it's a specific seed/dataset combo
   - May need to adjust hyperparameters or exclude that run

### Emergency Recovery

**If everything is stuck:**
1. Stop the notebook
2. Check Drive for what's actually saved
3. Run "Check Progress" to see status
4. Delete everything if needed:
   ```python
   import shutil
   shutil.rmtree("/content/drive/MyDrive/lora_scan_results/runs")
   ```
5. Start fresh

**To resume after cleanup:**
1. Re-run the "Restore previous results" cell
2. Run "Check Progress" to verify what's left
3. Continue the sweep

### Understanding the Auto-Save Logic

**Old behavior (before update):**
- ❌ Saved after EVERY run (including failed ones)
- ❌ No progress tracking
- ❌ Failed runs cluttered Drive

**New behavior:**
- ✅ Only saves successful runs
- ✅ Shows progress: X completed, Y failed
- ✅ Provides cleanup tools
- ✅ Clear status reporting

**How it works:**
1. Run completes (success or failure)
2. Script checks `metadata.json` for `"status"`
3. If `"status": "complete"` → Save to Drive ✓
4. If `"status": "failed"` → Skip save ✗
5. Print progress update

This means failed runs stay in local `/content/lora_scan/runs/` but don't pollute your Drive storage.
