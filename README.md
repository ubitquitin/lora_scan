# lora-scan

Config-driven SFT / GRPO LoRA experiments across models, datasets and **layer locations**.

> Configuration defines the experiment; a small set of reusable functions executes it;
> raw per-item outputs are the scientific record; analysis happens afterward.

## Requirements

**GPU Required**: This project requires a CUDA-capable GPU for training. The code will verify GPU availability at startup and fail with a clear error if none is detected.

- **Minimum**: 12GB VRAM (adjust batch sizes in `configs/defaults.yaml`)
- **Recommended**: 16-24GB VRAM (RTX 3090, 4090, A10, A100, etc.)
- PyTorch with CUDA support (verify with `python -c "import torch; print(torch.cuda.is_available())"`)

## Quick start

```bash
pip install -e ".[analysis,dev]"
pytest                                                                  # CPU-only, no downloads

python scripts/check_gpu.py                                                # Verify GPU is available
python scripts/run.py configs/experiments/qwen_gsm8k_grpo.yaml --dry-run   # run ids + resolved layers
python scripts/run.py configs/experiments/qwen_gsm8k_base.yaml             # untrained baseline (GPU check)
python scripts/run.py configs/experiments/qwen_gsm8k_grpo.yaml             # 5 locations x 3 seeds (requires GPU)
python scripts/aggregate.py                                                # -> results/runs.parquet
```

Useful flags: `--seeds 42`, `--only late random`, `--force`, `--runs-dir`.
Completed runs are skipped, so an interrupted sweep resumes by re-running the same command.

## Layout

```
configs/
  defaults.yaml      lora / data / evaluation / per-method training defaults
  locations.yaml     named LoRA locations (early, middle, late, random, every_other, all)
  models/ datasets/ prompts/ experiments/
src/lorascan/
  runs.py            config resolution, sweeps, run hash, run records, registry, collect_results
  layers.py          location name -> concrete layer indices
  train.py           model loading, LoRA, train_sft, train_grpo
  evaluate.py        dataset adapters, generation, evaluators (aggregate + per-item)
scripts/run.py       CLI;  scripts/aggregate.py  runs -> parquet
analysis/results.ipynb   late vs random, SFT vs GRPO, McNemar, CIs
predictions/registry.yaml  experiment_id -> seed -> run_id (auto-maintained)
runs/<run_id>/       config.yaml, metadata.json, checkpoint/ (adapter), predictions.jsonl, metrics.json
```

## An experiment

```yaml
name: qwen15_gsm8k_grpo
method: grpo                  # sft | grpo | none (baseline)
model: qwen25-1.5b            # -> configs/models/qwen25-1.5b.yaml
dataset: gsm8k
prompt: cot
lora: {rank: 1, alpha: 32, dropout: 0.0}
training: {learning_rate: 1.0e-5, max_steps: 500, num_generations: 4}
seeds: [42, 123, 456]
sweep:                        # optional: cartesian product of dotted keys
  lora.location: [early, middle, late, random, all]
```

* Everything under `training:` is passed straight to trl's `SFTConfig` / `GRPOConfig`.
* `sweep` works on any key (`model`, `prompt`, `lora.rank`, `training.learning_rate`, ...);
  variants are named `<name>_<value>`. See `scale_gsm8k_grpo.yaml` for a model-size sweep.
* `data.max_train_samples` / `data.max_eval_samples` subsample with a fixed seed, so every run
  sees the **same items** (required for paired tests).

## Run identity

`run_id = sha256(resolved config + seed + resolved layer indices + git commit)[:10]`.
A `random` location is resolved *before* hashing and recorded (name, seed, `resolved_layers`) in
`runs/<run_id>/config.yaml`, so the exact layers are part of identity. The experiment name is
excluded, so identical configs under different names dedupe. Set
`reproducibility.include_git_commit: false` if you want runs to survive code edits (metadata still
records the commit). Pin `revision:` in model/dataset configs to freeze HF revisions.

## Extending

| To add...        | Do this |
|------------------|---------|
| a model          | `configs/models/<name>.yaml` with `hf_id`, `num_layers`, `target_modules` |
| a dataset        | `configs/datasets/<name>.yaml`; reuse `gsm8k` / `multiple_choice` / `exact_match` evaluators, or add an adapter in `evaluate.FORMATTERS` + a class in `EVALUATORS` |
| a prompt         | `configs/prompts/<name>.yaml` (`template` with `{question}`, `target: full|short`) |
| a location       | entry in `configs/locations.yaml` (types: `proportion`, `stride`, `explicit`, `random`), or a `locations:` block in a model config for model-specific ones |

## Notes

* GRPO reward is binary correctness from the dataset's evaluator.
* trl's API moves between releases; `training:` keys are forwarded verbatim, so rename keys
  (e.g. `max_length`) in `defaults.yaml` if your trl version differs.
* Checkpoints are LoRA adapters only (small). McNemar/CIs belong in `analysis/`, not the evaluator.
