#!/usr/bin/env python
"""Run an experiment file: every sweep variant x every seed.

    python scripts/run.py configs/experiments/qwen_gsm8k_grpo.yaml --dry-run
    python scripts/run.py configs/experiments/qwen_gsm8k_grpo.yaml
    python scripts/run.py configs/experiments/qwen_gsm8k_sft.yaml --seeds 42 --only late
"""
import argparse
import gc
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lorascan import layers as layers_mod  # noqa: E402
from lorascan import runs  # noqa: E402


@dataclass
class Job:
    config: dict
    seed: int
    location: dict
    version: str | None
    run_id: str


def plan(path, seeds_override=None, only=None) -> list[Job]:
    base_registry = layers_mod.load_locations(runs.CONFIGS / "locations.yaml")
    jobs = []
    for config in runs.load_and_resolve(path):
        if only and not any(o in config["name"] for o in only):
            continue
        registry = {**base_registry, **config["model"].get("locations", {})}
        version = runs.code_version() if config["reproducibility"]["include_git_commit"] else None
        for seed in seeds_override or config["seeds"]:
            loc = layers_mod.resolve_location(config["lora"]["location"], config["model"]["num_layers"], seed, registry)
            jobs.append(Job(config, seed, loc, version, runs.run_hash(config, seed, loc, version)))
    return jobs


def execute(job: Job, runs_dir: Path, gpu_info: dict) -> None:
    from lorascan import evaluate, train  # heavy imports only when actually running

    cfg, run = job.config, runs.Run(runs_dir, job.run_id)
    run.start(cfg, job.seed, job.location, job.version)
    try:
        train.seed_everything(job.seed)
        print(f"  GPU: {gpu_info['device_name']} ({gpu_info['memory_gb']}GB) | "
              f"Precision: {'bf16' if gpu_info['supports_bf16'] else 'fp16'}")
        model, tok = train.load_model_and_tokenizer(cfg["model"])
        evaluator = evaluate.get_evaluator(cfg["dataset"]["evaluator"])
        train_items = []
        if cfg["method"] != "none":
            train_items = evaluate.load_items(cfg["dataset"], cfg["data"], "train", cfg["data"]["max_train_samples"])
        eval_items = evaluate.load_items(cfg["dataset"], cfg["data"], "eval", cfg["data"]["max_eval_samples"])

        model, info = train.train(cfg, job.seed, job.location["resolved_layers"], model, tok,
                                  train_items, evaluator, run.checkpoint_dir)
        preds = evaluate.generate_predictions(model, tok, eval_items, cfg["prompt"], cfg["evaluation"])
        run.finish(evaluator.per_item(preds), {"eval": evaluator.evaluate(preds), "train": info})
        runs.register_run(cfg["name"], job.seed, job.run_id)
    except BaseException as e:
        run.fail(f"{type(e).__name__}: {e}")
        raise
    finally:
        model = tok = None  # drop references so the next run starts with free GPU memory
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds in the file")
    ap.add_argument("--only", nargs="+", help="run only variants whose name contains one of these")
    ap.add_argument("--runs-dir", default=str(runs.RUNS))
    ap.add_argument("--force", action="store_true", help="re-run even if a completed run exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (run ids, resolved layers) and exit")
    a = ap.parse_args()

    # Verify GPU availability upfront (before loading models)
    if not a.dry_run:
        from lorascan import train
        try:
            gpu_info = train.verify_gpu()
            print(f"GPU detected: {gpu_info['device_name']} ({gpu_info['memory_gb']}GB, "
                  f"{'bf16' if gpu_info['supports_bf16'] else 'fp16'} precision)\n")
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        gpu_info = None

    jobs = plan(a.experiment, a.seeds, a.only)
    runs_dir = Path(a.runs_dir)
    failed = 0
    for i, job in enumerate(jobs, 1):
        done = runs.Run(runs_dir, job.run_id).complete
        tag = "DONE " if done and not a.force else "TODO "
        print(f"[{i}/{len(jobs)}] {tag}{job.run_id}  {job.config['name']}  seed={job.seed}  "
              f"location={job.location['name']} layers={job.location['resolved_layers']}"
              if job.config["method"] != "none" else
              f"[{i}/{len(jobs)}] {tag}{job.run_id}  {job.config['name']}  seed={job.seed}  (baseline)")
        if a.dry_run or (done and not a.force):
            continue
        # Failed runs are re-run into the same dir; a forced rerun overwrites it.
        try:
            execute(job, runs_dir, gpu_info)
        except Exception:
            failed += 1
            traceback.print_exc()
    if failed:
        print(f"{failed} run(s) failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
