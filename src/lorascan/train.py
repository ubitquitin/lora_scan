"""Model loading, LoRA construction, SFT and GRPO. Both trainers share one signature."""
from __future__ import annotations

import time
from pathlib import Path

from .evaluate import render_prompt, render_target, render_training_text, get_assistant_prefix


def seed_everything(seed: int) -> None:
    from transformers import set_seed
    set_seed(seed)


def verify_gpu() -> dict:
    """Verify GPU availability and return device info. Raises RuntimeError if no GPU."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-capable GPU detected. This project requires GPU for training.\n"
            "Please ensure:\n"
            "  - CUDA is installed\n"
            "  - PyTorch was installed with CUDA support\n"
            "  - A compatible GPU is available"
        )
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    return {
        "device_id": device_id,
        "device_name": device_name,
        "memory_gb": round(memory_gb, 2),
        "supports_bf16": torch.cuda.is_bf16_supported(),
    }


def _dtype():
    """Model loading dtype. Fixed to bf16 to match rank1 pilot."""
    import torch
    return torch.bfloat16  # Mentor's rank1 pilot uses bf16


def load_model_and_tokenizer(model_cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"revision": model_cfg.get("revision"), "torch_dtype": _dtype(), "device_map": "auto"}
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    tok = AutoTokenizer.from_pretrained(model_cfg["hf_id"], revision=model_cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **kwargs)
    return model, tok


def build_lora_config(lora_cfg: dict, model_cfg: dict, layers: list[int]):
    from peft import LoraConfig
    return LoraConfig(
        r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg.get("target_modules") or model_cfg["target_modules"],
        layers_to_transform=layers, use_rslora=lora_cfg.get("use_rslora", False),
        bias="none", task_type="CAUSAL_LM",
    )


# Removed _precision_flags() - now using explicit fp16 from config to match mentor
# def _precision_flags() -> dict:
#     import torch
#     return {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}


class ProgressCallback:
    """Custom callback for detailed progress reporting during training."""

    def __init__(self, max_steps: int):
        from transformers import TrainerCallback
        self.max_steps = max_steps
        self.start_time = None
        self.last_log = None

        class _Callback(TrainerCallback):
            def __init__(self, progress_tracker):
                self.tracker = progress_tracker

            def on_train_begin(self, args, state, control, **kwargs):
                self.tracker.start_time = time.time()
                self.tracker.last_log = self.tracker.start_time
                print(f"\n{'='*70}")
                print(f"  Training started: {self.tracker.max_steps} steps")
                print(f"{'='*70}")

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    current_step = state.global_step
                    elapsed = time.time() - self.tracker.start_time

                    # Use trainer's max_steps if available (it recalculates for num_train_epochs)
                    max_steps = state.max_steps if (state.max_steps is not None and state.max_steps > 0) else self.tracker.max_steps

                    # Calculate progress (handle None case)
                    if max_steps is None or max_steps <= 0:
                        progress_pct = 0
                    else:
                        progress_pct = (current_step / max_steps) * 100

                    # Estimate time remaining
                    if current_step > 0 and max_steps is not None and max_steps > 0:
                        avg_time_per_step = elapsed / current_step
                        remaining_steps = max_steps - current_step
                        eta_seconds = avg_time_per_step * remaining_steps

                        # Format times
                        elapsed_str = self._format_time(elapsed)
                        eta_str = self._format_time(eta_seconds)
                        throughput = current_step / elapsed if elapsed > 0 else 0

                        # Progress bar
                        bar_width = 40
                        filled = int(bar_width * current_step / max_steps)
                        bar = '█' * filled + '░' * (bar_width - filled)

                        # Print progress update
                        print(f"\r[{bar}] {progress_pct:5.1f}% | "
                              f"Step {current_step}/{max_steps} | "
                              f"Loss: {logs['loss']:.4f} | "
                              f"Elapsed: {elapsed_str} | "
                              f"ETA: {eta_str} | "
                              f"{throughput:.2f} steps/s", end='', flush=True)

            def on_train_end(self, args, state, control, **kwargs):
                elapsed = time.time() - self.tracker.start_time
                print(f"\n{'='*70}")
                print(f"  Training completed in {self._format_time(elapsed)}")
                print(f"{'='*70}\n")

            @staticmethod
            def _format_time(seconds):
                """Format seconds into human-readable time."""
                if seconds < 60:
                    return f"{seconds:.0f}s"
                elif seconds < 3600:
                    mins = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{mins}m {secs}s"
                else:
                    hours = int(seconds // 3600)
                    mins = int((seconds % 3600) // 60)
                    return f"{hours}h {mins}m"

        self.callback = _Callback(self)


def _info(trainer, output_dir: Path, time_train: float, vram_train: float | None) -> tuple:
    import math

    model = trainer.model
    model.save_pretrained(output_dir)  # adapter weights only
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]

    # Sanitize final loss (convert inf/nan to None)
    final_loss = losses[-1] if losses else None
    if final_loss is not None and (math.isinf(final_loss) or math.isnan(final_loss)):
        final_loss = None

    info = {
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_train_loss": final_loss,
        "steps": trainer.state.global_step,
        "time_train": round(time_train, 2),
        "vram_train": round(vram_train, 2) if vram_train is not None else None,
    }
    return model, info


def train_sft(*, model, tokenizer, dataset, prompt, lora_config, training_config, seed, output_dir):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback
    import trl

    # Prepare dataset in mentor's format: single "text" field with full chat template
    rows = [{"text": render_training_text(tokenizer, it, prompt)} for it in dataset]

    # Setup completion-only loss masking (matches mentor)
    # In TRL < 0.20: use DataCollatorForCompletionOnlyLM
    # In TRL >= 0.20: use completion_only_loss=True parameter
    assistant_prefix = get_assistant_prefix(tokenizer)
    response_template_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)

    print(f"  Response template: {repr(assistant_prefix)}")
    print(f"  Response template IDs: {response_template_ids}")

    # Check TRL version to determine approach
    trl_version = tuple(map(int, trl.__version__.split('.')[:2]))
    use_new_approach = trl_version >= (0, 20)

    if use_new_approach:
        print(f"  Using TRL {trl.__version__} completion_only_loss parameter")
        collator = None  # Not needed in new versions
        # Will add completion_only_loss to training_config below
    else:
        print(f"  Using TRL {trl.__version__} DataCollatorForCompletionOnlyLM")
        try:
            from transformers import DataCollatorForCompletionOnlyLM
        except ImportError:
            from trl import DataCollatorForCompletionOnlyLM

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )

    # Create progress callback - max_steps will be set by trainer if using num_train_epochs
    # For now, estimate it or wait for trainer to set it
    max_steps = training_config.get("max_steps")
    if max_steps is None and "num_train_epochs" in training_config:
        # Rough estimate: epochs * dataset_size / (batch_size * grad_accum)
        batch_size = training_config.get("per_device_train_batch_size", 4)
        grad_accum = training_config.get("gradient_accumulation_steps", 1)
        epochs = training_config["num_train_epochs"]
        max_steps = int(len(rows) * epochs / (batch_size * grad_accum))
    elif max_steps is None:
        max_steps = 500  # default fallback

    progress = ProgressCallback(max_steps)

    # Extract custom early stopping parameters (not recognized by SFTConfig)
    training_config = {**training_config}  # Copy to avoid mutating original
    early_stopping_enabled = training_config.pop("early_stopping_enabled", False)
    early_stopping_patience = training_config.pop("early_stopping_patience", 3)
    early_stopping_eval_steps = training_config.pop("early_stopping_eval_steps", 50)

    # Remove max_steps if it's None (let SFTConfig use num_train_epochs instead)
    if training_config.get("max_steps") is None:
        training_config.pop("max_steps", None)

    # Setup early stopping if enabled
    callbacks = [progress.callback]
    if early_stopping_enabled:
        # Add evaluation strategy for early stopping
        training_config["eval_strategy"] = "steps"
        training_config["eval_steps"] = early_stopping_eval_steps
        training_config["load_best_model_at_end"] = True
        training_config["metric_for_best_model"] = "loss"
        training_config["save_strategy"] = "steps"
        training_config["save_steps"] = early_stopping_eval_steps
        training_config["save_total_limit"] = 1

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"  Early stopping enabled: patience={early_stopping_patience}, eval_steps={early_stopping_eval_steps}")
    else:
        training_config["save_strategy"] = "no"

    # Match mentor exactly: use explicit fp16 from config, not dynamic detection
    # Add completion_only_loss for TRL >= 0.20 (new approach)
    if use_new_approach:
        training_config = {**training_config, "completion_only_loss": True}

    # Handle warmup_ratio compatibility (some TRL versions don't accept it)
    # Convert warmup_ratio to warmup_steps if needed
    training_config_copy = {**training_config}
    if "warmup_ratio" in training_config_copy and "warmup_steps" not in training_config_copy:
        # Calculate warmup_steps from warmup_ratio
        warmup_ratio = training_config_copy.pop("warmup_ratio")

        # Estimate total steps
        if training_config_copy.get("max_steps"):
            total_steps = training_config_copy["max_steps"]
        elif "num_train_epochs" in training_config_copy:
            batch_size = training_config_copy.get("per_device_train_batch_size", 4)
            grad_accum = training_config_copy.get("gradient_accumulation_steps", 1)
            epochs = training_config_copy["num_train_epochs"]
            total_steps = int(len(rows) * epochs / (batch_size * grad_accum))
        else:
            total_steps = 500  # fallback

        warmup_steps = int(total_steps * warmup_ratio)
        training_config_copy["warmup_steps"] = warmup_steps
        print(f"  Converted warmup_ratio {warmup_ratio} -> warmup_steps {warmup_steps} (total: {total_steps})")

    args = SFTConfig(output_dir=str(output_dir), seed=seed, report_to="none",
                     **training_config_copy)

    # Build trainer (with or without data_collator depending on TRL version)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": Dataset.from_list(rows),
        "eval_dataset": Dataset.from_list(rows[:min(100, len(rows))]) if early_stopping_enabled else None,
        "processing_class": tokenizer,
        "peft_config": lora_config,
        "callbacks": callbacks,
    }

    if not use_new_approach:
        trainer_kwargs["data_collator"] = collator  # Only needed for old TRL versions

    trainer = SFTTrainer(**trainer_kwargs)

    # Reset VRAM stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    trainer.train()

    # Capture training metrics
    time_train = time.time() - t0
    vram_train = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    print(f"  Training time: {time_train:.1f}s")
    if vram_train is not None:
        print(f"  Peak VRAM (train): {vram_train:.2f} GB")

    return _info(trainer, Path(output_dir), time_train, vram_train)


def train_grpo(*, model, tokenizer, dataset, prompt, lora_config, training_config, seed, output_dir, evaluator):
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = [{"prompt": render_prompt(tokenizer, it, prompt), "gold": it["gold"]} for it in dataset]

    def correctness_reward(completions, gold, **_):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [1.0 if evaluator.score(t, g) else 0.0 for t, g in zip(texts, gold)]

    # Create progress callback
    max_steps = training_config.get("max_steps", 500)
    progress = ProgressCallback(max_steps)

    # Use explicit precision from config (mentor doesn't use GRPO, but keep consistent)
    args = GRPOConfig(output_dir=str(output_dir), seed=seed, report_to="none", save_strategy="no",
                      **training_config)
    trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward], args=args,
                          train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                          peft_config=lora_config, callbacks=[progress.callback])

    # Reset VRAM stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    trainer.train()

    # Capture training metrics
    time_train = time.time() - t0
    vram_train = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    print(f"  Training time: {time_train:.1f}s")
    if vram_train is not None:
        print(f"  Peak VRAM (train): {vram_train:.2f} GB")

    return _info(trainer, Path(output_dir), time_train, vram_train)


def train(config: dict, seed: int, layers: list[int], model, tokenizer, dataset, evaluator, output_dir):
    """Dispatch on config['method']. Returns (model, info)."""
    method = config["method"]
    if method == "none":
        return model, {}
    kwargs = dict(model=model, tokenizer=tokenizer, dataset=dataset, prompt=config["prompt"],
                  lora_config=build_lora_config(config["lora"], config["model"], layers),
                  training_config=config["training"], seed=seed, output_dir=output_dir)
    if method == "sft":
        return train_sft(**kwargs)
    if method == "grpo":
        return train_grpo(**kwargs, evaluator=evaluator)
    raise ValueError(f"Unknown method {method!r}")
