"""Model loading, LoRA construction, SFT and GRPO. Both trainers share one signature."""
from __future__ import annotations

import time
from pathlib import Path

from .evaluate import render_prompt, render_target


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
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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


def _precision_flags() -> dict:
    import torch
    return {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}


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

                    # Calculate progress
                    progress_pct = (current_step / self.tracker.max_steps) * 100

                    # Estimate time remaining
                    if current_step > 0:
                        avg_time_per_step = elapsed / current_step
                        remaining_steps = self.tracker.max_steps - current_step
                        eta_seconds = avg_time_per_step * remaining_steps

                        # Format times
                        elapsed_str = self._format_time(elapsed)
                        eta_str = self._format_time(eta_seconds)
                        throughput = current_step / elapsed if elapsed > 0 else 0

                        # Progress bar
                        bar_width = 40
                        filled = int(bar_width * current_step / self.tracker.max_steps)
                        bar = '█' * filled + '░' * (bar_width - filled)

                        # Print progress update
                        print(f"\r[{bar}] {progress_pct:5.1f}% | "
                              f"Step {current_step}/{self.tracker.max_steps} | "
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


def _info(trainer, output_dir: Path) -> tuple:
    model = trainer.model
    model.save_pretrained(output_dir)  # adapter weights only
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    info = {
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_train_loss": losses[-1] if losses else None,
        "steps": trainer.state.global_step,
    }
    return model, info


def train_sft(*, model, tokenizer, dataset, prompt, lora_config, training_config, seed, output_dir):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    eos = tokenizer.eos_token or ""
    rows = [{"prompt": render_prompt(tokenizer, it, prompt),
             "completion": render_target(it, prompt) + eos} for it in dataset]

    # Create progress callback
    max_steps = training_config.get("max_steps", 500)
    progress = ProgressCallback(max_steps)

    # Setup early stopping if enabled
    callbacks = [progress.callback]
    early_stopping_enabled = training_config.get("early_stopping_enabled", False)
    if early_stopping_enabled:
        # Add evaluation strategy for early stopping
        training_config = {**training_config}  # Copy to avoid mutating original
        training_config["eval_strategy"] = "steps"
        training_config["eval_steps"] = training_config.get("early_stopping_eval_steps", 50)
        training_config["load_best_model_at_end"] = True
        training_config["metric_for_best_model"] = "loss"
        training_config["save_strategy"] = "steps"
        training_config["save_steps"] = training_config.get("early_stopping_eval_steps", 50)
        training_config["save_total_limit"] = 1

        early_stopping_patience = training_config.get("early_stopping_patience", 3)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"  Early stopping enabled: patience={early_stopping_patience}, eval_steps={training_config['eval_steps']}")
    else:
        training_config = {**training_config, "save_strategy": "no"}

    args = SFTConfig(output_dir=str(output_dir), seed=seed, report_to="none",
                     **_precision_flags(), **training_config)
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                         eval_dataset=Dataset.from_list(rows[:min(100, len(rows))]) if early_stopping_enabled else None,
                         processing_class=tokenizer, peft_config=lora_config,
                         callbacks=callbacks)
    trainer.train()
    return _info(trainer, Path(output_dir))


def train_grpo(*, model, tokenizer, dataset, prompt, lora_config, training_config, seed, output_dir, evaluator):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = [{"prompt": render_prompt(tokenizer, it, prompt), "gold": it["gold"]} for it in dataset]

    def correctness_reward(completions, gold, **_):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [1.0 if evaluator.score(t, g) else 0.0 for t, g in zip(texts, gold)]

    # Create progress callback
    max_steps = training_config.get("max_steps", 500)
    progress = ProgressCallback(max_steps)

    args = GRPOConfig(output_dir=str(output_dir), seed=seed, report_to="none", save_strategy="no",
                      **_precision_flags(), **training_config)
    trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward], args=args,
                          train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                          peft_config=lora_config, callbacks=[progress.callback])
    trainer.train()
    return _info(trainer, Path(output_dir))


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
