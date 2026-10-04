"""Model loading, LoRA construction, SFT and GRPO. Both trainers share one signature."""
from __future__ import annotations

from pathlib import Path

from .evaluate import render_prompt, render_target


def seed_everything(seed: int) -> None:
    from transformers import set_seed
    set_seed(seed)


def _dtype():
    import torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_and_tokenizer(model_cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"revision": model_cfg.get("revision"), "torch_dtype": _dtype()}
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    tok = AutoTokenizer.from_pretrained(model_cfg["hf_id"], revision=model_cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
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
    if not torch.cuda.is_available():
        return {}
    return {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}


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

    eos = tokenizer.eos_token or ""
    rows = [{"prompt": render_prompt(tokenizer, it, prompt),
             "completion": render_target(it, prompt) + eos} for it in dataset]
    args = SFTConfig(output_dir=str(output_dir), seed=seed, report_to="none", save_strategy="none",
                     **_precision_flags(), **training_config)
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                         processing_class=tokenizer, peft_config=lora_config)
    trainer.train()
    return _info(trainer, Path(output_dir))


def train_grpo(*, model, tokenizer, dataset, prompt, lora_config, training_config, seed, output_dir, evaluator):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = [{"prompt": render_prompt(tokenizer, it, prompt), "gold": it["gold"]} for it in dataset]

    def correctness_reward(completions, gold, **_):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [1.0 if evaluator.score(t, g) else 0.0 for t, g in zip(texts, gold)]

    args = GRPOConfig(output_dir=str(output_dir), seed=seed, report_to="none", save_strategy="no",
                      **_precision_flags(), **training_config)
    trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward], args=args,
                          train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                          peft_config=lora_config)
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
