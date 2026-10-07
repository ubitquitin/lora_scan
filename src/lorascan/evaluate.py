"""Dataset loading, generation, and evaluators.

Evaluators answer "was this completion correct?" and return BOTH aggregate metrics and
per-item records. Comparing runs (McNemar, CIs, ...) lives in analysis/, not here.
Heavy imports (torch, datasets) are lazy so the pure-python parts are testable anywhere.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ evaluators
_NUM = r"-?\d[\d,]*\.?\d*"


def _norm_number(s: str | None) -> str | None:
    import math
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        x = float(s)
    except ValueError:
        return None
    # Handle infinity and NaN - treat as invalid predictions
    if math.isinf(x) or math.isnan(x):
        return None
    return str(int(x)) if x == int(x) else str(x)


class Evaluator:
    """Subclass and implement `extract`; override `check` if equality isn't right."""

    def extract(self, completion: str) -> str | None:
        raise NotImplementedError

    def check(self, prediction: str | None, gold: str) -> bool:
        return prediction is not None and prediction == gold

    def score(self, completion: str, gold: str) -> bool:
        return self.check(self.extract(completion), gold)

    def per_item(self, predictions: list[dict]) -> list[dict]:
        rows = []
        for p in predictions:
            pred = self.extract(p["completion"])
            rows.append({
                "item_id": p["item_id"], "prediction": pred, "gold": p["gold"],
                "correct": bool(self.check(pred, p["gold"])),
                "completion_tokens": p.get("completion_tokens"),
                "completion": p["completion"],
            })
        return rows

    def evaluate(self, predictions: list[dict]) -> dict:
        rows = self.per_item(predictions)
        n = len(rows)
        n_correct = sum(r["correct"] for r in rows)
        toks = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
        return {
            "accuracy": n_correct / n if n else 0.0, "n": n, "n_correct": n_correct,
            "n_unparsed": sum(r["prediction"] is None for r in rows),
            "mean_completion_tokens": sum(toks) / len(toks) if toks else None,
        }


class ExactMatchEvaluator(Evaluator):
    def extract(self, completion):
        return completion.strip().splitlines()[-1].strip() if completion.strip() else None


class GSM8KEvaluator(Evaluator):
    """Final numeric answer: after '####', else after 'final answer', else last number."""

    def extract(self, completion):
        if "####" in completion:
            nums = re.findall(_NUM, completion.split("####")[-1])
            return _norm_number(nums[0]) if nums else None
        hits = re.findall(rf"final answer[^\d\-]*({_NUM})", completion, flags=re.I)
        if hits:
            return _norm_number(hits[-1])
        nums = re.findall(_NUM, completion)
        return _norm_number(nums[-1]) if nums else None


class MultipleChoiceEvaluator(Evaluator):
    """Extract a letter A-D."""

    def extract(self, completion):
        for pat in (r"answer\s*(?:is|:)?[\s:\(\*]*([A-D])\b", r"^\s*\(?([A-D])\)?[\.\):]?\s*$"):
            hits = re.findall(pat, completion, flags=re.I | re.M)
            if hits:
                return hits[-1].upper()
        hits = re.findall(r"\b([A-D])\b", completion)
        return hits[-1] if hits else None


EVALUATORS = {
    "exact_match": ExactMatchEvaluator,
    "multiple_choice": MultipleChoiceEvaluator,
    "gsm8k": GSM8KEvaluator,
}


def get_evaluator(name: str) -> Evaluator:
    if name not in EVALUATORS:
        raise KeyError(f"Unknown evaluator {name!r}. Known: {sorted(EVALUATORS)}")
    return EVALUATORS[name]()


# ------------------------------------------------------------------ dataset adapters
# One adapter per evaluator kind: raw HF example -> {question, gold, target, target_short}
def _fmt_gsm8k(ex: dict, cfg: dict) -> dict:
    reasoning, _, final = ex["answer"].partition("####")
    reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()
    gold = _norm_number(final.strip()) or final.strip()
    return {"question": ex["question"], "gold": gold,
            "target": f"{reasoning}\nThe final answer is {gold}.", "target_short": gold}


def _fmt_mc(ex: dict, cfg: dict) -> dict:
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", ex["choices"]))
    gold = "ABCD"[ex["answer"]]
    return {"question": f"{ex['question']}\n\n{opts}", "gold": gold,
            "target": f"The answer is {gold}.", "target_short": gold}


def _fmt_exact(ex: dict, cfg: dict) -> dict:
    gold = str(ex[cfg.get("answer_field", "answer")])
    return {"question": ex[cfg.get("question_field", "question")], "gold": gold,
            "target": gold, "target_short": gold}


FORMATTERS = {"gsm8k": _fmt_gsm8k, "multiple_choice": _fmt_mc, "exact_match": _fmt_exact}


def load_items(dataset_cfg: dict, data_cfg: dict, split: str, limit: int | None = None) -> list[dict]:
    """Load a split as a list of items. Subsampling uses a FIXED seed so every run sees the same items."""
    from datasets import load_dataset

    ds = load_dataset(dataset_cfg["hf_id"], dataset_cfg.get("subset"),
                      split=dataset_cfg[f"{split}_split"], revision=dataset_cfg.get("revision"))
    ds = ds.add_column("_idx", list(range(len(ds))))
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=data_cfg.get("subsample_seed", 0)).select(range(limit))
    fmt = FORMATTERS[dataset_cfg["evaluator"]]
    items = []
    for ex in ds:
        item = fmt(ex, dataset_cfg)
        item["item_id"] = f"{split}-{ex['_idx']}"
        items.append(item)
    return items


# ------------------------------------------------------------------ prompting + generation
def render_prompt(tokenizer, item: dict, prompt_cfg: dict) -> str:
    text = prompt_cfg["template"].replace("{question}", item["question"]).strip()
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        if prompt_cfg.get("system"):
            messages.insert(0, {"role": "system", "content": prompt_cfg["system"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "\n"


def render_target(item: dict, prompt_cfg: dict) -> str:
    return item["target_short"] if prompt_cfg.get("target", "full") == "short" else item["target"]


def generate_predictions(model, tokenizer, items: list[dict], prompt_cfg: dict, eval_cfg: dict) -> list[dict]:
    """Batched generation. Returns [{item_id, completion, gold, completion_tokens}, ...]."""
    import torch

    prompts = [render_prompt(tokenizer, it, prompt_cfg) for it in items]
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    temp = eval_cfg.get("temperature", 0.0)
    gen_kwargs = {"max_new_tokens": eval_cfg["max_new_tokens"], "do_sample": temp > 0,
                  "pad_token_id": tokenizer.pad_token_id}
    if temp > 0:
        gen_kwargs["temperature"] = temp
    add_special = getattr(tokenizer, "chat_template", None) is None

    prev_side, tokenizer.padding_side = tokenizer.padding_side, "left"
    preds, bs = [], eval_cfg["batch_size"]
    try:
        for i in range(0, len(items), bs):
            batch = tokenizer(prompts[i:i + bs], return_tensors="pt", padding=True,
                              add_special_tokens=add_special).to(device)
            with torch.no_grad():
                out = model.generate(**batch, **gen_kwargs)
            gen = out[:, batch["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for it, text, g in zip(items[i:i + bs], texts, gen):
                preds.append({"item_id": it["item_id"], "completion": text, "gold": it["gold"],
                              "completion_tokens": int((g != tokenizer.pad_token_id).sum())})
    finally:
        tokenizer.padding_side = prev_side
    return preds
