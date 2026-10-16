"""Dataset loading, generation, and evaluators.

Evaluators answer "was this completion correct?" and return BOTH aggregate metrics and
per-item records. Comparing runs (McNemar, CIs, ...) lives in analysis/, not here.
Heavy imports (torch, datasets) are lazy so the pure-python parts are testable anywhere.
"""
from __future__ import annotations

import re


# ------------------------------------------------------------------ chat template utilities
def get_assistant_prefix(tokenizer) -> str:
    """Extract the assistant turn prefix from tokenizer's chat template.

    Returns the string that appears between the user message and assistant message
    in a chat-formatted conversation. This is used for:
    1. Loss masking (DataCollatorForCompletionOnlyLM response_template)
    2. Evaluation prompts (where to append "Answer:")

    Matches mentor's implementation using sentinel tokens to avoid ambiguity.
    """
    _USER = '⚙USR⚙'
    _ASST = '⚙AST⚙'
    probe = tokenizer.apply_chat_template(
        [{'role': 'user',      'content': _USER},
         {'role': 'assistant', 'content': _ASST}],
        tokenize=False, add_generation_prompt=False
    )
    start = probe.rfind(_USER) + len(_USER)
    end   = probe.rfind(_ASST)
    return probe[start:end]

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
    """Final numeric answer: \boxed{}, then '####', then 'final answer', else last number.

    Priority order matches mentor's notebook:
    1. \boxed{number} format
    2. #### number format
    3. "final answer" followed by number
    4. Last number in text
    """

    def extract(self, completion):
        # Priority 1: \boxed{} format (mentor's notebook checks this first)
        boxes = re.findall(r'\\boxed\{(' + _NUM + r')\}', completion)
        if boxes:
            return _norm_number(boxes[-1])

        # Priority 2: #### format (standard GSM8K format)
        if "####" in completion:
            nums = re.findall(_NUM, completion.split("####")[-1])
            return _norm_number(nums[0]) if nums else None

        # Priority 3: "final answer" phrase
        hits = re.findall(rf"final answer[^\d\-]*({_NUM})", completion, flags=re.I)
        if hits:
            return _norm_number(hits[-1])

        # Priority 4: Last number anywhere in completion
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


class CombinedEvaluator(Evaluator):
    """Routes to the correct evaluator based on source_dataset tag."""

    def __init__(self):
        self.evaluators = {
            "mmlu": MultipleChoiceEvaluator(),
            "gsm8k": GSM8KEvaluator(),
        }

    def extract(self, completion):
        # This shouldn't be called directly on combined - use per_item instead
        raise NotImplementedError("CombinedEvaluator requires source_dataset in per_item")

    def per_item(self, predictions: list[dict]) -> list[dict]:
        """Evaluate each item with its appropriate evaluator."""
        rows = []
        for p in predictions:
            source = p.get("source_dataset")
            if not source or source not in self.evaluators:
                # Fallback - try both evaluators and use whichever succeeds
                pred = None
                for ev in self.evaluators.values():
                    pred = ev.extract(p["completion"])
                    if pred is not None:
                        break
            else:
                evaluator = self.evaluators[source]
                pred = evaluator.extract(p["completion"])

            rows.append({
                "item_id": p["item_id"],
                "prediction": pred,
                "gold": p["gold"],
                "correct": bool(pred is not None and pred == p["gold"]),
                "completion_tokens": p.get("completion_tokens"),
                "completion": p["completion"],
                "source_dataset": p.get("source_dataset"),
            })
        return rows

    def evaluate(self, predictions: list[dict]) -> dict:
        """Evaluate with breakdown by source dataset."""
        rows = self.per_item(predictions)

        # Overall metrics
        n = len(rows)
        n_correct = sum(r["correct"] for r in rows)
        toks = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]

        result = {
            "accuracy": n_correct / n if n else 0.0,
            "n": n,
            "n_correct": n_correct,
            "n_unparsed": sum(r["prediction"] is None for r in rows),
            "mean_completion_tokens": sum(toks) / len(toks) if toks else None,
        }

        # Add per-dataset breakdown
        for source in set(r.get("source_dataset") for r in rows if r.get("source_dataset")):
            subset = [r for r in rows if r.get("source_dataset") == source]
            n_sub = len(subset)
            n_correct_sub = sum(r["correct"] for r in subset)
            result[f"{source}_accuracy"] = n_correct_sub / n_sub if n_sub else 0.0
            result[f"{source}_n"] = n_sub
            result[f"{source}_n_correct"] = n_correct_sub

        return result


EVALUATORS = {
    "exact_match": ExactMatchEvaluator,
    "multiple_choice": MultipleChoiceEvaluator,
    "gsm8k": GSM8KEvaluator,
    "combined": CombinedEvaluator,
}

# Evaluators that should use logit-based evaluation instead of generation
LOGIT_BASED_EVALUATORS = {"multiple_choice"}


def get_evaluator(name: str) -> Evaluator:
    if name not in EVALUATORS:
        raise KeyError(f"Unknown evaluator {name!r}. Known: {sorted(EVALUATORS)}")
    return EVALUATORS[name]()


def evaluate_predictions(model, tokenizer, items: list[dict], prompt_cfg: dict, eval_cfg: dict,
                        dataset_cfg: dict) -> tuple[list[dict], dict]:
    """Route to appropriate evaluation method based on dataset type.

    - Multiple choice (MMLU): Uses logit-based evaluation (faster, matches mentor)
    - GSM8K: Uses generation-based evaluation
    - Combined: Routes each subset appropriately

    Returns (predictions, eval_metrics).
    """
    evaluator_name = dataset_cfg["evaluator"]

    # Handle combined datasets - need to evaluate subsets separately
    if evaluator_name == "combined":
        return _evaluate_combined(model, tokenizer, items, prompt_cfg, eval_cfg, dataset_cfg)

    # Route to logit-based or generation-based evaluation
    if evaluator_name in LOGIT_BASED_EVALUATORS:
        print(f"  Using logit-based evaluation for {evaluator_name}")
        return evaluate_multiple_choice_logits(model, tokenizer, items, prompt_cfg, eval_cfg)
    else:
        print(f"  Using generation-based evaluation for {evaluator_name}")
        return generate_predictions(model, tokenizer, items, prompt_cfg, eval_cfg)


def _evaluate_combined(model, tokenizer, items: list[dict], prompt_cfg: dict, eval_cfg: dict,
                       dataset_cfg: dict) -> tuple[list[dict], dict]:
    """Evaluate combined datasets by routing each subset to appropriate method."""
    import time
    import torch

    # Split items by source dataset
    items_by_source = {}
    for item in items:
        source = item.get("source_dataset", "unknown")
        items_by_source.setdefault(source, []).append(item)

    print(f"  Combined evaluation:")
    for source, source_items in items_by_source.items():
        print(f"    {source}: {len(source_items)} items")

    # Evaluate each subset with appropriate method
    all_preds = []
    total_time = 0.0
    max_vram = 0.0

    for source, source_items in sorted(items_by_source.items()):
        print(f"\n  Evaluating {source}...")

        # Determine evaluation method for this source
        # MMLU uses logit-based, GSM8K uses generation
        if source == "mmlu":
            preds, metrics = evaluate_multiple_choice_logits(model, tokenizer, source_items,
                                                            prompt_cfg, eval_cfg)
        else:  # gsm8k and others
            preds, metrics = generate_predictions(model, tokenizer, source_items,
                                                 prompt_cfg, eval_cfg)

        all_preds.extend(preds)
        total_time += metrics.get("time_eval", 0)
        max_vram = max(max_vram, metrics.get("vram_eval") or 0)

    combined_metrics = {
        "time_eval": round(total_time, 2),
        "vram_eval": round(max_vram, 2) if max_vram > 0 else None,
    }

    return all_preds, combined_metrics


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
    from datasets import load_dataset, concatenate_datasets
    from . import runs

    # Handle combined datasets
    if dataset_cfg.get("type") == "combined":
        all_items = []
        for dataset_name in dataset_cfg["datasets"]:
            sub_cfg = runs.load_component("datasets", dataset_name)
            # Use dataset-specific limit if defined (e.g., eval_limit for GSM8K)
            sub_limit = sub_cfg.get(f"{split}_limit") if split == "eval" else None
            sub_items = load_items(sub_cfg, data_cfg, split, limit=sub_limit)
            # Tag items with source dataset for debugging
            for item in sub_items:
                item["source_dataset"] = dataset_name
            all_items.extend(sub_items)

        # Shuffle combined items
        import random
        rng = random.Random(data_cfg.get("subsample_seed", 42))
        rng.shuffle(all_items)

        # Apply overall limit after combining (if specified)
        if limit and limit < len(all_items):
            all_items = all_items[:limit]

        # Debug: Show what we loaded
        if split == "eval":
            print(f"  Combined {split} dataset loaded:")
            source_counts = {}
            for item in all_items:
                source = item.get("source_dataset", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            for source, count in sorted(source_counts.items()):
                print(f"    {source}: {count} samples")
            print(f"    Total: {len(all_items)} samples")

        return all_items

    # Special handling for MMLU per-subject loading (mentor notebook style)
    # Applies to both train and eval when subset is 'per_subject'
    is_mmlu_per_subject = (dataset_cfg["hf_id"] == "cais/mmlu" and
                           dataset_cfg.get("subset") == "per_subject")

    if is_mmlu_per_subject:
        from datasets import concatenate_datasets

        # MMLU subject list (all 57 subjects)
        subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_medicine',
            'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics',
            'formal_logic', 'global_facts', 'high_school_biology',
            'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography',
            'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics',
            'high_school_us_history', 'high_school_world_history', 'human_aging',
            'human_sexuality', 'international_law', 'jurisprudence',
            'logical_fallacies', 'machine_learning', 'management', 'marketing',
            'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
            'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions'
        ]

        seed = data_cfg.get("subsample_seed", 42)
        all_parts = []

        if split == "train":
            # Train: Combine auxiliary_train + validation + dev per subject (mentor notebook)
            combine_splits = dataset_cfg.get("train_combine_splits", ["auxiliary_train"])
            print(f"  Loading MMLU train per-subject: combining {combine_splits} for 57 subjects")
            print(f"  This may take 5-10 minutes on first run (downloading datasets)...")

            for i, subject in enumerate(subjects, 1):
                if i % 10 == 1:
                    print(f"  Loading MMLU subjects: {i}/{len(subjects)}...", flush=True)

                # Load full subject dataset
                subject_full = load_dataset("cais/mmlu", subject, revision=dataset_cfg.get("revision"))

                # Combine specified splits for this subject
                subject_parts = []
                for split_name in combine_splits:
                    if split_name in subject_full:
                        ds_part = subject_full[split_name]
                        subject_parts.append(ds_part)

                if subject_parts:
                    subject_ds = concatenate_datasets(subject_parts)
                    subject_ds = subject_ds.add_column("_idx",
                                                      [f"{subject}-{j}" for j in range(len(subject_ds))])
                    all_parts.append(subject_ds)

            print(f"  Loaded all 57 MMLU subjects, combining...", flush=True)
            ds = concatenate_datasets(all_parts).shuffle(seed=seed)
            print(f"  MMLU train: {len(ds)} examples from {len(subjects)} subjects", flush=True)

        else:  # eval split
            # Eval: Sample N per subject (mentor notebook)
            samples_per_subject = dataset_cfg.get("eval_samples_per_subject", 8)
            print(f"  Loading MMLU eval per-subject: {samples_per_subject} samples × 57 subjects = {samples_per_subject * 57} total")
            print(f"  This may take 5-10 minutes on first run (downloading datasets)...")

            for i, subject in enumerate(subjects, 1):
                if i % 10 == 1:
                    print(f"  Loading MMLU subjects: {i}/{len(subjects)}...", flush=True)
                subject_ds = load_dataset("cais/mmlu", subject, split="test",
                                         revision=dataset_cfg.get("revision"))
                subject_ds = subject_ds.add_column("_idx",
                                                  [f"{subject}-{j}" for j in range(len(subject_ds))])
                # Shuffle and take first N samples per subject
                subject_ds = subject_ds.shuffle(seed=seed)
                n_samples = min(samples_per_subject, len(subject_ds))
                subject_ds = subject_ds.select(range(n_samples))
                all_parts.append(subject_ds)

            print(f"  Loaded all 57 MMLU subjects, combining...", flush=True)
            ds = concatenate_datasets(all_parts).shuffle(seed=seed)

        # Convert to items
        fmt = FORMATTERS[dataset_cfg["evaluator"]]
        items = []
        for ex in ds:
            item = fmt(ex, dataset_cfg)
            item["item_id"] = f"{split}-{ex['_idx']}"
            items.append(item)
        return items

    # Regular single dataset loading (all non-per-subject datasets)
    ds = load_dataset(dataset_cfg["hf_id"], dataset_cfg.get("subset"),
                      split=dataset_cfg[f"{split}_split"], revision=dataset_cfg.get("revision"))
    original_size = len(ds)
    ds = ds.add_column("_idx", list(range(len(ds))))
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=data_cfg.get("subsample_seed", 0)).select(range(limit))
        print(f"  {dataset_cfg['name']} {split}: Limited from {original_size} to {len(ds)} samples (limit={limit})", flush=True)
    fmt = FORMATTERS[dataset_cfg["evaluator"]]
    items = []
    for ex in ds:
        item = fmt(ex, dataset_cfg)
        item["item_id"] = f"{split}-{ex['_idx']}"
        items.append(item)
    return items


# ------------------------------------------------------------------ prompting + generation
def _get_prompt_for_item(item: dict, prompt_cfg: dict) -> dict:
    """Get the appropriate prompt config for an item, handling combined datasets."""
    # If prompt_cfg has a 'prompts' dict (for combined datasets), route by source_dataset
    if "prompts" in prompt_cfg:
        source = item.get("source_dataset")
        if source and source in prompt_cfg["prompts"]:
            return prompt_cfg["prompts"][source]
        # Fallback to first prompt if source not found
        return list(prompt_cfg["prompts"].values())[0]
    return prompt_cfg


def render_prompt(tokenizer, item: dict, prompt_cfg: dict) -> str:
    """Render prompt for evaluation (user message only, with generation_prompt=True)."""
    cfg = _get_prompt_for_item(item, prompt_cfg)
    text = cfg["template"].replace("{question}", item["question"]).strip()
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        if cfg.get("system"):
            messages.insert(0, {"role": "system", "content": cfg["system"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "\n"


def render_target(item: dict, prompt_cfg: dict) -> str:
    """Render target completion (assistant response text only, no template markers)."""
    cfg = _get_prompt_for_item(item, prompt_cfg)
    return item["target_short"] if cfg.get("target", "full") == "short" else item["target"]


def render_training_text(tokenizer, item: dict, prompt_cfg: dict) -> str:
    """Render full chat-formatted text for training (user + assistant messages).

    This is the format mentor's notebook uses: single "text" field with full conversation.
    Used with DataCollatorForCompletionOnlyLM to mask loss on prompt tokens.
    """
    cfg = _get_prompt_for_item(item, prompt_cfg)
    question_text = cfg["template"].replace("{question}", item["question"]).strip()
    answer_text = render_target(item, prompt_cfg)

    if getattr(tokenizer, "chat_template", None):
        messages = []
        if cfg.get("system"):
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": question_text})
        messages.append({"role": "assistant", "content": answer_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback for models without chat template
        return f"{question_text}\n{answer_text}"


def evaluate_multiple_choice_logits(model, tokenizer, items: list[dict], prompt_cfg: dict, eval_cfg: dict) -> tuple[list[dict], dict]:
    """Logit-based multiple choice evaluation (matches mentor's MMLU approach).

    Instead of generating text, extracts logits for A/B/C/D tokens and picks the highest.
    Much faster and more deterministic than generation-based evaluation.

    Returns (predictions, eval_metrics).
    """
    import torch
    import time

    CHOICES = ['A', 'B', 'C', 'D']

    # Render prompts - for multiple choice, we add "Answer:" at the end
    prompts = []
    for it in items:
        cfg = _get_prompt_for_item(it, prompt_cfg)
        text = cfg["template"].replace("{question}", it["question"]).strip()
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": text}]
            if cfg.get("system"):
                messages.insert(0, {"role": "system", "content": cfg["system"]})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = text + "\n"
        # Add "Answer:" suffix for multiple choice prompts
        prompt = prompt + "Answer:"
        prompts.append(prompt)

    model.eval()
    device = next(model.parameters()).device

    # Get token IDs for A, B, C, D (with space prefix to match typical tokenization)
    choice_ids = [tokenizer.encode(' ' + c, add_special_tokens=False)[-1] for c in CHOICES]
    print(f"    Multiple choice evaluation (logit-based)")
    print(f"    Choice token IDs: {list(zip(CHOICES, choice_ids))}")

    # Reset VRAM stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    prev_side, tokenizer.padding_side = tokenizer.padding_side, "right"
    preds, bs = [], eval_cfg["batch_size"]

    try:
        for i in range(0, len(items), bs):
            batch_prompts = prompts[i:i + bs]
            batch_items = items[i:i + bs]

            # Tokenize
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(device)

            # Logits diagnostics on first batch
            if i == 0:
                with torch.no_grad():
                    diag_out = model(**{k: v[:1] for k, v in enc.items()})
                    nan_count = torch.isnan(diag_out.logits).sum().item()
                    logit_min = diag_out.logits.min().item()
                    logit_max = diag_out.logits.max().item()
                    print(f"    Logits diagnostics (first batch):")
                    print(f"      NaNs in logits: {nan_count}")
                    print(f"      Logits min/max: {logit_min:.2f} / {logit_max:.2f}")

            # Forward pass
            with torch.no_grad():
                out = model(**enc)

            # Get last token position for each sequence (accounting for padding)
            attention_mask = enc['attention_mask']
            last_token_pos = attention_mask.sum(dim=1) - 1  # [batch_size]

            # Extract logits at last position for each sequence
            batch_indices = torch.arange(out.logits.shape[0], device=device)
            last_logits = out.logits[batch_indices, last_token_pos]  # [batch_size, vocab_size]

            # Get logits for A, B, C, D tokens only
            choice_logits = last_logits[:, choice_ids]  # [batch_size, 4]

            # Argmax to get predictions
            pred_indices = choice_logits.argmax(dim=-1).cpu().tolist()
            pred_letters = [CHOICES[idx] for idx in pred_indices]

            # Build prediction records
            for it, pred_letter in zip(batch_items, pred_letters):
                pred = {
                    "item_id": it["item_id"],
                    "completion": pred_letter,  # Store just the letter
                    "gold": it["gold"],
                    "completion_tokens": 1,  # Only "predicted" 1 token (the letter)
                }
                if "source_dataset" in it:
                    pred["source_dataset"] = it["source_dataset"]
                preds.append(pred)

            if (i + bs) % 100 == 0 or (i + bs) >= len(items):
                print(f"    Progress: {min(i + bs, len(items))}/{len(items)}")

    finally:
        tokenizer.padding_side = prev_side

    # Capture eval metrics
    time_eval = time.time() - t0
    vram_eval = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    eval_metrics = {
        "time_eval": round(time_eval, 2),
        "vram_eval": round(vram_eval, 2) if vram_eval is not None else None,
    }

    print(f"  Evaluation time: {time_eval:.1f}s")
    if vram_eval is not None:
        print(f"  Peak VRAM (eval): {vram_eval:.2f} GB")

    return preds, eval_metrics


def generate_predictions(model, tokenizer, items: list[dict], prompt_cfg: dict, eval_cfg: dict) -> tuple[list[dict], dict]:
    """Batched generation. Returns (predictions, eval_metrics).

    predictions: [{item_id, completion, gold, completion_tokens}, ...]
    eval_metrics: {time_eval, vram_eval}
    """
    import torch
    import time

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

    # Reset VRAM stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    prev_side, tokenizer.padding_side = tokenizer.padding_side, "left"
    preds, bs = [], eval_cfg["batch_size"]
    try:
        for i in range(0, len(items), bs):
            batch = tokenizer(prompts[i:i + bs], return_tensors="pt", padding=True,
                              add_special_tokens=add_special).to(device)

            # Logits diagnostics on first batch (match mentor's notebook)
            if i == 0:
                with torch.no_grad():
                    # Forward pass only for diagnostics
                    diag_out = model(**{k: v[:1] for k, v in batch.items()})  # First example only
                    nan_count = torch.isnan(diag_out.logits).sum().item()
                    logit_min = diag_out.logits.min().item()
                    logit_max = diag_out.logits.max().item()
                    print(f"    Logits diagnostics (first batch):")
                    print(f"      NaNs in logits: {nan_count}")
                    print(f"      Logits min/max: {logit_min:.2f} / {logit_max:.2f}")

            with torch.no_grad():
                out = model.generate(**batch, **gen_kwargs)
            gen = out[:, batch["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for it, text, g in zip(items[i:i + bs], texts, gen):
                pred = {"item_id": it["item_id"], "completion": text, "gold": it["gold"],
                        "completion_tokens": int((g != tokenizer.pad_token_id).sum())}
                # Carry through source_dataset for combined evaluators
                if "source_dataset" in it:
                    pred["source_dataset"] = it["source_dataset"]
                preds.append(pred)
    finally:
        tokenizer.padding_side = prev_side

    # Capture eval metrics
    time_eval = time.time() - t0
    vram_eval = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    eval_metrics = {
        "time_eval": round(time_eval, 2),
        "vram_eval": round(vram_eval, 2) if vram_eval is not None else None,
    }

    print(f"  Evaluation time: {time_eval:.1f}s")
    if vram_eval is not None:
        print(f"  Peak VRAM (eval): {vram_eval:.2f} GB")

    return preds, eval_metrics
