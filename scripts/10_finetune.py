"""
Step 10 — Safety fine-tuning via SFT, DPO, and/or RLHF.

Fine-tunes a model variant using LoRA, using PKU-SafeRLHF preference pairs as
the primary data source. The resulting checkpoint can be registered in
config/models.py and fed back through scripts 05–09 to compare probe
directions before vs. after fine-tuning.

Reads:
    data/raw/pku_saferlhf/pku_saferlhf_sample.csv
        columns: prompt, response_0, response_1,
                 is_response_0_safe, is_response_1_safe
    data/prompts/{lang}/harmless.jsonl  (SFT local fallback)
    data/prompts/{lang}/harmful.jsonl   (SFT refusal fallback)

Writes:
    models/finetuned/{model_key}_sft/          LoRA adapter + tokenizer
    models/finetuned/{model_key}_dpo/          LoRA adapter + tokenizer
    models/finetuned/{model_key}_rlhf_reward/  reward model adapter + tokenizer
    models/finetuned/{model_key}_rlhf/         PPO policy LoRA adapter + tokenizer

Requirements (not in base requirements.txt):
    pip install trl>=0.12.0 peft>=0.10.0 bitsandbytes>=0.43.0

Usage:
    # SFT only on base model, PKU data
    python scripts/10_finetune.py --model base --method sft

    # SFT then DPO on base model, PKU data
    python scripts/10_finetune.py --model base --method dpo

    # SFT then RLHF (reward model + PPO) on base model
    python scripts/10_finetune.py --model base --method rlhf

    # SFT then DPO sequentially on base model (alias for dpo)
    python scripts/10_finetune.py --model base --method both

    # Use local harmless/harmful prompt files instead of PKU
    python scripts/10_finetune.py --model base --method sft --data-source local --lang en

After finetuning:
    Add the saved adapter path to config/models.py as a new ModelConfig, then:
        python scripts/05_extract_activations.py --model <new_key> --lang en
        python scripts/06_train_probes.py         --model <new_key> --lang en
    to compare probe directions before vs. after fine-tuning.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES
from config.models import MODEL_ORDER, MODELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

SFT_TEMPLATE = "### Human: {prompt}\n### Assistant: {completion}"

# LoRA target modules for LLaMA/Aya-style architectures
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── training metadata ─────────────────────────────────────────────────────────

def save_training_info(out_dir: Path, info: dict) -> None:
    """Write training_info.json alongside the saved adapter."""
    info_path = out_dir / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Training info saved → {info_path}")


# ── data loading ──────────────────────────────────────────────────────────────

def load_pku_csv(pku_csv: Path) -> pd.DataFrame | None:
    if not pku_csv.exists():
        logger.warning(f"PKU-SafeRLHF file not found: {pku_csv}")
        logger.warning("Run scripts/02_download_datasets.py first, or use --data-source local")
        return None
    df = pd.read_csv(pku_csv)
    logger.info(f"Loaded PKU-SafeRLHF: {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_sft_data_pku(pku_csv: Path, max_samples: int) -> list[dict]:
    """
    Build SFT examples from PKU-SafeRLHF safe responses.
    For each row, pick the safe response as the completion target.
    """
    df = load_pku_csv(pku_csv)
    if df is None:
        return []

    records = []
    for _, row in df.iterrows():
        r0_safe = bool(row.get("is_response_0_safe", False))
        r1_safe = bool(row.get("is_response_1_safe", False))
        prompt   = str(row.get("prompt", "")).strip()
        r0       = str(row.get("response_0", "")).strip()
        r1       = str(row.get("response_1", "")).strip()

        if not prompt:
            continue
        if r0_safe and r0:
            records.append({"prompt": prompt, "completion": r0})
        elif r1_safe and r1:
            records.append({"prompt": prompt, "completion": r1})

    if max_samples and len(records) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in idx]

    logger.info(f"SFT (PKU): {len(records)} examples")
    return records


def load_sft_data_local(prompts_dir: Path, lang: str, max_samples: int) -> list[dict]:
    """
    Build SFT examples from local prompt files:
      - harmless prompts get an empty completion (model learns to be helpful)
      - harmful prompts get a refusal completion (model learns to decline)
    """
    def read_jsonl(p):
        if not p.exists():
            return []
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]

    harmless = read_jsonl(prompts_dir / lang / "harmless.jsonl")
    harmful  = read_jsonl(prompts_dir / lang / "harmful.jsonl")

    records = []
    for r in harmless:
        records.append({"prompt": r["text"], "completion": "Sure, I'd be happy to help with that."})
    for r in harmful:
        records.append({"prompt": r["text"], "completion": "I'm sorry, I can't assist with that request."})

    if not records:
        logger.warning(f"No local prompts found for lang={lang} at {prompts_dir / lang}")
        return []

    rng = np.random.default_rng(42)
    rng.shuffle(records)  # type: ignore[arg-type]
    records = records[:max_samples]

    logger.info(f"SFT (local/{lang}): {len(records)} examples "
                f"({len(harmless)} harmless + {len(harmful)} harmful)")
    return records


def load_sft_data_multilingual(path: Path, max_samples: int) -> list[dict]:
    """
    Load the balanced multilingual SFT set produced by 03c_translate_pku.py.
    Each line is {"prompt", "completion", "lang"} — PKU pairs partitioned across
    the 6 target languages (English kept verbatim, others Command-A translated).
    """
    if not path.exists():
        logger.warning(f"Multilingual SFT file not found: {path}")
        logger.warning("Run scripts/03c_translate_pku.py first.")
        return []

    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    records = [{"prompt": r["prompt"], "completion": r["completion"]}
               for r in records if r.get("prompt") and r.get("completion")]

    if max_samples and len(records) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in idx]

    logger.info(f"SFT (multilingual): {len(records)} examples")
    return records


def load_dpo_data_multilingual(path: Path, max_samples: int) -> list[dict]:
    """Load a multilingual DPO preference set from 03d_translate_pku_dpo.py:
    lines of {"prompt","chosen","rejected","lang"}."""
    if not path.exists():
        logger.warning(f"Multilingual DPO file not found: {path}")
        return []
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    records = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
               for r in records if r.get("prompt") and r.get("chosen") and r.get("rejected")]
    if max_samples and len(records) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in idx]
    logger.info(f"DPO (multilingual): {len(records)} preference pairs")
    return records


def load_dpo_data_pku(pku_csv: Path, max_samples: int) -> list[dict]:
    """
    Build DPO preference pairs from PKU-SafeRLHF.
    Chosen  = safe response, Rejected = unsafe response.
    Only keeps rows where one response is safe and one is not.
    """
    df = load_pku_csv(pku_csv)
    if df is None:
        return []

    records = []
    for _, row in df.iterrows():
        r0_safe = bool(row.get("is_response_0_safe", False))
        r1_safe = bool(row.get("is_response_1_safe", False))
        prompt  = str(row.get("prompt", "")).strip()
        r0      = str(row.get("response_0", "")).strip()
        r1      = str(row.get("response_1", "")).strip()

        if not prompt or not r0 or not r1:
            continue

        if r0_safe and not r1_safe:
            records.append({"prompt": prompt, "chosen": r0, "rejected": r1})
        elif r1_safe and not r0_safe:
            records.append({"prompt": prompt, "chosen": r1, "rejected": r0})
        # Skip rows where both or neither are safe (no clear preference)

    if max_samples and len(records) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in idx]

    logger.info(f"DPO (PKU): {len(records)} preference pairs")
    return records


# ── HuggingFace dataset helpers ───────────────────────────────────────────────

def records_to_hf_dataset(records: list[dict], method: str):
    """Convert a list of dicts to a HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(records)


def make_sft_formatting_fn():
    """Returns a formatting function for SFTTrainer."""
    def formatting_fn(example):
        text = SFT_TEMPLATE.format(
            prompt=example["prompt"],
            completion=example["completion"],
        )
        return text
    return formatting_fn


# ── LoRA setup ────────────────────────────────────────────────────────────────

def build_lora_model(hf_id: str, token: str | None, lora_r: int, lora_alpha: int,
                     lora_dropout: float, dtype_str: str):
    """Load model + tokenizer, apply QLoRA (4-bit NF4), return (model, tokenizer)."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if dtype_str == "bf16" else torch.float16,
    )

    logger.info(f"Loading {hf_id} with QLoRA 4-bit (dtype={dtype_str})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for SFT loss masking

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_cfg,
        token=token,
        device_map="auto",
    )
    model.config.use_cache = False   # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ── SFT ───────────────────────────────────────────────────────────────────────

def run_sft(
    model_key: str,
    records: list[dict],
    out_dir: Path,
    token: str | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype_str: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    save_steps: int | None = None,
    extra_info: dict | None = None,
):
    from trl import SFTTrainer, SFTConfig

    if not records:
        logger.error("No SFT training data available. Aborting SFT.")
        return

    cfg  = MODELS[model_key]
    model, tokenizer = build_lora_model(
        cfg.hf_id, token, lora_r, lora_alpha, lora_dropout, dtype_str)

    dataset = records_to_hf_dataset(records, "sft")
    fmt_fn  = make_sft_formatting_fn()
    dataset = dataset.map(lambda ex: {"text": fmt_fn(ex)})

    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = epochs * (len(dataset) // batch_size // grad_accum + 1)
    if save_steps is not None:
        _save_strategy = "steps"
        _save_steps    = save_steps
        _save_limit    = None  # keep all checkpoints
    else:
        _save_strategy = "epoch"
        _save_steps    = 500   # ignored when strategy="epoch"
        _save_limit    = 1

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * total_steps),
        logging_steps=10,
        save_strategy=_save_strategy,
        save_steps=_save_steps,
        save_total_limit=_save_limit,
        bf16=(dtype_str == "bf16"),
        fp16=(dtype_str == "fp16"),
        gradient_checkpointing=True,
        report_to=os.environ.get("FT_REPORT_TO", "none"),
        run_name=os.environ.get("WANDB_RUN_NAME"),
        seed=int(os.environ.get("FT_SEED", "42")),
        max_length=max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info(f"Starting SFT: {len(records)} examples, {epochs} epochs")
    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"SFT adapter saved → {out_dir}")

    save_training_info(out_dir, {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(os.environ.get("FT_SEED", "42")),
        "model_key": model_key,
        "base_model_hf_id": MODELS[model_key].hf_id,
        "method": "sft",
        "num_examples": len(records),
        "hyperparams": {
            "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "epochs": epochs, "batch_size": batch_size, "grad_accum": grad_accum,
            "lr": lr, "max_seq_len": max_seq_len, "dtype": dtype_str,
        },
        **(extra_info or {}),
    })


# ── DPO ───────────────────────────────────────────────────────────────────────

def run_dpo(
    model_key: str,
    records: list[dict],
    out_dir: Path,
    token: str | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype_str: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    beta: float,
    base_hf_id: str | None = None,
    save_steps: int | None = None,
    extra_info: dict | None = None,
):
    from trl import DPOTrainer, DPOConfig

    if not records:
        logger.error("No DPO training data available. Aborting DPO.")
        return

    import gc
    import torch

    hf_id = base_hf_id if base_hf_id is not None else MODELS[model_key].hf_id

    # DPOTrainer expects columns: prompt, chosen, rejected
    dataset = records_to_hf_dataset(records, "dpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_steps is not None:
        _save_strategy, _save_steps, _save_limit = "steps", save_steps, None
    else:
        _save_strategy, _save_steps, _save_limit = "epoch", 500, 1

    # DPO holds chosen+rejected activations per example, so long (often non-Latin,
    # multilingual) sequences can OOM at the default batch size. Rather than force
    # the caller to hand-tune, auto-back-off: on CUDA OOM, halve the per-device
    # batch and double grad-accum (effective batch unchanged), rebuild, and retry
    # down to batch size 1.
    bs, ga = batch_size, grad_accum
    oom_retries = 0
    model = None
    while True:
        model, tokenizer = build_lora_model(
            hf_id, token, lora_r, lora_alpha, lora_dropout, dtype_str)
        total_steps = epochs * (len(dataset) // max(bs, 1) // max(ga, 1) + 1)
        training_args = DPOConfig(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=int(0.05 * total_steps),
            logging_steps=10,
            save_strategy=_save_strategy,
            save_steps=_save_steps,
            save_total_limit=_save_limit,
            bf16=(dtype_str == "bf16"),
            fp16=(dtype_str == "fp16"),
            gradient_checkpointing=True,
            report_to=os.environ.get("FT_REPORT_TO", "none"),
            run_name=os.environ.get("WANDB_RUN_NAME"),
            seed=int(os.environ.get("FT_SEED", "42")),
            beta=beta,
            max_length=max_seq_len,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,   # None → use implicit reference (PEFT frozen base)
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        logger.info(f"Starting DPO: {len(records)} pairs, {epochs} epochs, "
                    f"beta={beta}, batch={bs}, grad_accum={ga}")
        try:
            trainer.train()
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if bs <= 1:
                logger.error("DPO OOM even at batch size 1 — aborting. "
                             "Try a shorter --max-seq-len or a smaller model.")
                raise
            oom_retries += 1
            new_bs, new_ga = max(1, bs // 2), ga * 2
            logger.warning(f"DPO CUDA OOM at batch {bs} → retrying at batch {new_bs} "
                           f"(grad_accum {new_ga}); effective batch unchanged.")
            del trainer, model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            bs, ga = new_bs, new_ga

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"DPO adapter saved → {out_dir} (final batch={bs}, grad_accum={ga}, "
                f"oom_retries={oom_retries})")

    save_training_info(out_dir, {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(os.environ.get("FT_SEED", "42")),
        "model_key": model_key,
        "base_model_hf_id": hf_id,
        "method": "dpo",
        "num_examples": len(records),
        "hyperparams": {
            "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "epochs": epochs, "batch_size": bs, "grad_accum": ga,
            "requested_batch_size": batch_size, "oom_retries": oom_retries,
            "lr": lr, "max_seq_len": max_seq_len, "dtype": dtype_str, "beta": beta,
        },
        **(extra_info or {}),
    })


# ── RLHF: reward model ────────────────────────────────────────────────────────

def run_reward_model(
    model_key: str,
    records: list[dict],
    reward_dir: Path,
    token: str | None,
    lora_r: int,
    lora_alpha: int,
    dtype_str: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    reward_lr: float,
    max_seq_len: int,
    base_hf_id: str | None = None,
):
    """Train a reward model (sequence classifier) on preference pairs."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardTrainer, RewardConfig
    from datasets import Dataset

    if not records:
        logger.error("No reward model training data available. Aborting.")
        return

    cfg = MODELS[model_key]
    dtype = torch.bfloat16 if dtype_str == "bf16" else (
            torch.float16 if dtype_str == "fp16" else torch.float32)

    hf_id = base_hf_id if base_hf_id is not None else cfg.hf_id
    logger.info(f"Loading reward model base: {hf_id} (dtype={dtype_str})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=1, torch_dtype=dtype, token=token, device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tokenize_pair(examples):
        tok_chosen = tokenizer(
            [SFT_TEMPLATE.format(prompt=p, completion=c)
             for p, c in zip(examples["prompt"], examples["chosen"])],
            truncation=True, max_length=max_seq_len, padding="max_length",
        )
        tok_rejected = tokenizer(
            [SFT_TEMPLATE.format(prompt=p, completion=r)
             for p, r in zip(examples["prompt"], examples["rejected"])],
            truncation=True, max_length=max_seq_len, padding="max_length",
        )
        return {
            "input_ids_chosen":        tok_chosen["input_ids"],
            "attention_mask_chosen":   tok_chosen["attention_mask"],
            "input_ids_rejected":      tok_rejected["input_ids"],
            "attention_mask_rejected": tok_rejected["attention_mask"],
        }

    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        tokenize_pair, batched=True,
        remove_columns=["prompt", "chosen", "rejected"],
    )

    reward_dir.mkdir(parents=True, exist_ok=True)

    training_args = RewardConfig(
        output_dir=str(reward_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=reward_lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * epochs * (len(dataset) // batch_size // grad_accum + 1)),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=(dtype_str == "bf16"),
        fp16=(dtype_str == "fp16"),
        gradient_checkpointing=True,
        report_to=os.environ.get("FT_REPORT_TO", "none"),
        run_name=os.environ.get("WANDB_RUN_NAME"),
        seed=int(os.environ.get("FT_SEED", "42")),
        max_length=max_seq_len,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(f"Training reward model: {len(records)} pairs, {epochs} epochs")
    trainer.train()

    model.save_pretrained(str(reward_dir))
    tokenizer.save_pretrained(str(reward_dir))
    logger.info(f"Reward model saved → {reward_dir}")


# ── RLHF: PPO policy fine-tuning ──────────────────────────────────────────────

def run_ppo(
    model_key: str,
    records: list[dict],
    reward_dir: Path,
    out_dir: Path,
    token: str | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype_str: str,
    batch_size: int,
    lr: float,
    max_seq_len: int,
    ppo_steps: int,
    ppo_mini_batch_size: int,
    base_hf_id: str | None = None,
):
    """PPO fine-tuning using the trained reward model."""
    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

    if not records:
        logger.error("No PPO prompt data available. Aborting PPO.")
        return

    cfg = MODELS[model_key]
    dtype = torch.bfloat16 if dtype_str == "bf16" else (
            torch.float16 if dtype_str == "fp16" else torch.float32)

    hf_id = base_hf_id if base_hf_id is not None else cfg.hf_id

    # Policy tokenizer (left-padding for generation)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Policy model wrapped with value head
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        hf_id,
        peft_config=lora_cfg,
        torch_dtype=dtype,
        token=token,
        device_map="auto",
    )

    # Reward model (trained in previous step)
    reward_tokenizer = AutoTokenizer.from_pretrained(str(reward_dir))
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        str(reward_dir), torch_dtype=dtype, device_map="auto",
    )
    reward_model.eval()

    ppo_config = PPOConfig(
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=ppo_mini_batch_size,
        gradient_accumulation_steps=max(1, batch_size // ppo_mini_batch_size),
        ppo_epochs=4,
        max_grad_norm=0.5,
        log_with=None,
    )
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model=None, tokenizer=tokenizer)

    prompts = [r["prompt"] for r in records]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting PPO: {ppo_steps} steps, batch_size={batch_size}")
    for step in range(ppo_steps):
        batch_prompts = prompts[
            (step * batch_size) % len(prompts):
            (step * batch_size) % len(prompts) + batch_size
        ]
        if not batch_prompts:
            break

        query_tensors = [
            tokenizer.encode(
                f"### Human: {p}\n### Assistant:",
                return_tensors="pt",
            ).squeeze(0)
            for p in batch_prompts
        ]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

        responses = [
            tokenizer.decode(r[q.shape[0]:], skip_special_tokens=True)
            for q, r in zip(query_tensors, response_tensors)
        ]

        rewards = []
        for prompt, response in zip(batch_prompts, responses):
            text = SFT_TEMPLATE.format(prompt=prompt, completion=response)
            inputs = reward_tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_seq_len,
            )
            inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                score = reward_model(**inputs).logits[0, 0].item()
            rewards.append(torch.tensor(score, dtype=torch.float32))

        ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % 10 == 0:
            mean_r = torch.stack(rewards).mean().item()
            logger.info(f"PPO step {step + 1}/{ppo_steps}: mean_reward={mean_r:.4f}")

    # Save only the LoRA adapter (not the value head)
    model.pretrained_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"RLHF (PPO) adapter saved → {out_dir}")

    save_training_info(out_dir, {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(os.environ.get("FT_SEED", "42")),
        "model_key": model_key,
        "base_model_hf_id": hf_id,
        "method": "rlhf_ppo",
        "num_examples": len(records),
        "hyperparams": {
            "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "lr": lr, "max_seq_len": max_seq_len, "dtype": dtype_str,
            "ppo_steps": ppo_steps, "ppo_mini_batch_size": ppo_mini_batch_size,
            "batch_size": batch_size,
        },
    })


# ── RLHF: orchestrator ────────────────────────────────────────────────────────

def run_rlhf(
    model_key: str,
    records: list[dict],
    out_root: Path,
    token: str | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype_str: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    reward_lr: float,
    ppo_steps: int,
    ppo_mini_batch_size: int,
    sft_dir: str | None = None,
):
    """Train reward model then run PPO — full RLHF pipeline."""
    reward_dir = out_root / f"{model_key}_rlhf_reward"
    policy_dir = out_root / f"{model_key}_rlhf"

    run_reward_model(
        model_key=model_key,
        records=records,
        reward_dir=reward_dir,
        token=token,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        dtype_str=dtype_str,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        reward_lr=reward_lr,
        max_seq_len=max_seq_len,
        base_hf_id=sft_dir,
    )

    run_ppo(
        model_key=model_key,
        records=records,
        reward_dir=reward_dir,
        out_dir=policy_dir,
        token=token,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        dtype_str=dtype_str,
        batch_size=batch_size,
        lr=lr,
        max_seq_len=max_seq_len,
        ppo_steps=ppo_steps,
        ppo_mini_batch_size=ppo_mini_batch_size,
        base_hf_id=sft_dir,
    )


# ── post-run instructions ─────────────────────────────────────────────────────

def print_next_steps(model_key: str, method: str, out_root: Path):
    print("\n" + "=" * 60)
    print("Fine-tuning complete. To include this model in the probe sweep:")
    print()
    print("1. Add a new ModelConfig entry in config/models.py, e.g.:")
    print()

    if method in ("sft", "both"):
        sft_path = out_root / f"{model_key}_sft"
        print(f'   "{model_key}_sft": ModelConfig(')
        print(f'       name="TinyAya-{model_key.capitalize()}-SFT",')
        print(f'       hf_id="{sft_path}",   # local adapter path')
        print(f'       stage="finetuned",')
        print(f'       probe_layers=[18, 24, 30],')
        print(f'   ),')
        print()

    if method in ("dpo", "both"):
        dpo_path = out_root / f"{model_key}_dpo"
        print(f'   "{model_key}_dpo": ModelConfig(')
        print(f'       name="TinyAya-{model_key.capitalize()}-DPO",')
        print(f'       hf_id="{dpo_path}",   # local adapter path')
        print(f'       stage="finetuned",')
        print(f'       probe_layers=[18, 24, 30],')
        print(f'   ),')
        print()

    if method == "rlhf":
        rlhf_path = out_root / f"{model_key}_rlhf"
        print(f'   "{model_key}_rlhf": ModelConfig(')
        print(f'       name="TinyAya-{model_key.capitalize()}-RLHF",')
        print(f'       hf_id="{rlhf_path}",   # local adapter path')
        print(f'       stage="finetuned",')
        print(f'       probe_layers=[18, 24, 30],')
        print(f'   ),')
        print()

    print("2. Re-extract activations and train probes:")
    suffixes = (
        ["_sft"] if method == "sft" else
        ["_dpo"] if method == "dpo" else
        ["_rlhf"] if method == "rlhf" else
        ["_sft", "_dpo"]
    )
    for suffix in suffixes:
        key = f"{model_key}{suffix}"
        print(f"   python scripts/05_extract_activations.py --model {key} --lang en")
        print(f"   python scripts/06_train_probes.py         --model {key} --lang en --alignment")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    # Reduce CUDA fragmentation OOMs (esp. DPO with long multilingual sequences).
    # Must be set before the first CUDA allocation, so do it at entry.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(
        description="Safety fine-tuning (SFT / DPO) with LoRA")

    # What to train
    parser.add_argument("--model",   choices=["base"], default="base",
                        help="Which model variant to fine-tune (default: base)")
    parser.add_argument("--method",  choices=["sft", "dpo", "rlhf", "both"], default="sft",
                        help="Fine-tuning method: sft, dpo, rlhf, or both (sft+dpo) (default: sft)")

    # Data
    parser.add_argument("--data-source", choices=["pku", "local", "multilingual"], default="pku",
                        help="Data source: pku=PKU-SafeRLHF CSV (English), "
                             "local=data/prompts/ files, "
                             "multilingual=balanced 6-lang PKU set from 03c_translate_pku.py")
    parser.add_argument("--multilingual-file",
                        default="data/processed/pku_translated/pku_sft_multilingual.jsonl",
                        help="Path to the combined multilingual SFT jsonl (for --data-source multilingual)")
    parser.add_argument("--multilingual-dpo-file",
                        default="data/processed/pku_translated/pku_dpo_multilingual_n3000.jsonl",
                        help="Path to the multilingual DPO preference jsonl (for --data-source multilingual --method dpo)")
    parser.add_argument("--lang",        choices=LANG_CODES, default="en",
                        help="Language for --data-source local (default: en)")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max training examples per method (default: 500)")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None,
                        help="Run multiple sample sizes in one go, each saved to its own subdir "
                             "(e.g. --sample-sizes 500 2000 5000 10000 20000). "
                             "Overrides --max-samples.")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Save a checkpoint every N optimizer steps (default: save per epoch only).")
    parser.add_argument("--save-every-n-samples", type=int, default=None,
                        help="Save a checkpoint every N training samples. Converted to steps "
                             "using batch-size × grad-accum. Overrides --save-steps.")
    parser.add_argument("--pku-csv",
                        default="data/raw/pku_saferlhf/pku_saferlhf_sample.csv",
                        help="Path to PKU-SafeRLHF CSV")
    parser.add_argument("--prompts-dir", default="data/prompts",
                        help="Root of local prompt files (for --data-source local)")

    # LoRA
    parser.add_argument("--lora-r",       type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--max-seq-len",type=int,   default=512)
    parser.add_argument("--beta",       type=float, default=0.1,
                        help="DPO beta (KL penalty coefficient, default: 0.1)")
    parser.add_argument("--skip-sft",   action="store_true",
                        help="For --method dpo: skip SFT and run DPO from the existing "
                             "{out-dir}/{model}_sft checkpoint (e.g. to resume after an OOM).")
    parser.add_argument("--reward-lr",  type=float, default=1e-4,
                        help="Learning rate for RLHF reward model (default: 1e-4)")
    parser.add_argument("--ppo-steps",  type=int,   default=100,
                        help="Number of PPO update steps (default: 100)")
    parser.add_argument("--ppo-mini-batch-size", type=int, default=2,
                        help="Mini-batch size for each PPO gradient update (default: 2)")
    parser.add_argument("--dtype",      choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Training dtype (default: bf16; use fp16 for older GPUs)")

    # I/O
    parser.add_argument("--out-dir", default="models/finetuned")
    parser.add_argument("--token",   default=None, help="HuggingFace token")
    parser.add_argument("--use-hf-mirror", action="store_true",
                        help="Use Hugging Face mirror as source for models")

    args = parser.parse_args()

    try:
        import trl   # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        print("ERROR: trl and peft are required. Install with:")
        print("  pip install trl>=0.12.0 peft>=0.10.0")
        sys.exit(1)

    # Set HF endpoint to mirror if specified
    if args.use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using Hugging Face mirror: https://hf-mirror.com")

    token = args.token or os.environ.get("HF_TOKEN")
    out_root    = ROOT / args.out_dir
    pku_csv     = ROOT / args.pku_csv
    prompts_dir = ROOT / args.prompts_dir

    logger.info(f"Model: {args.model}  Method: {args.method}  "
                f"Data: {args.data_source}  Dtype: {args.dtype}")

    # ── load data ─────────────────────────────────────────────────────────────

    sft_records, dpo_records, rlhf_records = [], [], []

    # Determine which sample sizes to run
    sample_sizes = args.sample_sizes if args.sample_sizes else [args.max_samples]

    # Load the full pool once (max of all requested sizes)
    max_needed = max(sample_sizes)
    if args.data_source == "pku":
        all_sft_records  = load_sft_data_pku(pku_csv, max_needed)
        all_dpo_records  = load_dpo_data_pku(pku_csv, max_needed) if args.method in ("dpo", "both", "rlhf") else []
        all_rlhf_records = all_dpo_records
    elif args.data_source == "multilingual":
        all_sft_records  = load_sft_data_multilingual(ROOT / args.multilingual_file, max_needed)
        all_dpo_records  = (load_dpo_data_multilingual(ROOT / args.multilingual_dpo_file, max_needed)
                            if args.method in ("dpo", "both") else [])
        all_rlhf_records = all_dpo_records
        if args.method in ("dpo", "both") and not all_dpo_records:
            logger.error("--method dpo needs a multilingual DPO file; run 03d_translate_pku_dpo.py.")
            sys.exit(1)
        if args.method == "rlhf":
            logger.warning("--data-source multilingual does not support rlhf.")
            sys.exit(1)
    else:
        all_sft_records  = load_sft_data_local(prompts_dir, args.lang, max_needed)
        all_dpo_records  = []
        all_rlhf_records = []
        if args.method in ("dpo", "both", "rlhf"):
            logger.warning("DPO/RLHF requires preference pairs; --data-source local only supports SFT.")
            if args.method != "sft":
                sys.exit(1)

    # ── train ─────────────────────────────────────────────────────────────────

    extra_info = {
        "data_source": args.data_source,
        "lang": args.lang if args.data_source == "local" else None,
    }

    # Resolve save_steps: --save-every-n-samples overrides --save-steps
    effective_save_steps = args.save_steps
    if args.save_every_n_samples is not None:
        effective_batch = args.batch_size * args.grad_accum
        effective_save_steps = max(1, args.save_every_n_samples // effective_batch)
        logger.info(f"--save-every-n-samples {args.save_every_n_samples} → "
                    f"save_steps={effective_save_steps} "
                    f"(effective batch={effective_batch})")

    common_kwargs = dict(
        model_key=args.model,
        token=token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype_str=args.dtype,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        save_steps=effective_save_steps,
        extra_info=extra_info,
    )

    if args.skip_sft:
        logger.info("--skip-sft: reusing existing SFT checkpoint(s); skipping SFT training.")
    else:
        for n in sample_sizes:
            sft_records  = all_sft_records[:n]

            # Use n-specific subdir when running multiple sizes
            suffix = f"_n{n}" if len(sample_sizes) > 1 else ""
            run_dir = out_root / f"{args.model}_sft{suffix}"

            logger.info(f"=== Sample size: {n} → {run_dir} ===")
            run_sft(records=sft_records, out_dir=run_dir, **common_kwargs)

    if args.method in ("dpo", "both") and all_dpo_records:
        for n in sample_sizes:
            suffix = f"_n{n}" if len(sample_sizes) > 1 else ""
            sft_dir = str(out_root / f"{args.model}_sft{suffix}")
            run_dpo(records=all_dpo_records[:n],
                    out_dir=out_root / f"{args.model}_dpo{suffix}",
                    beta=args.beta,
                    base_hf_id=sft_dir,
                    **common_kwargs)

    if args.method == "rlhf":
        for n in sample_sizes:
            suffix = f"_n{n}" if len(sample_sizes) > 1 else ""
            sft_dir = str(out_root / f"{args.model}_sft{suffix}")
            run_rlhf(
                records=all_rlhf_records[:n],
                out_root=out_root,
                reward_lr=args.reward_lr,
                ppo_steps=args.ppo_steps,
                ppo_mini_batch_size=args.ppo_mini_batch_size,
                sft_dir=sft_dir,
                **common_kwargs,
            )

    print_next_steps(args.model, args.method, out_root)


if __name__ == "__main__":
    main()
