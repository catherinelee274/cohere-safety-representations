"""
Step 10 — Safety fine-tuning via SFT and/or DPO.

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
    models/finetuned/{model_key}_sft/   LoRA adapter + tokenizer
    models/finetuned/{model_key}_dpo/   LoRA adapter + tokenizer

Requirements (not in base requirements.txt):
    pip install trl>=0.8.0 peft>=0.10.0

Usage:
    # SFT on base model, PKU data
    python scripts/10_finetune.py --model base --method sft

    # DPO on global model, PKU data
    python scripts/10_finetune.py --model global --method dpo

    # Both methods sequentially on base model
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


def make_sft_formatting_fn(tokenizer):
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
    """Load model + tokenizer, apply LoRA, return (model, tokenizer)."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if dtype_str == "bf16" else (
            torch.float16 if dtype_str == "fp16" else torch.float32)

    logger.info(f"Loading {hf_id} (dtype={dtype_str})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for SFT loss masking

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        token=token,
        device_map="auto",
    )
    model.config.use_cache = False   # required for gradient checkpointing

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
):
    from trl import SFTTrainer, SFTConfig

    if not records:
        logger.error("No SFT training data available. Aborting SFT.")
        return

    cfg  = MODELS[model_key]
    model, tokenizer = build_lora_model(
        cfg.hf_id, token, lora_r, lora_alpha, lora_dropout, dtype_str)

    dataset = records_to_hf_dataset(records, "sft")
    fmt_fn  = make_sft_formatting_fn(tokenizer)
    dataset = dataset.map(lambda ex: {"text": fmt_fn(ex)})

    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.05 * epochs * (len(dataset) // batch_size // grad_accum + 1)),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=(dtype_str == "bf16"),
        fp16=(dtype_str == "fp16"),
        gradient_checkpointing=True,
        report_to="none",
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
):
    from trl import DPOTrainer, DPOConfig

    if not records:
        logger.error("No DPO training data available. Aborting DPO.")
        return

    cfg  = MODELS[model_key]
    model, tokenizer = build_lora_model(
        cfg.hf_id, token, lora_r, lora_alpha, lora_dropout, dtype_str)

    # DPOTrainer expects columns: prompt, chosen, rejected
    dataset = records_to_hf_dataset(records, "dpo")

    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=(dtype_str == "bf16"),
        fp16=(dtype_str == "fp16"),
        gradient_checkpointing=True,
        report_to="none",
        beta=beta,
        max_length=max_seq_len,
        max_prompt_length=max_seq_len // 2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # None → use implicit reference (PEFT frozen base)
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info(f"Starting DPO: {len(records)} pairs, {epochs} epochs, beta={beta}")
    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"DPO adapter saved → {out_dir}")


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

    print("2. Re-extract activations and train probes:")
    for suffix in (["_sft"] if method == "sft" else
                   ["_dpo"] if method == "dpo" else ["_sft", "_dpo"]):
        key = f"{model_key}{suffix}"
        print(f"   python scripts/05_extract_activations.py --model {key} --lang en")
        print(f"   python scripts/06_train_probes.py         --model {key} --lang en --alignment")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Safety fine-tuning (SFT / DPO) with LoRA")

    # What to train
    parser.add_argument("--model",   choices=list(MODELS.keys()), default="base",
                        help="Which model variant to fine-tune (default: base)")
    parser.add_argument("--method",  choices=["sft", "dpo", "both"], default="sft",
                        help="Fine-tuning method (default: sft)")

    # Data
    parser.add_argument("--data-source", choices=["pku", "local"], default="pku",
                        help="Data source: pku=PKU-SafeRLHF CSV, local=data/prompts/ files")
    parser.add_argument("--lang",        choices=LANG_CODES, default="en",
                        help="Language for --data-source local (default: en)")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max training examples per method (default: 2000)")
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
        print("  pip install trl>=0.8.0 peft>=0.10.0")
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

    sft_records, dpo_records = [], []

    if args.method in ("sft", "both"):
        if args.data_source == "pku":
            sft_records = load_sft_data_pku(pku_csv, args.max_samples)
        else:
            sft_records = load_sft_data_local(prompts_dir, args.lang, args.max_samples)

    if args.method in ("dpo", "both"):
        if args.data_source == "pku":
            dpo_records = load_dpo_data_pku(pku_csv, args.max_samples)
        else:
            logger.warning("DPO requires preference pairs; --data-source local only supports SFT. "
                           "Skipping DPO.")
            if args.method == "dpo":
                sys.exit(1)

    # ── train ─────────────────────────────────────────────────────────────────

    common_kwargs = dict(
        model_key=args.model,
        out_dir=out_root / f"{args.model}_sft",   # overridden per method below
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
    )

    if args.method in ("sft", "both"):
        run_sft(records=sft_records,
                out_dir=out_root / f"{args.model}_sft",
                **{k: v for k, v in common_kwargs.items()
                   if k not in ("out_dir",)})

    if args.method in ("dpo", "both") and dpo_records:
        run_dpo(records=dpo_records,
                out_dir=out_root / f"{args.model}_dpo",
                beta=args.beta,
                **{k: v for k, v in common_kwargs.items()
                   if k not in ("out_dir",)})

    print_next_steps(args.model, args.method, out_root)


if __name__ == "__main__":
    main()
