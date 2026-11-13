"""Evaluate wrongful refusals on harmless prompts.

Outputs:
    results/behavioral/harmless_responses_{model}_{lang}.jsonl
    results/behavioral/overrefusal_summary.csv

Usage:
    python scripts/07b_overrefusal_eval.py --model base_dpo_ml --lang sw
    python scripts/07b_overrefusal_eval.py \
        --models base_sft_ml_big base_sft_en3000 base_dpo_ml base_dpo_en3000
"""

import argparse
import gc
import importlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES
from config.models import MODELS
from src.model_wrapper import ModelWrapper
from src.run_metadata import model_meta, save_run_info

is_refusal = importlib.import_module("scripts.07_behavioral_baseline").is_refusal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_pair(
    model_key: str,
    lang: str,
    prompts_dir: Path,
    out_dir: Path,
    token: str | None,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"harmless_responses_{model_key}_{lang}.jsonl"
    if out_path.exists():
        logger.info("Skipping %s/%s — already generated.", model_key, lang)
        return

    prompt_path = prompts_dir / lang / "harmless.jsonl"
    if not prompt_path.exists():
        logger.warning("No harmless prompts for lang=%s", lang)
        return

    with prompt_path.open() as handle:
        prompts = [json.loads(line) for line in handle if line.strip()]

    wrapper = ModelWrapper(MODELS[model_key].hf_id, layers=[], token=token)
    logger.info("Generating %s/%s (%d harmless prompts)...", model_key, lang, len(prompts))
    responses = wrapper.generate(
        [prompt["text"] for prompt in prompts],
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    records = []
    for prompt, response in zip(prompts, responses):
        records.append({
            **prompt,
            "response": response,
            "heuristic_refuse": is_refusal(response),
        })

    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    wrongful_refusals = sum(record["heuristic_refuse"] for record in records)
    rate = wrongful_refusals / len(records) if records else 0.0
    logger.info(
        "  %s/%s: overrefusal_rate=%.2f%% (%d/%d)",
        model_key,
        lang,
        100 * rate,
        wrongful_refusals,
        len(records),
    )
    save_run_info(out_dir, {
        **model_meta(model_key),
        "script": "07b_overrefusal_eval",
        "lang": lang,
        "prompt_label": "harmless",
        "n_prompts": len(records),
        "n_wrongful_refusals": wrongful_refusals,
        "overrefusal_rate": round(rate, 4),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
    })

    del wrapper
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in out_dir.glob("harmless_responses_*.jsonl"):
        stem = path.stem.removeprefix("harmless_responses_")
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model_key, lang = parts
        with path.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        wrongful_refusals = sum(record["heuristic_refuse"] for record in records)
        rows.append({
            "model": model_key,
            "model_hf_id": MODELS[model_key].hf_id if model_key in MODELS else None,
            "lang": lang,
            "n": len(records),
            "n_wrongful_refusals": wrongful_refusals,
            "overrefusal_rate": round(wrongful_refusals / max(len(records), 1), 4),
            "run_at": datetime.now(timezone.utc).isoformat(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model", "lang"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS))
    parser.add_argument("--lang", choices=LANG_CODES)
    parser.add_argument("--langs", nargs="+", choices=LANG_CODES)
    parser.add_argument("--prompts-dir", default="data/prompts")
    parser.add_argument("--out-dir", default="results/behavioral")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--token")
    args = parser.parse_args()

    if args.model and args.models:
        parser.error("Use either --model or --models, not both")

    models = args.models or ([args.model] if args.model else [
        "base_sft_ml_big", "base_sft_en3000", "base_dpo_ml", "base_dpo_en3000"
    ])
    langs = args.langs or ([args.lang] if args.lang else ["en", "fr", "ar", "hi", "sw", "yo"])

    for model_key in models:
        for lang in langs:
            evaluate_pair(
                model_key,
                lang,
                ROOT / args.prompts_dir,
                ROOT / args.out_dir,
                args.token or os.environ.get("HF_TOKEN"),
                args.batch_size,
                args.max_new_tokens,
            )

    summary = summarize(ROOT / args.out_dir)
    if not summary.empty:
        summary_path = ROOT / args.out_dir / "overrefusal_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nOver-refusal summary:\n{summary.to_string(index=False)}")
        print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
