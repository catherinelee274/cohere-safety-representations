"""
Step 5 — Extract residual-stream activations for all model × language pairs.

Reads harmful/harmless prompt sets from data/prompts/{lang}/
and saves activation arrays to data/processed/activations/{model}/{lang}/

Usage:
    # Single model + language (recommended for first sanity check)
    python scripts/05_extract_activations.py \\
        --model global --lang en --layers 18 24 30

    # Full sweep
    python scripts/05_extract_activations.py --all

Output:
    data/processed/activations/{model_key}/{lang}/
        layer_{n}.npy          shape (N_harmful + N_harmless, hidden_size)
        labels.npy             0 = harmless, 1 = harmful  (same N rows)
        prompts.jsonl          source prompts for debugging
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES
from config.models import MODEL_ORDER, MODELS
from src.model_wrapper import ModelWrapper

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_prompts(prompts_dir: Path, lang: str) -> tuple[list[dict], list[dict]]:
    def read(p):
        if not p.exists():
            return []
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]
    harmful  = read(prompts_dir / lang / "harmful.jsonl")
    harmless = read(prompts_dir / lang / "harmless.jsonl")
    return harmful, harmless


def run_extraction(
    model_key: str,
    lang: str,
    prompts_dir: Path,
    out_root: Path,
    token: str | None,
    batch_size: int,
    aggregation: str,
):
    cfg     = MODELS[model_key]
    out_dir = out_root / model_key / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if all layer files exist
    expected = [out_dir / f"layer_{l}.npy" for l in cfg.probe_layers]
    if all(p.exists() for p in expected):
        logger.info(f"Skipping {model_key}/{lang} — already extracted.")
        return

    harmful, harmless = load_prompts(prompts_dir, lang)
    if not harmful and not harmless:
        logger.warning(f"No prompts found for lang={lang}. Skipping.")
        return

    all_records = harmless + harmful
    texts  = [r["text"] for r in all_records]
    labels = np.array([0] * len(harmless) + [1] * len(harmful), dtype=np.int8)

    logger.info(f"Extracting {model_key}/{lang}: "
                f"{len(harmless)} harmless + {len(harmful)} harmful")

    wrapper = ModelWrapper(
        cfg.hf_id,
        layers=cfg.probe_layers,
        token=token,
    )

    acts = wrapper.get_activations(texts, batch_size=batch_size, aggregation=aggregation)

    for layer, arr in acts.items():
        np.save(out_dir / f"layer_{layer}.npy", arr)
        logger.info(f"  Saved layer {layer}: {arr.shape}")

    np.save(out_dir / "labels.npy", labels)

    with open(out_dir / "prompts.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"  Done → {out_dir}")

    # Free GPU memory between models
    del wrapper
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=list(MODELS.keys()), default=None)
    parser.add_argument("--lang",        choices=LANG_CODES,          default=None)
    parser.add_argument("--layers",      type=int, nargs="+",         default=None,
                        help="Override model's default probe_layers")
    parser.add_argument("--all",         action="store_true",
                        help="Run all model × language combinations")
    parser.add_argument("--prompts-dir", default="data/prompts")
    parser.add_argument("--out-dir",     default="data/processed/activations")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--aggregation", default="last",
                        choices=["last", "mean", "first"])
    parser.add_argument("--token",       default=None)
    args = parser.parse_args()

    import os
    token = args.token or os.environ.get("HF_TOKEN")

    prompts_dir = ROOT / args.prompts_dir
    out_root    = ROOT / args.out_dir

    if args.layers:
        for cfg in MODELS.values():
            cfg.probe_layers = args.layers

    pairs: list[tuple[str, str]] = []
    if args.all:
        pairs = [(m, l) for m in MODEL_ORDER for l in LANG_CODES]
    elif args.model and args.lang:
        pairs = [(args.model, args.lang)]
    elif args.model:
        pairs = [(args.model, l) for l in LANG_CODES]
    elif args.lang:
        pairs = [(m, args.lang) for m in MODEL_ORDER]
    else:
        # Default: sanity check — English + global only
        pairs = [("global", "en")]
        logger.info("No --model/--lang specified. Running sanity check: global × en")

    for model_key, lang in pairs:
        run_extraction(model_key, lang, prompts_dir, out_root, token,
                       args.batch_size, args.aggregation)

    print("\nDone. Next: scripts/06_train_probes.py")


if __name__ == "__main__":
    main()
