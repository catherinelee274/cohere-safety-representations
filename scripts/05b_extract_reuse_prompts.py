"""
Step 05b — Extract activations for a model reusing another model's saved prompt
set, so probe results are directly comparable on identical prompts.

The legacy activation sets (e.g. base_sft) were built from prompt files that have
since been regenerated, so re-running 05_extract_activations on the current files
would use a different (and unbalanced) prompt set.  To compare base_sft_ml against
base_sft fairly, we re-extract on the EXACT prompts saved in the reference model's
``prompts.jsonl`` (with its ``labels.npy``).

Reads:
    data/processed/activations/{ref_model}/{lang}/prompts.jsonl
    data/processed/activations/{ref_model}/{lang}/labels.npy
Writes:
    data/processed/activations/{model}/{lang}/layer_{n}.npy
    data/processed/activations/{model}/{lang}/labels.npy   (copied)
    data/processed/activations/{model}/{lang}/prompts.jsonl (copied)

Usage:
    python scripts/05b_extract_reuse_prompts.py --model base_sft_ml \
        --ref-model base_sft --langs en fr ar sw yo
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES
from config.models import MODELS
from src.model_wrapper import ModelWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--ref-model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--langs", nargs="+", default=LANG_CODES)
    parser.add_argument("--acts-dir", default="data/processed/activations")
    parser.add_argument("--aggregation", default="last")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    token   = args.token or os.environ.get("HF_TOKEN")
    acts    = ROOT / args.acts_dir
    cfg     = MODELS[args.model]

    wrapper = ModelWrapper(cfg.hf_id, layers=cfg.probe_layers, token=token)

    for lang in args.langs:
        ref_dir = acts / args.ref_model / lang
        out_dir = acts / args.model / lang
        prompts_path = ref_dir / "prompts.jsonl"
        labels_path  = ref_dir / "labels.npy"
        if not prompts_path.exists() or not labels_path.exists():
            print(f"  {lang}: SKIP — reference prompts/labels missing at {ref_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        expected = [out_dir / f"layer_{l}.npy" for l in cfg.probe_layers]
        if all(p.exists() for p in expected):
            print(f"  {lang}: SKIP — already extracted.")
            continue

        records = [json.loads(l) for l in open(prompts_path) if l.strip()]
        texts = [r["text"] for r in records]
        print(f"  {lang}: extracting {len(texts)} prompts (reuse {args.ref_model}) ...", flush=True)

        layer_acts = wrapper.get_activations(texts, batch_size=args.batch_size,
                                             aggregation=args.aggregation)
        for layer, arr in layer_acts.items():
            np.save(out_dir / f"layer_{layer}.npy", arr)
        shutil.copy(labels_path, out_dir / "labels.npy")
        shutil.copy(prompts_path, out_dir / "prompts.jsonl")
        print(f"    {lang}: saved layers {list(layer_acts.keys())} -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
