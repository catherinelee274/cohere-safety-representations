"""
Step 06b — Regenerate cross-lingual alignment matrices from existing probe
directions, without re-training probes.

The alignment_{model}_layer{n}.csv files were generated from an earlier language
set and are missing Hindi (hi).  All per-language probe directions already exist
on disk, so we just recompute the pairwise cosine matrices over the current
config.languages.LANG_CODES (en, fr, ar, hi, sw, yo).

Reads:
    results/probes/{model}/{lang}/layer_{n}_{probe_type}.pkl
Writes:
    results/probes/alignment_{model}_layer{n}.csv

Usage:
    python scripts/06b_regen_alignment.py                       # default model set
    python scripts/06b_regen_alignment.py --models base base_sft base_dpo base_sft_ml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import MODELS

# Reuse the canonical implementation from 06_train_probes.py
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_probes_06", ROOT / "scripts" / "06_train_probes.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
compute_alignment_matrix = _m.compute_alignment_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["base", "base_sft", "base_dpo", "base_sft_ml"])
    parser.add_argument("--probe-dir", default="results/probes")
    args = parser.parse_args()

    probe_dir = ROOT / args.probe_dir
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"  SKIP {model_key}: not in MODELS config.")
            continue
        cfg = MODELS[model_key]
        for layer in cfg.probe_layers:
            mat = compute_alignment_matrix(probe_dir, model_key, layer)
            if mat.empty:
                continue
            out = probe_dir / f"alignment_{model_key}_layer{layer}.csv"
            mat.to_csv(out)
            print(f"  {model_key} L{layer}: langs={list(mat.columns)} -> {out.name}")
    print("Done.")


if __name__ == "__main__":
    main()
