"""Run budget-matched multilingual and English-control FT for extra seeds.

Seed 42 already exists. This runner creates the two additional replications
(43 and 44 by default), each with both SFT and DPO checkpoints.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "ml": {
        "sft": "data/processed/pku_translated/pku_sft_multilingual_n3000.jsonl",
        "dpo": "data/processed/pku_translated/pku_dpo_multilingual_n3000.jsonl",
    },
    "en3000": {
        "sft": "data/processed/pku_translated/pku_sft_en_control_n3000.jsonl",
        "dpo": "data/processed/pku_translated/pku_dpo_en_control_n3000.jsonl",
    },
}


def command(seed: int, condition: str) -> list[str]:
    files = CONDITIONS[condition]
    return [
        sys.executable,
        "scripts/10_finetune.py",
        "--model", "base",
        "--method", "dpo",
        "--data-source", "multilingual",
        "--multilingual-file", files["sft"],
        "--multilingual-dpo-file", files["dpo"],
        "--max-samples", "3000",
        "--epochs", "3",
        "--batch-size", "4",
        "--grad-accum", "4",
        "--lr", "0.0002",
        "--max-seq-len", "512",
        "--out-dir", f"models/multiseed/seed{seed}/{condition}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(Path.home() / "hf_cache" / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(Path.home() / "hf_cache" / "datasets"))

    for seed in args.seeds:
        for condition in args.conditions:
            output = ROOT / f"models/multiseed/seed{seed}/{condition}/base_dpo"
            if (output / "adapter_config.json").exists():
                print(f"Skipping completed seed={seed} condition={condition}")
                continue
            cmd = command(seed, condition)
            print("FT_SEED=" + str(seed), *cmd, flush=True)
            if args.dry_run:
                continue
            run_env = {**env, "FT_SEED": str(seed)}
            subprocess.run(cmd, cwd=ROOT, env=run_env, check=True)


if __name__ == "__main__":
    main()
