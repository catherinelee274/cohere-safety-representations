"""Run behavioral and matched-probe evaluation for seed 43/44 models."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "base_sft_ml_s43", "base_sft_en3000_s43",
    "base_dpo_ml_s43", "base_dpo_en3000_s43",
    "base_sft_ml_s44", "base_sft_en3000_s44",
    "base_dpo_ml_s44", "base_dpo_en3000_s44",
]
PROBE_LANGS = ["en", "fr", "ar", "sw", "yo"]


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--skip-behavioral", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(Path.home() / "hf_cache" / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(Path.home() / "hf_cache" / "datasets"))

    if not args.skip_behavioral:
        for model in args.models:
            run([
                sys.executable, "scripts/07_behavioral_baseline.py",
                "--model", model, "--batch-size", str(args.batch_size),
            ], env, args.dry_run)

    if not args.skip_probes:
        for model in args.models:
            run([
                sys.executable, "scripts/05b_extract_reuse_prompts.py",
                "--model", model,
                "--ref-model", "base_sft_ml_big",
                "--langs", *PROBE_LANGS,
                "--batch-size", str(args.probe_batch_size),
            ], env, args.dry_run)
            run([
                sys.executable, "scripts/06_train_probes.py",
                "--model", model, "--alignment",
            ], env, args.dry_run)

    run([sys.executable, "scripts/28_aggregate_multiseed.py"], env, args.dry_run)


if __name__ == "__main__":
    main()
