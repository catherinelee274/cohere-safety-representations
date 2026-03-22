"""
Step 2 — Download raw safety datasets.

Datasets:
  - XSafety      : multilingual safety benchmark (11 categories, 10 langs)
  - Aya Red-teaming : multilingual red-teaming prompts (CohereForAI)
  - PKU-SafeRLHF : English preference pairs (used for DPO/translation)
  - LITMUS (ACCD): English axiom-based safety dataset (20k rows, 7 value dims)
                   columns: axiom, safety_label, source_dataset, input, target
                   English-only; used for en harmful/harmless pool + axiom analysis

Usage:
    python scripts/02_download_datasets.py [--data-dir data/raw] [--token HF_TOKEN]

Outputs (in data/raw/):
    xsafety/           XSafety CSV split by language
    aya_redteam/       Aya red-teaming JSONL
    pku_saferlhf/      PKU-SafeRLHF parquet files
    litmus/            LITMUS CSV (full + split by safety_label)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES


# ── helpers ──────────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


# ── dataset loaders ───────────────────────────────────────────────────────────

def download_xsafety(out_dir: Path):
    """
    XSafety: https://github.com/Jarviswang94/Multilingual_safety_benchmark
    Columns: prompt, language, category, label (safe/unsafe)
    Cloned via git as the dataset is only available through the GitHub repo.
    """
    print("\n[1/4] Downloading XSafety...")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/Jarviswang94/Multilingual_safety_benchmark"

    if out_dir.exists():
        print(f"  {out_dir} already exists, skipping clone.")
        return

    print(f"  git clone {repo_url} {out_dir}")
    result = subprocess.run(
        ["git", "clone", repo_url, str(out_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR cloning XSafety:\n{result.stderr}")
        return
    print(f"  Cloned → {out_dir}/")


def download_aya_redteam(out_dir: Path, token: str | None):
    """
    Aya Red-teaming dataset from CohereForAI.
    """
    print("\n[2/3] Downloading Aya Red-teaming...")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = [
        "CohereForAI/aya_redteaming",   # primary candidate
        "nthakur/multilingual-safety",  # fallback
    ]

    ds = None
    for did in dataset_ids:
        try:
            ds = load_dataset(did, token=token)
            print(f"  Loaded from {did}")
            break
        except Exception as e:
            print(f"  {did} unavailable: {e}")

    if ds is None:
        print("  Could not load Aya red-teaming dataset. Check dataset ID and access.")
        return

    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        out_path = out_dir / f"aya_redteam_{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  {split_name}: {len(df):,} rows → {out_path}")


def download_pku_saferlhf(out_dir: Path, token: str | None, n_sample: int = 500):
    """
    PKU-SafeRLHF: English preference pairs for harmful/harmless classification.
    We sample n_sample pairs; these will later be translated via NLLB.
    """
    print("\n[3/3] Downloading PKU-SafeRLHF...")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", token=token, split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    df = ds.to_pandas()
    print(f"  Total rows: {len(df):,}")

    # Keep useful columns; exact schema varies by version
    keep = []
    for col in ["prompt", "response_0", "response_1",
                "is_response_0_safe", "is_response_1_safe",
                "better_response_id", "safer_response_id"]:
        if col in df.columns:
            keep.append(col)
    df = df[keep]

    # Sample balanced harmful/harmless pairs
    # A "harmful" prompt is one where at least one response is unsafe
    if "is_response_0_safe" in df.columns:
        harmful   = df[~df["is_response_0_safe"]].sample(
            min(n_sample // 2, len(df)), random_state=42)
        harmless  = df[ df["is_response_0_safe"]].sample(
            min(n_sample // 2, len(df)), random_state=42)
        sampled   = pd.concat([harmful, harmless]).sample(frac=1, random_state=42)
    else:
        sampled = df.sample(min(n_sample, len(df)), random_state=42)

    out_path = out_dir / "pku_saferlhf_sample.csv"
    sampled.to_csv(out_path, index=False)
    print(f"  Sampled {len(sampled):,} rows → {out_path}")


def download_litmus(out_dir: Path, token: str | None):
    """
    LITMUS / ACCD (hasnat79/litmus): English axiom-based safety dataset.

    Columns used:
      - input        : the prompt / email text
      - safety_label : "safe" | "unsafe"
      - axiom        : value dimension (Wisdom, Well-being, Justice, Duty, Civility …)
      - source_dataset: origin dataset name

    English-only. Used as an additional source for English harmful/harmless pairs;
    axiom column enables stratified analysis by value dimension.
    ~44% of rows have missing 'target'; we keep those since we only need 'input'.
    """
    print("\n[4/4] Downloading LITMUS (ACCD)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("hasnat79/litmus", token=token)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    dfs = []
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        dfs.append(df)
        print(f"  {split_name}: {len(df):,} rows")

    full = pd.concat(dfs, ignore_index=True)

    # Normalise safety_label to lowercase
    if "safety_label" in full.columns:
        full["safety_label"] = full["safety_label"].str.lower().str.strip()

    full.to_csv(out_dir / "litmus_full.csv", index=False)
    print(f"  Total: {len(full):,} rows → {out_dir}/litmus_full.csv")

    # Split by label for easy loading downstream
    for label in ["safe", "unsafe"]:
        sub = full[full["safety_label"] == label]
        sub.to_csv(out_dir / f"litmus_{label}.csv", index=False)
        print(f"  {label}: {len(sub):,} rows")

    # Per-axiom breakdown
    if "axiom" in full.columns:
        print("  Axiom breakdown:")
        for axiom, grp in full.groupby("axiom"):
            n_unsafe = (grp["safety_label"] == "unsafe").sum()
            print(f"    {axiom:<30} {len(grp):>5} rows  ({n_unsafe} unsafe)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw",
                        help="Root directory for raw data")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token")
    parser.add_argument("--pku-sample", type=int, default=500,
                        help="Number of PKU-SafeRLHF pairs to sample")
    parser.add_argument("--use-hf-mirror", action="store_true",
                        help="Use Hugging Face mirror as source for datasets")
    args = parser.parse_args()

    # Set HF endpoint to mirror if specified
    if args.use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using Hugging Face mirror: https://hf-mirror.com")
    
    token = args.token or os.environ.get("HF_TOKEN")
    raw   = ROOT / args.data_dir

    download_xsafety(raw / "xsafety")
    download_aya_redteam(raw / "aya_redteam", token)
    download_pku_saferlhf(raw / "pku_saferlhf", token, n_sample=args.pku_sample)
    download_litmus(raw / "litmus", token)

    print("\nDone. Next: run scripts/03_translate_prompts.py")


if __name__ == "__main__":
    main()
