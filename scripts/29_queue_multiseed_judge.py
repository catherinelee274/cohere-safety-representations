"""Wait for seeded responses, then run the paid judge ensemble automatically."""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "base_sft_ml_s43", "base_sft_en3000_s43",
    "base_dpo_ml_s43", "base_dpo_en3000_s43",
    "base_sft_ml_s44", "base_sft_en3000_s44",
    "base_dpo_ml_s44", "base_dpo_en3000_s44",
]
LANGS = ["en", "fr", "ar", "hi", "sw", "yo"]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def responses_ready() -> bool:
    directory = ROOT / "results/behavioral"
    return all(
        (directory / f"responses_{model}_{lang}.jsonl").exists()
        for model in MODELS
        for lang in LANGS
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    while not responses_ready():
        print("Waiting for all 48 seeded response files...", flush=True)
        time.sleep(300)

    command = [
        sys.executable, "scripts/08_llm_judge.py",
        "--models", *MODELS,
        "--langs", *LANGS,
        "--limit", "300",
        "--workers", "8",
        "--judge-tag", "multiseed_ens",
    ]
    print("Seeded responses complete; starting judge ensemble.", flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
    subprocess.run(
        [sys.executable, "scripts/30_aggregate_multiseed_judge.py"],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )


if __name__ == "__main__":
    main()
