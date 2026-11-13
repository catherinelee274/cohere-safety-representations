"""
Step 04b — Guarantee a non-empty held-out harmful eval split for every language.

Background
----------
04_build_prompt_sets.py derives ``harmful_eval.jsonl`` from a held-out 20% of the
XSafety prompts.  XSafety only covers high-resource languages, so for languages
whose harmful prompts are entirely NLLB-translated (sw, yo) the XSafety eval
split is empty.  07_behavioral_baseline.py then loads 0 prompts and writes a
0-byte responses file, which is silently skipped on every subsequent run.

This script makes the eval split robust:

  * If ``harmful_eval.jsonl`` already exists and is non-empty -> leave it alone
    (en / fr / ar / hi keep their genuine XSafety held-out split).
  * Otherwise -> carve a deterministic held-out tail out of ``harmful.jsonl``
    (sorted by id for stability) so low-resource languages also get an eval set.

It is non-destructive: ``harmful.jsonl`` is never modified, so existing probe
results stay reproducible.  The minor train/eval overlap for the translated-only
languages does not affect probe cross-validation (a separate analysis); it only
provides prompts to generate on for the behavioral refusal-rate baseline.

Usage:
    python scripts/04b_eval_splits.py                 # all configured languages
    python scripts/04b_eval_splits.py --langs sw yo
    python scripts/04b_eval_splits.py --force         # rebuild even if present
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES

# Target eval size for carved splits — matches the ~558 XSafety eval size of the
# high-resource languages so refusal rates are comparable across languages.
EVAL_SIZE = 558
HELD_OUT_FRACTION = 0.20


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_eval_split(lang: str, prompts_dir: Path, force: bool) -> None:
    lang_dir    = prompts_dir / lang
    eval_path   = lang_dir / "harmful_eval.jsonl"
    harmful_path = lang_dir / "harmful.jsonl"

    existing = load_jsonl(eval_path) if eval_path.exists() else []
    if existing and not force:
        print(f"  {lang}: OK — {len(existing)} eval prompts already present.")
        return

    if not harmful_path.exists():
        print(f"  {lang}: SKIP — no harmful.jsonl to carve an eval split from.")
        return

    harmful = load_jsonl(harmful_path)
    if not harmful:
        print(f"  {lang}: SKIP — harmful.jsonl is empty.")
        return

    # Deterministic held-out tail (sorted by id for stability across runs).
    harmful_sorted = sorted(harmful, key=lambda r: r["id"])
    n_eval = min(EVAL_SIZE, max(1, int(len(harmful_sorted) * HELD_OUT_FRACTION)))
    eval_records = harmful_sorted[-n_eval:]

    save_jsonl(eval_records, eval_path)
    print(f"  {lang}: built {len(eval_records)} eval prompts "
          f"(held-out tail of {len(harmful_sorted)} translated prompts).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANG_CODES)
    parser.add_argument("--prompts-dir", default="data/prompts")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild eval splits even if a non-empty one exists.")
    args = parser.parse_args()

    prompts_dir = ROOT / args.prompts_dir
    print(f"Ensuring held-out harmful eval splits in {prompts_dir} ...")
    for lang in args.langs:
        build_eval_split(lang, prompts_dir, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
