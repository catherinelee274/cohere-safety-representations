"""
Step 8 — Run LLM judge ensemble over behavioral baseline responses.

Takes the JSONL files from scripts/07_behavioral_baseline.py and
re-scores each (prompt, response) pair with GPT-4o + Command-R.

Usage:
    # Validate inter-rater agreement first (required before full run)
    python scripts/08_llm_judge.py --validate --labeled data/judge_gold_50.jsonl

    # Judge a single file
    python scripts/08_llm_judge.py \\
        --input results/behavioral/responses_global_en.jsonl

    # Judge all files
    python scripts/08_llm_judge.py --all

Outputs:
    results/behavioral/judged_{model}_{lang}.jsonl
    results/behavioral/judge_summary.csv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.judge import judge_ensemble, validate_interrater

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def judge_file(
    in_path: Path,
    out_dir: Path,
    openai_key: str | None,
    cohere_key: str | None,
    rate_limit_sleep: float = 0.5,
) -> dict:
    stem     = in_path.stem.replace("responses_", "")
    out_path = out_dir / f"judged_{stem}.jsonl"

    if out_path.exists():
        logger.info(f"Skipping {in_path.name} — already judged.")
        with open(out_path) as f:
            judged = [json.loads(l) for l in f if l.strip()]
        return _compute_stats(judged, stem)

    with open(in_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    judged = []
    n_unsafe = 0

    for rec in tqdm(records, desc=f"Judging {stem}"):
        judgment = judge_ensemble(
            prompt=rec["text"],
            response=rec["response"],
            openai_key=openai_key,
            cohere_key=cohere_key,
        )
        row = {**rec, **judgment}
        judged.append(row)
        if judgment["ensemble_verdict"] == "UNSAFE":
            n_unsafe += 1

        import time; time.sleep(rate_limit_sleep)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = _compute_stats(judged, stem)
    agree_str = f"{stats['judge_agreement']:.2%}" if stats['judge_agreement'] is not None else "N/A"
    logger.info(f"  {stem}: unsafe_rate={stats['unsafe_rate']:.2%}  agreement={agree_str}")
    return stats


def _compute_stats(records: list[dict], stem: str) -> dict:
    parts = stem.rsplit("_", 1)
    model = parts[0] if len(parts) == 2 else stem
    lang  = parts[1] if len(parts) == 2 else "?"
    n = len(records)
    if n == 0:
        return {"model": model, "lang": lang, "n": 0}

    unsafe    = sum(r.get("ensemble_verdict") == "UNSAFE" for r in records)
    agree_vals = [r.get("agreement") for r in records if r.get("agreement") is not None]
    agreed     = sum(agree_vals) if agree_vals else None
    avg_score  = sum(r.get("ensemble_score", 0) or 0 for r in records) / n

    return {
        "model":           model,
        "lang":            lang,
        "n":               n,
        "unsafe_rate":     round(unsafe / n, 4),
        "avg_harm_score":  round(avg_score, 3),
        "judge_agreement": round(agreed / len(agree_vals), 4) if agree_vals else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default=None,
                        help="Single responses JSONL file to judge")
    parser.add_argument("--all",      action="store_true",
                        help="Judge all responses_*.jsonl files")
    parser.add_argument("--validate", action="store_true",
                        help="Run inter-rater agreement validation first")
    parser.add_argument("--labeled",  default="data/judge_gold_50.jsonl",
                        help="Gold-labeled JSONL for inter-rater validation")
    parser.add_argument("--behavioral-dir", default="results/behavioral")
    parser.add_argument("--openai-key",     default=None)
    parser.add_argument("--cohere-key",     default=None)
    parser.add_argument("--sleep",    type=float, default=0.5,
                        help="Sleep between API calls (seconds)")
    args = parser.parse_args()

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    cohere_key = args.cohere_key or os.environ.get("CO_API_KEY")
    behav_dir  = ROOT / args.behavioral_dir

    # ── Inter-rater validation ──────────────────────────────────────────────
    if args.validate:
        labeled_path = ROOT / args.labeled
        if not labeled_path.exists():
            print(f"ERROR: Gold labels file not found: {labeled_path}")
            print("Create a JSONL file with fields: prompt, response, gold_label")
            sys.exit(1)
        result = validate_interrater(
            str(labeled_path),
            openai_key=openai_key,
            cohere_key=cohere_key,
        )
        print(f"\nInter-rater agreement:\n{json.dumps(result, indent=2)}")
        if result["cohen_kappa"] < 0.7:
            print("\nWARNING: κ < 0.7. Review and refine judge prompt before full run.")
        else:
            print("\nAgreement OK (κ ≥ 0.7). Proceed with full judging.")
        return

    # ── Collect files to judge ──────────────────────────────────────────────
    if args.all:
        files = sorted(behav_dir.glob("responses_*.jsonl"))
    elif args.input:
        files = [Path(args.input)]
    else:
        files = sorted(behav_dir.glob("responses_global_en.jsonl"))
        if not files:
            print("No files found. Run scripts/07_behavioral_baseline.py first.")
            sys.exit(1)

    all_stats = []
    for fpath in files:
        stats = judge_file(fpath, behav_dir, openai_key, cohere_key, args.sleep)
        all_stats.append(stats)

    # ── Summary ─────────────────────────────────────────────────────────────
    if all_stats:
        df = pd.DataFrame(all_stats)
        summary_path = behav_dir / "judge_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nJudge summary:\n{df.to_string(index=False)}")
        print(f"\nSaved → {summary_path}")

    print("\nDone. Next: scripts/09_sae_analysis.py")


if __name__ == "__main__":
    main()
