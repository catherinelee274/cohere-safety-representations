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
from src.run_metadata import save_run_info

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def judge_file(
    in_path: Path,
    out_dir: Path,
    openai_key: str | None,
    cohere_key: str | None,
    rate_limit_sleep: float = 0.5,
    limit: int | None = None,
    workers: int = 1,
    out_tag: str = "",
) -> dict:
    stem     = in_path.stem.replace("responses_", "")
    suffix   = (f"_{out_tag}" if out_tag else "") + (f"_s{limit}" if limit else "")
    out_path = out_dir / f"judged_{stem}{suffix}.jsonl"

    if out_path.exists():
        logger.info(f"Skipping {in_path.name} — already judged.")
        with open(out_path) as f:
            judged = [json.loads(l) for l in f if l.strip()]
        return _compute_stats(judged, stem)

    with open(in_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Deterministic sample so refusal/unsafe rates stay comparable across models.
    if limit and len(records) > limit:
        import numpy as np
        idx = np.random.default_rng(42).choice(len(records), limit, replace=False)
        records = [records[i] for i in sorted(idx)]

    def _judge_one(rec):
        judgment = judge_ensemble(
            prompt=rec["text"],
            response=rec["response"],
            openai_key=openai_key,
            cohere_key=cohere_key,
        )
        return {**rec, **judgment}

    judged = []
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            judged = list(tqdm(ex.map(_judge_one, records),
                               total=len(records), desc=f"Judging {stem}"))
    else:
        for rec in tqdm(records, desc=f"Judging {stem}"):
            judged.append(_judge_one(rec))
            import time; time.sleep(rate_limit_sleep)
    n_unsafe = sum(r.get("ensemble_verdict") == "UNSAFE" for r in judged)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = _compute_stats(judged, stem)
    agree_str = f"{stats['judge_agreement']:.2%}" if stats['judge_agreement'] is not None else "N/A"
    logger.info(f"  {stem}: unsafe_rate={stats['unsafe_rate']:.2%}  agreement={agree_str}")

    save_run_info(out_dir, {
        "script":          "08_llm_judge",
        "input_file":      str(in_path),
        "model":           stats.get("model"),
        "lang":            stats.get("lang"),
        "n_judged":        stats.get("n"),
        "unsafe_rate":     stats.get("unsafe_rate"),
        "judge_agreement": stats.get("judge_agreement"),
        "judges":          ["gpt-4o", "command-r"],
    })
    return stats


def _compute_stats(records: list[dict], stem: str) -> dict:
    from datetime import datetime, timezone
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

    # Inter-judge Cohen's κ (Command-R vs Claude), gold-free — needs both verdicts.
    kappa = None
    cr = [r.get("command_r_verdict") for r in records]
    cl = [r.get("claude_verdict")    for r in records]
    pairs = [(a, b) for a, b in zip(cr, cl)
             if a in ("SAFE", "UNSAFE") and b in ("SAFE", "UNSAFE")]
    if len(pairs) >= 2 and len({a for a, _ in pairs} | {b for _, b in pairs}) > 1:
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = round(cohen_kappa_score([a for a, _ in pairs], [b for _, b in pairs]), 4)
        except Exception:
            kappa = None

    from config.models import MODELS
    hf_id = MODELS[model].hf_id if model in MODELS else None
    model_name = MODELS[model].name if model in MODELS else None
    return {
        "model":           model,
        "model_hf_id":     hf_id,
        "model_name":      model_name,
        "lang":            lang,
        "n":               n,
        "unsafe_rate":     round(unsafe / n, 4),
        "avg_harm_score":  round(avg_score, 3),
        "judge_agreement": round(agreed / len(agree_vals), 4) if agree_vals else None,
        "interjudge_kappa": kappa,
        "n_judges":        records[0].get("n_judges") if records else None,
        "run_at":          datetime.now(timezone.utc).isoformat(),
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
                        help="Sleep between API calls (seconds; ignored when --workers > 1)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Judge at most N responses per file (deterministic sample)")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Concurrent judge calls per file")
    parser.add_argument("--models",   nargs="+", default=None,
                        help="Only judge these model keys")
    parser.add_argument("--langs",    nargs="+", default=None,
                        help="Only judge these languages")
    parser.add_argument("--judge-tag", default="",
                        help="Suffix for output files + summary (e.g. 'ens' -> judged_*_ens.jsonl, "
                             "judge_summary_ens.csv). Avoids clobbering a single-judge run.")
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
    if args.all or args.models or args.langs:
        files = sorted(behav_dir.glob("responses_*.jsonl"))
    elif args.input:
        files = [Path(args.input)]
    else:
        files = sorted(behav_dir.glob("responses_global_en.jsonl"))
        if not files:
            print("No files found. Run scripts/07_behavioral_baseline.py first.")
            sys.exit(1)

    # Optional model / language filters (parse stem = responses_{model}_{lang})
    def _keep(fpath: Path) -> bool:
        stem = fpath.stem.replace("responses_", "")
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            return False
        model, lang = parts
        if args.models and model not in args.models:
            return False
        if args.langs and lang not in args.langs:
            return False
        return True

    if args.models or args.langs:
        files = [f for f in files if _keep(f)]
    logger.info(f"Judging {len(files)} file(s)"
                + (f", sampling {args.limit}/file" if args.limit else "")
                + (f", {args.workers} workers" if args.workers > 1 else ""))

    all_stats = []
    for fpath in files:
        stats = judge_file(fpath, behav_dir, openai_key, cohere_key,
                           args.sleep, limit=args.limit, workers=args.workers,
                           out_tag=args.judge_tag)
        all_stats.append(stats)

    # ── Summary (merge with existing, keyed by model+lang) ──────────────────
    if all_stats:
        df = pd.DataFrame(all_stats)
        summary_name = f"judge_summary_{args.judge_tag}.csv" if args.judge_tag else "judge_summary.csv"
        summary_path = behav_dir / summary_name
        if summary_path.exists():
            prev = pd.read_csv(summary_path)
            keys = set(zip(df.get("model", []), df.get("lang", [])))
            prev = prev[~prev.apply(lambda r: (r.get("model"), r.get("lang")) in keys, axis=1)]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(summary_path, index=False)
        print(f"\nJudge summary:\n{df.to_string(index=False)}")
        print(f"\nSaved → {summary_path}")

    print("\nDone. Next: scripts/09_sae_analysis.py")


if __name__ == "__main__":
    main()
