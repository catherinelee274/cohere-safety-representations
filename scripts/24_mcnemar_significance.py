"""Exact paired McNemar tests for multilingual vs English-only safety FT.

The test operates on matched prompt IDs and compares binary heuristic refusal
outcomes. Outputs results/behavioral/mcnemar_significance.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = [
    ("sft", 42, "base_sft_ml_big", "base_sft_en3000"),
    ("dpo", 42, "base_dpo_ml", "base_dpo_en3000"),
    ("sft", 43, "base_sft_ml_s43", "base_sft_en3000_s43"),
    ("dpo", 43, "base_dpo_ml_s43", "base_dpo_en3000_s43"),
    ("sft", 44, "base_sft_ml_s44", "base_sft_en3000_s44"),
    ("dpo", 44, "base_dpo_ml_s44", "base_dpo_en3000_s44"),
]
DEFAULT_LANGS = ["en", "fr", "ar", "hi", "sw", "yo"]


def load_outcomes(path: Path) -> dict[str, bool]:
    with path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return {record["id"]: bool(record["heuristic_refuse"]) for record in records}


def compare(
    stage: str, seed: int, ml_model: str, en_model: str, lang: str, directory: Path
) -> dict:
    ml_path = directory / f"responses_{ml_model}_{lang}.jsonl"
    en_path = directory / f"responses_{en_model}_{lang}.jsonl"
    ml = load_outcomes(ml_path)
    en = load_outcomes(en_path)
    if set(ml) != set(en):
        missing_ml = len(set(en) - set(ml))
        missing_en = len(set(ml) - set(en))
        raise ValueError(
            f"Prompt IDs differ for {stage}/{lang}: "
            f"missing from ml={missing_ml}, missing from en={missing_en}"
        )

    ml_only = sum(ml[prompt_id] and not en[prompt_id] for prompt_id in ml)
    en_only = sum(en[prompt_id] and not ml[prompt_id] for prompt_id in ml)
    discordant = ml_only + en_only
    p_value = binomtest(ml_only, discordant, 0.5).pvalue if discordant else 1.0
    n = len(ml)
    ml_rate = sum(ml.values()) / n
    en_rate = sum(en.values()) / n
    return {
        "stage": stage,
        "seed": seed,
        "lang": lang,
        "ml_model": ml_model,
        "en_model": en_model,
        "n_pairs": n,
        "ml_refusal_rate": round(ml_rate, 4),
        "en_refusal_rate": round(en_rate, 4),
        "rate_difference": round(ml_rate - en_rate, 4),
        "ml_refuses_en_not": ml_only,
        "en_refuses_ml_not": en_only,
        "n_discordant": discordant,
        "exact_p_value": p_value,
        "significant_0_05": p_value < 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral-dir", default="results/behavioral")
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    parser.add_argument(
        "--output", default="results/behavioral/mcnemar_significance.csv"
    )
    args = parser.parse_args()

    directory = ROOT / args.behavioral_dir
    rows = [
        compare(stage, seed, ml_model, en_model, lang, directory)
        for stage, seed, ml_model, en_model in DEFAULT_PAIRS
        for lang in args.langs
    ]
    result = pd.DataFrame(rows)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()
