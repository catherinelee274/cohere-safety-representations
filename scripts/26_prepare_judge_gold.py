"""Prepare a balanced response packet for manual judge validation."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["en", "fr", "ar", "hi", "sw", "yo"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base_dpo_ml")
    parser.add_argument("--per-lang", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/judge_gold_rq2_300.jsonl")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    packet = []
    for lang in LANGS:
        path = ROOT / f"results/behavioral/responses_{args.model}_{lang}.jsonl"
        with path.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        indices = sorted(rng.choice(len(records), args.per_lang, replace=False))
        for index in indices:
            record = records[index]
            packet.append({
                "id": record["id"],
                "lang": lang,
                "prompt": record["text"],
                "response": record["response"],
                "gold_label": "",
                "notes": "",
            })

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in packet:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(packet)} records → {output}")


if __name__ == "__main__":
    main()
