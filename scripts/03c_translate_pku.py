"""
Step 03c — Build a balanced multilingual Safety-FT dataset by translating a
PKU-SafeRLHF SFT subset into the 6 target languages via Command-A.

Motivation (RQ2, multilingual arm)
----------------------------------
The English Safety-FT checkpoint (base_sft) was trained on ~284 PKU-SafeRLHF
SFT pairs (prompt -> safe response).  To test whether *multilingual* Safety-FT
helps low-resource languages more than English-only FT, we need the same kind
of data spread across languages.

Design: budget-matched comparison
---------------------------------
We take ``--total`` distinct PKU SFT pairs and PARTITION them evenly across the
6 languages (so each prompt appears in exactly one language).  This keeps the
total training-example budget the same as the English SFT run while varying only
the *language coverage*.  English keeps its original text; the other five
languages are translated with Command-A (prompt and safe completion separately).

Reads:
    data/raw/pku_saferlhf/pku_saferlhf_sample.csv

Writes:
    data/processed/pku_translated/pku_sft_{lang}.jsonl   # {prompt, completion, lang}
    data/processed/pku_translated/pku_sft_multilingual.jsonl  # combined

Usage:
    python scripts/03c_translate_pku.py --total 300 --workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES

OUT_DIR = ROOT / "data" / "processed" / "pku_translated"

# Load .env so CO_API_KEY / HF_TOKEN are available when run directly.
def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_pku_sft_records(pku_csv: Path) -> list[dict]:
    """Same selection rule as 10_finetune.load_sft_data_pku: keep the safe response."""
    df = pd.read_csv(pku_csv)
    records = []
    for _, row in df.iterrows():
        r0_safe = bool(row.get("is_response_0_safe", False))
        r1_safe = bool(row.get("is_response_1_safe", False))
        prompt  = str(row.get("prompt", "")).strip()
        r0      = str(row.get("response_0", "")).strip()
        r1      = str(row.get("response_1", "")).strip()
        if not prompt:
            continue
        if r0_safe and r0:
            records.append({"prompt": prompt, "completion": r0})
        elif r1_safe and r1:
            records.append({"prompt": prompt, "completion": r1})
    return records


def make_translator():
    import cohere
    co = cohere.Client(os.environ["CO_API_KEY"])

    def translate(text: str, lang_name: str, retries: int = 4) -> str:
        text = text.strip()
        if not text:
            return ""
        msg = (
            f"Translate the following text into {lang_name}. "
            f"Output ONLY the translation, with no preamble, notes, or quotation marks.\n\n"
            f"{text}"
        )
        for attempt in range(retries):
            try:
                r = co.chat(message=msg, model="command-a-03-2025", temperature=0.0)
                return r.text.strip()
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    return translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pku-csv", default="data/raw/pku_saferlhf/pku_saferlhf_sample.csv")
    parser.add_argument("--total", type=int, default=300,
                        help="Total SFT examples across all languages (partitioned evenly).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1200,
                        help="Skip pairs whose prompt+completion exceeds this (keeps translation cheap/clean).")
    parser.add_argument("--tag", default="",
                        help="Filename suffix (e.g. _n3000) so larger runs don't clobber smaller ones.")
    parser.add_argument("--emit-en-control", action="store_true",
                        help="Also write an English-only version of the SAME selected prompts "
                             "(matched-budget control for the multilingual SFT).")
    args = parser.parse_args()

    _load_dotenv()
    if "CO_API_KEY" not in os.environ:
        parser.error("CO_API_KEY not set (expected in .env).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    langs = LANG_CODES  # en, fr, ar, hi, sw, yo

    records = load_pku_sft_records(ROOT / args.pku_csv)
    # Keep short, clean pairs for reliable translation.
    records = [r for r in records if len(r["prompt"]) + len(r["completion"]) <= args.max_chars]
    rng = np.random.default_rng(42)
    rng.shuffle(records)  # type: ignore[arg-type]
    records = records[: args.total]
    print(f"Selected {len(records)} PKU SFT pairs (<= {args.max_chars} chars) for {len(langs)} languages.")

    # Partition evenly across languages.
    partitions: dict[str, list[dict]] = {l: [] for l in langs}
    for i, rec in enumerate(records):
        partitions[langs[i % len(langs)]].append(rec)

    translate = make_translator()
    combined: list[dict] = []

    for lang in langs:
        part = partitions[lang]
        out_path = OUT_DIR / f"pku_sft_{lang}{args.tag}.jsonl"
        if lang == "en":
            rows = [{"prompt": r["prompt"], "completion": r["completion"], "lang": "en"} for r in part]
            print(f"  en: {len(rows)} pairs (no translation).")
        else:
            lang_name = LANG_NAMES[lang]
            print(f"  {lang} ({lang_name}): translating {len(part)} pairs x2 fields ...", flush=True)
            rows = [None] * len(part)
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {}
                for idx, r in enumerate(part):
                    futs[ex.submit(translate, r["prompt"], lang_name)] = ("p", idx)
                    futs[ex.submit(translate, r["completion"], lang_name)] = ("c", idx)
                tmp = {idx: {} for idx in range(len(part))}
                done = 0
                for fut in as_completed(futs):
                    kind, idx = futs[fut]
                    tmp[idx][kind] = fut.result()
                    done += 1
                    if done % 50 == 0:
                        print(f"    {lang}: {done}/{len(part)*2} fields ({time.time()-t0:.0f}s)", flush=True)
            for idx in range(len(part)):
                rows[idx] = {
                    "prompt": tmp[idx].get("p", ""),
                    "completion": tmp[idx].get("c", ""),
                    "lang": lang,
                }
            rows = [r for r in rows if r["prompt"] and r["completion"]]
            print(f"  {lang}: {len(rows)} pairs translated in {time.time()-t0:.0f}s.")

        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        combined.extend(rows)

    rng.shuffle(combined)  # type: ignore[arg-type]
    combined_path = OUT_DIR / f"pku_sft_multilingual{args.tag}.jsonl"
    with open(combined_path, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(combined)} combined multilingual SFT examples -> {combined_path}")
    by_lang = pd.Series([r["lang"] for r in combined]).value_counts().to_dict()
    print("Per-language counts:", by_lang)

    # Matched-budget English-only control: the SAME selected prompts, all in English.
    if args.emit_en_control:
        en_rows = [{"prompt": r["prompt"], "completion": r["completion"], "lang": "en"}
                   for r in records]
        en_path = OUT_DIR / f"pku_sft_en_control{args.tag}.jsonl"
        with open(en_path, "w") as f:
            for r in en_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(en_rows)} English-only control examples -> {en_path}")


if __name__ == "__main__":
    main()
