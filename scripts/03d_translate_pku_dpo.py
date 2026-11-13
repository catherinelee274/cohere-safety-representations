"""
Step 03d — Build a balanced multilingual DPO preference set by translating
PKU-SafeRLHF (prompt, chosen=safe, rejected=unsafe) triples into the 6 target
languages via Command-A. Mirror of 03c but for DPO preference pairs.

Design: same as 03c — partition N distinct preference pairs evenly across the 6
languages (500/lang by default), English verbatim, others Command-A translated.
Also emits a matched English-only control (the same pairs, all English).

Reads:  data/raw/pku_saferlhf/pku_saferlhf_sample.csv
Writes: data/processed/pku_translated/pku_dpo_multilingual{tag}.jsonl  {prompt,chosen,rejected,lang}
        data/processed/pku_translated/pku_dpo_en_control{tag}.jsonl

Usage:
    python scripts/03d_translate_pku_dpo.py --total 3000 --workers 10 --tag _n3000 --emit-en-control
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


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_pku_dpo_records(pku_csv: Path) -> list[dict]:
    """Same rule as 10_finetune.load_dpo_data_pku: chosen=safe, rejected=unsafe."""
    df = pd.read_csv(pku_csv)
    recs = []
    for _, row in df.iterrows():
        r0_safe = bool(row.get("is_response_0_safe", False))
        r1_safe = bool(row.get("is_response_1_safe", False))
        prompt  = str(row.get("prompt", "")).strip()
        r0      = str(row.get("response_0", "")).strip()
        r1      = str(row.get("response_1", "")).strip()
        if not prompt or not r0 or not r1:
            continue
        if r0_safe and not r1_safe:
            recs.append({"prompt": prompt, "chosen": r0, "rejected": r1})
        elif r1_safe and not r0_safe:
            recs.append({"prompt": prompt, "chosen": r1, "rejected": r0})
    return recs


def make_translator():
    import cohere
    co = cohere.Client(os.environ["CO_API_KEY"])

    def translate(text: str, lang_name: str, retries: int = 4) -> str:
        text = text.strip()
        if not text:
            return ""
        msg = (f"Translate the following text into {lang_name}. Output ONLY the "
               f"translation, no preamble or quotation marks.\n\n{text}")
        for attempt in range(retries):
            try:
                return co.chat(message=msg, model="command-a-03-2025",
                               temperature=0.0).text.strip()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    return translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pku-csv", default="data/raw/pku_saferlhf/pku_saferlhf_sample.csv")
    parser.add_argument("--total", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1000,
                        help="Skip triples whose prompt+chosen+rejected exceeds this.")
    parser.add_argument("--tag", default="_n3000")
    parser.add_argument("--emit-en-control", action="store_true")
    args = parser.parse_args()

    _load_dotenv()
    if "CO_API_KEY" not in os.environ:
        parser.error("CO_API_KEY not set (expected in .env).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    langs = LANG_CODES

    records = load_pku_dpo_records(ROOT / args.pku_csv)
    records = [r for r in records
               if len(r["prompt"]) + len(r["chosen"]) + len(r["rejected"]) <= args.max_chars]
    rng = np.random.default_rng(7)
    rng.shuffle(records)  # type: ignore[arg-type]
    records = records[: args.total]
    print(f"Selected {len(records)} PKU DPO pairs (<= {args.max_chars} chars).")

    partitions = {l: [] for l in langs}
    for i, rec in enumerate(records):
        partitions[langs[i % len(langs)]].append(rec)

    translate = make_translator()
    combined = []

    for lang in langs:
        part = partitions[lang]
        if lang == "en":
            rows = [{**r, "lang": "en"} for r in part]
            print(f"  en: {len(rows)} pairs (no translation).")
        else:
            name = LANG_NAMES[lang]
            print(f"  {lang} ({name}): translating {len(part)} pairs x3 fields ...", flush=True)
            t0 = time.time()
            tmp = {i: {} for i in range(len(part))}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {}
                for i, r in enumerate(part):
                    futs[ex.submit(translate, r["prompt"], name)]   = ("prompt", i)
                    futs[ex.submit(translate, r["chosen"], name)]   = ("chosen", i)
                    futs[ex.submit(translate, r["rejected"], name)] = ("rejected", i)
                done = 0
                for fut in as_completed(futs):
                    field, i = futs[fut]
                    tmp[i][field] = fut.result()
                    done += 1
                    if done % 100 == 0:
                        print(f"    {lang}: {done}/{len(part)*3} fields ({time.time()-t0:.0f}s)", flush=True)
            rows = []
            for i in range(len(part)):
                d = tmp[i]
                if d.get("prompt") and d.get("chosen") and d.get("rejected"):
                    rows.append({"prompt": d["prompt"], "chosen": d["chosen"],
                                 "rejected": d["rejected"], "lang": lang})
            print(f"  {lang}: {len(rows)} pairs in {time.time()-t0:.0f}s.")

        with open(OUT_DIR / f"pku_dpo_{lang}{args.tag}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        combined.extend(rows)

    rng.shuffle(combined)  # type: ignore[arg-type]
    out = OUT_DIR / f"pku_dpo_multilingual{args.tag}.jsonl"
    with open(out, "w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(combined)} multilingual DPO pairs -> {out}")
    print("Per-language:", pd.Series([r["lang"] for r in combined]).value_counts().to_dict())

    if args.emit_en_control:
        en_rows = [{"prompt": r["prompt"], "chosen": r["chosen"],
                    "rejected": r["rejected"], "lang": "en"} for r in records]
        en_out = OUT_DIR / f"pku_dpo_en_control{args.tag}.jsonl"
        with open(en_out, "w") as f:
            for r in en_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(en_rows)} English-only control DPO pairs -> {en_out}")


if __name__ == "__main__":
    main()
