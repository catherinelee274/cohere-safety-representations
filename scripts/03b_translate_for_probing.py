"""
Quick translation helper for probe data.

Uses NLLB-200-distilled-1.3B to translate harmful prompts for non-English languages.
Harmless prompts come from Aya Evaluation Suite (already multilingual) — loaded
directly in 04_build_prompt_sets.py, no translation needed here.

Writes translated harmful prompts to data/processed/en_translated/{lang}.jsonl,
which 04_build_prompt_sets.py picks up as an additional harmful source.

Usage:
    python scripts/03b_translate_for_probing.py --langs fr ar zh sw yo
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import NLLB_CODES, LANG_CODES

NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"
SRC_LANG   = "eng_Latn"


def make_id(text, lang):
    return hashlib.md5(f"{lang}::{text}".encode()).hexdigest()[:12]


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(records):>4} → {path}")


class Translator:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading NLLB on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        self.device = device

    def translate(self, texts, tgt_lang, batch_size=16):
        self.tokenizer.src_lang = SRC_LANG
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        results = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"→ {tgt_lang}", leave=False):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, forced_bos_token_id=tgt_id,
                    max_length=256, num_beams=4,
                )
            results += self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["fr", "ar", "zh", "sw", "yo"])
    parser.add_argument("--n-harmful",  type=int, default=10_000)
    args = parser.parse_args()

    raw_dir     = ROOT / "data/raw"
    prompts_dir = ROOT / "data/prompts"

    # --- load English harmful prompts for translation ---
    # XSafety en + PKU harmful prompts pooled
    en_harmful_texts = []
    xsafety_en_dir = raw_dir / "xsafety" / "en"
    if xsafety_en_dir.exists():
        for csv_path in sorted(xsafety_en_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
                col = next((c for c in df.columns
                            if "prompt" in c.lower()), df.columns[0])
                for text in df[col].dropna().astype(str).str.strip():
                    if text and text != "nan":
                        en_harmful_texts.append(text)
            except Exception:
                continue
    pku_path = raw_dir / "pku_saferlhf" / "pku_saferlhf_sample.csv"
    if pku_path.exists():
        df_pku = pd.read_csv(pku_path)
        safe_col = "is_response_0_safe" if "is_response_0_safe" in df_pku.columns else None
        for _, row in df_pku.iterrows():
            if safe_col and row.get(safe_col, True):
                continue
            text = str(row.get("prompt", "")).strip()
            if text and text != "nan":
                en_harmful_texts.append(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in en_harmful_texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    en_harmful_texts = deduped[:args.n_harmful]

    # --- load translator ---
    translator = Translator()

    for lang in args.langs:
        nllb_code = NLLB_CODES.get(lang)
        if not nllb_code:
            print(f"No NLLB code for {lang}, skipping.")
            continue

        print(f"\n=== {lang} ({nllb_code}) ===")

        # --- harmful: translate and save to processed/en_translated/ ---
        # 04_build_prompt_sets.py loads from this location as a source
        translated_harmful_path = ROOT / "data/processed/en_translated" / f"{lang}.jsonl"
        if translated_harmful_path.exists():
            existing_translated = sum(1 for _ in open(translated_harmful_path, encoding="utf-8"))
        else:
            existing_translated = 0

        if existing_translated >= args.n_harmful:
            print(f"  harmful: {existing_translated} translated already present, skipping.")
        elif en_harmful_texts:
            n_needed = args.n_harmful - existing_translated
            to_translate = en_harmful_texts[:n_needed]
            print(f"  Translating {len(to_translate)} harmful prompts")
            translated_harmful = translator.translate(to_translate, nllb_code)
            new_records = [
                {"id": make_id(t, lang), "text": t,
                 "source": "en_translated", "lang": lang, "label": "harmful"}
                for t in translated_harmful
            ]
            save_jsonl(new_records, translated_harmful_path)

    print("\nDone. Re-run scripts/04_build_prompt_sets.py if needed, then step 05.")


if __name__ == "__main__":
    main()
