"""
Step 3 — Translate English prompts to target languages via NLLB-200.

Model: facebook/nllb-200-distilled-1.3B  (good quality/speed tradeoff)
Swap for nllb-200-3.3B for higher quality on Yoruba/Swahili.

Usage:
    python scripts/03_translate_prompts.py \\
        --input  data/raw/pku_saferlhf/pku_saferlhf_sample.csv \\
        --output data/processed/pku_translated/ \\
        --langs  fr ar zh sw yo \\
        --batch-size 16

Outputs (data/processed/pku_translated/):
    pku_{lang}.csv    one file per target language
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, NLLB_CODES

NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"
SRC_LANG   = "eng_Latn"


# ── translation ───────────────────────────────────────────────────────────────

class NLLBTranslator:
    def __init__(self, model_name: str = NLLB_MODEL, device: str | None = None, use_hf_mirror: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set HF endpoint to mirror if specified
        if use_hf_mirror:
            import os
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("Using Hugging Face mirror: https://hf-mirror.com")
        
        print(f"Loading NLLB translator on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def translate_batch(
        self,
        texts: list[str],
        tgt_lang: str,
        src_lang: str = SRC_LANG,
        max_length: int = 256,
    ) -> list[str]:
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=4,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_column(
        self,
        texts: list[str],
        tgt_lang: str,
        batch_size: int = 16,
    ) -> list[str]:
        results = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"Translating → {tgt_lang}"):
            batch = texts[i : i + batch_size]
            results.extend(self.translate_batch(batch, tgt_lang))
        return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/raw/pku_saferlhf/pku_saferlhf_sample.csv")
    parser.add_argument("--output",     default="data/processed/pku_translated")
    parser.add_argument("--langs",      nargs="+", default=["fr", "ar", "zh", "sw", "yo"],
                        choices=LANG_CODES)
    parser.add_argument("--text-col",   default="prompt",
                        help="Column to translate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model",      default=NLLB_MODEL)
    parser.add_argument("--use-hf-mirror", action="store_true",
                        help="Use Hugging Face mirror as source for models")
    args = parser.parse_args()

    in_path  = ROOT / args.input
    out_dir  = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows from {in_path}")

    # Copy English source file as-is
    df_en = df.copy()
    df_en["lang"] = "en"
    df_en.to_csv(out_dir / "pku_en.csv", index=False)
    print(f"Saved English source → {out_dir}/pku_en.csv")

    translator = NLLBTranslator(model_name=args.model, use_hf_mirror=args.use_hf_mirror)

    texts = df[args.text_col].fillna("").tolist()

    for lang in args.langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            print(f"No NLLB code for '{lang}', skipping.")
            continue

        out_path = out_dir / f"pku_{lang}.csv"
        if out_path.exists():
            print(f"Skipping {lang} (already exists: {out_path})")
            continue

        translations = translator.translate_column(
            texts, nllb_code, batch_size=args.batch_size
        )

        df_out = df.copy()
        df_out[args.text_col + "_en"] = df_out[args.text_col]
        df_out[args.text_col]         = translations
        df_out["lang"]                = lang
        df_out.to_csv(out_path, index=False)
        print(f"Saved {lang} translations → {out_path}")

    print("\nDone. Run quality checks before proceeding — especially for Yoruba/Swahili.")
    print("Next: scripts/04_build_prompt_sets.py")


if __name__ == "__main__":
    main()
