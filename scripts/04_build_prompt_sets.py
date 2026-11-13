"""
Step 4 — Build matched harmful / harmless prompt sets per language.

Strategy:
  harmful  : from XSafety + Aya Red-teaming (already multilingual)
             + translated PKU-SafeRLHF harmful prompts
  harmless : LLM-assisted rephrase of harmful prompts → benign version
             on same topic; stored for human spot-check before use

Output format (data/prompts/{lang}/):
    harmful.jsonl     {"id": "...", "text": "...", "source": "...", "lang": "..."}
    harmless.jsonl    same schema

Usage:
    python scripts/04_build_prompt_sets.py \\
        --langs en fr ar zh sw yo \\
        --n-per-lang 430
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, N_HARMFUL, N_HARMLESS


# ── helpers ───────────────────────────────────────────────────────────────────

def make_id(text: str, lang: str) -> str:
    return hashlib.md5(f"{lang}::{text}".encode()).hexdigest()[:12]


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(records):>4} records → {path}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── source loaders ────────────────────────────────────────────────────────────

def load_xsafety(raw_dir: Path, lang: str) -> list[dict]:
    """Load XSafety prompts for a given language from per-category CSV files."""
    records = []
    lang_dir = raw_dir / "xsafety" / lang
    if not lang_dir.exists():
        return records
    for csv_path in sorted(lang_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        col = next((c for c in df.columns if "prompt" in c.lower()), df.columns[0])
        category = csv_path.stem
        for _, row in df.iterrows():
            text = str(row[col]).strip()
            if text and text != "nan":
                records.append({
                    "id":       make_id(text, lang),
                    "text":     text,
                    "source":   "xsafety",
                    "lang":     lang,
                    "label":    "harmful",
                    "category": category,
                })
    return records


_AYA_LANG_NAMES = {
    "en": "english", "fr": "french", "ar": "arabic", "zh": "chinese",
    "hi": "hindi",   "sw": "swahili", "yo": "yoruba", "fil": "filipino",
    "es": "spanish", "ru": "russian",
}

def load_aya_redteam(raw_dir: Path, lang: str) -> list[dict]:
    """Load Aya red-teaming prompts for a given language."""
    records = []
    lang_name = _AYA_LANG_NAMES.get(lang, lang)
    # Files are split per language: aya_redteam_{language}.csv
    candidate = raw_dir / "aya_redteam" / f"aya_redteam_{lang_name}.csv"
    if not candidate.exists():
        return records
    try:
        df = pd.read_csv(candidate)
    except Exception:
        return records
    text_col = next((c for c in df.columns if "prompt" in c.lower()), df.columns[0])
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if text and text != "nan":
            records.append({
                "id":     make_id(text, lang),
                "text":   text,
                "source": "aya_redteam",
                "lang":   lang,
                "label":  "harmful",
            })
    return records


# Map our lang codes to ISO 639-3 codes used as language column values in Aya Eval
_AYA_EVAL_ISO = {
    "en":  "eng",
    "fr":  "fra",
    "ar":  "arb",
    "zh":  "zho",
    "hi":  "hin",
    "sw":  "swh",
    "yo":  "yor",
    "fil": "tgl",
}

def load_aya_eval(raw_dir: Path, lang: str) -> list[dict]:
    """
    Load Aya Evaluation Suite (dolly_machine_translated) harmless prompts for a language.
    Files saved by 02_download_datasets.py as data/raw/aya_eval/{iso_code}.csv.
    """
    iso = _AYA_EVAL_ISO.get(lang)
    if not iso:
        return []

    p = raw_dir / "aya_eval" / f"{iso}.csv"
    if not p.exists():
        return []

    try:
        df = pd.read_csv(p)
    except Exception:
        return []

    records = []
    for _, row in df.iterrows():
        text = str(row.get("inputs", "")).strip()
        if text and text != "nan":
            records.append({
                "id":     make_id(text, lang),
                "text":   text,
                "source": "aya_eval",
                "lang":   lang,
                "label":  "harmless",
            })
    return records


def load_en_translated(processed_dir: Path, lang: str) -> list[dict]:
    """Load NLLB-translated harmful prompts produced by 03b."""
    p = processed_dir / "en_translated" / f"{lang}.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_pku_translated(processed_dir: Path, lang: str) -> list[dict]:
    """Load translated PKU harmful prompts."""
    p = processed_dir / "pku_translated" / f"pku_{lang}.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    safe_col = next((c for c in df.columns if "safe" in c.lower()), None)
    records = []
    for _, row in df.iterrows():
        # Only include rows flagged as unsafe
        if safe_col and row.get(safe_col, False):
            continue
        text = str(row.get("prompt", row.iloc[0])).strip()
        if text:
            records.append({
                "id":     make_id(text, lang),
                "text":   text,
                "source": "pku_saferlhf",
                "lang":   lang,
                "label":  "harmful",
            })
    return records


# ── harmless generation ───────────────────────────────────────────────────────

REPHRASE_TEMPLATE = """\
You will be given a harmful prompt. Your task is to rewrite it as a completely \
benign question on the SAME broad topic, suitable for a general audience.
Do NOT include any harmful content, instructions, or implications.
Output ONLY the rephrased question, nothing else.

Harmful prompt: {prompt}
Benign rephrasing:"""


def generate_harmless_batch(harmful_texts: list[str], lang: str) -> list[str]:
    """
    Generate harmless counterparts using a local model or OpenAI.
    Falls back to a simple template-based stub if no API key is set.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CO_API_KEY")

    if api_key and os.environ.get("OPENAI_API_KEY"):
        return _openai_rephrase(harmful_texts, lang, api_key)
    elif api_key and os.environ.get("CO_API_KEY"):
        return _cohere_rephrase(harmful_texts, lang, api_key)
    else:
        print("  WARNING: No API key found. Using stub harmless texts.")
        print("  Set OPENAI_API_KEY or CO_API_KEY and re-run for real harmless prompts.")
        return [f"[STUB] Tell me about {t[:40]}..." for t in harmful_texts]


def _openai_rephrase(texts: list[str], lang: str, api_key: str) -> list[str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    results = []
    for t in texts:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": REPHRASE_TEMPLATE.format(prompt=t)}],
            temperature=0.3,
            max_tokens=150,
        )
        results.append(resp.choices[0].message.content.strip())
    return results


def _cohere_rephrase(texts: list[str], lang: str, api_key: str) -> list[str]:
    import cohere
    co = cohere.Client(api_key)
    results = []
    for t in texts:
        resp = co.chat(
            message=REPHRASE_TEMPLATE.format(prompt=t),
            model="command-a-03-2025",
            temperature=0.3,
        )
        results.append(resp.text.strip())
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def build_for_lang(
    lang: str,
    raw_dir: Path,
    processed_dir: Path,
    prompts_dir: Path,
    n: int,
):
    print(f"\n--- {lang} ---")
    harmful_dir  = prompts_dir / lang
    harmless_dir = prompts_dir / lang

    # Collect harmful prompts — split XSafety 80/20 train/eval to avoid leakage
    xsafety_all = load_xsafety(raw_dir, lang)
    split_idx = int(len(xsafety_all) * 0.8)
    # Deterministic split: sort by id first so order is stable across runs
    xsafety_all_sorted = sorted(xsafety_all, key=lambda r: r["id"])
    xsafety_train = xsafety_all_sorted[:split_idx]
    xsafety_eval  = xsafety_all_sorted[split_idx:]

    harmful = []
    harmful += xsafety_train
    harmful += load_aya_redteam(raw_dir, lang)
    if lang != "en":
        harmful += load_pku_translated(processed_dir, lang)
        harmful += load_en_translated(processed_dir, lang)

    # Deduplicate harmful by id
    seen = set()
    deduped = []
    for r in harmful:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    harmful = deduped

    if len(harmful) < n:
        print(f"  WARNING: only {len(harmful)} harmful prompts (need {n}). "
              f"Consider adding more sources.")
    harmful = harmful[:n]
    save_jsonl(harmful, harmful_dir / "harmful.jsonl")

    # Save held-out XSafety eval split (used by 07_behavioral_baseline.py)
    save_jsonl(xsafety_eval, harmful_dir / "harmful_eval.jsonl")
    print(f"  XSafety split: {len(xsafety_train)} train / {len(xsafety_eval)} eval")

    # Generate / load harmless counterparts
    harmless_path = harmless_dir / "harmless.jsonl"
    if harmless_path.exists():
        harmless = load_jsonl(harmless_path)
        print(f"  Loaded {len(harmless)} existing harmless prompts.")
    else:
        # Seed from Aya Eval (already multilingual, no translation needed)
        seed_harmless = []
        seen_h = set()
        for r in load_aya_eval(raw_dir, lang):
            if r["id"] not in seen_h:
                seen_h.add(r["id"])
                seed_harmless.append(r)

        # Use Aya Eval only — native multilingual harmless, no LLM rephrase needed
        harmless = seed_harmless
        save_jsonl(harmless, harmless_path)
        print(f"  Harmless: {len(harmless)} from Aya Eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs",      nargs="+", default=LANG_CODES)
    parser.add_argument("--n-per-lang", type=int,  default=N_HARMFUL)
    parser.add_argument("--raw-dir",    default="data/raw")
    parser.add_argument("--proc-dir",   default="data/processed")
    parser.add_argument("--out-dir",    default="data/prompts")
    args = parser.parse_args()

    raw_dir       = ROOT / args.raw_dir
    processed_dir = ROOT / args.proc_dir
    prompts_dir   = ROOT / args.out_dir

    for lang in args.langs:
        build_for_lang(lang, raw_dir, processed_dir, prompts_dir, args.n_per_lang)

    print("\nDone. Next: scripts/05_extract_activations.py")


if __name__ == "__main__":
    main()
