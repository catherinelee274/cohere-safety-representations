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
    """Load XSafety prompts for a given language."""
    records = []
    # XSafety uses full language names or ISO codes depending on version
    for suffix in [lang, lang.upper()]:
        p = raw_dir / "xsafety" / f"xsafety_{suffix}.csv"
        if p.exists():
            df = pd.read_csv(p)
            col = next((c for c in df.columns if "prompt" in c.lower()), df.columns[0])
            for _, row in df.iterrows():
                text = str(row[col]).strip()
                if text:
                    records.append({
                        "id":     make_id(text, lang),
                        "text":   text,
                        "source": "xsafety",
                        "lang":   lang,
                        "label":  "harmful",
                    })
    return records


def load_aya_redteam(raw_dir: Path, lang: str) -> list[dict]:
    """Load Aya red-teaming prompts for a given language."""
    records = []
    for fname in (raw_dir / "aya_redteam").glob("*.csv"):
        try:
            df = pd.read_csv(fname)
        except Exception:
            continue
        lang_col = next((c for c in df.columns
                         if "lang" in c.lower() or "language" in c.lower()), None)
        if lang_col is None:
            continue
        sub = df[df[lang_col].str.lower().str.startswith(lang)]
        text_col = next((c for c in df.columns if "prompt" in c.lower()), df.columns[0])
        for _, row in sub.iterrows():
            text = str(row[text_col]).strip()
            if text:
                records.append({
                    "id":     make_id(text, lang),
                    "text":   text,
                    "source": "aya_redteam",
                    "lang":   lang,
                    "label":  "harmful",
                })
    return records


def load_litmus(raw_dir: Path, lang: str) -> tuple[list[dict], list[dict]]:
    """
    Load LITMUS harmful + harmless prompts (English only).
    Returns (harmful_records, harmless_records).
    """
    if lang != "en":
        return [], []

    harmful, harmless = [], []
    for label, bucket in [("unsafe", harmful), ("safe", harmless)]:
        p = raw_dir / "litmus" / f"litmus_{label}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        text_col = "input" if "input" in df.columns else df.columns[0]
        axiom_col = "axiom" if "axiom" in df.columns else None
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text or text == "nan":
                continue
            rec = {
                "id":     make_id(text, lang),
                "text":   text,
                "source": "litmus",
                "lang":   lang,
                "label":  "harmful" if label == "unsafe" else "harmless",
            }
            if axiom_col:
                rec["axiom"] = str(row[axiom_col])
            bucket.append(rec)

    return harmful, harmless


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
            model="command-r-plus",
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

    # Collect harmful prompts
    harmful = []
    harmful += load_xsafety(raw_dir, lang)
    harmful += load_aya_redteam(raw_dir, lang)
    if lang != "en":
        harmful += load_pku_translated(processed_dir, lang)

    # LITMUS: English-only; provides both harmful and pre-labelled harmless
    litmus_harmful, litmus_harmless = load_litmus(raw_dir, lang)
    harmful += litmus_harmful

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

    # Generate / load harmless counterparts
    harmless_path = harmless_dir / "harmless.jsonl"
    if harmless_path.exists():
        harmless = load_jsonl(harmless_path)
        print(f"  Loaded {len(harmless)} existing harmless prompts.")
    else:
        # For English: seed with LITMUS safe prompts before falling back to LLM rephrase
        seed_harmless = []
        seen_h = set()
        for r in litmus_harmless:
            if r["id"] not in seen_h:
                seen_h.add(r["id"])
                seed_harmless.append(r)

        n_still_needed = max(0, n - len(seed_harmless))
        if n_still_needed > 0:
            texts = [r["text"] for r in harmful[:n_still_needed]]
            rephrased = generate_harmless_batch(texts, lang)
            llm_harmless = [
                {
                    "id":     make_id(r, lang),
                    "text":   r,
                    "source": "llm_rephrase",
                    "lang":   lang,
                    "label":  "harmless",
                }
                for r in rephrased
            ]
        else:
            llm_harmless = []

        harmless = (seed_harmless + llm_harmless)[:n]
        save_jsonl(harmless, harmless_path)

        n_litmus = sum(1 for r in harmless if r["source"] == "litmus")
        n_llm    = sum(1 for r in harmless if r["source"] == "llm_rephrase")
        print(f"  Harmless: {n_litmus} from LITMUS, {n_llm} LLM-rephrased")
        if n_llm > 0:
            print(f"  NOTE: Spot-check LLM-rephrased entries in {harmless_path}")


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
