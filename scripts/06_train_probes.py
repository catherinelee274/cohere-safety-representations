"""
Step 6 — Train probes on extracted activations, compute silhouette scores
          and cross-lingual direction alignment.

Reads:   data/processed/activations/{model}/{lang}/layer_{n}.npy
Writes:  results/probes/{model}/{lang}/layer_{n}_{probe_type}.pkl
         results/probes/summary.csv

Usage:
    python scripts/06_train_probes.py \\
        --model global --lang en      # single pair (sanity check)
    python scripts/06_train_probes.py --all
"""

import argparse
import json
import logging
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES
from config.models import MODEL_ORDER, MODELS
from src.run_metadata import model_meta, save_run_info
from src.probe import (
    ProbeResult,
    cosine_alignment,
    diffmean_probe,
    logreg_probe,
    silhouette_score_probe,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_acts_labels(acts_dir: Path, model_key: str, lang: str, layer: int):
    d = acts_dir / model_key / lang
    arr_path = d / f"layer_{layer}.npy"
    lbl_path = d / "labels.npy"
    if not arr_path.exists() or not lbl_path.exists():
        return None, None
    return np.load(arr_path), np.load(lbl_path)


def save_probe(result: ProbeResult, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"layer_{result.layer}_{result.probe_type}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(result, f)
    return fname


def load_probe(path: Path) -> ProbeResult:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── main logic ────────────────────────────────────────────────────────────────

def run_probes_for(
    model_key: str,
    lang: str,
    acts_dir: Path,
    probe_dir: Path,
) -> list[dict]:
    """Train both probe types for every layer. Return list of result rows."""
    from datetime import datetime, timezone
    cfg  = MODELS[model_key]
    rows = []
    run_at = datetime.now(timezone.utc).isoformat()

    for layer in cfg.probe_layers:
        acts, labels = load_acts_labels(acts_dir, model_key, lang, layer)
        if acts is None:
            logger.warning(f"Missing activations for {model_key}/{lang}/layer_{layer}")
            continue

        sil = silhouette_score_probe(acts, labels)

        for probe_fn, ptype in [(diffmean_probe, "diffmean"), (logreg_probe, "logreg")]:
            result = probe_fn(acts, labels,
                              model_key=model_key, lang=lang, layer=layer)
            out_dir = probe_dir / model_key / lang
            fpath   = save_probe(result, out_dir)

            row = {
                "model":        model_key,
                "model_hf_id":  cfg.hf_id,
                "model_name":   cfg.name,
                "lang":         lang,
                "layer":        layer,
                "probe_type":   ptype,
                "accuracy":     round(result.accuracy, 4),
                "auc":          round(result.auc, 4),
                "silhouette":   round(sil, 4),
                "run_at":       run_at,
            }
            if result.fold_accs:
                row["acc_std"] = round(float(np.std(result.fold_accs)), 4)
            rows.append(row)
            logger.info(f"  {model_key}/{lang}/layer{layer}/{ptype}: "
                        f"acc={row['accuracy']:.3f}  auc={row['auc']:.3f}  "
                        f"sil={row['silhouette']:.3f}")

    if rows:
        save_run_info(probe_dir / model_key / lang, {
            **model_meta(model_key),
            "script":  "06_train_probes",
            "lang":    lang,
            "layers":  cfg.probe_layers,
            "n_probes": len(rows),
        })
    return rows


def compute_alignment_matrix(probe_dir: Path, model_key: str, layer: int,
                              probe_type: str = "diffmean") -> pd.DataFrame:
    """
    For a given model and layer, compute pairwise cosine alignment of
    probe directions across languages.
    """
    dirs = {}
    for lang in LANG_CODES:
        p = probe_dir / model_key / lang / f"layer_{layer}_{probe_type}.pkl"
        if p.exists():
            result  = load_probe(p)
            dirs[lang] = result.direction

    langs  = sorted(dirs.keys())
    matrix = pd.DataFrame(index=langs, columns=langs, dtype=float)
    for l1, l2 in combinations(langs, 2):
        sim = cosine_alignment(dirs[l1], dirs[l2])
        matrix.loc[l1, l2] = sim
        matrix.loc[l2, l1] = sim
    for l in langs:
        matrix.loc[l, l] = 1.0

    return matrix


STAGE_ORDER = ["base", "base_sft", "base_dpo"]


def compute_cross_stage_alignment(
    probe_dir: Path,
    lang: str,
    layer: int,
    probe_type: str = "diffmean",
    stages: list[str] = STAGE_ORDER,
) -> list[dict]:
    """
    For a given language and layer, compute pairwise cosine alignment of the
    harmful probe direction across training stages (base → sft → dpo).
    """
    dirs = {}
    for stage in stages:
        p = probe_dir / stage / lang / f"layer_{layer}_{probe_type}.pkl"
        if p.exists():
            dirs[stage] = load_probe(p).direction

    rows = []
    available = [s for s in stages if s in dirs]
    for s1, s2 in combinations(available, 2):
        sim = cosine_alignment(dirs[s1], dirs[s2])
        rows.append({
            "lang":       lang,
            "layer":      layer,
            "probe_type": probe_type,
            "stage_a":    s1,
            "stage_b":    s2,
            "cosine_sim": round(sim, 4),
        })
        logger.info(f"  cross-stage [{probe_type}] {s1}↔{s2} "
                    f"lang={lang} layer={layer}: {sim:.3f}")
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     choices=list(MODELS.keys()), default=None)
    parser.add_argument("--lang",      choices=LANG_CODES,          default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--acts-dir",  default="data/processed/activations")
    parser.add_argument("--probe-dir", default="results/probes")
    parser.add_argument("--alignment", action="store_true",
                        help="Compute cross-lingual alignment matrices after probing")
    parser.add_argument("--cross-stage", action="store_true",
                        help="Compute harmful-direction alignment across training stages")
    args = parser.parse_args()

    acts_dir  = ROOT / args.acts_dir
    probe_dir = ROOT / args.probe_dir

    if args.all:
        pairs = [(m, l) for m in MODEL_ORDER for l in LANG_CODES]
    elif args.model and args.lang:
        pairs = [(args.model, args.lang)]
    elif args.model:
        pairs = [(args.model, l) for l in LANG_CODES]
    elif args.lang:
        pairs = [(m, args.lang) for m in MODEL_ORDER]
    else:
        pairs = [("global", "en")]
        logger.info("Default: sanity check on global × en")

    all_rows = []
    for model_key, lang in pairs:
        rows = run_probes_for(model_key, lang, acts_dir, probe_dir)
        all_rows.extend(rows)

    # Save summary CSV — merge with existing to avoid losing prior runs
    if all_rows:
        summary_path = ROOT / "results" / "probes" / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(all_rows)
        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            # Drop rows for models we just re-ran, then append fresh results
            existing = existing[~existing["model"].isin(new_df["model"].unique())]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved → {summary_path}")
        print(new_df.to_string(index=False))

    # Cross-stage alignment (harmful direction across base → sft → dpo)
    if args.cross_stage:
        print("\n--- Cross-stage direction alignment (harmful) ---")
        cs_rows = []
        processed_langs = set(l for _, l in pairs)
        for lang in processed_langs:
            layers = set()
            for stage in STAGE_ORDER:
                if stage in MODELS:
                    layers.update(MODELS[stage].probe_layers)
            for layer in sorted(layers):
                for ptype in ("diffmean", "logreg"):
                    cs_rows.extend(
                        compute_cross_stage_alignment(probe_dir, lang, layer, ptype)
                    )
        if cs_rows:
            cs_path = ROOT / "results" / "probes" / "cross_stage_alignment.csv"
            cs_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cs_rows).to_csv(cs_path, index=False)
            print(pd.DataFrame(cs_rows).to_string(index=False))
            logger.info(f"Cross-stage alignment → {cs_path}")

    # Cross-lingual alignment
    if args.alignment:
        print("\n--- Cross-lingual direction alignment ---")
        for model_key in MODEL_ORDER:
            cfg = MODELS[model_key]
            for layer in cfg.probe_layers:
                mat = compute_alignment_matrix(probe_dir, model_key, layer)
                if mat.empty:
                    continue
                align_path = (ROOT / "results" / "probes" /
                              f"alignment_{model_key}_layer{layer}.csv")
                mat.to_csv(align_path)
                print(f"\n{model_key} / layer {layer}:\n{mat.round(3).to_string()}")

    print("\nDone. Next: scripts/07_behavioral_baseline.py")


if __name__ == "__main__":
    main()
