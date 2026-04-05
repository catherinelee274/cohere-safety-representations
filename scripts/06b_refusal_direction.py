"""
Step 6b — Refusal direction probes + cross-stage alignment.

For each model × language × layer:
  1. Load harmful-input activations (already extracted by script 05).
  2. Join to script-07 behavioral outputs to get a refusal label
     (1 = model refused, 0 = model complied) per harmful prompt.
  3. Train a DiffMean "refusal" probe on those activations.
  4. Compute cosine alignment between:
       • harmful_direction vs refusal_direction  (within model/layer)
       • harmful_direction across training stages (base → sft → dpo)
       • refusal_direction across training stages

Reads:
    data/processed/activations/{model}/{lang}/layer_{n}.npy
    data/processed/activations/{model}/{lang}/labels.npy
    data/processed/activations/{model}/{lang}/prompts.jsonl
    results/behavioral/responses_{model}_{lang}.jsonl   (from script 07)

Writes:
    results/probes/{model}/{lang}/layer_{n}_refusal.pkl
    results/probes/refusal_summary.csv
    results/probes/cross_stage_alignment.csv

Usage:
    python scripts/06b_refusal_direction.py --model base --lang en
    python scripts/06b_refusal_direction.py --all
    python scripts/06b_refusal_direction.py --all --cross-stage
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
from src.probe import ProbeResult, cosine_alignment, diffmean_probe

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Training stages for cross-stage comparison (ordered)
STAGE_ORDER = ["base", "base_sft", "base_dpo"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_refusal_labels(
    acts_dir: Path,
    behavioral_dir: Path,
    model_key: str,
    lang: str,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Returns (harmful_indices, refusal_labels) where:
      harmful_indices  — row indices into layer_{n}.npy that correspond to harmful prompts
      refusal_labels   — 1 if model refused, 0 if complied, aligned to harmful_indices
    Returns (None, None) if data is missing.
    """
    act_base = acts_dir / model_key / lang
    labels_path  = act_base / "labels.npy"
    prompts_path = act_base / "prompts.jsonl"
    resp_path    = behavioral_dir / f"responses_{model_key}_{lang}.jsonl"

    for p in (labels_path, prompts_path, resp_path):
        if not p.exists():
            logger.warning(f"Missing: {p}")
            return None, None

    labels = np.load(labels_path)

    with open(prompts_path) as f:
        all_prompts = [json.loads(l) for l in f if l.strip()]

    with open(resp_path) as f:
        responses = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}

    harmful_indices = np.where(labels == 1)[0]
    harmful_prompts = [all_prompts[i] for i in harmful_indices]

    refusal_labels = []
    missing = 0
    for p in harmful_prompts:
        resp = responses.get(p["id"])
        if resp is None:
            missing += 1
            refusal_labels.append(-1)   # sentinel; dropped below
        else:
            refusal_labels.append(int(resp["heuristic_refuse"]))

    refusal_labels = np.array(refusal_labels, dtype=np.int8)

    if missing:
        logger.warning(f"{model_key}/{lang}: {missing} harmful prompts had no "
                       "behavioral response — dropping them from refusal probe")
        keep = refusal_labels != -1
        harmful_indices  = harmful_indices[keep]
        refusal_labels   = refusal_labels[keep]

    n_refused  = refusal_labels.sum()
    n_complied = len(refusal_labels) - n_refused
    if n_refused < 5 or n_complied < 5:
        logger.warning(f"{model_key}/{lang}: too few refused ({n_refused}) or "
                       f"complied ({n_complied}) samples — skipping refusal probe")
        return None, None

    logger.info(f"{model_key}/{lang}: {n_refused} refused / {n_complied} complied")
    return harmful_indices, refusal_labels


def load_acts(acts_dir: Path, model_key: str, lang: str, layer: int) -> np.ndarray | None:
    p = acts_dir / model_key / lang / f"layer_{layer}.npy"
    if not p.exists():
        return None
    return np.load(p)


# ── probe helpers ─────────────────────────────────────────────────────────────

def save_probe(result: ProbeResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"layer_{result.layer}_{result.probe_type}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(result, f)
    return fname


def load_probe(path: Path) -> ProbeResult | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── refusal probe ─────────────────────────────────────────────────────────────

def run_refusal_probes_for(
    model_key: str,
    lang: str,
    acts_dir: Path,
    behavioral_dir: Path,
    probe_dir: Path,
) -> list[dict]:
    harmful_indices, refusal_labels = load_refusal_labels(
        acts_dir, behavioral_dir, model_key, lang
    )
    if harmful_indices is None:
        return []

    cfg  = MODELS[model_key]
    rows = []

    for layer in cfg.probe_layers:
        acts = load_acts(acts_dir, model_key, lang, layer)
        if acts is None:
            logger.warning(f"Missing activations: {model_key}/{lang}/layer_{layer}")
            continue

        acts_harmful = acts[harmful_indices]   # only harmful-input rows

        result = diffmean_probe(
            acts_harmful, refusal_labels,
            model_key=model_key, lang=lang, layer=layer,
        )
        # Override probe_type so it's saved as a distinct file
        result = ProbeResult(
            model_key=result.model_key,
            lang=result.lang,
            layer=result.layer,
            probe_type="refusal",
            direction=result.direction,
            accuracy=result.accuracy,
            auc=result.auc,
            scores=result.scores,
        )

        out_dir = probe_dir / model_key / lang
        save_probe(result, out_dir)

        row = {
            "model":      model_key,
            "lang":       lang,
            "layer":      layer,
            "probe_type": "refusal",
            "accuracy":   round(result.accuracy, 4),
            "auc":        round(result.auc, 4),
        }
        rows.append(row)
        logger.info(f"  {model_key}/{lang}/layer{layer}/refusal: "
                    f"acc={row['accuracy']:.3f}  auc={row['auc']:.3f}")

    return rows


# ── cross-stage alignment ─────────────────────────────────────────────────────

def compute_cross_stage_alignment(
    probe_dir: Path,
    lang: str,
    layer: int,
    stages: list[str] = STAGE_ORDER,
) -> list[dict]:
    """
    For a given language and layer, load harmful (diffmean) and refusal probes
    for each stage and compute pairwise cosine alignment between stages.
    """
    rows = []

    for probe_type in ("diffmean", "refusal"):
        dirs = {}
        for stage in stages:
            p = probe_dir / stage / lang / f"layer_{layer}_{probe_type}.pkl"
            result = load_probe(p)
            if result is not None:
                dirs[stage] = result.direction

        available = [s for s in stages if s in dirs]
        for s1, s2 in combinations(available, 2):
            sim = cosine_alignment(dirs[s1], dirs[s2])
            rows.append({
                "lang":        lang,
                "layer":       layer,
                "probe_type":  probe_type,
                "stage_a":     s1,
                "stage_b":     s2,
                "cosine_sim":  round(sim, 4),
            })
            logger.info(f"  cross-stage [{probe_type}] {s1}↔{s2} "
                        f"lang={lang} layer={layer}: {sim:.3f}")

    return rows


def compute_harmful_vs_refusal_alignment(
    probe_dir: Path,
    model_key: str,
    lang: str,
    layer: int,
) -> dict | None:
    """
    Cosine alignment between the harmful direction and refusal direction
    for the same model, language, and layer.
    """
    harmful_p  = probe_dir / model_key / lang / f"layer_{layer}_diffmean.pkl"
    refusal_p  = probe_dir / model_key / lang / f"layer_{layer}_refusal.pkl"

    harmful  = load_probe(harmful_p)
    refusal  = load_probe(refusal_p)

    if harmful is None or refusal is None:
        return None

    sim = cosine_alignment(harmful.direction, refusal.direction)
    return {
        "model":      model_key,
        "lang":       lang,
        "layer":      layer,
        "cosine_sim": round(sim, 4),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          choices=list(MODELS.keys()), default=None)
    parser.add_argument("--lang",           choices=LANG_CODES,          default=None)
    parser.add_argument("--all",            action="store_true")
    parser.add_argument("--cross-stage",    action="store_true",
                        help="Compute cross-stage direction alignment after probing")
    parser.add_argument("--acts-dir",       default="data/processed/activations")
    parser.add_argument("--behavioral-dir", default="results/behavioral")
    parser.add_argument("--probe-dir",      default="results/probes")
    args = parser.parse_args()

    acts_dir       = ROOT / args.acts_dir
    behavioral_dir = ROOT / args.behavioral_dir
    probe_dir      = ROOT / args.probe_dir

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

    # ── 1. Train refusal probes ───────────────────────────────────────────────
    all_rows = []
    for model_key, lang in pairs:
        rows = run_refusal_probes_for(
            model_key, lang, acts_dir, behavioral_dir, probe_dir
        )
        all_rows.extend(rows)

    if all_rows:
        refusal_summary_path = probe_dir / "refusal_summary.csv"
        refusal_summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(refusal_summary_path, index=False)
        logger.info(f"Refusal summary → {refusal_summary_path}")

    # ── 2. Harmful vs refusal alignment (within model) ───────────────────────
    hvr_rows = []
    for model_key, lang in pairs:
        cfg = MODELS[model_key]
        for layer in cfg.probe_layers:
            row = compute_harmful_vs_refusal_alignment(
                probe_dir, model_key, lang, layer
            )
            if row:
                hvr_rows.append(row)
                logger.info(f"  harmful↔refusal {model_key}/{lang}/layer{layer}: "
                            f"{row['cosine_sim']:.3f}")

    if hvr_rows:
        hvr_path = probe_dir / "harmful_vs_refusal_alignment.csv"
        pd.DataFrame(hvr_rows).to_csv(hvr_path, index=False)
        logger.info(f"Harmful-vs-refusal alignment → {hvr_path}")

    # ── 3. Cross-stage alignment ──────────────────────────────────────────────
    if args.cross_stage:
        cs_rows = []
        # Collect unique (lang, layer) pairs from what we just processed
        processed_langs  = set(l for _, l in pairs)
        processed_models = set(m for m, _ in pairs)
        for lang in processed_langs:
            # Use union of all layers across stages we care about
            layers = set()
            for stage in STAGE_ORDER:
                if stage in processed_models and stage in MODELS:
                    layers.update(MODELS[stage].probe_layers)
            for layer in sorted(layers):
                cs_rows.extend(
                    compute_cross_stage_alignment(probe_dir, lang, layer)
                )

        if cs_rows:
            cs_path = probe_dir / "cross_stage_alignment.csv"
            pd.DataFrame(cs_rows).to_csv(cs_path, index=False)
            logger.info(f"Cross-stage alignment → {cs_path}")
            df = pd.DataFrame(cs_rows)
            print(f"\nCross-stage alignment:\n{df.to_string(index=False)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
