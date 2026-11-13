"""
Step 21 — Train diffmean + logreg probes for checkpoint and n-variant sweep models.

These models are not registered in config/models.py MODELS dict (checkpoints are
dynamic; n-variants are registered but use fixed probe_layers). This script scans
activations/ for any directory matching the sweep patterns and trains probes on
whatever layer files exist.

Writes:
  results/probes/{model_key}/{lang}/layer_{n}_diffmean.pkl
  results/probes/{model_key}/{lang}/layer_{n}_logreg.pkl
  results/probes/sweep_summary.csv

Run:
  python scripts/21_train_sweep_probes.py                    # all sweep models
  python scripts/21_train_sweep_probes.py --pattern ckpt     # only checkpoints
  python scripts/21_train_sweep_probes.py --pattern nvar     # only n-variants
  python scripts/21_train_sweep_probes.py --model base_sft_ckpt96  # single model
"""

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probe import diffmean_probe, logreg_probe, silhouette_score_probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACTS_DIR  = ROOT / "data" / "processed" / "activations"
PROBE_DIR = ROOT / "results" / "probes"

CKPT_RE = re.compile(r"^(base_sft|base_dpo)_ckpt\d+$")
NVAR_RE = re.compile(r"^base_sft_n\d+$")

SKIP_LANGS = {"fil"}


def discover_models(pattern: str | None = None) -> list[str]:
    models = []
    for d in sorted(ACTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        is_ckpt = bool(CKPT_RE.match(name))
        is_nvar = bool(NVAR_RE.match(name))
        if not (is_ckpt or is_nvar):
            continue
        if pattern == "ckpt" and not is_ckpt:
            continue
        if pattern == "nvar" and not is_nvar:
            continue
        models.append(name)
    return models


def get_available_layers(model_dir: Path, lang: str) -> list[int]:
    lang_dir = model_dir / lang
    if not lang_dir.exists():
        return []
    layers = []
    for f in lang_dir.glob("layer_*.npy"):
        m = re.match(r"layer_(\d+)\.npy", f.name)
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def load_acts_labels(model_key: str, lang: str, layer: int):
    d = ACTS_DIR / model_key / lang
    acts_path   = d / f"layer_{layer}.npy"
    labels_path = d / "labels.npy"
    if not acts_path.exists() or not labels_path.exists():
        return None, None
    acts   = np.load(acts_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int8)
    return acts, labels


def save_probe(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"layer_{result.layer}_{result.probe_type}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(result, f)


def already_trained(model_key: str, lang: str, layer: int) -> bool:
    d = PROBE_DIR / model_key / lang
    return (d / f"layer_{layer}_diffmean.pkl").exists() and \
           (d / f"layer_{layer}_logreg.pkl").exists()


def run_probes_for(model_key: str, lang: str) -> list[dict]:
    layers = get_available_layers(ACTS_DIR / model_key, lang)
    if not layers:
        return []

    rows = []
    for layer in layers:
        if already_trained(model_key, lang, layer):
            logger.info(f"  {model_key}/{lang}/layer{layer}: already trained, skipping")
            continue

        acts, labels = load_acts_labels(model_key, lang, layer)
        if acts is None:
            continue
        if len(np.unique(labels)) < 2:
            logger.warning(f"  {model_key}/{lang}/layer{layer}: only one class, skipping")
            continue

        sil = silhouette_score_probe(acts, labels)
        out_dir = PROBE_DIR / model_key / lang

        for probe_fn, ptype in [(diffmean_probe, "diffmean"), (logreg_probe, "logreg")]:
            result = probe_fn(acts, labels, model_key=model_key, lang=lang, layer=layer)
            save_probe(result, out_dir)
            row = {
                "model":      model_key,
                "lang":       lang,
                "layer":      layer,
                "probe_type": ptype,
                "accuracy":   round(result.accuracy, 4),
                "auc":        round(result.auc, 4),
                "silhouette": round(sil, 4),
            }
            rows.append(row)
            logger.info(
                f"  {model_key}/{lang}/layer{layer}/{ptype}: "
                f"acc={result.accuracy:.3f}  auc={result.auc:.3f}  sil={sil:.3f}"
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None, help="Single model key to run")
    parser.add_argument("--pattern", choices=["ckpt", "nvar"], default=None)
    parser.add_argument("--force",   action="store_true", help="Re-train even if probes exist")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = discover_models(args.pattern)

    if not models:
        logger.error("No matching models found in %s", ACTS_DIR)
        return

    logger.info("Models to process: %s", models)

    all_rows = []
    for model_key in models:
        model_dir = ACTS_DIR / model_key
        langs = [d.name for d in sorted(model_dir.iterdir())
                 if d.is_dir() and d.name not in SKIP_LANGS]
        logger.info(f"=== {model_key}: langs={langs}")
        for lang in langs:
            rows = run_probes_for(model_key, lang)
            all_rows.extend(rows)

    if not all_rows:
        logger.info("Nothing new trained (all probes already exist).")
        return

    new_df = pd.DataFrame(all_rows)
    out_csv = PROBE_DIR / "sweep_summary.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Drop rows for models we just ran (may have re-run subset of layers)
        existing = existing[~existing["model"].isin(new_df["model"].unique())]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_csv, index=False)
    logger.info(f"Saved → {out_csv}")
    print(new_df.to_string(index=False))


if __name__ == "__main__":
    main()
