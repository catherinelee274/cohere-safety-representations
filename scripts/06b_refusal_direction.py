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

    # Checkpoint trajectory: refusal direction at every saved checkpoint
    python scripts/06b_refusal_direction.py --checkpoints-for base_sft_n5000
    python scripts/06b_refusal_direction.py --checkpoints-for base_sft_n5000 --lang en

Checkpoint mode requires activations and behavioral responses to be extracted
for each checkpoint first:
    python scripts/05_extract_activations.py --model base_sft_n5000_ckpt500 --lang en
    python scripts/07_behavioral_baseline.py --model base_sft_n5000_ckpt500 --lang en
(repeat for each checkpoint step)
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
from config.models import MODEL_ORDER, MODELS, get_checkpoint_configs
from src.probe import ProbeResult, cosine_alignment, diffmean_probe

# Activation roots, searched in order (eval split preferred — IDs match behavioral responses)
_ACTS_FALLBACK = [
    ROOT / "data" / "processed" / "activations_eval",
    ROOT / "data" / "processed" / "activations",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Training stages for cross-stage comparison (ordered)
STAGE_ORDER = ["base", "base_sft", "base_dpo"]


# ── data loading ──────────────────────────────────────────────────────────────

def _resolve_acts_dir(primary_acts_dir: Path, model_key: str, lang: str) -> Path | None:
    """Return the activation directory with the best ID overlap against behavioral responses."""
    candidates = [primary_acts_dir / model_key / lang] + [
        r / model_key / lang for r in _ACTS_FALLBACK
        if (r / model_key / lang) != (primary_acts_dir / model_key / lang)
    ]
    for d in candidates:
        if d.exists() and (d / "labels.npy").exists():
            return d
    return None


def load_refusal_labels(
    acts_dir: Path,
    behavioral_dir: Path,
    model_key: str,
    lang: str,
) -> tuple[np.ndarray, np.ndarray, Path] | tuple[None, None, None]:
    """
    Returns (harmful_indices, refusal_labels, resolved_acts_dir) where:
      harmful_indices    — row indices into layer_{n}.npy for harmful prompts
      refusal_labels     — 1 if model refused, 0 if complied, aligned to harmful_indices
      resolved_acts_dir  — the activation dir whose IDs matched the behavioral responses
    Returns (None, None, None) if data is missing.
    """
    resp_path = behavioral_dir / f"responses_{model_key}_{lang}.jsonl"
    if not resp_path.exists():
        logger.warning(f"Missing: {resp_path}")
        return None, None, None

    # Find the activation directory whose prompt IDs best match the behavioral responses
    with open(resp_path) as f:
        beh_records = [json.loads(l) for l in f if l.strip()]
    beh_ids = {r["id"] for r in beh_records}

    best_dir = None
    best_overlap = 0
    candidates = [acts_dir / model_key / lang] + [
        r / model_key / lang for r in _ACTS_FALLBACK
    ]
    seen = set()
    for d in candidates:
        key = str(d)
        if key in seen or not d.exists() or not (d / "labels.npy").exists():
            seen.add(key)
            continue
        seen.add(key)
        try:
            with open(d / "prompts.jsonl") as f:
                ids = {json.loads(l)["id"] for l in f if l.strip()}
            overlap = len(ids & beh_ids)
            if overlap > best_overlap:
                best_overlap = overlap
                best_dir = d
        except Exception:
            continue

    if best_dir is None or best_overlap == 0:
        logger.warning(f"{model_key}/{lang}: no activation dir with matching IDs found")
        return None, None, None

    act_base = best_dir
    labels_path  = act_base / "labels.npy"
    prompts_path = act_base / "prompts.jsonl"

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
        return None, None, None

    logger.info(f"{model_key}/{lang}: {n_refused} refused / {n_complied} complied "
                f"[acts: {best_dir}]")
    return harmful_indices, refusal_labels, best_dir


def load_acts(acts_dir: Path, model_key: str, lang: str, layer: int,
              resolved_dir: Path | None = None) -> np.ndarray | None:
    """Load layer activations, preferring resolved_dir if provided."""
    dirs_to_try = []
    if resolved_dir is not None:
        dirs_to_try.append(resolved_dir)
    dirs_to_try.append(acts_dir / model_key / lang)
    dirs_to_try += [r / model_key / lang for r in _ACTS_FALLBACK]
    seen = set()
    for d in dirs_to_try:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        p = d / f"layer_{layer}.npy"
        if p.exists():
            return np.load(p)
    return None


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
    harmful_indices, refusal_labels, resolved_acts_dir = load_refusal_labels(
        acts_dir, behavioral_dir, model_key, lang
    )
    if harmful_indices is None:
        return []

    cfg  = MODELS[model_key]
    rows = []

    for layer in cfg.probe_layers:
        acts = load_acts(acts_dir, model_key, lang, layer, resolved_dir=resolved_acts_dir)
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

def compute_checkpoint_trajectory(
    base_key: str,
    lang: str,
    probe_dir: Path,
    acts_dir: Path,
    behavioral_dir: Path,
) -> list[dict]:
    """
    Load or train refusal probes for every checkpoint of *base_key*, then compute
    cosine alignment of each checkpoint's refusal direction against the final
    (fully-trained) model's refusal direction.  Returns one row per
    (checkpoint_key, lang, layer).
    """
    ckpt_list = get_checkpoint_configs(base_key)
    if not ckpt_list:
        logger.warning(f"No checkpoints found for {base_key!r}")
        return []

    # Collect all (step, key, cfg) with their probe results
    ckpt_probes: dict[str, dict[int, ProbeResult]] = {}  # key → layer → ProbeResult
    for ckpt_key, ckpt_cfg in ckpt_list:
        MODELS[ckpt_key] = ckpt_cfg  # register temporarily so helpers can find it
        rows = run_refusal_probes_for(ckpt_key, lang, acts_dir, behavioral_dir, probe_dir)
        if not rows:
            continue
        layer_probes: dict[int, ProbeResult] = {}
        for layer in ckpt_cfg.probe_layers:
            p = probe_dir / ckpt_key / lang / f"layer_{layer}_refusal.pkl"
            result = load_probe(p)
            if result is not None:
                layer_probes[layer] = result
        ckpt_probes[ckpt_key] = layer_probes

    # Also load the final model's refusal directions as the reference
    final_probes: dict[int, ProbeResult] = {}
    cfg_final = MODELS[base_key]
    for layer in cfg_final.probe_layers:
        p = probe_dir / base_key / lang / f"layer_{layer}_refusal.pkl"
        result = load_probe(p)
        if result is not None:
            final_probes[layer] = result

    trajectory_rows = []
    for ckpt_key, layer_probes in ckpt_probes.items():
        step = int(ckpt_key.split("_ckpt")[-1])
        for layer, result in layer_probes.items():
            row = {
                "base_model":   base_key,
                "checkpoint":   ckpt_key,
                "step":         step,
                "lang":         lang,
                "layer":        layer,
                "accuracy":     round(result.accuracy, 4),
                "auc":          round(result.auc, 4),
                "cosine_vs_final": None,
            }
            if layer in final_probes:
                row["cosine_vs_final"] = round(
                    cosine_alignment(result.direction, final_probes[layer].direction), 4
                )
            trajectory_rows.append(row)

    return trajectory_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",             choices=list(MODELS.keys()), default=None)
    parser.add_argument("--lang",              choices=LANG_CODES,          default=None)
    parser.add_argument("--all",               action="store_true")
    parser.add_argument("--checkpoints-for",   default=None, metavar="BASE_KEY",
                        help="Train refusal probes at every checkpoint of BASE_KEY "
                             "and write a step-level trajectory CSV")
    parser.add_argument("--cross-stage",       action="store_true",
                        help="Compute cross-stage direction alignment after probing")
    parser.add_argument("--acts-dir",          default="data/processed/activations")
    parser.add_argument("--behavioral-dir",    default="results/behavioral")
    parser.add_argument("--probe-dir",         default="results/probes")
    args = parser.parse_args()

    acts_dir       = ROOT / args.acts_dir
    behavioral_dir = ROOT / args.behavioral_dir
    probe_dir      = ROOT / args.probe_dir

    # ── checkpoint trajectory mode ────────────────────────────────────────────
    if args.checkpoints_for:
        base_key = args.checkpoints_for
        if base_key not in MODELS:
            parser.error(f"Unknown base model key: {base_key!r}")
        langs = [args.lang] if args.lang else list(LANG_CODES)
        traj_rows = []
        for lang in langs:
            traj_rows.extend(
                compute_checkpoint_trajectory(
                    base_key, lang, probe_dir, acts_dir, behavioral_dir
                )
            )
        if traj_rows:
            traj_path = probe_dir / f"checkpoint_trajectory_{base_key}.csv"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            import pandas as _pd
            _pd.DataFrame(traj_rows).to_csv(traj_path, index=False)
            logger.info(f"Checkpoint trajectory → {traj_path}")
            print(f"\n{_pd.DataFrame(traj_rows).to_string(index=False)}")
        print("\nDone.")
        return

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
