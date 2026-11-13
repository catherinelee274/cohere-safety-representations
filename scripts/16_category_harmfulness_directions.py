"""
Step 16 — Per-category harmfulness directions (replicates Zhao et al. 2025 Appendix D).

Tests whether the harmfulness direction in representation space is category-specific
(different XSafety categories → different directions) or universal.

For each model × language × layer:
  1. Compute per-category DiffMean direction:
       d_cat = μ(harmful_in_cat) − μ(harmless_all)
  2. Build N_cat × N_cat cross-category cosine similarity matrix
  3. Compare each category's harmfulness direction to the refusal direction

Also computes the refusal direction from behavioral labels:
  d_refusal = μ(refused) − μ(accepted)
and reports cos_sim(d_cat, d_refusal) per category.

Uses activations from data/processed/activations_eval/ (aggregation=last).

Run:
  python scripts/16_category_harmfulness_directions.py --model base --lang en
  python scripts/16_category_harmfulness_directions.py --model base --all
  python scripts/16_category_harmfulness_directions.py --all

Writes:
  results/probes/category_directions_{model}_{lang}_layer{L}.csv
  results/figures/fig_category_sim_{model}_{lang}_layer{L}.png
  results/probes/category_refusal_cosine.csv   (all models × langs × layers)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES
from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR  = ROOT / "results" / "figures"
PROB_DIR = ROOT / "results" / "probes"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROB_DIR.mkdir(parents=True, exist_ok=True)

ACTS_ROOT = ROOT / "data" / "processed" / "activations_eval"
RESP_DIR  = ROOT / "results" / "behavioral"

import re as _re

def _norm_cat(cat: str) -> str:
    """Strip language suffix (_en, _n, _fr, etc.) for cross-language comparison."""
    return _re.sub(r"_[a-z]+$", "", cat)

SHORT_CAT = {
    "Crimes_And_Illegal_Activities_en": "Crimes",
    "Crimes_And_Illegal_Activities":    "Crimes",
    "Ethics_And_Morality_n":            "Ethics",
    "Ethics_And_Morality":              "Ethics",
    "Goal_Hijacking_n":                 "GoalHij",
    "Goal_Hijacking":                   "GoalHij",
    "Inquiry_With_Unsafe_Opinion_n":    "UnsafeOp",
    "Inquiry_With_Unsafe_Opinion":      "UnsafeOp",
    "Insult_n":                         "Insult",
    "Insult":                           "Insult",
    "Mental_Health_n":                  "MentalH",
    "Mental_Health":                    "MentalH",
    "Physical_Harm_n":                  "PhysHarm",
    "Physical_Harm":                    "PhysHarm",
    "Privacy_And_Property_n":           "Privacy",
    "Privacy_And_Property":             "Privacy",
    "Prompt_Leaking_n":                 "PromptLk",
    "Prompt_Leaking":                   "PromptLk",
    "Reverse_Exposure_n":               "RevExp",
    "Reverse_Exposure":                 "RevExp",
    "Role_Play_Instruction_n":          "RolePlay",
    "Role_Play_Instruction":            "RolePlay",
    "Unfairness_And_Discrimination_n":  "Unfair",
    "Unfairness_And_Discrimination":    "Unfair",
    "Unsafe_Instruction_Topic_n":       "UnsafeTp",
    "Unsafe_Instruction_Topic":         "UnsafeTp",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-8 or bn < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def load_prompts(model_key: str, lang: str) -> list[dict]:
    p = ACTS_ROOT / model_key / lang / "prompts.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_acts(model_key: str, lang: str, layer: int) -> np.ndarray | None:
    p = ACTS_ROOT / model_key / lang / f"layer_{layer}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def load_refusal_labels(model_key: str, lang: str) -> dict[str, bool]:
    path = RESP_DIR / f"responses_{model_key}_{lang}.jsonl"
    if not path.exists():
        return {}
    labels = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels[d["id"]] = bool(d.get("heuristic_refuse", False))
    return labels


# ── analysis ─────────────────────────────────────────────────────────────────

def analyse_layer(
    acts: np.ndarray,
    prompts: list[dict],
    refusal_labels: dict[str, bool],
) -> tuple[dict[str, np.ndarray], np.ndarray | None, list[dict]]:
    """
    Returns:
        cat_dirs: {category → DiffMean direction vector}
        refusal_dir: μ(refused) − μ(accepted), or None
        rows: list of per-category stats dicts
    """
    id_to_idx = {p["id"]: i for i, p in enumerate(prompts)}

    # Harmless indices
    harmless_idx = [i for i, p in enumerate(prompts) if p.get("label") == "harmless"]
    if not harmless_idx:
        return {}, None, []
    mu_harmless = acts[harmless_idx].mean(0)

    # Discover categories from data (handles both suffixed and bare names)
    present_cats = sorted({
        p.get("category", "")
        for p in prompts
        if p.get("label") == "harmful" and p.get("category")
    })

    # Per-category directions
    cat_dirs: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for cat in present_cats:
        cat_idx = [i for i, p in enumerate(prompts) if p.get("category") == cat]
        if len(cat_idx) < 3:
            continue
        mu_cat = acts[cat_idx].mean(0)
        d = mu_cat - mu_harmless
        cat_dirs[cat] = d
        rows.append({"category": cat, "n": len(cat_idx)})

    # Refusal direction (from behavioral labels)
    refused_idx  = [id_to_idx[pid] for pid, ref in refusal_labels.items() if ref     and pid in id_to_idx]
    accepted_idx = [id_to_idx[pid] for pid, ref in refusal_labels.items() if not ref and pid in id_to_idx]
    refusal_dir = None
    if len(refused_idx) >= 3 and len(accepted_idx) >= 3:
        refusal_dir = acts[refused_idx].mean(0) - acts[accepted_idx].mean(0)

    return cat_dirs, refusal_dir, rows


def build_cosine_matrix(cat_dirs: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    cats = list(cat_dirs.keys())
    n = len(cats)
    mat = np.full((n, n), np.nan)
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            mat[i, j] = cosine(cat_dirs[ci], cat_dirs[cj])
    return mat, cats


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_cosine_matrix(mat: np.ndarray, cats: list[str], title: str, out: Path):
    labels = [SHORT_CAT.get(c, c) for c in cats]
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title, fontsize=9)
    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=5,
                        color="black" if abs(mat[i, j]) < 0.6 else "white")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_refusal_cosines(rows_df: pd.DataFrame, title: str, out: Path):
    """Bar plot: cos_sim(category_harm_dir, refusal_dir) per category."""
    df = rows_df.dropna(subset=["cos_refusal"]).sort_values("cos_refusal", ascending=False)
    if df.empty:
        return
    labels = [SHORT_CAT.get(c, c) for c in df["category"]]
    colors = ["#EE4444" if v >= 0 else "#4477AA" for v in df["cos_refusal"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, df["cos_refusal"].values, color=colors)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("cos_sim(category_harm_dir, refusal_dir)")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── per model / language ──────────────────────────────────────────────────────

def analyse_model_lang(model_key: str, lang: str) -> list[dict]:
    cfg = MODELS[model_key]
    prompts = load_prompts(model_key, lang)
    if not prompts:
        logger.warning(f"No prompts for {model_key}/{lang}")
        return []
    refusal_labels = load_refusal_labels(model_key, lang)
    logger.info(f"{model_key}/{lang}: {len(prompts)} prompts, "
                f"{len(refusal_labels)} with refusal labels")

    all_rows = []
    for layer in cfg.probe_layers:
        acts = load_acts(model_key, lang, layer)
        if acts is None:
            logger.debug(f"  Missing layer {layer}")
            continue

        cat_dirs, refusal_dir, base_rows = analyse_layer(acts, prompts, refusal_labels)
        if not cat_dirs:
            continue

        mat, cats = build_cosine_matrix(cat_dirs)

        # Save cosine matrix CSV
        mat_df = pd.DataFrame(mat, index=cats, columns=cats)
        mat_csv = PROB_DIR / f"category_directions_{model_key}_{lang}_layer{layer}.csv"
        mat_df.to_csv(mat_csv)

        # Plot similarity matrix
        model_name = cfg.name
        lang_name  = LANG_NAMES.get(lang, lang)
        title = f"Cross-category harmfulness directions\n{model_name} / {lang_name} / Layer {layer}"
        plot_cosine_matrix(
            mat, cats, title,
            FIG_DIR / f"fig_category_sim_{model_key}_{lang}_layer{layer}.png",
        )

        # Build per-category rows with refusal cosine
        cat_to_n = {r["category"]: r["n"] for r in base_rows}
        for cat, d in cat_dirs.items():
            cos_ref = cosine(d, refusal_dir) if refusal_dir is not None else float("nan")
            cat_list = list(cat_dirs.keys())
            other_cos = [
                cosine(d, cat_dirs[c2])
                for c2 in cat_list if c2 != cat
            ]
            mean_cross = float(np.nanmean(other_cos)) if other_cos else float("nan")
            all_rows.append({
                "model":      model_key,
                "lang":       lang,
                "layer":      layer,
                "category":   cat,
                "category_norm": _norm_cat(cat),
                "n":          cat_to_n.get(cat, 0),
                "cos_refusal": round(cos_ref,   4),
                "mean_cross_cat_cos": round(mean_cross, 4),
            })

        # Plot refusal alignment bars
        rows_df = pd.DataFrame(all_rows).query(
            f"model == '{model_key}' and lang == '{lang}' and layer == {layer}"
        )
        if not rows_df.empty and refusal_dir is not None:
            plot_refusal_cosines(
                rows_df, title,
                FIG_DIR / f"fig_category_refusal_cos_{model_key}_{lang}_layer{layer}.png",
            )

    return all_rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--lang",  choices=LANG_CODES,          default=None)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    if args.all and args.model:
        models = [args.model]
        langs  = LANG_CODES
    elif args.all:
        models = list(MODELS.keys())
        langs  = LANG_CODES
    elif args.model and args.lang:
        models = [args.model]
        langs  = [args.lang]
    elif args.model:
        models = [args.model]
        langs  = LANG_CODES
    else:
        models = ["base"]
        langs  = ["en"]
        logger.info("Defaulting to base × en")

    all_rows = []
    for model_key in models:
        for lang in langs:
            rows = analyse_model_lang(model_key, lang)
            all_rows.extend(rows)

    if all_rows:
        out_csv = PROB_DIR / "category_refusal_cosine.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        logger.info(f"Saved → {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
