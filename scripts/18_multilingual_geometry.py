"""
Step 18 — Multilingual geometry of refusal representations.
Replicates Wang et al. 2026 (2505.17306) Figure 4 and Figure 5.

Figure 4 (Wang): PCA visualization of multilingual harmful/harmless representations.
  - Project activations from multiple languages into 2D PCA space (fit on English only)
  - Plot harmful vs harmless clusters per language, colour-coded
  - Arrows indicate the refusal/harmfulness direction per language

Figure 5 (Wang): Cross-lingual cosine similarity heatmap.
  - Compute cosine similarity between the refusal (DiffMean) direction of every
    (model, lang_source) pair and every (model, lang_target) pair, at each probe layer
  - Show as language × language heatmap per layer (Wang Fig 5 equivalent)

Run:
  python scripts/18_multilingual_geometry.py --model base
  python scripts/18_multilingual_geometry.py --all

Writes:
  results/figures/fig4_pca_{model}_layer{L}.png
  results/figures/fig5_crosslingual_cosine_{model}_layer{L}.png
  results/probes/crosslingual_cosine_{model}.csv
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES
from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR  = ROOT / "results" / "figures"
PROB_DIR = ROOT / "results" / "probes"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ACTS_ROOT = ROOT / "data" / "processed" / "activations_eval"

# Language colours for scatter (Wang Fig 4 style)
LANG_COLORS = {
    "en": "#4477AA",
    "fr": "#66CCEE",
    "ar": "#228833",
    "hi": "#CCBB44",
    "sw": "#EE6677",
    "yo": "#AA3377",
    "zh": "#BBBBBB",
    "de": "#44AA99",
}
LABEL_MARKERS = {"harmful": "x", "harmless": "o"}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_activations(model_key: str, lang: str, layer: int) -> tuple[np.ndarray, list[dict]] | None:
    acts_path = ACTS_ROOT / model_key / lang / f"layer_{layer}.npy"
    prompts_path = ACTS_ROOT / model_key / lang / "prompts.jsonl"
    if not acts_path.exists() or not prompts_path.exists():
        return None
    acts = np.load(acts_path).astype(np.float32)
    with open(prompts_path) as f:
        prompts = [json.loads(l) for l in f if l.strip()]
    return acts, prompts


def load_direction(model_key: str, lang: str, layer: int, probe_type: str = "diffmean") -> np.ndarray | None:
    path = PROB_DIR / model_key / lang / f"layer_{layer}_{probe_type}.pkl"
    if not path.exists():
        return None
    res = pickle.load(open(path, "rb"))
    d = res.direction.astype(np.float32)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-8 else d


# ── Figure 4: PCA ─────────────────────────────────────────────────────────────

def plot_pca_multilingual(model_key: str, layer: int, langs: list[str]):
    """
    Wang et al. Fig 4: PCA projection of harmful/harmless activations across languages.
    PCA is fit on English harmful+harmless, then all languages are projected.
    Arrows show the DiffMean direction per language in PCA space.
    """
    result = load_activations(model_key, "en", layer)
    if result is None:
        logger.warning(f"No English activations for {model_key} layer {layer}")
        return
    en_acts, en_prompts = result

    # Fit PCA on English
    pca = PCA(n_components=2, random_state=42)
    pca.fit(en_acts)

    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    n_langs = len([l for l in langs if load_activations(model_key, l, layer) is not None])
    ncols = min(4, n_langs)
    nrows = max(1, (n_langs + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_langs == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    ax_idx = 0
    for lang in langs:
        result = load_activations(model_key, lang, layer)
        if result is None:
            continue

        acts, prompts = result
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        ax_idx += 1

        proj = pca.transform(acts)
        lang_name = LANG_NAMES.get(lang, lang)
        color = LANG_COLORS.get(lang, "#888888")

        for label, marker in LABEL_MARKERS.items():
            idx = [i for i, p in enumerate(prompts) if p.get("label") == label]
            if idx:
                ax.scatter(proj[idx, 0], proj[idx, 1],
                           c=color, marker=marker, s=12, alpha=0.5,
                           label=label)

        # DiffMean direction arrow in PCA space
        d = load_direction(model_key, lang, layer, "diffmean")
        if d is not None:
            origin = proj.mean(axis=0)
            d_proj = pca.transform(d.reshape(1, -1))[0]
            scale = 3.0
            ax.annotate("", xy=origin + scale * d_proj, xytext=origin,
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        ax.set_title(f"{lang_name}", fontsize=8)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, markerscale=1.5)

    # Hide unused axes
    for i in range(ax_idx, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(f"PCA: harmful vs harmless ({model_name} / Layer {layer})\n"
                 f"PCA fit on English; arrows = DiffMean direction", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"fig4_pca_{model_key}_layer{layer}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── Figure 5: Cross-lingual cosine similarity heatmap ────────────────────────

def compute_crosslingual_cosine(model_key: str, langs: list[str], layers: list[int]) -> pd.DataFrame:
    """
    For each (layer, lang_src, lang_tgt) pair, compute cosine similarity between
    the DiffMean harmfulness directions.
    """
    rows = []
    for layer in layers:
        for src in langs:
            d_src = load_direction(model_key, src, layer, "diffmean")
            if d_src is None:
                continue
            for tgt in langs:
                d_tgt = load_direction(model_key, tgt, layer, "diffmean")
                if d_tgt is None:
                    cos = float("nan")
                else:
                    cos = float(np.dot(d_src, d_tgt))  # already normalized
                rows.append({
                    "model": model_key,
                    "layer": layer,
                    "lang_src": src,
                    "lang_tgt": tgt,
                    "cosine": round(cos, 4),
                })
    return pd.DataFrame(rows)


def plot_crosslingual_heatmap(df: pd.DataFrame, model_key: str, layers: list[int]):
    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    sub = df[df["model"] == model_key]
    if sub.empty:
        return

    n = len(layers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for li, layer in enumerate(layers):
        ax = axes[li // ncols][li % ncols]
        layer_sub = sub[sub["layer"] == layer]
        if layer_sub.empty:
            ax.set_visible(False)
            continue

        pivot = layer_sub.pivot(index="lang_src", columns="lang_tgt", values="cosine")
        langs_present = list(pivot.index)
        lang_labels = [LANG_NAMES.get(l, l) for l in langs_present]

        im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(langs_present)))
        ax.set_yticks(range(len(langs_present)))
        ax.set_xticklabels(lang_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(lang_labels, fontsize=7)
        ax.set_title(f"Layer {layer}", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(langs_present)):
            for j in range(len(langs_present)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if abs(v) < 0.6 else "white")

    for i in range(li + 1, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(f"Cross-lingual DiffMean cosine similarity\n{model_name}", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"fig5_crosslingual_cosine_{model_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    if args.all:
        models = list(MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        models = ["base"]
        logger.info("Defaulting to base")

    # Only process models that actually have activations_eval data
    models = [m for m in models if (ACTS_ROOT / m).exists()]

    all_cos_rows = []
    for model_key in models:
        cfg = MODELS[model_key]
        # Find available languages
        available_langs = [l for l in LANG_CODES if (ACTS_ROOT / model_key / l).exists()]
        if not available_langs:
            continue
        logger.info(f"{model_key}: langs={available_langs}")

        # Fig 4: PCA at each probe layer
        for layer in cfg.probe_layers:
            plot_pca_multilingual(model_key, layer, available_langs)

        # Fig 5: cross-lingual cosine heatmap
        cos_df = compute_crosslingual_cosine(model_key, available_langs, cfg.probe_layers)
        if not cos_df.empty:
            all_cos_rows.append(cos_df)
            plot_crosslingual_heatmap(cos_df, model_key, cfg.probe_layers)

    if all_cos_rows:
        combined = pd.concat(all_cos_rows, ignore_index=True)
        out_csv = PROB_DIR / "crosslingual_cosine.csv"
        combined.to_csv(out_csv, index=False)
        logger.info(f"Saved → {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
