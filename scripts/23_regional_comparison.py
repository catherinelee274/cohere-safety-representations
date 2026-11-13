"""
Step 23 — Regional model comparison: Earth / Fire / Water vs Global.

Core hypothesis: Regional models (trained on region-specific data) have more
language-specific safety directions than the Global model, i.e. their harmfulness
directions for the focal region's languages diverge more from English and are
more internally consistent within the region.

Analyses
--------
1. Probe accuracy comparison (LogReg AUC, DiffMean AUC) across models × languages × layers
2. Within-model crosslingual cosine matrices per model (replicates script 18 for each regional model)
3. Cross-model direction alignment: for each language, cosine(regional_dir, global_dir)
4. Behavioural refusal rates (where available)
5. Summary figure: heatmap of cross-model direction cosines per language

Regional model coverage:
  earth  → Africa / West Asia  (Arabic, Swahili focus)
  fire   → South Asia          (Hindi focus)
  water  → Asia-Pacific / EU   (French, Chinese focus; zh not in eval set)

Writes
------
  results/probes/regional_probe_comparison.csv
  results/probes/crosslingual_cosine_{model}.csv   (for earth/fire/water/global)
  results/probes/cross_model_direction_cosine.csv
  results/figures/fig_regional_probe_auc.png
  results/figures/fig_regional_crosslingual_{model}.png
  results/figures/fig_cross_model_direction.png

Run
---
  python scripts/23_regional_comparison.py
  python scripts/23_regional_comparison.py --skip-plots
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROBE_DIR = ROOT / "results" / "probes"
FIG_DIR   = ROOT / "results" / "figures"
BSLN_DIR  = ROOT / "results" / "behavioral"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGIONAL_MODELS = ["global", "earth", "fire", "water"]
FOCAL_LANGS = {
    "global": ["en", "fr", "ar", "hi"],
    "earth":  ["ar", "hi", "en", "fr"],   # Arabic / Swahili focus; sw has 0 harmful in eval
    "fire":   ["hi", "ar", "en", "fr"],   # Hindi focus
    "water":  ["fr", "en", "ar", "hi"],   # French / Chinese focus; zh missing from eval
}
LAYERS = [4, 9, 18, 24, 30, 35]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_probe(model_key: str, lang: str, layer: int, probe_type: str):
    path = PROBE_DIR / model_key / lang / f"layer_{layer}_{probe_type}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_direction(model_key: str, lang: str, layer: int) -> np.ndarray | None:
    p = load_probe(model_key, lang, layer, "diffmean")
    if p is None:
        return None
    d = p.direction.astype(np.float32)
    n = np.linalg.norm(d)
    return d / n if n > 1e-8 else d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Analysis 1: probe accuracy comparison ─────────────────────────────────────

def collect_probe_metrics() -> pd.DataFrame:
    rows = []
    for model in REGIONAL_MODELS:
        for lang in FOCAL_LANGS[model]:
            for layer in LAYERS:
                for ptype in ["logreg", "diffmean"]:
                    p = load_probe(model, lang, layer, ptype)
                    if p is None:
                        continue
                    rows.append({
                        "model": model,
                        "lang": lang,
                        "layer": layer,
                        "probe_type": ptype,
                        "accuracy": round(p.accuracy, 4),
                        "auc": round(p.auc, 4),
                    })
    return pd.DataFrame(rows)


# ── Analysis 2: within-model crosslingual cosines ─────────────────────────────

def compute_crosslingual_cosines(model_key: str, layer: int) -> pd.DataFrame | None:
    langs = FOCAL_LANGS[model_key]
    dirs = {}
    for lang in langs:
        d = load_direction(model_key, lang, layer)
        if d is not None:
            dirs[lang] = d

    if len(dirs) < 2:
        return None

    lang_list = sorted(dirs)
    records = []
    for src in lang_list:
        for tgt in lang_list:
            records.append({
                "model": model_key,
                "layer": layer,
                "lang_src": src,
                "lang_tgt": tgt,
                "cosine": round(cosine(dirs[src], dirs[tgt]), 4),
            })
    return pd.DataFrame(records)


# ── Analysis 3: cross-model direction alignment ────────────────────────────────

def compute_cross_model_cosines() -> pd.DataFrame:
    """
    For each (lang, layer), compute cosine between each regional model's
    harmfulness direction and the global model's direction.
    """
    rows = []
    all_langs = sorted({l for langs in FOCAL_LANGS.values() for l in langs})
    for lang in all_langs:
        d_global = None
        for layer in LAYERS:
            d_global = load_direction("global", lang, layer)
            for model in ["earth", "fire", "water"]:
                d_reg = load_direction(model, lang, layer)
                if d_global is None or d_reg is None:
                    cos = None
                else:
                    cos = round(cosine(d_global, d_reg), 4)
                rows.append({
                    "model": model,
                    "lang": lang,
                    "layer": layer,
                    "cos_to_global": cos,
                })
    return pd.DataFrame(rows)


# ── Behavioural data ──────────────────────────────────────────────────────────

def load_refusal_rates() -> pd.DataFrame:
    path = BSLN_DIR / "baseline_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["model"].isin(REGIONAL_MODELS)]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_probe_auc(df: pd.DataFrame):
    """
    Grid: rows = languages, cols = models.
    Line plot of LogReg AUC across layers for each model.
    """
    langs = sorted(df["lang"].unique())
    n_langs = len(langs)
    n_models = len(REGIONAL_MODELS)

    fig, axes = plt.subplots(n_langs, n_models,
                             figsize=(3.5 * n_models, 2.8 * n_langs),
                             sharex=True, sharey=True)
    if n_langs == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    colors = {"logreg": "#1f77b4", "diffmean": "#ff7f0e"}

    for ri, lang in enumerate(langs):
        for ci, model in enumerate(REGIONAL_MODELS):
            ax = axes[ri][ci]
            sub = df[(df["lang"] == lang) & (df["model"] == model)]
            for ptype, grp in sub.groupby("probe_type"):
                grp = grp.sort_values("layer")
                ax.plot(grp["layer"], grp["auc"],
                        marker="o", ms=4, lw=1.5,
                        color=colors.get(ptype, "gray"),
                        label=ptype)
            if ri == 0:
                ax.set_title(model, fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{lang}\nAUC", fontsize=8)
            ax.set_ylim(0.5, 1.0)
            ax.axhline(0.736, color="gray", lw=0.7, ls="--")  # base rate
            ax.tick_params(labelsize=7)
            if ri == n_langs - 1:
                ax.set_xlabel("Layer", fontsize=8)

    handles = [plt.Line2D([0], [0], color=c, marker="o", ms=4, lw=1.5, label=l)
               for l, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Probe AUC by model and language (regional vs global)", fontsize=10)
    fig.subplots_adjust(hspace=0.35, wspace=0.2, top=0.93, right=0.88)
    out = FIG_DIR / "fig_regional_probe_auc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_crosslingual_heatmap(model_key: str, layer: int, df: pd.DataFrame):
    if df is None or df.empty:
        return
    pivot = df.pivot(index="lang_src", columns="lang_tgt", values="cosine")
    langs = pivot.index.tolist()
    mat = pivot.values

    fig, ax = plt.subplots(figsize=(len(langs) * 0.9 + 1, len(langs) * 0.9 + 0.5))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels(langs, fontsize=9)
    ax.set_yticklabels(langs, fontsize=9)
    for i in range(len(langs)):
        for j in range(len(langs)):
            val = mat[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="cosine")
    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    ax.set_title(f"Crosslingual direction cosines — {model_name} / Layer {layer}", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / f"fig_regional_crosslingual_{model_key}_layer{layer}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_cross_model_cosines(df: pd.DataFrame):
    """
    For each language, show how cos(regional, global) varies across layers.
    One subplot per language.
    """
    langs = sorted(df["lang"].unique())
    models = ["earth", "fire", "water"]
    colors = {"earth": "#1b7837", "fire": "#d6604d", "water": "#4393c3"}

    n_langs = len(langs)
    ncols = min(4, n_langs)
    nrows = (n_langs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, lang in enumerate(langs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["lang"] == lang]
        for model in models:
            msub = sub[sub["model"] == model].dropna(subset=["cos_to_global"]).sort_values("layer")
            if not msub.empty:
                ax.plot(msub["layer"], msub["cos_to_global"],
                        marker="o", ms=4, lw=1.5,
                        color=colors[model], label=model)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(lang, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("cos(regional, global)", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Layer", fontsize=8)

    for idx in range(n_langs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [plt.Line2D([0], [0], color=c, marker="o", ms=4, lw=1.5, label=m)
               for m, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Cross-model direction alignment (regional vs global)\n"
                 "cos(regional_harmfulness_dir, global_harmfulness_dir)",
                 fontsize=9)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.88, right=0.87)
    out = FIG_DIR / "fig_cross_model_direction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    # 1. Probe metrics
    logger.info("Collecting probe metrics …")
    probe_df = collect_probe_metrics()
    if not probe_df.empty:
        out = PROBE_DIR / "regional_probe_comparison.csv"
        probe_df.to_csv(out, index=False)
        logger.info(f"Saved → {out}")
        logger.info("\n" + probe_df[probe_df["probe_type"] == "logreg"].pivot_table(
            index=["model", "lang"], columns="layer", values="auc"
        ).round(3).to_string())
    else:
        logger.warning("No probe metrics found.")

    # 2. Within-model crosslingual cosines
    logger.info("Computing crosslingual cosines …")
    all_cross = []
    for model in REGIONAL_MODELS:
        for layer in LAYERS:
            df = compute_crosslingual_cosines(model, layer)
            if df is not None:
                all_cross.append(df)
                if not args.skip_plots and layer in [18, 24]:
                    plot_crosslingual_heatmap(model, layer, df)
    if all_cross:
        cross_df = pd.concat(all_cross, ignore_index=True)
        out = PROBE_DIR / "crosslingual_cosine_regional.csv"
        cross_df.to_csv(out, index=False)
        logger.info(f"Saved → {out}")

        # Print layer-18 summary
        logger.info("\n=== Layer 18 crosslingual cosines (src=en) ===")
        sub = cross_df[(cross_df["layer"] == 18) & (cross_df["lang_src"] == "en")]
        if not sub.empty:
            print(sub.pivot_table(index="model", columns="lang_tgt", values="cosine").round(3).to_string())

    # 3. Cross-model direction alignment
    logger.info("Computing cross-model direction alignment …")
    cmd_df = compute_cross_model_cosines()
    if not cmd_df.empty:
        out = PROBE_DIR / "cross_model_direction_cosine.csv"
        cmd_df.to_csv(out, index=False)
        logger.info(f"Saved → {out}")

        logger.info("\n=== Layer 18 cross-model cosines vs global ===")
        sub = cmd_df[cmd_df["layer"] == 18].dropna(subset=["cos_to_global"])
        if not sub.empty:
            print(sub.pivot_table(index="model", columns="lang", values="cos_to_global").round(3).to_string())

    # 4. Behavioural refusal rates
    logger.info("Loading behavioural refusal rates …")
    bsln_df = load_refusal_rates()
    if not bsln_df.empty:
        logger.info("\n=== Refusal rates by model and language ===")
        print(bsln_df.pivot_table(index="model", columns="lang", values="refusal_rate").round(3).to_string())
    else:
        logger.info("No behavioural data for regional models yet.")

    # 5. Plots
    if not args.skip_plots:
        if not probe_df.empty:
            plot_probe_auc(probe_df)
        if not cmd_df.empty:
            plot_cross_model_cosines(cmd_df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
