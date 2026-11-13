"""
Step 19 — Evolution of safety representations across fine-tuning stages.

Research question:
  How do harmfulness, harmlessness, and refusal representations change
  before and after fine-tuning? Does alignment training:
    (1) preserve separation between harmful and harmless prompts?
    (2) change where refusal is encoded (by layer)?
    (3) make harmfulness and refusal more or less entangled?

Uses:
  - results/probes/{model}/{lang}/layer_*_{diffmean,logreg,refusal}.pkl
  - data/processed/activations_eval/{model}/{lang}/
  - results/behavioral/baseline_summary.csv

Run:
  python scripts/19_finetuning_evolution.py --lang en
  python scripts/19_finetuning_evolution.py --all

Writes:
  results/probes/finetuning_evolution.csv
  results/figures/fig_evolution_probes_{lang}.png        (4-panel: accuracy + entanglement + drift)
  results/figures/fig_evolution_pca_{lang}_layer{L}.png  (side-by-side PCA across stages)
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES
from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR  = ROOT / "results" / "figures"
PROB_DIR = ROOT / "results" / "probes"
# Primary path for base/sft/dpo; regional models (global/earth/water) live in activations/
ACTS_ROOTS = [
    ROOT / "data" / "processed" / "activations_eval",
    ROOT / "data" / "processed" / "activations",
]
BEHAV_CSV = ROOT / "results" / "behavioral" / "baseline_summary.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Training-stage order to compare (global included for cross-model reference)
STAGE_MODELS = ["base", "global", "base_sft", "base_dpo"]
STAGE_LABELS = {
    "base": "Base", "base_sft": "Base+SFT", "base_dpo": "Base+DPO",
    "global": "Global (multilingual IT)", "earth": "Earth", "water": "Water",
    "fire": "Fire",
}
STAGE_COLORS = {
    "base": "#888888", "base_sft": "#4477AA", "base_dpo": "#EE4444",
    "global": "#228833", "earth": "#CCBB44", "water": "#66CCEE",
    "fire": "#AA3377",
}


# ── data loaders ──────────────────────────────────────────────────────────────

def load_probe(model_key: str, lang: str, layer: int, probe_type: str):
    path = PROB_DIR / model_key / lang / f"layer_{layer}_{probe_type}.pkl"
    return pickle.load(open(path, "rb")) if path.exists() else None


def _acts_dir(model_key: str, lang: str) -> Path | None:
    for root in ACTS_ROOTS:
        d = root / model_key / lang
        if d.exists():
            return d
    return None


def load_acts(model_key: str, lang: str, layer: int):
    d = _acts_dir(model_key, lang)
    if d is None:
        return None
    p = d / f"layer_{layer}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def load_prompts(model_key: str, lang: str) -> list[dict]:
    d = _acts_dir(model_key, lang)
    if d is None:
        return []
    p = d / "prompts.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── per-layer metrics ─────────────────────────────────────────────────────────

def compute_metrics(model_key: str, lang: str, ref_model: str = "base") -> pd.DataFrame:
    """
    Returns a DataFrame with per-layer metrics for one model/language:
      - diffmean_acc, logreg_acc, logreg_auc  — harmful/harmless probe quality
      - refusal_acc                            — refusal probe quality
      - harm_refusal_cos                       — cos_sim(harmfulness dir, refusal dir)
      - harm_drift_cos                         — cos_sim(this model harm dir, ref_model harm dir)
      - silhouette                             — harmful vs harmless cluster separation
    """
    cfg = MODELS.get(model_key)
    if cfg is None:
        return pd.DataFrame()

    prompts = load_prompts(model_key, lang)
    if not prompts:
        return pd.DataFrame()
    labels_arr = np.array([1 if p.get("label") == "harmful" else 0 for p in prompts])
    has_both = labels_arr.sum() > 0 and (1 - labels_arr).sum() > 0

    rows = []
    for layer in cfg.probe_layers:
        row = {"model": model_key, "lang": lang, "layer": layer}

        # ── probe accuracy ────────────────────────────────────────────────────
        dm = load_probe(model_key, lang, layer, "diffmean")
        lr = load_probe(model_key, lang, layer, "logreg")
        rf = load_probe(model_key, lang, layer, "refusal")

        row["diffmean_acc"] = round(dm.accuracy, 4) if dm else float("nan")
        row["logreg_acc"]   = round(lr.accuracy, 4) if lr else float("nan")
        row["logreg_auc"]   = round(lr.auc,      4) if lr else float("nan")
        row["refusal_acc"]  = round(rf.accuracy,  4) if rf else float("nan")

        acts = load_acts(model_key, lang, layer)

        # ── sign-canonicalize directions ──────────────────────────────────────
        # DiffMean sign is arbitrary: canonicalize so harmful projects higher.
        # Refusal direction: canonicalize so refused projects higher.
        def canon_harm(d, acts, labels):
            if d is None or acts is None:
                return d
            d = d / (np.linalg.norm(d) + 1e-8)
            mu_harm = acts[labels == 1].mean(0) if (labels == 1).any() else None
            mu_less = acts[labels == 0].mean(0) if (labels == 0).any() else None
            if mu_harm is None or mu_less is None:
                return d
            if np.dot(mu_harm, d) < np.dot(mu_less, d):
                d = -d
            return d

        def canon_refusal(d, acts, refused_mask):
            if d is None or acts is None or refused_mask.sum() == 0:
                return d
            d = d / (np.linalg.norm(d) + 1e-8)
            mu_ref = acts[refused_mask].mean(0)
            mu_acc = acts[~refused_mask].mean(0)
            if np.dot(mu_ref, d) < np.dot(mu_acc, d):
                d = -d
            return d

        d_harm = canon_harm(
            dm.direction.copy() if dm else None, acts, labels_arr
        )

        # Refusal direction: need refused/accepted labels for this model
        rf_direction = rf.direction.copy() if rf else None
        # Use behavioral refusal labels if available, otherwise skip canonicalization
        refusal_mask = np.zeros(len(labels_arr), dtype=bool)  # default: none refused
        if acts is not None and rf_direction is not None:
            # Load behavioral labels to canonicalize refusal direction
            resp_path = ROOT / "results" / "behavioral" / f"responses_{model_key}_{lang}.jsonl"
            if resp_path.exists():
                import json as _json
                id_to_idx = {p["id"]: i for i, p in enumerate(load_prompts(model_key, lang))}
                with open(resp_path) as _f:
                    for _line in _f:
                        _d = _json.loads(_line)
                        if _d["id"] in id_to_idx and _d.get("heuristic_refuse"):
                            refusal_mask[id_to_idx[_d["id"]]] = True
        d_ref = canon_refusal(rf_direction, acts, refusal_mask)

        # ── harmfulness ↔ refusal entanglement ───────────────────────────────
        if d_harm is not None and d_ref is not None:
            row["harm_refusal_cos"] = round(float(np.dot(d_harm, d_ref)), 4)
        else:
            row["harm_refusal_cos"] = float("nan")

        # ── direction drift from base (centroid-based, avoids pkl sign artifacts) ─
        # Recompute DiffMean directly from cluster centroids for both models so
        # sign is always canonical and consistent.
        if model_key != ref_model:
            acts_ref = load_acts(ref_model, lang, layer)
            ref_prompts = load_prompts(ref_model, lang)
            if acts_ref is not None and ref_prompts:
                ref_labels = np.array([1 if p.get("label") == "harmful" else 0
                                       for p in ref_prompts])
                def centroid_dir(a, lab):
                    pos = a[lab == 1].mean(0) if (lab == 1).any() else None
                    neg = a[lab == 0].mean(0) if (lab == 0).any() else None
                    if pos is None or neg is None:
                        return None
                    d = pos - neg
                    return d / (np.linalg.norm(d) + 1e-8)

                d_this_centroid = centroid_dir(acts, labels_arr)
                d_ref_centroid  = centroid_dir(acts_ref, ref_labels)
                if d_this_centroid is not None and d_ref_centroid is not None:
                    row["harm_drift_cos"] = round(float(np.dot(d_this_centroid, d_ref_centroid)), 4)
                else:
                    row["harm_drift_cos"] = float("nan")
            else:
                row["harm_drift_cos"] = float("nan")
        else:
            row["harm_drift_cos"] = 1.0

        # ── silhouette score (harmful vs harmless) ────────────────────────────
        # acts already loaded above
        if acts is not None and has_both and len(set(labels_arr)) == 2:
            try:
                # Subsample to max 1000 for speed
                n = len(acts)
                if n > 1000:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n, 1000, replace=False)
                    sil = silhouette_score(acts[idx], labels_arr[idx], metric="cosine")
                else:
                    sil = silhouette_score(acts, labels_arr, metric="cosine")
                row["silhouette"] = round(float(sil), 4)
            except Exception:
                row["silhouette"] = float("nan")
        else:
            row["silhouette"] = float("nan")

        rows.append(row)
    return pd.DataFrame(rows)


# ── plotting: 4-panel evolution figure ────────────────────────────────────────

def plot_evolution(df: pd.DataFrame, lang: str):
    lang_name = LANG_NAMES.get(lang, lang)
    layers = sorted(df["layer"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax_harm, ax_sil, ax_ref, ax_ent = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    present_models = [m for m in df["model"].unique()
                      if (df["model"] == m).any() and m in STAGE_LABELS]
    for model_key in present_models:
        sub = df[(df["model"] == model_key) & (df["lang"] == lang)].sort_values("layer")
        if sub.empty:
            continue
        label = STAGE_LABELS.get(model_key, model_key)
        color = STAGE_COLORS.get(model_key, "#888888")

        # Panel 1: harmful/harmless probe accuracy (logreg preferred, fallback diffmean)
        acc = sub["logreg_acc"].where(sub["logreg_acc"].notna(), sub["diffmean_acc"])
        ax_harm.plot(sub["layer"], acc, marker="o", markersize=5,
                     color=color, label=label)

        # Panel 2: silhouette score
        ax_sil.plot(sub["layer"], sub["silhouette"], marker="o", markersize=5,
                    color=color, label=label)

        # Panel 3: refusal probe accuracy
        ax_ref.plot(sub["layer"], sub["refusal_acc"], marker="o", markersize=5,
                    color=color, label=label)

        # Panel 4: harmfulness-refusal entanglement (cos similarity)
        ax_ent.plot(sub["layer"], sub["harm_refusal_cos"], marker="o", markersize=5,
                    color=color, label=label)

    ax_harm.set_title("Harmful/Harmless separation\n(LogReg probe accuracy)", fontsize=9)
    ax_harm.set_ylabel("Accuracy")
    ax_harm.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax_harm.legend(fontsize=8)
    ax_harm.set_ylim(0.45, 1.02)

    ax_sil.set_title("Harmful/Harmless cluster separation\n(Silhouette score, cosine)", fontsize=9)
    ax_sil.set_ylabel("Silhouette score")
    ax_sil.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_sil.legend(fontsize=8)

    ax_ref.set_title("Refusal encoding\n(Refusal probe accuracy)", fontsize=9)
    ax_ref.set_ylabel("Accuracy")
    ax_ref.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax_ref.set_xlabel("Layer")
    ax_ref.legend(fontsize=8)
    ax_ref.set_ylim(0.45, 1.02)

    ax_ent.set_title("Harmfulness ↔ Refusal entanglement\ncos_sim(harm dir, refusal dir)", fontsize=9)
    ax_ent.set_ylabel("Cosine similarity")
    ax_ent.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_ent.set_xlabel("Layer")
    ax_ent.legend(fontsize=8)

    for ax in [ax_harm, ax_sil, ax_ref, ax_ent]:
        ax.set_xticks(layers)

    fig.suptitle(f"Safety representation evolution: Base → SFT → DPO\n{lang_name}", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / f"fig_evolution_probes_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_evolution_drift(df: pd.DataFrame, lang: str):
    """Direction drift: how much does the harmfulness direction change from Base?"""
    lang_name = LANG_NAMES.get(lang, lang)
    layers = sorted(df["layer"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    non_base = [m for m in df["model"].unique() if m != "base"]
    for model_key in non_base:
        sub = df[(df["model"] == model_key) & (df["lang"] == lang)].sort_values("layer")
        if sub.empty or sub["harm_drift_cos"].isna().all():
            continue
        label = f"Base → {STAGE_LABELS.get(model_key, model_key)}"
        ax.plot(sub["layer"], sub["harm_drift_cos"], marker="o", markersize=5,
                color=STAGE_COLORS.get(model_key, "#888888"), label=label)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="No change (cos=1)")
    ax.axhline(0.0, color="lightgray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos_sim(finetuned harm dir, base harm dir)")
    ax.set_title(f"Harmfulness direction drift from Base\n{lang_name}", fontsize=9)
    ax.set_xticks(layers)
    ax.legend(fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    out = FIG_DIR / f"fig_evolution_drift_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── PCA side-by-side across stages ───────────────────────────────────────────

def plot_evolution_pca(lang: str, layer: int):
    """Side-by-side PCA for base / base_sft / base_dpo at one layer."""
    lang_name = LANG_NAMES.get(lang, lang)

    # Fit PCA on Base activations (harmful + harmless together)
    base_acts = load_acts("base", lang, layer)
    base_prompts = load_prompts("base", lang)
    if base_acts is None or not base_prompts:
        return

    pca = PCA(n_components=2, random_state=42)
    pca.fit(base_acts)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, model_key in zip(axes, STAGE_MODELS):
        acts = load_acts(model_key, lang, layer)
        prompts = load_prompts(model_key, lang)
        if acts is None or not prompts:
            ax.set_title(f"{STAGE_LABELS[model_key]}\n(no data)")
            continue

        proj = pca.transform(acts)
        for label, color, marker in [("harmful", "#EE4444", "x"), ("harmless", "#4477AA", "o")]:
            idx = [i for i, p in enumerate(prompts) if p.get("label") == label]
            if idx:
                ax.scatter(proj[idx, 0], proj[idx, 1],
                           c=color, marker=marker, s=12, alpha=0.4, label=label)

        # DiffMean direction arrow
        dm = load_probe(model_key, lang, layer, "diffmean")
        if dm is not None:
            d = dm.direction / (np.linalg.norm(dm.direction) + 1e-8)
            origin = proj.mean(axis=0)
            d_proj = pca.transform(d.reshape(1, -1))[0]
            ax.annotate("", xy=origin + 3 * d_proj, xytext=origin,
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

        ax.set_title(f"{STAGE_LABELS[model_key]}", fontsize=9)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=7)

    fig.suptitle(f"Cluster evolution: Base → SFT → DPO  ({lang_name}, Layer {layer})\n"
                 f"PCA fit on Base; arrow = DiffMean harmfulness direction", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"fig_evolution_pca_{lang}_layer{layer}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── RQ2 summary: multi-language heatmap at best layer ─────────────────────────

def plot_rq2_summary(full_df: pd.DataFrame, layer: int = 24):
    """
    Heatmap grid: rows = model × col = language, showing four key metrics:
      harm_refusal_cos, harm_drift_cos, logreg_acc, silhouette
    One row of subplots per metric.
    """
    models = [m for m in ["base", "global", "base_sft", "base_dpo"]
              if m in full_df["model"].values]
    langs  = [l for l in ["en", "fr", "ar", "hi", "sw", "yo"]
              if l in full_df["lang"].values]
    metrics = [
        ("harm_refusal_cos",  "Harmful↔Refusal cos",   -1, 1,   "RdBu",   True),
        ("harm_drift_cos",    "Harm dir drift (→Base)", -1, 1,   "RdBu",   True),
        ("logreg_acc",        "LogReg accuracy",         0.5, 1, "Blues",  False),
        ("silhouette",        "Silhouette",             -0.1, 0.2, "Greens", False),
    ]

    sub = full_df[full_df["layer"] == layer]

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(8, len(langs) * 1.4), len(metrics) * 2.5))
    if len(metrics) == 1:
        axes = [axes]

    import matplotlib.colors as mcolors

    for ax, (col, title, vmin, vmax, cmap, center_zero) in zip(axes, metrics):
        mat = np.full((len(models), len(langs)), np.nan)
        for ri, m in enumerate(models):
            for ci, l in enumerate(langs):
                row = sub[(sub["model"] == m) & (sub["lang"] == l)]
                if not row.empty and not np.isnan(row[col].values[0]):
                    mat[ri, ci] = row[col].values[0]

        if center_zero:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels(langs, fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([STAGE_LABELS.get(m, m) for m in models], fontsize=8)
        ax.set_title(f"{title}  (Layer {layer})", fontsize=9, pad=3)

        for ri in range(len(models)):
            for ci in range(len(langs)):
                val = mat[ri, ci]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.6 * max(abs(vmin), abs(vmax)) else "black"
                    ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)
                else:
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=7, color="#aaaaaa")

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    fig.suptitle(f"RQ2: Safety representation evolution across fine-tuning stages  (Layer {layer})",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / f"fig_rq2_summary_layer{layer}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANG_CODES, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list (e.g. --models global earth water base)")
    args = parser.parse_args()

    langs = LANG_CODES if args.all else ([args.lang] if args.lang else ["en"])
    if not args.lang and not args.all:
        logger.info("Defaulting to en")

    models = args.models if args.models else STAGE_MODELS

    all_rows = []
    for lang in langs:
        for model_key in models:
            if _acts_dir(model_key, lang) is None:
                continue
            logger.info(f"Computing metrics: {model_key}/{lang}")
            df = compute_metrics(model_key, lang)
            if not df.empty:
                all_rows.append(df)

    if not all_rows:
        logger.error("No data found.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    # Append to existing CSV if running a different set of models
    out_csv = PROB_DIR / "finetuning_evolution.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Drop rows for models we're re-computing, then append
        existing = existing[~existing["model"].isin(models)]
        full_df = pd.concat([existing, full_df], ignore_index=True)
    full_df.to_csv(out_csv, index=False)
    logger.info(f"Saved → {out_csv}")

    # Use only the models present in this run's data for plotting
    for lang in langs:
        lang_df = full_df[full_df["lang"] == lang]
        if lang_df.empty:
            continue
        plot_evolution(lang_df, lang)
        plot_evolution_drift(lang_df, lang)
        # PCA at best probe layer (18) and middle layer (24)
        for layer in [18, 24]:
            plot_evolution_pca(lang, layer)

    # RQ2 summary heatmap at layer 24 (and 18 for comparison)
    for layer in [18, 24]:
        plot_rq2_summary(full_df, layer=layer)

    print("\nDone.")


if __name__ == "__main__":
    main()
