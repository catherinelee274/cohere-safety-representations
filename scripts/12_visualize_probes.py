"""
Step 12 — Visualize probe results.

Reads:
  results/probes/summary.csv
  results/probes/alignment_{model}_layer{n}.csv   (optional, from --alignment flag)
  results/probes/cross_stage_alignment.csv         (optional, from --cross-stage flag)
  data/processed/activations/{model}/{lang}/layer_{n}.npy  (for UMAP)

Writes:
  results/figures/accuracy_heatmap.png
  results/figures/layer_sweep.png
  results/figures/alignment_{model}_layer{n}.png
  results/figures/cross_stage_alignment.png
  results/figures/umap_{model}_{lang}_layer{n}.png

Usage:
    python scripts/12_visualize_probes.py                    # all plots from summary.csv
    python scripts/12_visualize_probes.py --umap             # also generate UMAP plots
    python scripts/12_visualize_probes.py --probe-type logreg
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_NAMES, LOW_RESOURCE
from config.models import MODEL_ORDER, MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent language label (mark low-resource with *)
def lang_label(code: str) -> str:
    name = LANG_NAMES.get(code, code)
    return f"{name}*" if LOW_RESOURCE.get(code) else name


# ── 1. Accuracy heatmap: language × model ─────────────────────────────────────

def plot_accuracy_heatmap(df: pd.DataFrame, probe_type: str, layer: int | None = None):
    sub = df[df["probe_type"] == probe_type].copy()
    if layer is not None:
        sub = sub[sub["layer"] == layer]
    else:
        # Use best layer per (model, lang)
        sub = sub.loc[sub.groupby(["model", "lang"])["accuracy"].idxmax()]

    # Keep only models present in data
    models_present = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    langs_present  = sorted(sub["lang"].unique(), key=lambda l: LANG_NAMES.get(l, l))

    pivot = sub.pivot_table(index="lang", columns="model", values="accuracy")
    pivot = pivot.reindex(index=langs_present, columns=models_present)

    fig, ax = plt.subplots(figsize=(max(8, len(models_present) * 1.2), max(4, len(langs_present) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f", vmin=0.5, vmax=1.0,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        yticklabels=[lang_label(l) for l in langs_present],
        xticklabels=[MODELS[m].name if m in MODELS else m for m in models_present],
    )
    layer_str = f"layer {layer}" if layer is not None else "best layer"
    ax.set_title(f"Probe accuracy ({probe_type}, {layer_str})")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = FIG_DIR / f"accuracy_heatmap_{probe_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 2. Layer sweep: accuracy vs layer, one line per language ──────────────────

def plot_layer_sweep(df: pd.DataFrame, probe_type: str, model_key: str):
    sub = df[(df["probe_type"] == probe_type) & (df["model"] == model_key)]
    if sub.empty:
        logger.warning(f"No data for model={model_key} probe_type={probe_type}")
        return

    langs = sorted(sub["lang"].unique())
    layers = sorted(sub["layer"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for lang in langs:
        row = sub[sub["lang"] == lang].sort_values("layer")
        ls = "--" if LOW_RESOURCE.get(lang) else "-"
        ax.plot(row["layer"], row["accuracy"], marker="o", linestyle=ls, label=lang_label(lang))

    ax.axhline(0.85, color="gray", linestyle=":", linewidth=1, label="85% threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    ax.set_title(f"Layer sweep — {model_name} ({probe_type})")
    ax.set_xticks(layers)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out = FIG_DIR / f"layer_sweep_{model_key}_{probe_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 3. Cross-lingual alignment heatmap ────────────────────────────────────────

def plot_alignment_heatmap(matrix_path: Path):
    mat = pd.read_csv(matrix_path, index_col=0)
    if mat.empty:
        return

    langs = list(mat.index)
    labels = [lang_label(l) for l in langs]

    fig, ax = plt.subplots(figsize=(max(5, len(langs) * 0.9), max(4, len(langs) * 0.8)))
    sns.heatmap(
        mat.astype(float),
        annot=True, fmt=".2f", vmin=-1.0, vmax=1.0,
        cmap="coolwarm", linewidths=0.5, ax=ax,
        xticklabels=labels, yticklabels=labels,
    )
    # Infer title from filename
    stem = matrix_path.stem  # e.g. alignment_global_layer24
    parts = stem.split("_")
    model_part = parts[1] if len(parts) > 1 else ""
    layer_part = parts[2] if len(parts) > 2 else ""
    model_name = MODELS[model_part].name if model_part in MODELS else model_part
    ax.set_title(f"Cross-lingual direction alignment\n{model_name} / {layer_part}")
    plt.tight_layout()

    out = FIG_DIR / f"{matrix_path.stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 4. Cross-stage alignment bar chart ────────────────────────────────────────

def plot_cross_stage_alignment(cs_path: Path, probe_type: str = "diffmean"):
    df = pd.read_csv(cs_path)
    sub = df[df["probe_type"] == probe_type].copy()
    if sub.empty:
        return

    sub["pair"] = sub["stage_a"] + " → " + sub["stage_b"]
    sub["lang_label"] = sub["lang"].map(lang_label)

    pairs = sub["pair"].unique()
    langs = sorted(sub["lang_label"].unique())
    x = np.arange(len(langs))
    width = 0.8 / max(len(pairs), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.2), 5))
    for i, pair in enumerate(pairs):
        vals = []
        for lang_lbl in langs:
            row = sub[(sub["pair"] == pair) & (sub["lang_label"] == lang_lbl)]
            vals.append(row["cosine_sim"].values[0] if not row.empty else np.nan)
        ax.bar(x + i * width, vals, width, label=pair)

    ax.set_xticks(x + width * (len(pairs) - 1) / 2)
    ax.set_xticklabels(langs, rotation=20, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.7, color="gray", linestyle=":", linewidth=1, label="κ=0.7 threshold")
    ax.set_title(f"Harmful direction alignment across training stages ({probe_type})")
    ax.legend()
    plt.tight_layout()

    out = FIG_DIR / f"cross_stage_alignment_{probe_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 5. Fig-2 style: layer-wise clustering score s^l across layers ────────────
#
# s^l(h) = cos_sim(h, μ_harmful) - cos_sim(h, μ_harmless)
# Plots mean ± std band for harmful vs harmless prompts across layers.
# Mirrors Figure 2 (top row) from Zhao et al. 2025 (LLMs Encode Harmfulness
# and Refusal Separately).

def plot_layerwise_cluster_score(model_key: str, lang: str):
    acts_dir = ROOT / "data" / "processed" / "activations"
    lbl_path = acts_dir / model_key / lang / "labels.npy"
    if not lbl_path.exists():
        logger.warning(f"Missing labels for cluster score: {model_key}/{lang}")
        return

    labels = np.load(lbl_path)
    cfg    = MODELS[model_key] if model_key in MODELS else None
    if cfg is None:
        return

    layers = sorted(cfg.probe_layers)
    scores_harmful  = {l: None for l in layers}
    scores_harmless = {l: None for l in layers}

    for layer in layers:
        arr_path = acts_dir / model_key / lang / f"layer_{layer}.npy"
        if not arr_path.exists():
            continue
        acts = np.load(arr_path).astype(np.float32)

        mu_harmful  = acts[labels == 1].mean(0)
        mu_harmless = acts[labels == 0].mean(0)

        # Normalise cluster centres
        mu_harmful  /= (np.linalg.norm(mu_harmful)  + 1e-8)
        mu_harmless /= (np.linalg.norm(mu_harmless) + 1e-8)
        acts_norm    = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)

        s = acts_norm @ mu_harmful - acts_norm @ mu_harmless
        scores_harmful[layer]  = s[labels == 1]
        scores_harmless[layer] = s[labels == 0]

    valid_layers = [l for l in layers if scores_harmful[l] is not None]
    if not valid_layers:
        return

    means_h  = [scores_harmful[l].mean()  for l in valid_layers]
    stds_h   = [scores_harmful[l].std()   for l in valid_layers]
    means_hl = [scores_harmless[l].mean() for l in valid_layers]
    stds_hl  = [scores_harmless[l].std()  for l in valid_layers]

    x = np.array(valid_layers)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, means_h,  color="#EE6677", label="harmful")
    ax.fill_between(x,
                    np.array(means_h) - np.array(stds_h),
                    np.array(means_h) + np.array(stds_h),
                    color="#EE6677", alpha=0.2)

    ax.plot(x, means_hl, color="#4477AA", label="harmless")
    ax.fill_between(x,
                    np.array(means_hl) - np.array(stds_hl),
                    np.array(means_hl) + np.array(stds_hl),
                    color="#4477AA", alpha=0.2)

    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$s^l(h^l)$")
    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    ax.set_title(f"Layer-wise clustering score — {model_name} / {LANG_NAMES.get(lang, lang)}")
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()

    out = FIG_DIR / f"cluster_score_{model_key}_{lang}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 6. Fig-7 style: Δ_harmful across layers for training-stage sweep ──────────
#
# For each model in `stage_models`, compute mean s^l for harmful prompts at
# each layer and plot as lines — shows how the harmfulness signal grows with
# more SFT data. Mirrors Figure 7 from Zhao et al. 2025.

def plot_sft_sweep_cluster_score(
    lang: str,
    stage_models: list[str],
    layer_subset: list[int] | None = None,
):
    acts_dir = ROOT / "data" / "processed" / "activations"

    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False

    for model_key in stage_models:
        if model_key not in MODELS:
            continue
        cfg      = MODELS[model_key]
        layers   = sorted(layer_subset or cfg.probe_layers)
        lbl_path = acts_dir / model_key / lang / "labels.npy"
        if not lbl_path.exists():
            continue
        labels = np.load(lbl_path)

        means = []
        valid = []
        for layer in layers:
            arr_path = acts_dir / model_key / lang / f"layer_{layer}.npy"
            if not arr_path.exists():
                continue
            acts = np.load(arr_path).astype(np.float32)
            mu_harmful  = acts[labels == 1].mean(0)
            mu_harmless = acts[labels == 0].mean(0)
            mu_harmful  /= (np.linalg.norm(mu_harmful)  + 1e-8)
            mu_harmless /= (np.linalg.norm(mu_harmless) + 1e-8)
            acts_norm    = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
            s = acts_norm @ mu_harmful - acts_norm @ mu_harmless
            means.append(s[labels == 1].mean())
            valid.append(layer)

        if valid:
            label = cfg.name
            ax.plot(valid, means, marker="o", markersize=4, label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\Delta_\mathrm{harmful}$ (mean $s^l$ on harmful prompts)")
    ax.set_title(f"Harmfulness signal by SFT data size — {LANG_NAMES.get(lang, lang)}")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    model_tag = "_".join(m.replace("base_sft_", "") for m in stage_models if m in MODELS)
    out = FIG_DIR / f"sft_sweep_cluster_{lang}_{model_tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 7. UMAP of activations ────────────────────────────────────────────────────

def plot_umap(model_key: str, lang: str, layer: int):
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP plots. pip install umap-learn")
        return

    acts_dir = ROOT / "data" / "processed" / "activations"
    arr_path = acts_dir / model_key / lang / f"layer_{layer}.npy"
    lbl_path = acts_dir / model_key / lang / "labels.npy"
    if not arr_path.exists() or not lbl_path.exists():
        logger.warning(f"Missing activations for UMAP: {model_key}/{lang}/layer_{layer}")
        return

    acts   = np.load(arr_path)
    labels = np.load(lbl_path)

    reducer  = UMAP(n_components=2, random_state=42, verbose=False)
    embedded = reducer.fit_transform(acts)

    fig, ax = plt.subplots(figsize=(7, 6))
    for val, color, label in [(0, "#4477AA", "harmless"), (1, "#EE6677", "harmful")]:
        mask = labels == val
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=color, label=label, s=8, alpha=0.6, rasterized=True)

    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    ax.set_title(f"UMAP — {model_name} / {LANG_NAMES.get(lang, lang)} / layer {layer}")
    ax.legend(markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    out = FIG_DIR / f"umap_{model_key}_{lang}_layer{layer}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 8. Refusal rates by model × language ─────────────────────────────────────

def plot_refusal_rates(behavioral_path: Path):
    if not behavioral_path.exists():
        logger.warning(f"Behavioral summary not found: {behavioral_path}")
        return

    df = pd.read_csv(behavioral_path)
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    langs_present  = sorted(df["lang"].unique(), key=lambda l: LANG_NAMES.get(l, l))

    pivot = df.pivot_table(index="lang", columns="model", values="refusal_rate")
    pivot = pivot.reindex(index=langs_present, columns=models_present)

    fig, ax = plt.subplots(figsize=(max(8, len(models_present) * 1.2), max(4, len(langs_present) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f", vmin=0.0, vmax=1.0,
        cmap="Blues", linewidths=0.5, ax=ax,
        yticklabels=[lang_label(l) for l in langs_present],
        xticklabels=[MODELS[m].name if m in MODELS else m for m in models_present],
    )
    ax.set_title("Refusal rate (behavioral)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = FIG_DIR / "refusal_rates.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 9. Harmful vs refusal direction cosine alignment by layer ─────────────────

def plot_harmful_vs_refusal(hvr_path: Path):
    if not hvr_path.exists():
        logger.warning(f"harmful_vs_refusal_alignment.csv not found: {hvr_path}")
        return

    df = pd.read_csv(hvr_path)
    if df.empty:
        return

    models = df["model"].unique()
    fig, ax = plt.subplots(figsize=(8, 4))
    for model_key in models:
        sub = df[df["model"] == model_key].sort_values("layer")
        model_name = MODELS[model_key].name if model_key in MODELS else model_key
        ax.plot(sub["layer"], sub["cosine_sim"], marker="o", label=model_name)

    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Harmful direction ↔ Refusal direction alignment")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out = FIG_DIR / "harmful_vs_refusal_alignment.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── 10. Checkpoint trajectory: probe accuracy and cosine-vs-final over steps ──

def plot_checkpoint_trajectory(traj_path: Path):
    if not traj_path.exists():
        logger.warning(f"Checkpoint trajectory not found: {traj_path}")
        return

    df = pd.read_csv(traj_path)
    if df.empty:
        return

    base_model = df["base_model"].iloc[0]
    layers = sorted(df["layer"].unique())
    langs  = sorted(df["lang"].unique())

    fig, axes = plt.subplots(
        2, len(layers),
        figsize=(max(10, len(layers) * 3), 6),
        sharey="row",
    )
    if len(layers) == 1:
        axes = axes.reshape(2, 1)

    for col, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        for lang in langs:
            lang_sub = sub[sub["lang"] == lang].sort_values("step")
            ls = "--" if LOW_RESOURCE.get(lang) else "-"
            axes[0, col].plot(lang_sub["step"], lang_sub["accuracy"],
                              marker="o", markersize=3, linestyle=ls, label=lang_label(lang))
            axes[1, col].plot(lang_sub["step"], lang_sub["cosine_vs_final"],
                              marker="o", markersize=3, linestyle=ls, label=lang_label(lang))

        axes[0, col].set_title(f"Layer {layer}", fontsize=9)
        axes[0, col].axhline(0.85, color="gray", linestyle=":", linewidth=1)
        axes[1, col].axhline(1.0,  color="gray", linestyle=":", linewidth=1)
        axes[0, col].set_xlabel("Step")
        axes[1, col].set_xlabel("Step")

    axes[0, 0].set_ylabel("Probe accuracy")
    axes[1, 0].set_ylabel("Cosine vs final")
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", fontsize=7,
               bbox_to_anchor=(1.0, 1.0))
    model_name = MODELS[base_model].name if base_model in MODELS else base_model
    fig.suptitle(f"Checkpoint trajectory — {model_name}", y=1.01)
    plt.tight_layout()

    stem = traj_path.stem  # checkpoint_trajectory_base_sft
    out  = FIG_DIR / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-type", choices=["diffmean", "logreg", "both"],
                        default="both")
    parser.add_argument("--model",  default=None,
                        help="Model key for layer sweep (default: all models in data)")
    parser.add_argument("--layer",  type=int, default=None,
                        help="Fix a specific layer for accuracy heatmap")
    parser.add_argument("--umap",         action="store_true",
                        help="Generate UMAP scatter plots (requires umap-learn)")
    parser.add_argument("--cluster-score", action="store_true",
                        help="Generate layer-wise clustering score plots (Fig 2 style)")
    parser.add_argument("--sft-sweep",    action="store_true",
                        help="Generate SFT data-size sweep plot (Fig 7 style)")
    parser.add_argument("--summary",  default="results/probes/summary.csv")
    parser.add_argument("--align-dir", default="results/probes",
                        help="Directory containing alignment_*.csv files")
    args = parser.parse_args()

    summary_path = ROOT / args.summary
    if not summary_path.exists():
        logger.error(f"summary.csv not found at {summary_path}. Run 06_train_probes.py first.")
        sys.exit(1)

    df = pd.read_csv(summary_path)
    logger.info(f"Loaded summary: {len(df)} rows, models={df['model'].unique().tolist()}")

    probe_types = ["diffmean", "logreg"] if args.probe_type == "both" else [args.probe_type]

    # 1. Accuracy heatmaps
    for pt in probe_types:
        plot_accuracy_heatmap(df, pt, layer=args.layer)

    # 2. Layer sweeps
    models_in_data = df["model"].unique().tolist()
    sweep_models = [args.model] if args.model else models_in_data
    for m in sweep_models:
        for pt in probe_types:
            plot_layer_sweep(df, pt, m)

    # 3. Cross-lingual alignment heatmaps
    align_dir = ROOT / args.align_dir
    for align_csv in sorted(align_dir.glob("alignment_*.csv")):
        plot_alignment_heatmap(align_csv)

    # 4. Cross-stage alignment
    cs_path = ROOT / "results" / "probes" / "cross_stage_alignment.csv"
    if cs_path.exists():
        for pt in probe_types:
            plot_cross_stage_alignment(cs_path, pt)

    # 5. Layer-wise clustering score (Fig 2 style)
    if args.cluster_score:
        from config.languages import LANG_CODES
        for model_key in sweep_models:
            if model_key not in MODELS:
                continue
            for lang in LANG_CODES:
                plot_layerwise_cluster_score(model_key, lang)

    # 6. SFT sweep Δ_harmful (Fig 7 style)
    if args.sft_sweep:
        from config.languages import LANG_CODES
        sweep_stage_models = ["base_sft", "base_sft_n500", "base_sft_n2000",
                              "base_sft_n5000", "base_sft_n10000", "base_sft_n20000"]
        for lang in LANG_CODES:
            plot_sft_sweep_cluster_score(lang, sweep_stage_models)

    # 8. Refusal rates heatmap
    behavioral_path = ROOT / "results" / "behavioral" / "baseline_summary.csv"
    plot_refusal_rates(behavioral_path)

    # 9. Harmful vs refusal direction alignment
    hvr_path = ROOT / "results" / "probes" / "harmful_vs_refusal_alignment.csv"
    plot_harmful_vs_refusal(hvr_path)

    # 10. Checkpoint trajectories
    for traj_csv in sorted((ROOT / "results" / "probes").glob("checkpoint_trajectory_*.csv")):
        plot_checkpoint_trajectory(traj_csv)

    # 7. UMAP (optional)
    if args.umap:
        for model_key in (sweep_models if args.model else models_in_data):
            if model_key not in MODELS:
                continue
            for lang in df[df["model"] == model_key]["lang"].unique():
                best_layer = (
                    df[(df["model"] == model_key) & (df["lang"] == lang) & (df["probe_type"] == "logreg")]
                    .sort_values("accuracy", ascending=False)["layer"]
                    .iloc[0] if not df.empty else MODELS[model_key].probe_layers[-1]
                )
                plot_umap(model_key, lang, int(best_layer))

    print(f"\nDone. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
