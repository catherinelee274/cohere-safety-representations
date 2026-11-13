"""
Step 22 — Checkpoint trajectory and data ablation analysis.

Two analyses using the same core metrics (logreg_acc, harm_drift_cos, harm_refusal_cos):

1. Checkpoint trajectory: how safety representations evolve step-by-step during SFT/DPO.
   Models: base → base_sft_ckpt{96,312,624,936,1248,1560,1872,2169} → base_sft
           base → base_dpo_ckpt{15,312,540} → base_dpo

2. Data ablation: how training set size affects safety representations.
   Models: base_sft_n500 → base_sft_n2000 → base_sft_n20000 → base_sft

Reads:
  results/probes/{model}/{lang}/layer_{n}_{diffmean,logreg,refusal}.pkl
  data/processed/activations/{model}/{lang}/

Writes:
  results/probes/trajectory_summary.csv
  results/probes/ablation_summary.csv
  results/figures/fig_trajectory_{sft|dpo}_{lang}.png
  results/figures/fig_ablation_{lang}.png

Run:
  python scripts/22_trajectory_ablation.py             # both analyses, all langs
  python scripts/22_trajectory_ablation.py --analysis trajectory
  python scripts/22_trajectory_ablation.py --analysis ablation
  python scripts/22_trajectory_ablation.py --lang en
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Search activations_eval/ first (base/sft/dpo use clean eval set), then activations/
ACTS_DIRS = [
    ROOT / "data" / "processed" / "activations_eval",
    ROOT / "data" / "processed" / "activations",
]
PROBE_DIR = ROOT / "results" / "probes"
FIG_DIR   = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _acts_dir(model_key: str, lang: str):
    for root in ACTS_DIRS:
        d = root / model_key / lang
        if d.exists():
            return d
    return None

PROBE_LAYERS = [4, 9, 18, 24, 30, 35]
FOCUS_LAYER  = 18

SFT_CHECKPOINTS = [
    (0,    "base"),
    (96,   "base_sft_ckpt96"),
    (312,  "base_sft_ckpt312"),
    (624,  "base_sft_ckpt624"),
    (936,  "base_sft_ckpt936"),
    (1248, "base_sft_ckpt1248"),
    (1560, "base_sft_ckpt1560"),
    (1872, "base_sft_ckpt1872"),
    (2169, "base_sft_ckpt2169"),
    (2169, "base_sft"),
]

DPO_CHECKPOINTS = [
    (0,   "base"),
    (15,  "base_dpo_ckpt15"),
    (312, "base_dpo_ckpt312"),
    (540, "base_dpo_ckpt540"),
    (540, "base_dpo"),
]

N_VARIANTS = [
    (500,   "base_sft_n500"),
    (2000,  "base_sft_n2000"),
    (20000, "base_sft_n20000"),
    (None,  "base_sft"),
]

SKIP_LANGS = {"fil"}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_probe(model_key: str, lang: str, layer: int, probe_type: str):
    path = PROBE_DIR / model_key / lang / f"layer_{layer}_{probe_type}.pkl"
    return pickle.load(open(path, "rb")) if path.exists() else None


def load_acts(model_key: str, lang: str, layer: int):
    d = _acts_dir(model_key, lang)
    if d is None:
        return None
    p = d / f"layer_{layer}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def load_labels(model_key: str, lang: str):
    d = _acts_dir(model_key, lang)
    if d is None:
        return None
    p = d / "labels.npy"
    return np.load(p).astype(np.int8) if p.exists() else None


def canon_harm_dir(direction, acts, labels):
    d = direction / (np.linalg.norm(direction) + 1e-8)
    if (labels == 1).any() and (labels == 0).any():
        mu_h = acts[labels == 1].mean(0)
        mu_l = acts[labels == 0].mean(0)
        if np.dot(mu_h, d) < np.dot(mu_l, d):
            d = -d
    return d


# ── metric computation ────────────────────────────────────────────────────────

def compute_metrics_for(model_key: str, lang: str, ref_key: str = "base") -> list[dict]:
    labels = load_labels(model_key, lang)
    if labels is None:
        logger.warning(f"No labels: {model_key}/{lang}")
        return []

    ref_labels = load_labels(ref_key, lang)

    rows = []
    for layer in PROBE_LAYERS:
        dm   = load_probe(model_key, lang, layer, "diffmean")
        lr   = load_probe(model_key, lang, layer, "logreg")
        rf   = load_probe(model_key, lang, layer, "refusal")
        acts = load_acts(model_key, lang, layer)

        row = {"model": model_key, "lang": lang, "layer": layer}
        row["logreg_acc"]   = round(lr.accuracy, 4) if lr else float("nan")
        row["logreg_auc"]   = round(lr.auc,      4) if lr else float("nan")
        row["diffmean_acc"] = round(dm.accuracy, 4) if dm else float("nan")
        row["refusal_acc"]  = round(rf.accuracy, 4) if rf else float("nan")

        if dm and rf and acts is not None:
            d_harm = canon_harm_dir(dm.direction.copy(), acts, labels)
            d_ref  = rf.direction / (np.linalg.norm(rf.direction) + 1e-8)
            row["harm_refusal_cos"] = round(float(np.dot(d_harm, d_ref)), 4)
        else:
            row["harm_refusal_cos"] = float("nan")

        # harm_drift_cos: recompute DiffMean directly from activations (like script 19)
        # so that probe-pkl sign artifacts and prompt-set mismatches don't affect the result.
        if model_key != ref_key and acts is not None and ref_labels is not None:
            ref_acts = load_acts(ref_key, lang, layer)
            if ref_acts is not None:
                def centroid_dir(a, lbl):
                    if (lbl == 1).any() and (lbl == 0).any():
                        d = a[lbl == 1].mean(0) - a[lbl == 0].mean(0)
                        n = np.linalg.norm(d)
                        return d / n if n > 1e-8 else d
                    return None
                d_this = centroid_dir(acts, labels)
                d_ref2 = centroid_dir(ref_acts, ref_labels)
                if d_this is not None and d_ref2 is not None:
                    row["harm_drift_cos"] = round(float(np.dot(d_this, d_ref2)), 4)
                else:
                    row["harm_drift_cos"] = float("nan")
            else:
                row["harm_drift_cos"] = float("nan")
        else:
            row["harm_drift_cos"] = 1.0

        rows.append(row)
        logger.info(
            f"  {model_key}/{lang}/layer{layer}: "
            f"lr_acc={row['logreg_acc']:.3f}  "
            f"harm_ref_cos={row['harm_refusal_cos']:.3f}  "
            f"drift={row['harm_drift_cos']:.3f}"
        )
    return rows


# ── trajectory analysis ───────────────────────────────────────────────────────

def run_trajectory(stage: str, checkpoints: list, lang: str) -> pd.DataFrame:
    all_rows = []
    seen_models = set()
    for step, model_key in checkpoints:
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        if _acts_dir(model_key, lang) is None:
            logger.warning(f"Missing activations: {model_key}/{lang}")
            continue
        logger.info(f"=== trajectory {stage} step={step} {model_key}/{lang}")
        rows = compute_metrics_for(model_key, lang)
        for r in rows:
            r["stage"]    = stage
            r["step"]     = step
            r["is_final"] = model_key in ("base_sft", "base_dpo")
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def plot_trajectory(df: pd.DataFrame, stage: str, lang: str):
    lang_name = LANG_NAMES.get(lang, lang)
    focus = df[df["layer"] == FOCUS_LAYER].copy()
    if focus.empty:
        return

    max_ckpt_step = focus[~focus["is_final"]]["step"].max()
    x_final = max_ckpt_step * 1.1

    focus["x"] = focus.apply(
        lambda r: x_final if r["is_final"] else r["step"], axis=1
    )

    color = "#4477AA" if stage == "sft" else "#EE4444"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ["logreg_acc", "harm_drift_cos", "harm_refusal_cos"]
    titles  = ["LogReg Accuracy", "Harm Direction Drift (vs Base)", "Harm↔Refusal Cosine"]

    for ax, metric, title in zip(axes, metrics, titles):
        sub = focus[~focus["is_final"]].sort_values("x")
        ax.plot(sub["x"], sub[metric], marker="o", color=color, linewidth=1.5, markersize=5)
        final = focus[focus["is_final"]]
        if not final.empty:
            ax.scatter([x_final], final[metric].values, marker="*", s=120,
                       color=color, zorder=5, label=f"base_{stage} (final)")
        base_rows = focus[focus["model"] == "base"]
        if not base_rows.empty:
            ax.axhline(base_rows[metric].values[0], color="gray",
                       linestyle=":", linewidth=0.8, label="base")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)

    stage_label = "SFT" if stage == "sft" else "DPO"
    fig.suptitle(f"{stage_label} checkpoint trajectory — layer {FOCUS_LAYER} — {lang_name}", fontsize=10)
    fig.subplots_adjust(wspace=0.35, top=0.88)
    out = FIG_DIR / f"fig_trajectory_{stage}_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def plot_trajectory_layers(df: pd.DataFrame, stage: str, lang: str):
    pivot = df.pivot_table(index="model", columns="layer", values="logreg_acc")
    if pivot.empty:
        return

    checkpoints = SFT_CHECKPOINTS if stage == "sft" else DPO_CHECKPOINTS
    order = [mk for _, mk in checkpoints if mk in pivot.index]
    pivot = pivot.reindex([m for m in order if m in pivot.index])

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.55)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"L{l}" for l in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if v > 0.75 else "white")
    plt.colorbar(im, ax=ax, label="LogReg Accuracy")
    stage_label = "SFT" if stage == "sft" else "DPO"
    ax.set_title(f"{stage_label} checkpoint × layer logreg_acc — {lang}", fontsize=9)
    fig.subplots_adjust(left=0.25)
    out = FIG_DIR / f"fig_trajectory_{stage}_heatmap_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── data ablation ─────────────────────────────────────────────────────────────

def run_ablation(lang: str) -> pd.DataFrame:
    all_rows = []
    for n_train, model_key in N_VARIANTS:
        if _acts_dir(model_key, lang) is None:
            logger.warning(f"Missing activations: {model_key}/{lang}")
            continue
        logger.info(f"=== ablation {model_key}/{lang} (n={n_train})")
        rows = compute_metrics_for(model_key, lang)
        for r in rows:
            r["n_train"] = n_train if n_train is not None else "full"
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def plot_ablation(df: pd.DataFrame, lang: str):
    lang_name = LANG_NAMES.get(lang, lang)
    focus = df[df["layer"] == FOCUS_LAYER].copy()
    if focus.empty:
        return

    full_x = 30000

    def get_x(row):
        return full_x if row["n_train"] == "full" else int(row["n_train"])

    focus["x"] = focus.apply(get_x, axis=1)

    metrics = ["logreg_acc", "harm_drift_cos", "harm_refusal_cos"]
    titles  = ["LogReg Accuracy", "Harm Direction Drift (vs Base)", "Harm↔Refusal Cosine"]

    sub_mask  = focus["n_train"] != "full"
    full_mask = focus["n_train"] == "full"
    sub_xs    = sorted(focus.loc[sub_mask, "x"].dropna().unique())
    use_log   = len(sub_xs) > 0 and all(x > 0 for x in sub_xs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, metric, title in zip(axes, metrics, titles):
        sub  = focus[sub_mask].sort_values("x")
        full = focus[full_mask]

        if not sub.empty:
            ax.plot(sub["x"], sub[metric], marker="o", color="#4477AA",
                    linewidth=1.5, markersize=5)
        if not full.empty:
            ax.scatter([full_x], full[metric].values, marker="*", s=120,
                       color="#4477AA", zorder=5, label="base_sft (full)")
        base_rows = focus[focus["model"] == "base"]
        if not base_rows.empty:
            ax.axhline(base_rows[metric].values[0], color="gray",
                       linestyle=":", linewidth=0.8, label="base")

        if use_log:
            ax.set_xlim(left=min(sub_xs) * 0.5, right=full_x * 1.5)
            ax.set_xscale("log")
            ax.set_xlabel("Training samples (log scale)")
        else:
            ax.set_xlabel("Training samples")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)

    fig.suptitle(f"Data ablation — SFT training size — layer {FOCUS_LAYER} — {lang_name}", fontsize=10)
    fig.subplots_adjust(wspace=0.35, top=0.88)
    out = FIG_DIR / f"fig_ablation_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=["trajectory", "ablation"], default=None)
    parser.add_argument("--lang", choices=LANG_CODES, default=None)
    args = parser.parse_args()

    do_traj     = args.analysis in (None, "trajectory")
    do_ablation = args.analysis in (None, "ablation")

    langs = [args.lang] if args.lang else [l for l in LANG_CODES if l not in SKIP_LANGS]

    # ── trajectory ────────────────────────────────────────────────────────────
    if do_traj:
        traj_rows = []
        for lang in langs:
            for stage, checkpoints in [("sft", SFT_CHECKPOINTS), ("dpo", DPO_CHECKPOINTS)]:
                df = run_trajectory(stage, checkpoints, lang)
                if df.empty:
                    continue
                traj_rows.append(df)
                plot_trajectory(df, stage, lang)
                plot_trajectory_layers(df, stage, lang)

        if traj_rows:
            full_df = pd.concat(traj_rows, ignore_index=True)
            out = PROBE_DIR / "trajectory_summary.csv"
            full_df.to_csv(out, index=False)
            logger.info(f"Saved → {out}")
            print("\n=== Trajectory layer-18 summary ===")
            print(full_df[full_df["layer"] == FOCUS_LAYER][
                ["stage", "step", "model", "lang", "logreg_acc", "harm_drift_cos", "harm_refusal_cos"]
            ].to_string(index=False))

    # ── ablation ──────────────────────────────────────────────────────────────
    if do_ablation:
        abl_rows = []
        for lang in langs:
            df = run_ablation(lang)
            if df.empty:
                continue
            abl_rows.append(df)
            plot_ablation(df, lang)

        if abl_rows:
            full_df = pd.concat(abl_rows, ignore_index=True)
            out = PROBE_DIR / "ablation_summary.csv"
            full_df.to_csv(out, index=False)
            logger.info(f"Saved → {out}")
            print("\n=== Ablation layer-18 summary ===")
            print(full_df[full_df["layer"] == FOCUS_LAYER][
                ["n_train", "model", "lang", "logreg_acc", "harm_drift_cos", "harm_refusal_cos"]
            ].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
