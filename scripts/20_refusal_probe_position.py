"""
Step 20 — Refusal probe: instruction position vs full-prompt position + balanced sampling.

Open question from script 19: refusal_acc ≈ 0.55 at the instruction position.
Two candidate explanations:
  (a) Refusal is not linearly encoded at input positions (true null result)
  (b) Class imbalance (15–25% refused) masked the signal

This script tests both by:
  1. Comparing t_inst (last user-turn token) vs eval (last input token = assistant
     turn-start marker) — the only two positions that actually differ in the data.
     NOTE: activations_postinst == activations_eval (same last-input-token position),
     so they are excluded from comparison.
  2. Using balanced LogReg (class_weight='balanced') and centre-threshold DiffMean
     so class imbalance cannot mask signal.
  3. Reporting AUC (threshold-independent) as the primary metric.

Run:
  python scripts/20_refusal_probe_position.py --lang en
  python scripts/20_refusal_probe_position.py --all

Writes:
  results/probes/refusal_position_comparison.csv
  results/figures/fig_refusal_position_{lang}.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES
from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR  = ROOT / "results" / "figures"
PROB_DIR = ROOT / "results" / "probes"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# t_inst = last token of user turn (add_generation_prompt=False)
# eval   = last token of full input with assistant turn-start (add_generation_prompt=True)
# NOTE: activations_postinst is identical to activations_eval (both capture last input token)
ACTS_ROOTS = {
    "t_inst": ROOT / "data" / "processed" / "activations_tinst",
    "eval":   ROOT / "data" / "processed" / "activations_eval",
}

STAGE_MODELS = ["base", "base_sft", "base_dpo"]
STAGE_LABELS = {"base": "Base", "base_sft": "Base+SFT", "base_dpo": "Base+DPO"}
STAGE_COLORS = {"base": "#888888", "base_sft": "#4477AA", "base_dpo": "#EE4444"}
POS_STYLES   = {"t_inst": "-", "eval": "--"}
POS_LABELS   = {"t_inst": "t_inst (last user token)", "eval": "eval (asst turn-start)"}


# ── data helpers ──────────────────────────────────────────────────────────────

def load_acts(acts_root: Path, model_key: str, lang: str, layer: int):
    p = acts_root / model_key / lang / f"layer_{layer}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def load_prompts(acts_root: Path, model_key: str, lang: str) -> list[dict]:
    p = acts_root / model_key / lang / "prompts.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_refusal_map(model_key: str, lang: str) -> dict[str, bool]:
    path = ROOT / "results" / "behavioral" / f"responses_{model_key}_{lang}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = bool(d.get("heuristic_refuse", False))
    return out


# ── probe training ────────────────────────────────────────────────────────────

def diffmean_balanced_threshold(acts: np.ndarray, labels: np.ndarray):
    """DiffMean with midpoint-of-class-centroids threshold (not all-scores mean)."""
    mu_pos = acts[labels == 1].mean(0)
    mu_neg = acts[labels == 0].mean(0)
    d = mu_pos - mu_neg
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return float("nan"), float("nan"), float("nan")
    d = d / norm
    scores = acts @ d
    threshold = (scores[labels == 1].mean() + scores[labels == 0].mean()) / 2
    preds = (scores > threshold).astype(int)
    return (
        float(accuracy_score(labels, preds)),
        float(balanced_accuracy_score(labels, preds)),
        float(roc_auc_score(labels, scores)),
    )


def logreg_balanced_cv(acts: np.ndarray, labels: np.ndarray, n_folds: int = 5):
    """LogReg with class_weight='balanced' and stratified CV."""
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    scaler = StandardScaler()
    X = scaler.fit_transform(acts)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs, fold_baccs, fold_aucs = [], [], []
    for train_idx, val_idx in skf.split(X, labels):
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"
        )
        clf.fit(X[train_idx], labels[train_idx])
        preds = clf.predict(X[val_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        fold_accs.append(accuracy_score(labels[val_idx], preds))
        fold_baccs.append(balanced_accuracy_score(labels[val_idx], preds))
        fold_aucs.append(roc_auc_score(labels[val_idx], probs))
    return float(np.mean(fold_accs)), float(np.mean(fold_baccs)), float(np.mean(fold_aucs))


# ── per model/lang/position ───────────────────────────────────────────────────

def analyse_one(model_key: str, lang: str, position: str) -> pd.DataFrame:
    acts_root = ACTS_ROOTS[position]
    if not (acts_root / model_key / lang).exists():
        return pd.DataFrame()

    prompts = load_prompts(acts_root, model_key, lang)
    if not prompts:
        return pd.DataFrame()

    refusal_map = load_refusal_map(model_key, lang)
    if not refusal_map:
        logger.warning(f"No behavioral responses: {model_key}/{lang}")
        return pd.DataFrame()

    harmful_idx, refusal_labels = [], []
    for i, p in enumerate(prompts):
        if p.get("label") != "harmful":
            continue
        pid = p.get("id", "")
        if pid not in refusal_map:
            continue
        harmful_idx.append(i)
        refusal_labels.append(1 if refusal_map[pid] else 0)

    harmful_idx    = np.array(harmful_idx, dtype=int)
    refusal_labels = np.array(refusal_labels, dtype=np.int8)
    n_refused  = int(refusal_labels.sum())
    n_complied = int(len(refusal_labels) - n_refused)

    logger.info(f"{model_key}/{lang}/{position}: {n_refused} refused / {n_complied} complied")
    if n_refused < 5 or n_complied < 5:
        logger.warning(f"Skipping — too few examples")
        return pd.DataFrame()

    cfg = MODELS.get(model_key)
    if cfg is None:
        return pd.DataFrame()

    rows = []
    for layer in cfg.probe_layers:
        acts = load_acts(acts_root, model_key, lang, layer)
        if acts is None:
            continue
        acts_harmful = acts[harmful_idx]

        dm_acc, dm_bacc, dm_auc = diffmean_balanced_threshold(acts_harmful, refusal_labels)
        lr_acc, lr_bacc, lr_auc = logreg_balanced_cv(acts_harmful, refusal_labels)

        rows.append({
            "model":      model_key,
            "lang":       lang,
            "layer":      layer,
            "position":   position,
            "n_refused":  n_refused,
            "n_complied": n_complied,
            "refusal_pct": round(n_refused / (n_refused + n_complied), 3),
            "dm_acc":     round(dm_acc,  4) if not np.isnan(dm_acc)  else float("nan"),
            "dm_bacc":    round(dm_bacc, 4) if not np.isnan(dm_bacc) else float("nan"),
            "dm_auc":     round(dm_auc,  4) if not np.isnan(dm_auc)  else float("nan"),
            "lr_acc":     round(lr_acc,  4) if not np.isnan(lr_acc)  else float("nan"),
            "lr_bacc":    round(lr_bacc, 4) if not np.isnan(lr_bacc) else float("nan"),
            "lr_auc":     round(lr_auc,  4) if not np.isnan(lr_auc)  else float("nan"),
        })
        logger.info(
            f"  layer{layer}: DM auc={dm_auc:.3f} bacc={dm_bacc:.3f} | "
            f"LR auc={lr_auc:.3f} bacc={lr_bacc:.3f}"
        )

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_position_comparison(df: pd.DataFrame, lang: str):
    lang_name = LANG_NAMES.get(lang, lang)
    layers = sorted(df["layer"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_auc, ax_bacc = axes

    for model_key in STAGE_MODELS:
        for position in ["t_inst", "eval"]:
            sub = df[
                (df["model"] == model_key) &
                (df["lang"] == lang) &
                (df["position"] == position)
            ].sort_values("layer")
            if sub.empty:
                continue
            color = STAGE_COLORS[model_key]
            ls    = POS_STYLES[position]
            label = f"{STAGE_LABELS[model_key]} / {position}"
            ax_auc.plot(sub["layer"],  sub["lr_auc"],  color=color, ls=ls,
                        marker="o", markersize=4, label=label)
            ax_bacc.plot(sub["layer"], sub["lr_bacc"], color=color, ls=ls,
                         marker="o", markersize=4, label=label)

    for ax, title, ylabel in [
        (ax_auc,  "Balanced LogReg AUC", "AUC"),
        (ax_bacc, "Balanced LogReg Balanced Accuracy", "Balanced Accuracy"),
    ]:
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(layers)
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(0.44, 0.85)

    fig.suptitle(
        f"Refusal probe: t_inst (user turn end) vs eval (asst turn-start)\n"
        f"{lang_name}  —  solid=t_inst, dashed=eval", fontsize=9
    )
    plt.tight_layout()
    out = FIG_DIR / f"fig_refusal_position_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANG_CODES, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    langs = LANG_CODES if args.all else ([args.lang] if args.lang else ["en"])
    if not args.lang and not args.all:
        logger.info("Defaulting to en")

    all_rows = []
    for lang in langs:
        for model_key in STAGE_MODELS:
            for position, acts_root in ACTS_ROOTS.items():
                if not (acts_root / model_key / lang).exists():
                    continue
                logger.info(f"--- {model_key}/{lang}/{position} ---")
                df = analyse_one(model_key, lang, position)
                if not df.empty:
                    all_rows.append(df)

    if not all_rows:
        logger.error("No data found.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    out_csv = PROB_DIR / "refusal_position_comparison.csv"
    full_df.to_csv(out_csv, index=False)
    logger.info(f"Saved → {out_csv}")

    # Print AUC summary
    summary = full_df[full_df["layer"] == 18][
        ["model", "lang", "position", "n_refused", "refusal_pct", "dm_auc", "lr_auc", "lr_bacc"]
    ].sort_values(["lang", "model", "position"])
    print(f"\nLayer-18 summary:\n{summary.to_string(index=False)}")

    for lang in langs:
        lang_df = full_df[full_df["lang"] == lang]
        if lang_df.empty:
            continue
        plot_position_comparison(lang_df, lang)

    print("\nDone.")


if __name__ == "__main__":
    main()
