"""
Step 15 — Replicate Zhao et al. (2025) Figure 2 for TinyAya.

Tests the core claim: at t_inst hidden states cluster by harmfulness,
at t_post_inst hidden states cluster by refusal/acceptance.

Requires activations extracted at TWO positions:
  1. t_inst    → data/processed/activations_tinst/{model}/{lang}/
  2. post_inst → data/processed/activations_postinst/{model}/{lang}/

Run extraction first:
  python scripts/05_extract_activations.py \\
      --model base --harmful-file harmful_eval.jsonl \\
      --aggregation t_inst --out-dir data/processed/activations_tinst

  python scripts/05_extract_activations.py \\
      --model base --harmful-file harmful_eval.jsonl \\
      --aggregation post_inst --out-dir data/processed/activations_postinst

Then:
  python scripts/15_harmfulness_refusal_clustering.py --model base --lang en
  python scripts/15_harmfulness_refusal_clustering.py --model base --all

Writes:
  results/probes/harmfulness_refusal_clustering_{model}_{lang}.csv
  results/figures/fig2_{model}_{lang}.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES, LANG_NAMES
from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR  = ROOT / "results" / "figures"
PROB_DIR = ROOT / "results" / "probes"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── data loading ──────────────────────────────────────────────────────────────

def load_acts(acts_root: Path, model_key: str, lang: str, layer: int):
    p = acts_root / model_key / lang / f"layer_{layer}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def load_prompts(acts_root: Path, model_key: str, lang: str) -> list[dict]:
    p = acts_root / model_key / lang / "prompts.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_refusal_labels(resp_dir: Path, model_key: str, lang: str) -> dict[str, bool]:
    """Return {prompt_id: heuristic_refuse} from responses_{model}_{lang}.jsonl."""
    path = resp_dir / f"responses_{model_key}_{lang}.jsonl"
    if not path.exists():
        return {}
    labels = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels[d["id"]] = bool(d.get("heuristic_refuse", False))
    return labels


def build_groups(
    prompts: list[dict],
    refusal_labels: dict[str, bool],
) -> dict[str, np.ndarray]:
    """
    Return index arrays for four groups.
    Harmless prompts: all treated as accepted (no response tracking).
    Harmful prompts: split by heuristic_refuse label.
    """
    refused_harmful  = []
    accepted_harmful = []
    accepted_harmless = []

    for i, p in enumerate(prompts):
        if p["label"] == "harmless":
            accepted_harmless.append(i)
        elif p["id"] in refusal_labels:
            if refusal_labels[p["id"]]:
                refused_harmful.append(i)
            else:
                accepted_harmful.append(i)
        # harmful prompts not in refusal_labels → skip (no behavioral data)

    return {
        "refused_harmful":   np.array(refused_harmful,   dtype=int),
        "accepted_harmful":  np.array(accepted_harmful,  dtype=int),
        "accepted_harmless": np.array(accepted_harmless, dtype=int),
    }


# ── clustering score ──────────────────────────────────────────────────────────

def cluster_score(
    acts: np.ndarray,
    idx: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
) -> tuple[float, float]:
    """
    s^l(h) = cos_sim(h, mu_pos) - cos_sim(h, mu_neg)
    Returns (mean, std) for the given subset.
    """
    if len(idx) == 0:
        return float("nan"), float("nan")
    h = acts[idx]
    h_n     = h     / (np.linalg.norm(h,     axis=1, keepdims=True) + 1e-8)
    mu_p_n  = mu_pos / (np.linalg.norm(mu_pos) + 1e-8)
    mu_n_n  = mu_neg / (np.linalg.norm(mu_neg) + 1e-8)
    s = h_n @ mu_p_n - h_n @ mu_n_n
    return float(s.mean()), float(s.std())


# ── per-model / per-language analysis ─────────────────────────────────────────

def analyse_model_lang(
    model_key: str,
    lang: str,
    tinst_root: Path,
    postinst_root: Path,
    resp_dir: Path,
) -> pd.DataFrame:
    cfg = MODELS[model_key]

    # Load prompts from t_inst dir (same prompts for both positions)
    prompts = load_prompts(tinst_root, model_key, lang)
    if not prompts:
        logger.warning(f"No prompts for {model_key}/{lang} in {tinst_root}")
        return pd.DataFrame()

    refusal_labels = load_refusal_labels(resp_dir, model_key, lang)
    groups = build_groups(prompts, refusal_labels)

    n_rh = len(groups["refused_harmful"])
    n_ah = len(groups["accepted_harmful"])
    n_al = len(groups["accepted_harmless"])
    logger.info(f"{model_key}/{lang}: refused_harmful={n_rh}, "
                f"accepted_harmful={n_ah}, accepted_harmless={n_al}")

    if n_rh == 0 or n_al == 0:
        logger.warning(f"Not enough groups for {model_key}/{lang} — need refused_harmful + accepted_harmless")
        return pd.DataFrame()

    rows = []
    for layer in cfg.probe_layers:
        tinst_acts    = load_acts(tinst_root,    model_key, lang, layer)
        postinst_acts = load_acts(postinst_root, model_key, lang, layer)

        for position, acts in [("t_inst", tinst_acts), ("post_inst", postinst_acts)]:
            if acts is None:
                logger.debug(f"  Missing {position} layer {layer}")
                continue

            # Cluster centres: refused_harmful vs accepted_harmless
            mu_refused_harmful  = acts[groups["refused_harmful"]].mean(0)
            mu_accepted_harmless = acts[groups["accepted_harmless"]].mean(0)

            for group_name, idx in groups.items():
                mean_s, std_s = cluster_score(
                    acts, idx, mu_refused_harmful, mu_accepted_harmless
                )
                rows.append({
                    "model":    model_key,
                    "lang":     lang,
                    "layer":    layer,
                    "position": position,
                    "group":    group_name,
                    "mean_s":   round(mean_s, 4),
                    "std_s":    round(std_s,  4),
                    "n":        len(idx),
                })

    df = pd.DataFrame(rows)

    # Fig 3 scatter: per-prompt belief scores
    if n_rh >= 3 and n_al >= 3:
        belief_df = compute_beliefs(
            prompts, groups, tinst_root, postinst_root, model_key, lang, cfg.probe_layers
        )
        plot_fig3_scatter(belief_df, model_key, lang)
        belief_df.to_csv(PROB_DIR / f"belief_scores_{model_key}_{lang}.csv", index=False)

    return df


# ── per-prompt belief scores (Zhao et al. Fig 3) ─────────────────────────────

def compute_beliefs(
    prompts: list[dict],
    groups: dict[str, np.ndarray],
    tinst_root: Path,
    postinst_root: Path,
    model_key: str,
    lang: str,
    layers: list[int],
) -> pd.DataFrame:
    """
    Compute Δ_harmful and Δ_refuse per individual prompt (averaged over layers),
    following Zhao et al. Equations 3 & 4.

    Δ_harmful = mean_l [ cos(h_tinst,    μ_refused_harmful_tinst)
                        - cos(h_tinst,    μ_accepted_harmless_tinst) ]
    Δ_refuse  = mean_l [ cos(h_postinst, μ_refused_harmful_postinst)
                        - cos(h_postinst, μ_accepted_harmless_postinst) ]
    """
    harm_scores: dict[int, np.ndarray] = {}   # prompt_idx → per-layer score
    ref_scores:  dict[int, np.ndarray] = {}

    n = len(prompts)
    harm_mat = np.full((n, len(layers)), np.nan)
    ref_mat  = np.full((n, len(layers)), np.nan)

    for li, layer in enumerate(layers):
        tinst_acts    = load_acts(tinst_root,    model_key, lang, layer)
        postinst_acts = load_acts(postinst_root, model_key, lang, layer)

        for position, acts, mat in [
            ("t_inst",    tinst_acts,    harm_mat),
            ("post_inst", postinst_acts, ref_mat),
        ]:
            if acts is None:
                continue
            mu_rh = acts[groups["refused_harmful"]].mean(0)
            mu_al = acts[groups["accepted_harmless"]].mean(0)
            mu_rh_n = mu_rh / (np.linalg.norm(mu_rh) + 1e-8)
            mu_al_n = mu_al / (np.linalg.norm(mu_al) + 1e-8)
            h_n = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
            mat[:, li] = h_n @ mu_rh_n - h_n @ mu_al_n

    delta_harm = np.nanmean(harm_mat, axis=1)
    delta_ref  = np.nanmean(ref_mat,  axis=1)

    # Assign group label per prompt
    group_label = np.full(n, "other", dtype=object)
    for g_name, idx in groups.items():
        group_label[idx] = g_name

    rows = []
    for i, p in enumerate(prompts):
        rows.append({
            "id":          p.get("id", i),
            "label":       p.get("label", ""),
            "group":       group_label[i],
            "delta_harm":  round(float(delta_harm[i]), 4),
            "delta_refuse": round(float(delta_ref[i]), 4),
        })
    return pd.DataFrame(rows)


def plot_fig3_scatter(belief_df: pd.DataFrame, model_key: str, lang: str):
    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    lang_name  = LANG_NAMES.get(lang, lang)

    SCATTER_STYLE = {
        "refused_harmful":   {"color": "#EE4444", "marker": "o",  "label": "refused harmful",   "alpha": 0.6, "s": 25},
        "accepted_harmful":  {"color": "#EE4444", "marker": "+",  "label": "accepted harmful",  "alpha": 0.6, "s": 40},
        "accepted_harmless": {"color": "#4477AA", "marker": "s",  "label": "accepted harmless", "alpha": 0.5, "s": 20},
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    for group, style in SCATTER_STYLE.items():
        sub = belief_df[belief_df["group"] == group]
        if sub.empty:
            continue
        ax.scatter(sub["delta_harm"], sub["delta_refuse"],
                   c=style["color"], marker=style["marker"],
                   label=style["label"], alpha=style["alpha"], s=style["s"])

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\Delta_\mathrm{harmful}$")
    ax.set_ylabel(r"$\Delta_\mathrm{refuse}$")
    ax.set_title(f"Belief of harmfulness vs refusal\n{model_name} / {lang_name}", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()

    out = FIG_DIR / f"fig3_belief_scatter_{model_key}_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── plotting ──────────────────────────────────────────────────────────────────

GROUP_STYLE = {
    "refused_harmful":   {"color": "#EE4444", "ls": "-",  "label": "refused harmful"},
    "accepted_harmful":  {"color": "#EE4444", "ls": "--", "label": "accepted harmful"},
    "accepted_harmless": {"color": "#4477AA", "ls": "-",  "label": "accepted harmless"},
}


def plot_fig2(df: pd.DataFrame, model_key: str, lang: str):
    model_name = MODELS[model_key].name if model_key in MODELS else model_key
    lang_name  = LANG_NAMES.get(lang, lang)
    layers = sorted(df["layer"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for col, position in enumerate(["t_inst", "post_inst"]):
        ax = axes[col]
        sub = df[df["position"] == position]
        if sub.empty:
            ax.set_title(f"{position} (no data)")
            continue

        for group, style in GROUP_STYLE.items():
            grp_sub = sub[sub["group"] == group].sort_values("layer")
            if grp_sub.empty:
                continue
            means = grp_sub["mean_s"].values
            stds  = grp_sub["std_s"].values
            x     = grp_sub["layer"].values
            ax.plot(x, means, color=style["color"], ls=style["ls"],
                    marker="o", markersize=4, label=style["label"])
            ax.fill_between(x, means - stds, means + stds,
                            color=style["color"], alpha=0.12)

        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$s^l(h^l)$")
        title = "t_inst (encodes harmfulness?)" if position == "t_inst" \
                else "post_inst (encodes refusal?)"
        ax.set_title(title, fontsize=9)
        ax.set_xticks(layers)
        ax.legend(fontsize=7)

    fig.suptitle(f"Harmfulness vs Refusal clustering\n{model_name} / {lang_name}", fontsize=10)
    plt.tight_layout()

    out = FIG_DIR / f"fig2_{model_key}_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=list(MODELS.keys()), default=None)
    parser.add_argument("--lang",   choices=LANG_CODES,          default=None)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--tinst-dir",    default="data/processed/activations_tinst")
    parser.add_argument("--postinst-dir", default="data/processed/activations_postinst")
    parser.add_argument("--responses-dir", default="results/behavioral")
    args = parser.parse_args()

    tinst_root    = ROOT / args.tinst_dir
    postinst_root = ROOT / args.postinst_dir
    resp_dir      = ROOT / args.responses_dir

    if args.all and args.model:
        langs = LANG_CODES
        models = [args.model]
    elif args.all:
        langs  = LANG_CODES
        models = list(MODELS.keys())
    elif args.model and args.lang:
        langs  = [args.lang]
        models = [args.model]
    elif args.model:
        langs  = LANG_CODES
        models = [args.model]
    else:
        models = ["base"]
        langs  = ["en"]
        logger.info("No --model/--lang — defaulting to base × en")

    all_rows = []
    for model_key in models:
        for lang in langs:
            df = analyse_model_lang(
                model_key, lang, tinst_root, postinst_root, resp_dir
            )
            if df.empty:
                continue
            all_rows.append(df)
            plot_fig2(df, model_key, lang)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_csv = PROB_DIR / "harmfulness_refusal_clustering.csv"
        combined.to_csv(out_csv, index=False)
        logger.info(f"Saved → {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
