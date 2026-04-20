"""
Step 14 — Correlation analysis linking representation metrics to behavioral refusal rates.

Two correlations from the scope:

  Corr 1 — Similarity-to-English → refusal rate gap from English
    H: The closer a language's safety direction is to English's,
       the closer its refusal rate is to English's (which is high).

  Corr 2 — Harmful ↔ Refusal direction alignment → refusal rate
    H: Languages where harmful and refusal directions are well-aligned
       (low angular distance) show higher refusal rates on harmful prompts.

Reads:
  results/probes/alignment_{model}_layer{n}.csv
  results/probes/{model}/{lang}/layer_{n}_diffmean.pkl
  results/probes/{model}/{lang}/layer_{n}_refusal.pkl
  results/behavioral/baseline_summary.csv

Writes:
  results/probes/corr1_lang_sim_vs_refusal_gap.csv
  results/probes/corr2_hvr_alignment_vs_refusal_rate.csv
  results/figures/corr1_scatter.png
  results/figures/corr2_scatter.png
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_NAMES
from config.models import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROBE_DIR  = ROOT / "results" / "probes"
FIG_DIR    = ROOT / "results" / "figures"
BEH_PATH   = ROOT / "results" / "behavioral" / "baseline_summary.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def spearman(x, y):
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return round(float(r), 4), round(float(p), 4)


# ── Correlation 1 ─────────────────────────────────────────────────────────────

def build_corr1(behavioral: pd.DataFrame) -> pd.DataFrame:
    """
    For each model × layer that has an alignment_*.csv, extract
    cosine_sim(lang → en) and |refusal_lang - refusal_en|.
    Returns a long-form DataFrame.
    """
    rows = []
    for align_csv in sorted(PROBE_DIR.glob("alignment_*_layer*.csv")):
        stem   = align_csv.stem                          # alignment_global_layer24
        # last part is always "layer{n}"
        import re
        m = re.search(r"_layer(\d+)$", stem)
        if m is None:
            continue
        layer     = int(m.group(1))
        model_key = re.sub(r"^alignment_", "", stem[:m.start()])

        mat = pd.read_csv(align_csv, index_col=0)
        if "en" not in mat.columns:
            continue

        # refusal rates for this model
        beh_model = behavioral[behavioral["model"] == model_key]
        en_rate   = beh_model.loc[beh_model["lang"] == "en", "refusal_rate"]
        if en_rate.empty:
            continue
        en_rate = en_rate.iloc[0]

        for lang in mat.index:
            if lang == "en":
                continue
            cos_sim_to_en = float(mat.loc[lang, "en"])
            beh_lang = beh_model.loc[beh_model["lang"] == lang, "refusal_rate"]
            if beh_lang.empty:
                continue
            lang_rate    = beh_lang.iloc[0]
            refusal_gap  = abs(lang_rate - en_rate)
            cos_dist     = 1.0 - cos_sim_to_en

            rows.append({
                "model": model_key,
                "layer": layer,
                "lang": lang,
                "cos_sim_to_en": round(cos_sim_to_en, 4),
                "cos_dist_to_en": round(cos_dist, 4),
                "refusal_rate": round(lang_rate, 4),
                "en_refusal_rate": round(en_rate, 4),
                "refusal_gap": round(refusal_gap, 4),
            })

    return pd.DataFrame(rows)


def compute_corr1_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation per model × layer."""
    results = []
    for (model, layer), grp in df.groupby(["model", "layer"]):
        r, p = spearman(grp["cos_dist_to_en"], grp["refusal_gap"])
        results.append({"model": model, "layer": layer,
                        "n_langs": len(grp), "spearman_r": r, "p_value": p})
    return pd.DataFrame(results).sort_values(["model", "layer"])


def plot_corr1(df: pd.DataFrame):
    models = df["model"].unique()
    layers = sorted(df["layer"].unique())

    fig, axes = plt.subplots(
        len(models), len(layers),
        figsize=(max(10, len(layers) * 3), max(6, len(models) * 2.5)),
        squeeze=False,
    )

    for row, model_key in enumerate(sorted(models)):
        model_name = MODELS[model_key].name if model_key in MODELS else model_key
        for col, layer in enumerate(layers):
            ax = axes[row, col]
            sub = df[(df["model"] == model_key) & (df["layer"] == layer)]
            if sub.empty:
                ax.set_visible(False)
                continue
            ax.scatter(sub["cos_dist_to_en"], sub["refusal_gap"],
                       s=50, alpha=0.8, color="#4477AA")
            for _, pt in sub.iterrows():
                ax.annotate(LANG_NAMES.get(pt["lang"], pt["lang"]),
                            (pt["cos_dist_to_en"], pt["refusal_gap"]),
                            fontsize=6, ha="left", va="bottom")
            r, p = spearman(sub["cos_dist_to_en"], sub["refusal_gap"])
            ax.set_title(f"{model_name} / L{layer}\nρ={r}, p={p}", fontsize=7)
            ax.set_xlabel("Cos dist from EN", fontsize=7)
            ax.set_ylabel("|refusal gap|", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "corr1_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── Correlation 2 ─────────────────────────────────────────────────────────────

def build_corr2(behavioral: pd.DataFrame) -> pd.DataFrame:
    """
    For each model/lang/layer with both diffmean and refusal pkl files,
    compute cosine_sim(harmful_dir, refusal_dir) and join with refusal rate.
    """
    rows = []
    for diffmean_pkl in sorted(PROBE_DIR.glob("*/*/layer_*_diffmean.pkl")):
        parts      = diffmean_pkl.parts
        # …/probes/{model}/{lang}/layer_{n}_diffmean.pkl
        model_key  = parts[-3]
        lang       = parts[-2]
        layer_str  = diffmean_pkl.stem.split("_")[1]   # "layer_24_diffmean" → "24"
        layer      = int(layer_str)

        refusal_pkl = diffmean_pkl.parent / f"layer_{layer}_refusal.pkl"
        if not refusal_pkl.exists():
            continue

        harm_probe   = load_pkl(diffmean_pkl)
        refusal_probe = load_pkl(refusal_pkl)
        if harm_probe is None or refusal_probe is None:
            continue
        if harm_probe.direction is None or refusal_probe.direction is None:
            continue

        cos = cosine_sim(harm_probe.direction, refusal_probe.direction)

        beh_row = behavioral[(behavioral["model"] == model_key) &
                              (behavioral["lang"] == lang)]
        refusal_rate = beh_row["refusal_rate"].iloc[0] if not beh_row.empty else float("nan")

        rows.append({
            "model": model_key,
            "lang": lang,
            "layer": layer,
            "hvr_cosine": round(cos, 4),
            "refusal_rate": round(refusal_rate, 4) if not np.isnan(refusal_rate) else float("nan"),
        })

    return pd.DataFrame(rows)


def compute_corr2_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation per model × layer across languages."""
    results = []
    valid = df.dropna(subset=["refusal_rate"])
    for (model, layer), grp in valid.groupby(["model", "layer"]):
        r, p = spearman(grp["hvr_cosine"], grp["refusal_rate"])
        results.append({"model": model, "layer": layer,
                        "n_langs": len(grp), "spearman_r": r, "p_value": p})
    return pd.DataFrame(results).sort_values(["model", "layer"])


def plot_corr2(df: pd.DataFrame):
    models = [m for m in df["model"].unique() if not ("ckpt" in m)]
    layers = sorted(df["layer"].unique())
    valid  = df.dropna(subset=["refusal_rate"])

    fig, axes = plt.subplots(
        len(models), len(layers),
        figsize=(max(10, len(layers) * 3), max(6, len(models) * 2.5)),
        squeeze=False,
    )

    for row, model_key in enumerate(sorted(models)):
        model_name = MODELS[model_key].name if model_key in MODELS else model_key
        for col, layer in enumerate(layers):
            ax = axes[row, col]
            sub = valid[(valid["model"] == model_key) & (valid["layer"] == layer)]
            if sub.empty:
                ax.set_visible(False)
                continue
            ax.scatter(sub["hvr_cosine"], sub["refusal_rate"],
                       s=50, alpha=0.8, color="#EE6677")
            for _, pt in sub.iterrows():
                ax.annotate(LANG_NAMES.get(pt["lang"], pt["lang"]),
                            (pt["hvr_cosine"], pt["refusal_rate"]),
                            fontsize=6, ha="left", va="bottom")
            r, p = spearman(sub["hvr_cosine"], sub["refusal_rate"])
            ax.set_title(f"{model_name} / L{layer}\nρ={r}, p={p}", fontsize=7)
            ax.set_xlabel("Harmful↔Refusal cosine", fontsize=7)
            ax.set_ylabel("Refusal rate", fontsize=7)
            ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    out = FIG_DIR / "corr2_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not BEH_PATH.exists():
        logger.error(f"Behavioral summary not found: {BEH_PATH}")
        sys.exit(1)

    behavioral = pd.read_csv(BEH_PATH)

    # ── Correlation 1 ─────────────────────────────────────────────────────────
    logger.info("Building Correlation 1 (representation similarity to EN vs refusal gap)…")
    df1 = build_corr1(behavioral)
    if df1.empty:
        logger.warning("No data for Corr 1 — check alignment_*.csv files.")
    else:
        out1 = PROBE_DIR / "corr1_lang_sim_vs_refusal_gap.csv"
        df1.to_csv(out1, index=False)
        logger.info(f"Saved → {out1} ({len(df1)} rows)")

        stats1 = compute_corr1_stats(df1)
        logger.info("\n" + stats1.to_string(index=False))
        stats_out1 = PROBE_DIR / "corr1_stats.csv"
        stats1.to_csv(stats_out1, index=False)
        logger.info(f"Saved → {stats_out1}")

        plot_corr1(df1)

    # ── Correlation 2 ─────────────────────────────────────────────────────────
    logger.info("\nBuilding Correlation 2 (harmful↔refusal direction alignment vs refusal rate)…")
    df2 = build_corr2(behavioral)
    if df2.empty:
        logger.warning("No data for Corr 2 — need paired diffmean+refusal pkl files.")
    else:
        out2 = PROBE_DIR / "corr2_hvr_alignment_vs_refusal_rate.csv"
        df2.to_csv(out2, index=False)
        logger.info(f"Saved → {out2} ({len(df2)} rows)")

        stats2 = compute_corr2_stats(df2)
        logger.info("\n" + stats2.to_string(index=False))
        stats_out2 = PROBE_DIR / "corr2_stats.csv"
        stats2.to_csv(stats_out2, index=False)
        logger.info(f"Saved → {stats_out2}")

        plot_corr2(df2)

    print("\nDone.")


if __name__ == "__main__":
    main()
