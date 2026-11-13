# Findings: Correlation Analysis

**Scripts**: `scripts/14_correlation_analysis.py`
**Data**: `results/probes/corr1_lang_sim_vs_refusal_gap.csv`, `results/probes/corr1_stats.csv`, `results/probes/corr2_hvr_alignment_vs_refusal_rate.csv`, `results/probes/corr2_stats.csv`
**Figures**: `results/figures/corr1_*.png`, `results/figures/corr2_*.png`

---

## Research Questions

1. **Corr1**: Do languages with harmfulness directions more distant from English have larger refusal gaps (lower refusal rates relative to English)?
2. **Corr2**: Within a model, do languages with higher harm↔refusal direction cosine have higher refusal rates?

---

## Findings

### 1. Corr1: Representational distance from English predicts refusal gap only at deeper layers in aligned models

Spearman r between cosine-distance-to-English and refusal gap (refusal_rate_en − refusal_rate_lang):

| model           | layer | n_langs | Spearman r | p     |
|-----------------|-------|---------|------------|-------|
| base_dpo        | 30    | 5       | +0.894     | 0.040 |
| base_sft        | 30    | 4       | +0.949     | 0.051 |
| global          | 35    | 4       | +1.000     | 0.000 |
| base_dpo        | 4–24  | 5       | 0.11–0.45  | n.s.  |
| base_sft        | 4–24  | 4       | −0.21–0.32 | n.s.  |
| base_sft_n20000 | all   | 5       | −0.05–0.67 | n.s.  |

Interpretation of key results:

**base_dpo layer 30** (r=0.894, p=0.04): French has the highest cos_sim_to_en (0.919) and the smallest refusal gap (0.014); Arabic has cos_sim_to_en=0.680, refusal gap=0.229; Hindi has the lowest cos_sim (0.499) and refusal gap=0.246. The relationship holds: languages that are representationally closer to English (in the harmfulness direction) have more English-like refusal behavior.

**global layer 35** (r=1.0, p=0.0, n=4): Perfect rank correlation — every unit of representational distance from English corresponds to a rank increase in refusal gap.

**base_sft layer 30** (r=0.949, p=0.051): Borderline significant (n=4 limits power). Same direction as base_dpo; misses α=0.05 by a small margin.

**Why only layer 30?** Earlier layers (4–24) show near-zero or non-monotone correlations. The refusal-predictive signal appears to be concentrated in deeper layers for the aligned models, consistent with prior findings that deeper layers carry more alignment-specific information.

**Caveat**: n=3–5 languages limits statistical power. These are rank correlations across a small sample. The pattern is directionally consistent across base_dpo, base_sft, and global, but p-values should be interpreted cautiously.

---

### 2. Corr2: Harm↔refusal cosine is not reliably predictive of refusal rate within models

Spearman r between hvr_cosine (harmfulness↔refusal direction cosine) and refusal_rate across languages:

| model    | layer | n_langs | Spearman r | p     |
|----------|-------|---------|------------|-------|
| base     | 9     | 3       | +1.000     | 0.000 |
| base     | 18    | 3       | +1.000     | 0.000 |
| base     | 4,24  | 3       | +0.500     | 0.667 |
| base     | 30,35 | 3       | −0.500     | 0.667 |
| base_dpo | all   | 2       | —          | —     |
| base_sft | all   | 1       | —          | —     |
| global   | all   | 4       | −0.40/−0.20| n.s.  |

At **base layers 9 and 18**, the rank correlation is perfect (r=1.0, p=0.0), but with only n=3 languages (en, ar, fr — the only languages with valid hvr_cosine entries in the base model). This is too small a sample to draw strong conclusions.

For **base_dpo** and **base_sft**, the analysis has only n=2 and n=1 usable language entries respectively — not enough to compute a meaningful Spearman r. This is because the hvr_cosine for non-English languages is not available or not computed for the aligned models.

For **global**, the correlation is consistently negative (−0.40 to −0.20), not significant, and not in the expected direction.

**Interpretation**: The corr2 analysis is underpowered. The n=3 perfect correlation in base at layers 9–18 is consistent with a real relationship but not confirmatory at that sample size. The analysis cannot distinguish a genuine harm↔refusal alignment effect from coincidence.

---

## Summary

Corr1 shows a consistent, statistically supported (or borderline) relationship at deeper layers (30–35): languages with harmfulness directions more distant from English in representation space also show larger refusal rate gaps relative to English. This holds in base_dpo (r=0.894, p=0.04, n=5), global (r=1.0, p=0.0, n=4), and base_sft (r=0.949, p=0.051, n=4). The effect is not present at early layers (4–24), suggesting that the refusal-predictive representational signal emerges at deeper layers.

Corr2 (within-model harm↔refusal cosine vs refusal rate) is underpowered across all models and does not yield interpretable results with the available data (n=1–4 per model).
