# Findings: Multilingual Geometry

**Scripts**: `scripts/18_multilingual_geometry.py`
**Data**: `results/probes/crosslingual_cosine.csv`, `results/probes/cross_stage_alignment.csv`
**Figures**: `results/figures/alignment_*.png`

---

## Research Questions

1. Are harmfulness directions shared across languages within a model?
2. Does fine-tuning increase or decrease cross-lingual alignment?
3. Is the refusal direction preserved across training stages?

---

## Findings

### 1. English harmfulness direction is not universal — it is anti-correlated with Arabic and Swahili in base

Crosslingual cosines at layer 18, **base model**:

| lang_src | ar     | en     | fr     | hi     | sw     | yo     |
|----------|--------|--------|--------|--------|--------|--------|
| ar       | 1.000  | −0.420 | 0.740  | 0.586  | 0.903  | 0.303  |
| en       | −0.420 | 1.000  | 0.094  | 0.099  | −0.301 | 0.364  |
| fr       | 0.740  | 0.094  | 1.000  | 0.691  | 0.814  | 0.581  |
| hi       | 0.586  | 0.099  | 0.691  | 1.000  | 0.671  | 0.622  |
| sw       | 0.903  | −0.301 | 0.814  | 0.671  | 1.000  | 0.425  |
| yo       | 0.303  | 0.364  | 0.581  | 0.622  | 0.425  | 1.000  |

Two clusters in base:
- **Non-English cluster**: ar/fr/hi/sw/yo all share broadly positive cosines (0.30–0.90). Arabic and Swahili are particularly tightly aligned (0.903).
- **English is an outlier**: strongly anti-correlated with Arabic (−0.42) and Swahili (−0.30), near-zero with French and Hindi.

English harmfulness representations in the base model occupy a distinct orientation from every other language. The harmfulness signal in ar/fr/sw/hi/yo points in a shared non-English direction.

### 2. Fine-tuning aligns non-English languages toward English — not the other way around

Crosslingual cosines at layer 18, **base_sft** and **base_dpo**:

| Pair     | base   | base_sft | base_dpo |
|----------|--------|----------|----------|
| en–fr    | +0.094 | +0.942   | +0.939   |
| en–sw    | −0.301 | +0.929   | +0.919   |
| en–yo    | +0.364 | +0.848   | +0.844   |
| en–ar    | −0.420 | +0.712   | +0.680   |
| ar–sw    | +0.903 | +0.744   | +0.738   |
| ar–fr    | +0.740 | +0.611   | +0.608   |

After SFT/DPO:
- English aligns strongly with fr/sw/yo (0.84–0.94) and moderately with ar (0.68–0.71)
- The tight ar–sw cluster (0.903 in base) loosens slightly to ~0.74 — fine-tuning pulls languages toward English, partially disrupting the original non-English cluster

**Hindi** (base_dpo only, layer 18): en–hi = 0.498, hi–fr = 0.527. Hindi moves less than other languages toward English, consistent with the DPO English-centric alignment finding.

### 3. English harmfulness direction flips sign after fine-tuning relative to base

From `cross_stage_alignment.csv`, DiffMean probe cosines (en):

| Layer | base→base_sft | base→base_dpo | base_sft→base_dpo |
|-------|--------------|--------------|------------------|
| 4     | −0.363        | −0.334        | +0.989            |
| 9     | −0.508        | −0.503        | +0.989            |
| 18    | −0.420        | −0.398        | +0.966            |
| 24    | −0.118        | −0.035        | +0.930            |
| 30    | +0.186        | +0.276        | +0.922            |
| 35    | +0.098        | +0.198        | +0.898            |

The DiffMean probe direction at early layers (4–18) is **sign-reversed** between base and both aligned models. base_sft and base_dpo are highly coherent with each other (+0.97–0.99). This sign reversal is a direction sign convention artifact for English: the fine-tuned model's harmful cluster projects on the other side of the harmless cluster compared to base, while centroid-based drift (script 19) still shows high cosines (~0.97). The clusters move together, but the DiffMean convention reverses.

For non-English languages, the sign reversal at early layers also appears (not shown here), though less consistently.

### 4. Refusal direction is broadly preserved across training stages

Refusal probe cosines across stages (en):

| Layer | base→base_sft | base→base_dpo | base_sft→base_dpo |
|-------|--------------|--------------|------------------|
| 4     | +0.875        | +0.903        | +0.769            |
| 9     | +0.858        | +0.830        | +0.668            |
| 18    | +0.835        | +0.737        | +0.591            |
| 24    | +0.756        | +0.724        | +0.544            |
| 30    | +0.741        | +0.764        | +0.596            |
| 35    | +0.709        | +0.673        | +0.527            |

The refusal direction is preserved from base into both aligned models (0.71–0.90 across layers). SFT and DPO produce somewhat divergent refusal directions (0.53–0.77), suggesting they implement refusal via related but non-identical mechanisms. Preservation decreases at deeper layers.

---

## Summary

The base TinyAya model represents harmfulness in an English-specific direction that is anti-correlated with Arabic and Swahili (−0.42, −0.30). The non-English languages share a broadly aligned cluster (ar/fr/hi/sw/yo cosines 0.30–0.90). Fine-tuning (SFT/DPO) aligns all languages toward the English direction (post-SFT en–fr = 0.94, en–sw = 0.93), partially collapsing the multi-language geometry into a single shared axis. Arabic aligns least (0.68–0.71). Hindi (DPO only) also aligns less. The refusal direction is broadly preserved across training stages (0.71–0.90), with SFT and DPO producing related but distinct refusal axes.
