# Findings: Regional Model Comparison (Earth / Fire / Water vs Global)

**Scripts**: `scripts/23_regional_comparison.py`
**Data**: `results/probes/regional_probe_comparison.csv`, `results/probes/crosslingual_cosine_regional.csv`, `results/probes/cross_model_direction_cosine.csv`
**Figures**: `results/figures/fig_regional_probe_auc.png`, `results/figures/fig_regional_crosslingual_*.png`, `results/figures/fig_cross_model_direction.png`

---

## Background

The core hypothesis: regional models (Earth = Africa/WestAsia, Fire = South Asia, Water = AsiaPacific/Europe) have more language-specific safety representations than the Global model. Specifically, we expected:
- Earth's Arabic direction to be more distinct from English, and more tightly clustered with other Afro-Asian languages
- Fire's Hindi direction to be more separable and internally consistent
- Water's European/Asian languages to cluster differently

**Note**: Swahili and Yoruba have 0 harmful examples in the eval activation set for all regional models and cannot be probed. Analysis covers en, fr, ar, hi across all four models.

---

## Findings

### 1. Probe accuracy is nearly identical across all four models

LogReg AUC at layer 18, all models × languages:

| model  | en    | fr    | ar    | hi    |
|--------|-------|-------|-------|-------|
| global | 0.882 | 0.869 | 0.877 | 0.860 |
| earth  | 0.898 | 0.876 | 0.874 | 0.863 |
| fire   | 0.894 | 0.870 | 0.874 | 0.844 |
| water  | 0.895 | 0.878 | 0.870 | 0.848 |

Differences across models are within ±0.016 for every language. There is no evidence that Earth improves Arabic separability, or that Fire improves Hindi separability. Regional training does not make the harmfulness representations more linearly separable for the focal languages.

---

### 2. Crosslingual geometry is preserved identically across all models

Crosslingual cosines (en as source, layer 18):

| model  | en→ar  | en→fr  | en→hi  |
|--------|--------|--------|--------|
| global | −0.825 | +0.614 | −0.183 |
| earth  | −0.818 | +0.613 | −0.178 |
| fire   | −0.807 | +0.613 | −0.150 |
| water  | −0.820 | +0.613 | −0.176 |

The English anti-correlation with Arabic (≈ −0.82) and Hindi (≈ −0.15 to −0.18) is present in every model at essentially the same value. The positive English–French alignment (≈ +0.61) is also identical. Regional training produces no change in the relative geometry of language directions.

---

### 3. Regional model harmfulness directions are near-identical to Global's

Cross-model cosines (regional vs global) at layer 18:

| model | en    | fr    | ar    | hi    |
|-------|-------|-------|-------|-------|
| earth | 1.000 | 0.996 | 0.999 | 0.984 |
| fire  | 0.998 | 0.985 | 0.998 | 0.942 |
| water | 1.000 | 0.996 | 0.999 | 0.982 |

All cosines are ≥ 0.942. The harmfulness directions learned by regional models are essentially the same as those in Global. 

The only notable deviation is **fire/hi = 0.942** — the smallest cross-model cosine observed. Fire is the model specialised for South Asia (Hindi focus), and Hindi is its focal language. Even so, 0.942 indicates extremely high alignment — this is not evidence for a distinct safety direction, just the smallest deviation among an otherwise uniform null result.

---

## Summary

**The original hypothesis is not supported.** Regional specialization during pre-training or instruction-tuning does not produce more language-specific safety representations. The harmfulness directions for Arabic, Hindi, French, and English are:
1. **Near-identical across all four models** (cosines 0.942–1.000 vs Global)
2. **No more separable** in regional models than in Global (probe AUC within ±0.016)
3. **Geometrically identical**: the English anti-correlation with Arabic/Hindi and alignment with French is preserved at the same values in Earth, Fire, Water, and Global

The safety representation geometry appears to be determined by the base model weights (shared across all TinyAya variants) and the pretraining data distribution, not by regional specialization. Regional fine-tuning shifts the content/style of model outputs but does not reshape the geometry of the safety-relevant subspace.

---

## Caveats

1. **Swahili and Yoruba** (the most under-resourced languages; sw is Earth's focal language for Africa) have 0 harmful examples in the eval set and cannot be included. The most extreme test of the Earth hypothesis (African-language safety directions) is unavailable.
2. **Behavioral data pending**: Refusal rates for earth/fire/water have not been collected yet. It is possible the behavioral gap exists without a representational gap.
3. **Layer sensitivity**: Results reported at layer 18. Cross-model cosines are not substantially different at other layers (the pattern holds at layers 4, 9, 24, 30, 35).
