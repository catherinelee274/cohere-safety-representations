# Findings: SAE Analysis

**Scripts**: `scripts/09_sae_analysis.py`
**Data**: `results/sae/global/en/layer_24/safety_features.json`, `feature_activations.npy`
**Figures**: `results/figures/sae_*.png`

---

## Research Questions

1. Which sparse autoencoder features are most discriminative between harmful and harmless prompts?
2. Do harmful prompts activate a small, interpretable set of features, or a diffuse pattern?

---

## Setup

- **Model**: TinyAya-Global
- **Language**: English
- **Layer**: 24 (mid-late; best DiffMean AUC layer from Phase 2)
- **SAE architecture**: TopK sparse autoencoder, 8× expansion (2048 → 16384 features), k=32
- **Data**: 860 prompts (430 harmful XSafety + 430 harmless LITMUS/translated)
- Safety features identified by Cohen's d between harmful and harmless activation distributions

---

## Findings

### 1. A small set of sparse features discriminate harmful from harmless inputs

Top harmful-enriched features (positive Cohen's d = more active on harmful):

| feature_idx | cohen_d | mean_harmful | mean_harmless | sparsity |
|-------------|---------|--------------|---------------|----------|
| 12081       | +1.151  | 4.159        | 0.880         | 0.399    |
| 8051        | +1.082  | 4.319        | 0.983         | 0.393    |
| 6864        | +0.960  | 3.071        | 0.587         | 0.312    |
| 6254        | +0.919  | 4.509        | 1.599         | 0.457    |
| 16187       | +0.898  | 2.931        | 0.565         | 0.293    |
| 10525       | +0.858  | 2.623        | 0.581         | 0.292    |
| 13385       | +0.839  | 1.959        | 0.226         | 0.204    |
| 3323        | +0.836  | 2.171        | 0.331         | 0.227    |
| 12153       | +0.812  | 2.798        | 0.778         | 0.324    |
| 7416        | +0.807  | 3.728        | 1.346         | 0.412    |

~11 features have |d| > 0.8 with positive sign (harmful-enriched). Sparsity ranges from 0.20–0.46, meaning these features activate on 20–46% of all prompts. They are not highly sparse — they fire broadly.

### 2. A large cluster of harmless-enriched features shows suspicious uniformity

Top harmless-enriched features (negative Cohen's d):

| feature_idx | cohen_d | mean_harmful | mean_harmless | sparsity |
|-------------|---------|--------------|---------------|----------|
| 6504        | −0.731  | 3.519        | 6.995         | 0.521    |
| 16189       | −0.726  | 2.602        | 5.159         | 0.517    |
| 9998        | −0.724  | 3.619        | 7.172         | 0.517    |
| 8768        | −0.724  | 3.770        | 7.470         | 0.517    |
| 14485       | −0.724  | 4.424        | 8.765         | 0.517    |
| (30+ more)  | ≈−0.722 | —            | —             | ≈0.516   |

There are 30+ features with near-identical cohen_d ≈ −0.722 and near-identical sparsity ≈ 0.516. This is almost certainly a **degenerate feature cluster** — likely a group of near-duplicate features that the SAE learned to use interchangeably for some common token type or structural property. These likely do not represent interpretable safety-relevant concepts.

These features are also the *most active overall* (mean activation 5–7, sparsity 0.52), suggesting they fire on roughly half of all tokens regardless of content, with higher activation on harmless prompts.

### 3. The harmfulness signal in SAE space is sparse but not highly concentrated

The full feature activation matrix has shape (860, 16384) with nonzero fraction = 0.00195 (exactly 1/512, consistent with k=32 active features per token in a 16384-feature SAE: 32/16384 = 0.00195). The TopK constraint is working correctly.

The harmful-discriminative features (d > 0.8) number ~11 out of 16384 (0.07% of all features). Their mean harmful activation is 1.4–4.5 with sparsity 0.20–0.46 — they activate on a moderate fraction of harmful prompts but not exclusively on harmful prompts.

---

## Limitations

1. **Single model, single layer, single language**: Results may not generalize to other layers (L18 or L30) or other languages.
2. **Degenerate cluster**: The ~30 near-identical harmless-enriched features suggest a SAE training artifact. Filtering these out would be needed before interpreting the harmless-enriched side.
3. **No semantic interpretation**: Feature indices alone are uninterpretable. To understand what these features represent, they would need to be decoded (e.g., by finding maximally activating examples in a broader dataset or using a feature decoder).
4. **SAE coverage**: The SAE was trained on a limited dataset (860 prompts). A larger, more diverse training corpus would improve reconstruction quality and feature quality.

---

## Summary

At layer 24 of TinyAya-Global (English), ~11 SAE features show Cohen's d > 0.8 between harmful and harmless prompts, representing a small fraction of the 16384-feature dictionary (0.07%). The top harmful-enriched feature (#12081, d=1.15) has roughly 5× higher mean activation on harmful vs harmless inputs. A large degenerate cluster of 30+ harmless-enriched features with near-identical statistics likely represents a SAE training artifact rather than semantically meaningful content. The harmful signal in SAE space is sparse but moderately broad (features activate on 20–46% of prompts), suggesting the harmfulness signal is not concentrated in a handful of highly-specific features.
