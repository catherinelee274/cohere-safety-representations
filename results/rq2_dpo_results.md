# RQ2 Results — Base → SFT → DPO (PKU-SafeRLHF)

**Date:** 2026-03-31  
**Status:** COMPLETE — all 6 languages  
**Chain:** CohereLabs/tiny-aya-base → SFT (284 examples, 3 epochs) → DPO (74 pairs, 3 epochs, beta=0.1)  
**Languages:** en, fr, ar, zh, sw, yo  
**Layers probed:** 4, 9, 18, 24, 30, 35  
**N per language:** 860 (430 harmful XSafety + 430 harmless LITMUS/translated)

---

## 1. DPO Training Summary

| Metric | Epoch 2 | Epoch 3 |
|--------|---------|---------|
| Loss | 0.429 | 0.318 |
| rewards/accuracies | 0.717 | **1.000** |
| rewards/margins | 1.102 | **2.952** |
| logps/chosen | −190.3 | −185.4 |
| logps/rejected | −227.7 | −237.3 |

DPO converged strongly — reward accuracy reached 100% by epoch 3, with a large margin (2.95) between chosen and rejected log-probs.

---

## 2. DiffMean AUC — Base-DPO (best layer per language)

| Lang | L4    | L9    | L18   | L24   | L30    | L35   | Best   |
|------|-------|-------|-------|-------|--------|-------|--------|
| en   | 0.648 | 0.796 | 0.691 | 0.752 | **0.810** | 0.756 | L30 |
| fr   | 0.667 | 0.691 | 0.697 | **0.729** | 0.727 | 0.662 | L24 |
| ar   | 0.688 | **0.784** | 0.757 | 0.677 | 0.654 | 0.707 | L9  |
| zh   | 0.721 | **0.821** | 0.790 | 0.765 | 0.757 | 0.816 | L9  |
| sw   | 0.673 | **0.779** | 0.666 | 0.676 | 0.744 | 0.656 | L9  |
| yo   | 0.648 | 0.632 | 0.634 | **0.728** | 0.708 | 0.715 | L24 |

## 3. LogReg AUC — Base-DPO (best layer per language)

| Lang | L4    | L9    | L18   | L24   | L30   | L35   | Best |
|------|-------|-------|-------|-------|-------|-------|------|
| en   | 0.959 | 0.958 | 0.954 | 0.960 | 0.952 | 0.940 | L24 |
| fr   | 0.934 | 0.951 | **0.953** | 0.945 | 0.913 | 0.926 | L18 |
| ar   | 0.939 | **0.942** | 0.943 | 0.930 | 0.918 | 0.910 | L18 |
| zh   | 0.967 | 0.972 | 0.978 | **0.986** | 0.956 | 0.964 | L24 |
| sw   | 0.903 | **0.927** | 0.930 | 0.906 | 0.895 | 0.899 | L18 |
| yo   | 0.896 | 0.909 | **0.913** | 0.890 | 0.880 | 0.877 | L18 |

---

## 4. Three-way comparison: Base vs SFT vs DPO (DiffMean AUC at best layer)

| Lang | Base       | SFT        | DPO        | Δ SFT−Base | Δ DPO−SFT |
|------|------------|------------|------------|------------|------------|
| en   | 0.759 (L9) | 0.771 (L18)| **0.810** (L30) | +0.012 | +0.039 |
| fr   | 0.709 (L24)| 0.751 (L18)| **0.729** (L24) | +0.042 | −0.022 |
| ar   | 0.787 (L24)| 0.759 (L35)| **0.784** (L9)  | −0.028 | +0.025 |
| zh   | 0.786 (L35)| 0.834 (L35)| **0.821** (L9)  | +0.048 | −0.013 |
| sw   | 0.695 (L24)| 0.765 (L18)| **0.779** (L9)  | +0.070 | +0.014 |
| yo   | 0.708 (L4) | 0.760 (L4) | **0.728** (L24) | +0.052 | −0.032 |

### Three-way comparison: LogReg AUC at best layer

| Lang | Base       | SFT        | DPO        | Δ SFT−Base | Δ DPO−SFT |
|------|------------|------------|------------|------------|------------|
| en   | 0.938      | **0.970**  | 0.960      | +0.032     | −0.010     |
| fr   | 0.933      | 0.951      | **0.953**  | +0.018     | +0.002     |
| ar   | 0.924      | 0.945      | **0.943**  | +0.021     | −0.002     |
| zh   | 0.965      | 0.979      | **0.986**  | +0.014     | +0.007     |
| sw   | 0.928      | **0.938**  | 0.930      | +0.010     | −0.008     |
| yo   | 0.927      | 0.914      | **0.913**  | −0.013     | −0.001     |

---

## 5. Silhouette at L24: Base vs SFT vs DPO

| Lang | Base  | SFT   | DPO   | Δ SFT | Δ DPO |
|------|-------|-------|-------|-------|-------|
| en   | 0.121 | 0.114 | 0.121 | −0.007 | +0.007 |
| zh   | 0.099 | 0.104 | 0.107 | +0.005 | +0.003 |
| sw   | 0.050 | 0.051 | 0.052 | +0.001 | +0.001 |
| fr   | 0.043 | 0.045 | 0.045 | +0.002 | +0.000 |
| ar   | 0.043 | 0.045 | 0.045 | +0.002 | +0.000 |
| yo   | 0.029 | 0.031 | 0.033 | +0.002 | +0.002 |

---

## 6. Emerging Findings (partial — yo pending)

1. **DPO further sharpens DiffMean for en and sw** — English L30 AUC improves from 0.771 (SFT) to 0.810 (DPO), the strongest single score across all models. Swahili also gains (+0.014).

2. **Layer shift continues: L18→L30 for English under DPO** — The best DiffMean layer moves later with each training stage (Base: L9, SFT: L18, DPO: L30), suggesting safety information migrates to deeper layers as alignment training progresses.

3. **DPO is less consistent than SFT cross-linguistically** — French and Chinese DiffMean AUC slightly decrease post-DPO (−0.022, −0.013), while Arabic partially recovers (+0.025). This may reflect DPO's smaller dataset (74 pairs vs 284 SFT examples) or the English-only nature of training.

4. **Arabic partially recovers under DPO** — Arabic was the sole SFT regression; DPO brings it back to near-base levels (0.787→0.759→0.784), suggesting DPO's contrastive objective helps where SFT's supervision alone was insufficient.

5. **Yoruba regresses under DPO** (0.760→0.728, −0.032) — Yoruba had the largest SFT gain but loses ground under DPO, possibly because 74 preference pairs are too few to maintain cross-lingual transfer for a very low-resource language. LogReg is barely affected (−0.001), so the linear subspace is preserved but the mean-direction degrades.

5. **LogReg AUC is broadly flat SFT→DPO** — Changes are within ±0.01, meaning DPO does not substantially reshape the linear separability of the full representation space, only the mean-difference direction.

6. **Silhouette recovers to base levels under DPO** — English silhouette at L24 returns to 0.121 (same as base), erasing the small SFT dip. Suggests DPO re-expands the geometric separation that SFT compressed slightly.

---

## 7. Next Steps

- [x] All 6 language probes complete
- [ ] Run behavioral baseline (07) + LLM judge (08) for base_dpo to get refusal rates
- [ ] Cross-lingual alignment matrices for base_dpo (do directions converge further?)
- [ ] SAE analysis at L24 comparing base / base_sft / base_dpo feature activations
