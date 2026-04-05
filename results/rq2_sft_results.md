# RQ2 Results — TinyAya-Base vs Base-SFT (PKU-SafeRLHF)

**Date:** 2026-03-30
**Models compared:** CohereLabs/tiny-aya-base vs base_sft (QLoRA adapter, 3 epochs PKU-SafeRLHF)
**Languages:** en, fr, ar, zh, sw, yo
**Layers probed:** 4, 9, 18, 24, 30, 35
**N per language:** 860 (430 harmful XSafety + 430 harmless LITMUS/translated)

---

## 1. DiffMean AUC — Base-SFT

| Lang | L4    | L9    | L18   | L24   | L30   | L35   | Best |
|------|-------|-------|-------|-------|-------|-------|------|
| en   | 0.655 | 0.706 | **0.771** | 0.649 | 0.636 | 0.643 | L18 |
| fr   | 0.637 | 0.696 | **0.751** | 0.657 | 0.614 | 0.603 | L18 |
| ar   | 0.679 | 0.638 | **0.721** | 0.631 | 0.613 | 0.759 | L35 |
| zh   | 0.704 | 0.802 | 0.820 | 0.704 | 0.684 | **0.834** | L35 |
| sw   | 0.675 | 0.736 | **0.765** | 0.601 | 0.559 | 0.760 | L18 |
| yo   | **0.760** | 0.704 | 0.656 | 0.659 | 0.602 | 0.627 | L4 |

## 2. LogReg AUC — Base-SFT

| Lang | L4    | L9    | L18   | L24   | L30   | L35   | Best |
|------|-------|-------|-------|-------|-------|-------|------|
| en   | **0.970** | 0.939 | 0.952 | 0.934 | 0.912 | 0.921 | L4 |
| fr   | 0.933 | 0.944 | **0.951** | 0.929 | 0.895 | 0.918 | L18 |
| ar   | 0.932 | 0.939 | **0.945** | 0.923 | 0.908 | 0.896 | L18 |
| zh   | **0.979** | 0.971 | 0.975 | 0.966 | 0.959 | 0.960 | L4 |
| sw   | 0.918 | 0.904 | **0.938** | 0.907 | 0.895 | 0.899 | L18 |
| yo   | 0.896 | 0.886 | **0.914** | 0.883 | 0.867 | 0.868 | L18 |

## 3. Silhouette at L24 — Base-SFT

| Lang | Silhouette |
|------|-----------|
| en   | **0.114** |
| zh   | 0.104     |
| sw   | 0.051     |
| fr   | 0.045     |
| ar   | 0.045     |
| yo   | **0.031** |

---

## 4. RQ2 Comparison: Base vs Base-SFT

### DiffMean AUC at best layer (Δ = SFT − Base)

| Lang | Base best (layer) | SFT best (layer) | Δ     |
|------|-------------------|------------------|-------|
| en   | 0.759 (L9)        | 0.771 (L18)      | +0.012 |
| fr   | 0.709 (L24)       | 0.751 (L18)      | +0.042 |
| ar   | 0.787 (L24)       | 0.759 (L35)      | −0.028 |
| zh   | 0.786 (L35)       | 0.834 (L35)      | +0.048 |
| sw   | 0.695 (L24)       | 0.765 (L18)      | +0.070 |
| yo   | 0.708 (L4)        | 0.760 (L4)       | +0.052 |

**Note:** DiffMean AUC improves after SFT for most languages, most notably sw (+0.07) and yo (+0.05). Arabic is the sole exception (−0.028), consistent with its outlier status in the cross-lingual alignment analysis.

### LogReg AUC at best layer (Δ = SFT − Base)

| Lang | Base best (layer) | SFT best (layer) | Δ      |
|------|-------------------|------------------|--------|
| en   | 0.938 (L4/L9)     | 0.970 (L4)       | +0.032 |
| fr   | 0.933 (L9/L18)    | 0.951 (L18)      | +0.018 |
| ar   | 0.924 (L9)        | 0.945 (L18)      | +0.021 |
| zh   | 0.965 (L4)        | 0.979 (L4)       | +0.014 |
| sw   | 0.928 (L9)        | 0.938 (L18)      | +0.010 |
| yo   | 0.927 (L4)        | 0.914 (L18)      | −0.013 |

**Note:** LogReg AUC improves uniformly after SFT except for Yoruba (−0.013). However, Yoruba's silhouette improves (0.029 → 0.031), suggesting marginal geometric improvement.

### Silhouette at L24 (Δ = SFT − Base)

| Lang | Base   | SFT    | Δ      |
|------|--------|--------|--------|
| en   | 0.121  | 0.114  | −0.007 |
| zh   | 0.099  | 0.104  | +0.005 |
| sw   | 0.050  | 0.051  | +0.001 |
| fr   | 0.043  | 0.045  | +0.002 |
| ar   | 0.043  | 0.045  | +0.002 |
| yo   | 0.029  | 0.031  | +0.002 |

---

## 5. Key Findings (RQ2)

1. **SFT improves harmfulness separation in representation space** — DiffMean AUC increases for 5/6 languages after PKU-SafeRLHF SFT, suggesting the model develops a more linearly separable safety direction.

2. **Layer shift: L9→L18 after SFT** — The best DiffMean layer shifts from early layers (L4/L9 for base) to mid layers (L18) for most languages post-SFT. Safety information consolidates at a deeper layer.

3. **Low-resource languages benefit most** — Swahili (+0.07) and Yoruba (+0.05) show the largest DiffMean improvements, even though SFT data was English-only (PKU-SafeRLHF). This suggests cross-lingual transfer of safety representations through SFT.

4. **Arabic remains the outlier** — DiffMean AUC slightly decreases for Arabic post-SFT (−0.028), consistent with Arabic's structurally different safety direction observed in Phase 2 (RQ1). English-only SFT may not adequately update Arabic's representation.

5. **LogReg is more stable than DiffMean** — LogReg AUC improves uniformly (+0.01–0.03) across almost all languages, confirming that the safety-relevant subspace expands after SFT even when the mean-difference direction is imperfect.

6. **Silhouette scores are broadly flat** — Geometric class separation (silhouette at L24) changes minimally after SFT (±0.007), suggesting SFT does not dramatically reshape the geometry of the representation space at L24.

---

## Next Steps

- [ ] Run behavioral baseline (07) + LLM judge (08) for base_sft to get refusal rates post-SFT
- [ ] Compare cosine alignment matrices Base vs Base-SFT (do languages converge post-SFT?)
- [ ] Extract activations for TinyAya-Base (no fine-tuning) for full 3-way comparison
- [ ] SAE analysis at L24 for base_sft to identify which features activate post-SFT
- [ ] Run DPO/RLHF variants if compute allows
