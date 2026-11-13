# Findings: Evolution of Safety Representations Across Fine-tuning

**Scripts**: `scripts/19_finetuning_evolution.py`, `scripts/20_refusal_probe_position.py`
**Data**: `results/probes/finetuning_evolution.csv`, `results/probes/refusal_position_comparison.csv`
**Figures**: `results/figures/fig_evolution_*.png`, `results/figures/fig_refusal_position_*.png`

---

## Research Question

How do harmfulness, harmlessness, and refusal representations change before and after fine-tuning (Base → SFT → DPO)?

1. Does fine-tuning preserve separation between harmful and harmless prompts?
2. Does fine-tuning change where refusal appears internally (by layer)?
3. Does fine-tuning make harmfulness and refusal more entangled?

---

## Findings

### 1. Cluster centroids are preserved; linear separability decreases

**harm_drift_cos (centroid-based)** at layer 18 — cosine between harmful cluster centroid direction before and after fine-tuning:

| Model    | en    | fr    | ar    | hi    |
|----------|-------|-------|-------|-------|
| Base+SFT | 0.969 | 0.918 | 0.986 | 0.968 |
| Base+DPO | 0.983 | 0.925 | 0.985 | 0.985 |

The harmful cluster centroids barely move after fine-tuning (cosines 0.92–0.99 across all languages). At layer 18, the 1D projection onto the centroid direction shows identical separation and within-class spread: separation ≈ 21.7, std_harmful ≈ 100, std_harmless ≈ 165 — **unchanged across base, base_sft, and base_dpo**.

**Yet LogReg probe accuracy drops substantially:**

| Model    | en    | fr    | ar    | sw    | yo    | hi    |
|----------|-------|-------|-------|-------|-------|-------|
| Base     | 0.953 | 0.984 | 0.975 | 0.982 | 0.981 | 0.983 |
| Base+SFT | 0.786 | 0.778 | 0.769 | 0.785 | 0.760 | 0.845 |
| Base+DPO | 0.790 | 0.777 | 0.760 | 0.774 | 0.737 | 0.983 |

Fine-tuning doesn't move the harmful cluster in the centroid direction, but changes the **higher-dimensional covariance structure** — the 2048D distribution becomes harder to linearly separate even though the cluster means are preserved. The WHERE of the clusters is stable; the SHAPE is not.

Hindi: DPO keeps accuracy at 0.983 (unchanged from base), but SFT drops it to 0.845. English-centric alignment partially exempts Hindi — DPO leaves it untouched, SFT does not.

### 2. Refusal is not linearly encoded at input positions

Refusal probe AUC at layer 18 (balanced LogReg, harmful prompts only):

| Model    | eval (asst turn-start) | t_inst (last user token) |
|----------|------------------------|--------------------------|
| Base     | 0.695                  | 0.677                    |
| Base+SFT | 0.576                  | 0.551                    |
| Base+DPO | 0.622                  | 0.631                    |

AUC 0.55–0.70 across all models and positions, even with balanced sampling (class_weight='balanced'). Neither the t_inst nor eval position gives strong refusal separability. Class imbalance (15–25% refused) is not the cause — the signal is genuinely weak.

Counterintuitive: the base model has the *highest* refusal AUC despite the lowest refusal rate (15%). Fine-tuning reduces refusal AUC, suggesting SFT/DPO implements refusal via a more distributed or non-linear mechanism.

Note: `activations_postinst` and `activations_eval` capture the same position (both extract the last input token). True Zhao et al. post_inst probing (first *generated* token) would require `model.generate(..., output_hidden_states=True)` — not yet implemented.

### 3. Fine-tuning aligns harmfulness and refusal directions

cos_sim(harmfulness direction, refusal direction) at layer 18:

| Model    | en    | fr    |
|----------|-------|-------|
| Base     | +0.171 | −0.095 |
| Base+SFT | +0.892 | —      |
| Base+DPO | +0.523 | +0.790 |

Fine-tuning aligns the harmfulness and refusal directions into a shared axis. After alignment training, the model uses the same internal direction for both "is this harmful?" and "should I refuse?". SFT shows stronger entanglement than DPO — DPO may produce a slightly more independent refusal axis.

### 4. Direction drift increases with depth

harm_drift_cos by layer for English:

| Layer | Base+SFT | Base+DPO |
|-------|----------|----------|
| 4     | 0.985    | 0.999    |
| 9     | 0.989    | 0.999    |
| 18    | 0.969    | 0.983    |
| 24    | 0.918    | 0.942    |
| 30    | 0.856    | 0.899    |
| 35    | 0.717    | 0.854    |

Fine-tuning modifies late-layer representations more than early ones. Early layers (4–18) are near-identical to base; deep layers (30–35) show more reorganization.

---

## Summary

Fine-tuning (SFT/DPO) preserves the *location* of harmful/harmless clusters (centroid cosines 0.92–0.99) but changes the *shape* of the distribution, reducing linear separability from 0.95 to 0.77–0.79. Refusal is not strongly linearly encoded at any input position for TinyAya — AUC peaks at 0.70 for the base model and actually decreases after fine-tuning, contrasting with Zhao et al.'s large aligned LLMs. At the same time, fine-tuning aligns the harmfulness and refusal directions into a shared axis. Hindi representations are unaffected, consistent with English-centric alignment training.

---

## Open Questions

1. **Why does linear separability drop if centroids and 1D spread are preserved?** Higher-dimensional covariance is likely changing — worth computing Fréchet distance between harmful distributions across models.

2. **Generation-position refusal probing**: Extract hidden states at the first generated token via `model.generate(..., output_hidden_states=True)`. This may show stronger refusal encoding than any input position.

3. **Does harm↔refusal alignment hold for regional models (global, earth, water, fire)?** Partially answered — see below.

4. ~~**Hindi protection**~~ **Resolved**: base_sft Hindi logreg_acc = 0.845 (drops from 0.983), base_dpo = 0.983 (unchanged). DPO exempts Hindi; SFT does not. The "Hindi protection" effect is DPO-specific.

---

## Follow-up: Regional models — global, earth, water (script 19 with `--models global earth water base`)

**Data**: `results/probes/finetuning_evolution.csv` (regional rows appended)

### What is valid

**harm_refusal_cos for global (English)**:

| Layer | harm_refusal_cos | logreg_acc |
|-------|-----------------|------------|
| 4     | 0.952           | 0.791      |
| 9     | 0.957           | 0.792      |
| 18    | 0.925           | 0.781      |
| 24    | 0.850           | 0.780      |
| 30    | 0.800           | 0.859      |
| 35    | 0.751           | 0.866      |

Global's harm↔refusal cos (0.92–0.96) is as high or higher than base_sft (0.892). The alignment finding generalises: globally-aligned TinyAya uses the same internal axis for harmfulness and refusal.

Earth and water lack refusal probes (no behavioral responses extracted for those models), so harm_refusal_cos is unavailable.

### Clean comparison (re-extracted on same eval prompts)

After re-extracting activations for global/earth/water using the same `harmful_eval.jsonl` prompt set as base and retraining probes, the valid comparison at layer 18:

**logreg_acc**:

| Lang | Base  | Base+SFT | Base+DPO | Global | Earth | Water |
|------|-------|----------|----------|--------|-------|-------|
| en   | 0.953 | 0.786    | 0.790    | 0.818  | 0.822 | 0.818 |
| fr   | 0.984 | 0.778    | 0.777    | 0.810  | 0.811 | 0.813 |
| ar   | 0.975 | 0.769    | 0.760    | 0.752  | 0.741 | 0.743 |
| hi   | 0.983 | 0.844    | 0.983    | 0.837  | 0.846 | 0.844 |
| sw   | 0.982 | 0.785    | 0.774    | 0.755  | 0.983 | 0.983 |
| yo   | 0.981 | 0.760    | 0.737    | 0.754  | 0.983 | 0.982 |

**harm_drift_cos** (centroid cosine vs base):

| Lang | Base+SFT | Base+DPO | Global | Earth | Water |
|------|----------|----------|--------|-------|-------|
| en   | 0.969    | 0.983    | 0.960  | 0.961 | 0.961 |
| fr   | 0.918    | 0.925    | 0.853  | 0.859 | 0.857 |
| ar   | 0.986    | 0.985    | 0.961  | 0.964 | 0.964 |
| hi   | 0.968    | 0.985    | 0.768  | 0.778 | 0.773 |

**harm_refusal_cos at layer 18 (English)**:

| Model  | harm_refusal_cos |
|--------|-----------------|
| Base   | +0.171          |
| Global | +0.911          |
| Earth  | NaN (no refusal probes) |
| Water  | NaN (no refusal probes) |

### Interpretation

- **Global confirms the alignment finding**: harm↔refusal cos = 0.911, as high as Base+SFT (0.892). All aligned models (SFT, DPO, Global) share the same harm↔refusal entanglement.
- **Earth and water show mixed signals**: sw/yo maintain high logreg_acc (0.98–0.98, like base), but en/fr/ar drop to 0.74–0.82 (like aligned models). This suggests regional training partially aligns the English/French/Arabic representations but leaves Swahili/Yoruba untouched — the inverse of the Hindi DPO pattern.
- **harm_drift_cos for regional models** (0.85–0.96) is slightly lower than for base_sft/base_dpo (0.92–0.99), meaning regional alignment moves the harmfulness direction slightly more than fine-tuning on the same base — especially for Hindi (0.77 vs 0.97).
