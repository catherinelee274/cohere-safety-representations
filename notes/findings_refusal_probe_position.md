# Findings: Refusal Probe Position

**Scripts**: `scripts/20_refusal_probe_position.py`
**Data**: `results/probes/refusal_position_comparison.csv`
**Figures**: `results/figures/refusal_position_*.png`

---

## Research Questions

1. Does the token position at which activations are extracted affect the refusal probe's ability to classify refused vs complied responses?
2. Does fine-tuning (SFT/DPO) change position sensitivity?

---

## Setup

Probe trained to classify **refused vs complied** responses (not harmful vs harmless) using activations at two positions:
- **t_inst**: the instruction token position (end of user turn, before model generates)
- **eval**: the evaluation position (after some generation — typically the first assistant token)

Metrics: DiffMean accuracy/AUC and LogReg accuracy/AUC, at layers 4, 9, 18, 24, 30, 35.
Labels: refused_harmful (n=85/105/137) vs complied_harmful+harmless (n=473/453/421) depending on model.

---

## Findings

### 1. Position (t_inst vs eval) has minimal effect — both are similarly weak

Base model / English:

| layer | pos    | dm_auc | lr_auc |
|-------|--------|--------|--------|
| 4     | t_inst | 0.708  | 0.703  |
| 4     | eval   | 0.704  | 0.681  |
| 9     | t_inst | 0.702  | 0.687  |
| 9     | eval   | 0.699  | 0.676  |
| 18    | t_inst | 0.682  | 0.677  |
| 18    | eval   | 0.684  | 0.695  |
| 24    | t_inst | 0.704  | 0.702  |
| 24    | eval   | 0.700  | 0.669  |

At all layers and both positions, DiffMean AUC is 0.68–0.71 and LogReg AUC is 0.67–0.70. The two positions give essentially the same probe performance. There is no clear advantage to extracting from the evaluation position over the instruction position.

---

### 2. Aligned models show substantially lower refusal separability than base

| model    | pos    | layer 18 dm_auc | layer 18 lr_auc |
|----------|--------|-----------------|-----------------|
| base     | t_inst | 0.682           | 0.677           |
| base     | eval   | 0.684           | 0.695           |
| base_sft | t_inst | 0.636           | 0.551           |
| base_sft | eval   | 0.619           | 0.576           |
| base_dpo | t_inst | 0.626           | 0.631           |
| base_dpo | eval   | 0.632           | 0.622           |

After SFT, both DiffMean and LogReg AUC drop by ~0.05–0.12. After DPO, there is partial recovery relative to SFT for LogReg (0.63 vs 0.55), but still below base.

This means the linear harmfulness direction is a *worse* predictor of which inputs will be refused after fine-tuning than it is in the base model. Fine-tuning does not make the refused/complied distinction more linearly separable — it makes it less so. This is consistent with refusal being mediated through a more complex (non-linear) or different mechanism in the aligned models.

---

### 3. Refusal separability peaks at early layers (4–9) and is lower at mid layers

For base at t_inst:

| layer | dm_auc | lr_auc |
|-------|--------|--------|
| 4     | 0.708  | 0.703  |
| 9     | 0.702  | 0.687  |
| 18    | 0.682  | 0.677  |
| 24    | 0.704  | 0.702  |
| 30    | 0.688  | 0.689  |
| 35    | 0.650  | 0.684  |

Refusal probe AUC peaks at layer 4 and is relatively flat through layer 24 before declining. This contrasts with harmfulness probe AUC (script 06), which peaks at mid layers (18–24) in the base model. The refusal decision appears to be encoded somewhat earlier in the network than the harmfulness representation, though both are present across layers.

---

## Summary

Token position (t_inst vs eval) does not meaningfully affect refusal probe accuracy — both positions give AUC ≈ 0.68–0.71 in the base model. Fine-tuning substantially reduces the linear separability of the refused vs complied distinction (−0.05 to −0.12 AUC), suggesting that the harmfulness direction is not the primary axis along which refusal decisions are computed in the aligned models. Refusal-separating signal peaks at early layers (4–9) in the base model, earlier than the harmfulness direction (which peaks at 18–24).
