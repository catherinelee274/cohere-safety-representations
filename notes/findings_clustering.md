# Findings: Refusal Cluster Structure

**Scripts**: `scripts/15_refusal_clustering.py`
**Data**: `results/probes/harmfulness_refusal_clustering.csv`
**Figures**: `results/figures/refusal_cluster_*.png`

---

## Research Questions

1. Are refused harmful prompts more refusal-aligned in representation space than accepted harmful prompts?
2. Does fine-tuning (SFT) sharpen or compress the separation between refused/accepted harmful/harmless clusters?
3. Is there a clean threshold along the refusal direction that separates these groups?

---

## Methodology

Each prompt's activation is projected onto the refusal direction (the DiffMean direction from `refused` vs `complied` behavioral data). The projection scalar `s` is reported as mean ± std per group per layer. Three groups:
- **refused_harmful** (n=85): harmful prompts the model refused
- **accepted_harmful** (n=473): harmful prompts the model answered
- **accepted_harmless** (n=200): harmless prompts the model answered

Position: `post_inst` (activation after the instruction token) is the primary measure; `t_inst` is also available.

---

## Findings

### 1. Refused harmful prompts project higher on the refusal direction than accepted harmful — but overlap is large

base model / English / post_inst position:

| layer | group             | mean_s | std_s | n   |
|-------|-------------------|--------|-------|-----|
| 4     | refused_harmful   | +0.120 | 0.125 | 85  |
| 4     | accepted_harmful  | +0.021 | 0.159 | 473 |
| 4     | accepted_harmless | +0.008 | 0.172 | 200 |
| 9     | refused_harmful   | +0.227 | 0.169 | 85  |
| 9     | accepted_harmful  | +0.092 | 0.217 | 473 |
| 9     | accepted_harmless | +0.069 | 0.238 | 200 |
| 18    | refused_harmful   | +0.279 | 0.186 | 85  |
| 18    | accepted_harmful  | +0.137 | 0.232 | 473 |
| 18    | accepted_harmless | +0.103 | 0.263 | 200 |
| 24    | refused_harmful   | +0.164 | 0.125 | 85  |
| 24    | accepted_harmful  | +0.067 | 0.157 | 473 |
| 24    | accepted_harmless | +0.042 | 0.180 | 200 |
| 30    | refused_harmful   | +0.127 | 0.163 | 85  |
| 30    | accepted_harmful  | +0.014 | 0.186 | 473 |
| 30    | accepted_harmless | +0.000 | 0.195 | 200 |
| 35    | refused_harmful   | +0.075 | 0.105 | 85  |
| 35    | accepted_harmful  | +0.005 | 0.118 | 473 |
| 35    | accepted_harmless | −0.005 | 0.125 | 200 |

At **layer 18**, the separation peaks: refused_harmful mean = +0.279 vs accepted_harmful mean = +0.137 — a difference of 0.14. However, with std ≈ 0.18–0.23, there is substantial overlap between groups. The effect size (Cohen's d ≈ 0.14/0.21 ≈ 0.67) is moderate but not sufficient for clean classification.

Accepted harmful and accepted harmless prompts are nearly indistinguishable along this axis (mean difference ≈ 0.034 at layer 18).

**The refusal direction captures a real but small signal**: refused harmful prompts sit higher on average, but the distributions overlap too much for a scalar threshold to cleanly separate them.

---

### 2. Layer 18 shows the strongest separation; separation decreases at deeper layers

The mean_s difference between refused and accepted harmful peaks at layer 9–18 and diminishes at layers 24–35. This is consistent with the harm↔refusal cosine peaking at mid layers (script 19) and declining toward deeper layers.

---

### 3. Fine-tuning (SFT) compresses all three groups toward positive values but reduces inter-group spread

base_sft model / English / post_inst:

| layer | group             | mean_s | std_s | n   |
|-------|-------------------|--------|-------|-----|
| 18    | refused_harmful   | +0.200 | 0.103 | 105 |
| 18    | accepted_harmful  | +0.158 | 0.140 | 453 |
| 18    | accepted_harmless | +0.113 | 0.182 | 200 |

After SFT at layer 18: refused_harmful mean = +0.200 vs accepted_harmful = +0.158 — a difference of only 0.042 (vs 0.142 in base). The groups shift to be more uniformly positive, and the std decreases (0.10 vs 0.19 for refused_harmful), but the inter-group gap narrows substantially.

**Interpretation**: After SFT, the refusal direction no longer discriminates well between refused and accepted harmful prompts within the post-instruction position. This is consistent with the aligned model computing refusal at a different stage or via a different mechanism than the probe direction captured.

---

## Summary

In the base model, refused harmful prompts project systematically higher onto the refusal direction than accepted harmful prompts (mean difference ≈ 0.14 at layer 18), with peak separation at mid layers (9–18) and convergence at deeper layers. The separation is statistically real but too noisy for threshold-based classification (large standard deviations, substantial overlap). Accepted harmful and accepted harmless prompts are nearly indistinguishable along this axis. After SFT, all groups shift toward positive projections and inter-group spread compresses — the refusal direction as computed from the base model's behavioral data no longer cleanly indexes refusal behavior in the fine-tuned model.
