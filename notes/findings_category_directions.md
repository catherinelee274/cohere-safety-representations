# Findings: Category Directions

**Scripts**: `scripts/16_category_directions.py`
**Data**: `results/probes/category_directions_*.csv`, `results/probes/category_refusal_cosine.csv`
**Figures**: `results/figures/category_directions_*.png`

---

## Research Questions

1. Do different harm categories (crimes, hate speech, mental health, etc.) project in different directions in representation space?
2. Which categories are most aligned with the refusal direction?
3. Does fine-tuning change which categories are refusal-aligned?

---

## Findings

### 1. Harm categories split into two clusters, with Prompt_Leaking as a structural outlier

Pairwise cosines between per-category DiffMean directions, base model / English / layer 18:

**Main harmful cluster** — {Crimes, Ethics, Reverse_Exposure, Insult, Physical_Harm, Unfairness, Unsafe_Instruction}:
Most pairs have cosines 0.59–1.00 (e.g., Unfairness↔Unsafe_Instruction = 0.990, Crimes↔Reverse_Exposure = 0.942, Ethics↔Reverse_Exposure = 0.956).

**Secondary cluster** — {Goal_Hijacking, Role_Play_Instruction, commonsense}:
Internal cosines 0.74–0.85 (e.g., Goal_Hijacking↔Role_Play = 0.854, commonsense↔Role_Play = 0.738).
These overlap moderately with the main cluster (0.34–0.77) but are most similar to each other.

**Structural outliers**:
- **Prompt_Leaking**: strongly anti-correlated with the main cluster (−0.45 to −0.75 vs Crimes, Ethics, Goal_Hijacking). Cosine with Role_Play = −0.819; with commonsense = −0.669. Positive only with Inquiry_With_Unsafe_Opinion (+0.792). Prompt_Leaking occupies a near-opposite direction from most other harm categories.
- **Mental_Health**: weakly correlated with all clusters (0.17–0.45 with main cluster; near-zero with Unfairness/Unsafe_Instruction). Not clearly in either cluster.
- **Inquiry_With_Unsafe_Opinion**: near-zero or negative with the main cluster (−0.22 vs Crimes, −0.44 vs Physical_Harm), positive with Prompt_Leaking (+0.792). Aligns with the "boundary-probing" categories rather than the clearly harmful ones.
- **Privacy_And_Property**: moderately positive with main cluster (0.41–0.55) but also positive with Inquiry_With_Unsafe_Opinion (+0.613). Intermediate position.

---

### 2. Category-refusal alignment varies widely in the base model

cos_refusal at layer 18, English, base model (sorted descending):

| Category                      | cos_refusal |
|-------------------------------|-------------|
| Unfairness_And_Discrimination | +0.908      |
| Unsafe_Instruction_Topic      | +0.907      |
| Reverse_Exposure              | +0.759      |
| Ethics_And_Morality           | +0.746      |
| Crimes_And_Illegal_Activities | +0.657      |
| Insult                        | +0.518      |
| Privacy_And_Property          | +0.498      |
| Physical_Harm                 | +0.361      |
| Prompt_Leaking                | +0.081      |
| Inquiry_With_Unsafe_Opinion   | +0.047      |
| Goal_Hijacking                | +0.038      |
| Role_Play_Instruction         | −0.105      |
| commonsense                   | −0.193      |
| Mental_Health                 | −0.219      |

The top-5 categories (Unfairness, Unsafe_Instruction, Reverse_Exposure, Ethics, Crimes) are all strongly refusal-aligned (>0.65). The bottom cluster (Role_Play, commonsense, Mental_Health) is negatively aligned — these categories project *away* from the refusal direction in the base model.

---

### 3. SFT increases refusal alignment broadly; DPO decreases it

cos_refusal at layer 18, English, across models:

| Category                      | base   | base_sft | base_dpo |
|-------------------------------|--------|----------|----------|
| Unfairness_And_Discrimination | +0.908 | +0.891   | +0.543   |
| Unsafe_Instruction_Topic      | +0.907 | +0.893   | +0.575   |
| Reverse_Exposure              | +0.759 | +0.880   | +0.466   |
| Ethics_And_Morality           | +0.746 | +0.878   | +0.441   |
| Crimes_And_Illegal_Activities | +0.657 | +0.852   | +0.495   |
| Insult                        | +0.518 | +0.794   | +0.317   |
| Physical_Harm                 | +0.361 | +0.645   | +0.183   |
| Goal_Hijacking                | +0.038 | +0.370   | +0.065   |
| Privacy_And_Property          | +0.498 | +0.363   | +0.469   |
| Role_Play_Instruction         | −0.105 | +0.258   | +0.028   |
| commonsense                   | −0.193 | +0.090   | −0.326   |
| Mental_Health                 | −0.219 | −0.002   | −0.083   |
| Inquiry_With_Unsafe_Opinion   | +0.047 | −0.218   | +0.184   |
| Prompt_Leaking                | +0.081 | −0.300   | +0.085   |

**After SFT**: Most categories gain substantial refusal alignment. Previously negative categories (Role_Play, commonsense) flip to slightly positive. The main harmful cluster all rise to 0.64–0.89 — fine-tuning aligns the representation of "harmful" with the refusal direction across most categories. Exceptions: Inquiry_With_Unsafe_Opinion and Prompt_Leaking become *more negative* after SFT (−0.22, −0.30 respectively), consistent with their outlier position in the pairwise cosine structure.

**After DPO**: All categories show lower refusal alignment than SFT. The main cluster drops to 0.18–0.58. commonsense becomes strongly negative (−0.326). DPO appears to decouple category harmfulness representations from the refusal direction more than SFT does. This is consistent with the script 19 finding that DPO's overall harm↔refusal cosine (hvr_cosine) is lower than SFT's.

---

## Summary

In the base model, harm categories cluster into two groups: a main "clearly harmful" cluster (Crimes, Ethics, Unfairness, etc.) that is also refusal-aligned, and a secondary "boundary-probing" cluster (Goal_Hijacking, Role_Play, commonsense) that is not. Prompt_Leaking is a structural outlier, projecting in a near-opposite direction to most harmful categories. SFT increases refusal alignment across the main cluster (to 0.64–0.89), while DPO does not — DPO decreases alignment to 0.18–0.58, with commonsense going strongly negative. The rank order of categories is largely preserved across training stages: Unfairness and Unsafe_Instruction_Topic are consistently the most refusal-aligned; Mental_Health and the boundary-probing categories are consistently the least.
