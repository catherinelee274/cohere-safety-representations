# Findings: Activation Steering

**Scripts**: `scripts/17_activation_steering.py`
**Data**: `results/steering/steering_summary.csv`, `results/steering/steering_*.jsonl`
**Figures**: `results/figures/steering_*.png`

---

## Research Questions

1. Can adding the harmfulness/refusal direction to model activations increase or decrease refusal rates?
2. Is activation steering a viable causal tool for probing safety representations in these models?

---

## Setup

Two experiments:
- **base_sft, layer 24**: alpha ∈ {0, 10, 20, 40, 80}, n=100 English prompts
- **base_dpo, layer 18**: alpha ∈ {0, 20, 50, 100, 200}, n=100 English prompts

Steering vector: the DiffMean harmfulness direction at the specified layer. The direction is added to the residual stream at that layer during generation with scale factor alpha.

Refusal detection: heuristic (keyword-based).

---

## Findings

### Null result: steering has no effect on refusal rate

| model    | layer | alpha | n   | refused | rate  |
|----------|-------|-------|-----|---------|-------|
| base_sft | 24    | 0     | 100 | 2       | 0.020 |
| base_sft | 24    | 10    | 100 | 2       | 0.020 |
| base_sft | 24    | 20    | 100 | 2       | 0.020 |
| base_sft | 24    | 40    | 100 | 2       | 0.020 |
| base_sft | 24    | 80    | 100 | 2       | 0.020 |
| base_dpo | 18    | 0     | 100 | 3       | 0.030 |
| base_dpo | 18    | 20    | 100 | 3       | 0.030 |
| base_dpo | 18    | 50    | 100 | 3       | 0.030 |
| base_dpo | 18    | 100   | 100 | 3       | 0.030 |
| base_dpo | 18    | 200   | 100 | 3       | 0.030 |

The refusal count is **identical across all alpha values**, including alpha=0 (the baseline with no steering). This is a clean null result: adding the harmfulness DiffMean direction to activations at the tested layers and scales does not change the model's refusal behavior as measured by heuristic detection.

---

## Interpretation

Several explanations are consistent with this null result:

1. **Wrong layer**: The steering was applied at a single layer (24 for base_sft, 18 for base_dpo). Refusal decisions may be computed at a different layer or distributed across layers, making single-layer steering ineffective.

2. **Wrong direction**: The DiffMean direction encodes the *harmfulness* distinction (harmful vs harmless), not the *refusal* decision. Steering the harmfulness direction may not causally affect the refusal circuit. The harm↔refusal cosine is non-trivial but <1 (0.89 for base_sft), meaning the directions are related but not identical.

3. **Alpha too small or too large**: At alpha=200 for base_dpo (a very large perturbation), there is still no change, suggesting the issue is not scale. However, at very large alpha the model may be producing incoherent outputs that still don't trigger the heuristic refusal detector.

4. **Heuristic detection is inadequate**: The heuristic keyword-based detector may miss refusals that are expressed differently after steering (e.g., non-English refusal phrasing, or novel phrasing from the perturbed distribution). An LLM judge would be needed to confirm the null result.

5. **Prompt set composition**: The 100-prompt test set may have insufficient harmful prompts to show an effect on refusal rate. The baseline refusal rates are already very low (2–3%), meaning only 2–3 prompts are being refused at alpha=0. Small absolute numbers make it hard to detect changes.

---

## Summary

Activation steering of the harmfulness DiffMean direction at the tested layers (18–24) and scales (alpha up to 200) produces no change in heuristic refusal rate for either base_sft or base_dpo. This is a null result, but it is not conclusive evidence that the harmfulness direction is non-causal — the failure modes (wrong layer, wrong direction, inadequate detection, underpowered test set) cannot be ruled out with the current experiment. Extending the steering experiment with the refusal direction (rather than harmfulness direction), multiple layers, and LLM-judge evaluation would be needed to draw stronger conclusions.
