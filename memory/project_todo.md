---
name: Project TODO / Deferred Work
description: Tasks deferred mid-session that need to be picked up later
type: project
---

## DPO sweep re-run (deferred)

DPO training was killed (2026-04-08) to free GPU for activation extraction.
Need to run DPO on top of each `finetuned_sweep/base_sft_n{500,2000,5000,10000,20000}` checkpoint to produce matching DPO variants for the data-size sweep.

**Why:** Completing the sweep with DPO lets us compare SFT-only vs SFT→DPO representations at each data size — stronger RQ2 story.

**How to apply:** After DPO sweep is done, register the DPO sweep models in `config/models.py` (similar to how SFT sweep was added), extract activations, and train probes.

## Visualization TODOs

- Verify `12_visualize_probes.py` correctly shows harmful/harmless separation (UMAP plots)
- After verifying, add refusal direction overlay to UMAP in `12_visualize_probes.py` (blocked on 06b, which needs 07 first)

## Pipeline execution order (blocked steps)

**Can run now (activations exist):**
- `python scripts/06_train_probes.py --all` — harmful/harmless probe directions

**Blocked on step 07 (behavioral baseline):**
- `06b_refusal_direction.py` requires `results/behavioral/responses_{model}_{lang}.jsonl` from step 07 to get per-prompt refusal labels before it can train refusal direction probes
- Run order: `07_behavioral_baseline.py --all` → `08_llm_judge.py --all` → `06b_refusal_direction.py --all --cross-stage`
