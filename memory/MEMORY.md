# Project Memory: cohere-safety-representations

## What this project is
Mechanistic interpretability study of safety representations in TinyAya multilingual models.
Core hypothesis: Regional models (Earth/Fire/Water) have more language-specific safety directions than Global.

## Critical path
1. Confirm HuggingFace model access (`scripts/01_check_model_access.py`) — BLOCKS everything
2. Update `config/models.py` HF IDs once real TinyAya slugs are available (currently placeholders)

## Languages: en, fr, ar, zh, sw, yo
- sw (Swahili) and yo (Yoruba) are low-resource — translation quality needs extra QC
- NLLB codes in `config/languages.py`

## Pipeline order
01 check access → 02 download → 03 translate → 04 build prompts
→ 05 extract activations → 06 train probes → 07 behavioral baseline
→ 08 llm judge → 09 SAE analysis

## Architecture decisions
- Activation extraction: HuggingFace forward hooks (`src/model_wrapper.py`)
  - Supports LlamaForCausalLM / Aya style (model.model.layers) + GPT-2 style
- Probe types: DiffMean + LogReg 5-fold CV (`src/probe.py`)
- Probe layers: [4, 9, 18, 24, 30, 35] (configurable per model in config/models.py)
- SAE: TopK sparse autoencoder, 8× expansion, k=32 (`scripts/09_sae_analysis.py`)
- SAE layer: 24 (mid-late; update after probe sweep)
- Judge: GPT-4o + Command-R ensemble (`src/judge.py`) — validate κ ≥ 0.7 before full run

## Key file locations
- config/models.py     — TinyAya HF IDs + probe layers (UPDATE HF IDs)
- config/languages.py  — 6 languages + NLLB codes
- src/model_wrapper.py — activation extraction via hooks
- src/probe.py         — DiffMean, LogReg, silhouette, cosine alignment
- src/judge.py         — GPT-4o + Command-R judge ensemble
- notebooks/00_sanity_check.ipynb — validate on en × global before full sweep

## Sanity check pass criteria (English × Global)
- LogReg probe accuracy > 85%
- Silhouette score > 0
- DiffMean ↔ LR direction cosine > 0.7

## Data sources
- XSafety: HF `Libra-learnmore/xsafety` (multilingual safety benchmark)
- Aya Red-teaming: HF `CohereForAI/aya_redteaming` (check access)
- PKU-SafeRLHF: HF `PKU-Alignment/PKU-SafeRLHF` (500 pairs sampled, then NLLB translated)
- Harmless prompts: LLM-assisted rephrase (GPT-4o-mini or Command-R+); spot-check required

## Activation extraction status (2026-04-08)
- Extracted (860 samples, 430/430 balanced, d_model=2048): global, base_sft, base_dpo × all 6 langs
- Pending: base, base_sft_20k, base_sft_n500/n2000/n5000/n10000/n20000 (sweep models registered 2026-04-08)

## Deferred work
- [project_todo.md](project_todo.md) — DPO sweep re-run needed for finetuned_sweep SFT checkpoints
