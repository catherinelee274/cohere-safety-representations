# cohere-safety-representations

Mechanistic study of how safety representations form across TinyAya model variants
(Base → Global → Regional) and across 6 languages (en, fr, ar, zh, sw, yo).

## Quick start

```bash
pip install -r requirements.txt

# 1. Check model access (critical path)
python scripts/01_check_model_access.py --token $HF_TOKEN

# 2. Download datasets
python scripts/02_download_datasets.py --token $HF_TOKEN

# 3. Translate PKU-SafeRLHF to target languages via NLLB
python scripts/03_translate_prompts.py

# 4. Build matched harmful/harmless prompt sets
python scripts/04_build_prompt_sets.py

# 5. Sanity check — English × Global only (2-3 hours)
python scripts/05_extract_activations.py --model global --lang en --layers 18 24 30
python scripts/06_train_probes.py        --model global --lang en
jupyter lab notebooks/00_sanity_check.ipynb

# 6. Full sweep (after sanity check passes)
python scripts/07_behavioral_baseline.py --all
python scripts/08_llm_judge.py           --validate --labeled data/judge_gold_50.jsonl
python scripts/08_llm_judge.py           --all
python scripts/05_extract_activations.py --all
python scripts/06_train_probes.py        --all --alignment
python scripts/09_sae_analysis.py        --model global --lang en --layer 24 --train --analyse
```

## Experiment order (recommended)

1. **Behavioral baselines** (`07_behavioral_baseline.py`) — confirm unsafe/refusal gap exists
2. **Probe accuracy + silhouette** (`06_train_probes.py`) — main hypothesis test
3. **Cross-lingual direction alignment** (`06_train_probes.py --alignment`) — language-specificity
4. **SAE analysis** (`09_sae_analysis.py`) — most expensive; run last

## Project structure

```
config/          Language + model configs (update HF IDs once access confirmed)
data/
  raw/           Downloaded datasets (XSafety, Aya Red-teaming, PKU-SafeRLHF)
  processed/     Translations + extracted activations
  prompts/       Final harmful/harmless prompt sets per language
scripts/         Numbered pipeline scripts (01–09)
src/             Shared utilities (ModelWrapper, probes, judge)
notebooks/       00_sanity_check.ipynb
results/         Output: probes/, behavioral/, sae/, figures/
```

## Key decisions

| Choice | Decision | Rationale |
|---|---|---|
| Activation extraction | HuggingFace hooks | Broad model support; no TransformerLens porting needed |
| Probe types | DiffMean + LogReg (5-fold CV) | DiffMean = interpretable; LR = stronger accuracy signal |
| Translation | NLLB-200-distilled-1.3B | Good Swahili/Yoruba quality; swap for 3.3B if needed |
| Judge ensemble | GPT-4o + Command-R | Diverse, validated with Cohen's κ ≥ 0.7 before full run |
| SAE | TopK sparse autoencoder | Standard; 8× expansion, k=32 active features |
| SAE layer | 24 | Mid-late layers concentrate safety features (prior work) |
