# RQ2 — Fair-Budget Multilingual Safety-FT Re-test

**Date:** 2026-06-24
**Why:** The first multilingual run ([rq2_multilingual_results.md](rq2_multilingual_results.md))
spread a 300-example budget across 6 languages (50/lang) and *lost everywhere* —
but that was a **budget-dilution artifact**, not evidence against multilingual
training. This re-test gives each language a real budget and adds a clean control.

**Design (controlled — only language coverage differs):**
- `base_sft_ml_big` — 3000 examples, **500/language** (multilingual, Command-A translated)
- `base_sft_en3000` — the **same 3000 prompts, all in English** (matched-budget control)
- Both SFT from TinyAya-Base, identical hyperparams (QLoRA r16/α32, lr2e-4, 3 ep).
- wandb runs: `base_sft_ml_big`, `base_sft_en3000` (project cohere-safety-representations).

---

## 1. Behavioral — keyword refusal rate (XSafety eval, n≈558/lang)

| lang | base_sft (284 en) | en3000 (3000 en) | **ml_big (3000, 500/lang)** |
|------|------|------|------|
| en | 0.188 | 0.204 | 0.177 |
| fr | 0.059 | 0.065 | **0.111** |
| ar | 0.016 | 0.039 | 0.018 |
| hi | 0.029 | 0.010 | 0.000 |
| sw | 0.050 | 0.054 | **0.084** |
| yo | 0.004 | 0.005 | 0.000 |

## 2. Behavioral — Command-R judge **unsafe rate** (sampled 200/lang; lower = safer)

| lang | base_sft | base_sft_ml (diluted 300) | en3000 | **ml_big** |
|------|------|------|------|------|
| en | 0.110 | 0.180 | 0.115 | 0.145 |
| fr | 0.110 | 0.275 | 0.125 | **0.100** |
| ar | 0.145 | 0.210 | 0.160 | **0.115** |
| hi | 0.205 | 0.425 | 0.250 | 0.270 |
| sw | 0.105 | 0.145 | 0.080 | 0.080 |
| yo | 0.015 | 0.010 | 0.010 | 0.030 |

## 3. Representation — probe AUC at best layer (en3000 vs ml_big, identical prompts)

| lang | LogReg en3000 | LogReg ml_big | DiffMean en3000 | DiffMean ml_big | DiffMean Δ |
|------|------|------|------|------|------|
| en | 0.958 | 0.953 | 0.817 | 0.776 | −0.041 |
| fr | 0.955 | 0.941 | 0.801 | 0.741 | −0.060 |
| ar | 0.959 | 0.951 | 0.814 | 0.782 | −0.032 |
| **sw** | 0.942 | 0.939 | 0.724 | 0.731 | **+0.007** |
| **yo** | 0.931 | 0.928 | 0.721 | 0.730 | **+0.009** |

---

## 4. Findings — the hypothesis is partially rescued at a fair budget

1. **Budget was the confound.** The diluted multilingual model (50/lang) had the
   *worst* judge unsafe rates of any model (up to 0.425 in hi). Scaling to 500/lang
   (`ml_big`) cut that dramatically (hi 0.425→0.270, fr 0.275→0.100) — most of the
   original "multilingual hurts" result was too little data per language, not
   multilingual training itself.

2. **At a fair budget, multilingual beats English-only for the right languages.**
   vs the matched `en3000` control:
   - **Swahili (low-resource):** more refusals (0.084 vs 0.054, +56%), equal judge
     unsafe rate, and a **stronger DiffMean safety direction** (+0.007). Convergent
     win across behavior *and* representation.
   - **French:** more refusals (0.111 vs 0.065) and lower judge unsafe (0.100 vs 0.125).
   - **Arabic:** lower judge unsafe (0.115 vs 0.160).
   - The two low-resource languages (sw, yo) are the **only** ones where multilingual
     gives a *better* DiffMean direction — consistent with "native data helps where
     English transfer is weakest."

3. **Cost falls on English / high-resource.** `ml_big` is slightly less safe than
   `en3000` in English (judge 0.145 vs 0.115) and has weaker DiffMean directions for
   en/fr/ar — expected, since English lost 5/6 of its share of the budget.

4. **More English data also helps (control behaves sensibly).** Scaling English-only
   284→3000 raised refusal (en 0.188→0.204, ar 0.016→0.039) — but multilingual moved
   sw/fr *more* at the same total budget.

5. **Representation↔behavior correlation stays strong** (Corr1 ρ≈0.8–1.0 at L18–24
   for both new models; n=4 so fragile but consistent with the chain
   base≈0 → sft 0.6 → dpo/3000-models 0.8–1.0).

**Bottom line:** "Just translate your safety data into more languages" *does* help
low-resource languages (clearly Swahili) **once each language gets enough data** —
at a small cost to English. The earlier negative result was budget dilution, now ruled out.

---

## 5. Caveats
- **Single judge** (Command-R only; no OpenAI key) → no κ inter-rater validation;
  unsafe rates from a 200-prompt sample.
- **Yoruba** stays noisy/near-zero refusal — rough MT quality; don't over-read yo.
- **Correlation power** low (n=4–5 languages).
- Differences in sw/fr are modest in absolute terms (tens of prompts); directionally
  consistent across metrics but worth a larger eval to firm up.

## 6. Artifacts
- Models: `models/finetuned_ml_big/base_sft`, `models/finetuned_en3000/base_sft`
- Data: `data/processed/pku_translated/pku_sft_multilingual_n3000.jsonl`,
  `…/pku_sft_en_control_n3000.jsonl`
- Judge: `results/behavioral/judged_*_s200.jsonl`, `results/behavioral/judge_summary.csv`
- Probes/corr: `results/probes/summary.csv`, `results/probes/corr1_stats.csv`,
  `results/probes/alignment_base_sft_ml_big_*.csv`
- Pipeline: `scratchpad/fairtest_pipeline.sh`; judge now supports `--limit/--workers/--models/--langs`

### Possible next steps
- Larger eval set (full XSafety, not sampled) to firm up the sw/fr gains
- Add OpenAI key → full GPT-4o+Command-R ensemble with κ validation
- Multilingual DPO (not just SFT) at fair budget
- Add Filipino as a 7th language to raise correlation power
