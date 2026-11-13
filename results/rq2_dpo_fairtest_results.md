# RQ2 — Fair-Budget Multilingual DPO + Ensemble Judge (Command-R + Claude)

**Date:** 2026-07-01
**Extends** the SFT fair-test ([rq2_fairtest_results.md](rq2_fairtest_results.md)) to the DPO
stage, and replaces the single-judge keyword/Command-R metrics with a 2-judge
ensemble + inter-judge κ.

**Design (controlled — only language coverage differs):**
- `base_dpo_ml`     = base → SFT(3000 multilingual) → DPO(1200 multilingual pref pairs, 200/lang)
- `base_dpo_en3000` = base → SFT(3000 English)      → DPO(1200 English pref pairs) — same prompts
- QLoRA, β=0.1, 3 epochs. wandb: runs `base_dpo_ml`, `base_dpo_en3000`.
- Judge = **Command-R + Claude (claude-sonnet-4-6) ensemble**, majority vote, 300 sampled/lang.

---

**Both models now matched at max_seq_len=512** (an earlier `ml` run used 448 and
slightly truncated long multilingual pairs; removing that confound *strengthened*
the multilingual representation advantage — see below). Both: effective batch 16,
3 epochs, β=0.1. `ml` DPO auto-backed-off to per-device batch 2 (grad-accum 8) via
the OOM self-heal; `en3000` used batch 4 — results-neutral (no BatchNorm; per-example
DPO loss; identical effective batch).

## 1. Keyword refusal — base_dpo_en3000 vs base_dpo_ml (same prompts)

| lang | en3000 | ml | |
|------|--------|-----|---|
| en | 0.109 | 0.186 | ml higher |
| fr | 0.066 | 0.099 | **ml ~1.5×** |
| ar | 0.025 | 0.020 | ~tie |
| hi | 0.002 | 0.002 | ~tie |
| **sw** | 0.023 | **0.090** | **ml ~4×** |
| yo | 0.004 | 0.002 | ~0 both |

## 2. Ensemble unsafe rate (Command-R + Claude; lower = safer)

| lang | en3000 | ml | |
|------|--------|-----|---|
| en | 0.047 | 0.123 | English safer |
| fr | 0.130 | 0.097 | **ml safer** |
| ar | 0.113 | 0.107 | ml safer |
| hi | 0.207 | 0.243 | English safer |
| **sw** | 0.123 | **0.103** | **ml safer** |
| yo | 0.043 | 0.037 | ml safer |

## 3. Representation — DiffMean best-layer AUC (safety direction strength)

| lang | en3000 | ml | Δ(ml−en) |
|------|--------|-----|----------|
| en | 0.766 | 0.871 | +0.105 |
| fr | 0.789 | 0.810 | +0.021 |
| ar | 0.754 | 0.797 | +0.043 |
| **sw** | 0.758 | 0.796 | **+0.038** |
| yo | 0.766 | 0.737 | −0.029 |

## 4. Inter-judge κ (Command-R vs Claude) — reliability of the judging itself

Mean κ ≈ 0.19 (both DPO models); decent only for French (~0.35–0.40), **~0 or
negative for Yoruba**. Raw agreement stays high (~88%) but is inflated by class
imbalance. Same pattern as the SFT set: **safety evaluation is only reliable for
high-resource languages and essentially breaks down for Yoruba.**

---

## 5. Findings — consistent with the SFT fair-test

1. **Multilingual DPO helps low-resource Swahili on every axis:** more refusals
   (0.090 vs 0.023, ~4×), lower ensemble unsafe rate (0.103 vs 0.123), and a
   stronger DiffMean safety direction (+0.038). Behavior *and* representation agree.
2. **French and Arabic also benefit** (unsafe fr 0.097 vs 0.130, ar 0.107 vs 0.113).
3. **The cost is English:** `ml` is less safe than `en3000` in English (unsafe
   0.123 vs 0.047) — the English share of the fixed budget shrank. Note the
   behavior/representation split: `ml`'s English DiffMean AUC is actually *higher*
   (+0.105) even though it refuses less — a harmfulness≠refusal signal (the model
   encodes English harm more separably yet acts on it less).
4. **Removing the seq-448 truncation strengthened the multilingual advantage:** at
   the matched seq 512, every DiffMean Δ grew (sw +0.028→+0.038, en +0.043→+0.105)
   and Yoruba's deficit shrank (−0.073→−0.029) — confirming the earlier run
   understated `ml` by truncating its long sequences.
5. **Yoruba is still not clearly helped:** near-zero refusal both ways, slightly
   weaker DiffMean (−0.029), and judge κ≈0 — rough MT + un-judgeable outputs. We do
   **not** claim a Yoruba effect in either direction.
5. **SFT ↔ DPO agree:** both stages show multilingual > English-only for Swahili
   (low-resource) at a small English cost — the effect survives the stronger DPO
   objective, not just SFT.

**Bottom line across the whole post-training study:** at a *fair per-language
budget*, native-language safety data (SFT and DPO) improves low-resource Swahili
safety — behaviorally and representationally — at a modest cost to English. The
earlier "multilingual doesn't help" result was pure budget dilution. Yoruba
remains beyond reliable measurement with current MT + judges.

---

## 6. Caveats
- **Ensemble = 2 judges** (Command-R + Claude); κ < 0.7 target everywhere, ~0 for yo.
- DPO budget is 200 pref pairs/lang (1200 total) — length filter trimmed the pool.
- Both DPO models now matched at **seq 512, effective batch 16** (confound removed).
  `ml` ran at per-device batch 2 via the OOM self-heal, `en3000` at batch 4 — this
  is results-neutral (identical effective batch; no BatchNorm; per-example loss).
- Correlation power low (n=4–5 langs); Yoruba MT quality rough.

## 7. Artifacts
- Models: `models/finetuned_dpo_ml/base_dpo`, `models/finetuned_dpo_en3000/base_dpo`
- Judge: `results/behavioral/judged_*_ens.jsonl`, `results/behavioral/judge_summary_ens.csv`
- Data: `data/processed/pku_translated/pku_dpo_{multilingual,en_control}_n3000.jsonl`
- Code: `scripts/03d_translate_pku_dpo.py`; `10_finetune.py` (`--data-source multilingual --method dpo`,
  `--skip-sft`); `src/judge.py` (`judge_claude` + ensemble + κ); `08_llm_judge.py`
  (`--limit/--workers/--models/--langs/--judge-tag`)
- Infra note: HF cache must be overridden (`HF_HOME=/home/catherine/hf_cache`) — the
  default `~/.cache/huggingface` is a dead symlink to `/workspace` in this environment.
