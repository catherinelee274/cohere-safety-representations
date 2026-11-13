# RQ2 — Multilingual Safety-FT & Representation↔Behavior Link (all 6 languages)

**Date:** 2026-06-24
**Scope:** Closes the three gaps in the post-training (RQ2) goal —
(1) low-resource behavioral refusal rates, (2) the multilingual Safety-FT arm,
(3) the representation↔refusal correlation across the training chain.
**Models:** base → base_sft (English PKU) → base_dpo (English PKU); plus
**base_sft_ml** (multilingual PKU, new). **Languages:** en, fr, ar, hi, sw, yo.
**wandb:** training is now wandb-enabled; base_sft_ml run backfilled →
`catherinelee274-stanford-university/cohere-safety-representations/runs/69fpqa5g`.

---

## 1. Behavioral refusal rates (heuristic keyword judge, XSafety eval split, n≈558/lang)

| model        | en    | fr    | ar    | hi    | sw    | yo    |
|--------------|-------|-------|-------|-------|-------|-------|
| base         | 0.152 | 0.048 | 0.018 | 0.000 | 0.007 | 0.004 |
| base_sft     | 0.188 | 0.059 | 0.016 | 0.029 | 0.050 | 0.004 |
| base_sft_ml  | 0.115 | 0.041 | 0.011 | 0.000 | 0.025 | 0.000 |
| base_dpo     | 0.246 | 0.231 | 0.016 | 0.000 | 0.061 | 0.000 |

(sw/yo were previously empty — XSafety has no low-resource coverage, so the eval
split was empty and the generator wrote 0-byte files. Fixed via
`04b_eval_splits.py`, which carves a held-out tail of the translated prompts.)

**Findings**
1. **Low-resource behavioral gap confirmed.** The base model essentially never
   refuses in sw/yo (0.4–0.7%) vs 15% in English — the gap the study targets.
2. **English-only SFT transfers cross-lingually, unevenly.** SFT lifts Swahili
   refusal 0.7%→5.0% with English-only data, but Yoruba stays flat (~0.4%).
3. **DPO sharpens refusal most where there is already signal** (en 24.6%,
   fr 23.1%, sw 6.1%) but leaves hi/yo at 0% — alignment doesn't manufacture a
   refusal behavior the base model never had in the lowest-resource languages.

---

## 2. Multilingual vs English-only SFT (budget-matched: 300 examples each)

base_sft_ml was trained on a **balanced 6-language PKU subset** (50 pairs/lang,
English kept verbatim, others Command-A translated) — the *same* 300-example
budget as the English base_sft, so the comparison isolates **language coverage**.

**Probe AUC at best layer (base_sft → base_sft_ml), same prompts:**

| lang | LogReg base_sft | LogReg base_sft_ml | Δ | DiffMean base_sft | DiffMean base_sft_ml | Δ |
|------|------|------|------|------|------|------|
| en | 0.970 | 0.973 | +0.003 | 0.771 | 0.755 | −0.016 |
| fr | 0.951 | 0.946 | −0.005 | 0.751 | 0.674 | −0.077 |
| ar | 0.944 | 0.952 | +0.008 | 0.759 | 0.738 | −0.021 |
| sw | 0.935 | 0.936 | +0.001 | 0.764 | 0.694 | −0.070 |
| yo | 0.913 | 0.925 | +0.012 | 0.760 | 0.718 | −0.042 |

**Finding (honest / nuanced — hypothesis NOT supported at this budget):**
At a matched 300-example budget, multilingual SFT **did not beat** English-only
SFT. Refusal rates are *lower* (Section 1) and the **DiffMean safety direction is
weaker across every language** (Δ −0.02 to −0.08). Interpretation: spreading the
budget across 6 languages (50 each) gives each language too few safety examples
to sharpen its mean safety direction, whereas concentrating 284 examples in
English produces a stronger, cross-lingually transferable direction — a
**budget-dilution effect**. The full-subspace LogReg probe is robust to this
(AUC within ±0.01), and the only positive movement is the lowest-resource
language **Yoruba (+0.012 LogReg AUC)**. A fair test of the multilingual
hypothesis needs a *larger* per-language budget, not a re-partitioned one.

---

## 3. Representation ↔ behavior correlation (Corr1, across the training chain)

Spearman ρ between cosine **distance** of each language's safety direction from
English and its **refusal-rate gap** from English (scope's Corr1 hypothesis:
farther from English ⇒ larger refusal gap ⇒ positive ρ).

| model     | L9 | L18 | L24 | L30 |
|-----------|----|-----|-----|-----|
| base      | −0.20 | −0.60 | −0.20 | 0.00 |
| base_sft  | 0.50 | 0.60 | 0.60 | 0.20 |
| base_dpo  | 0.72 | **0.82** | **0.82** | **0.87** |

**Key positive result:** the representation↔behavior link **emerges with
alignment training** — absent (even negative) in the base model, moderate after
SFT, and strong after DPO (ρ≈0.82–0.87, p≈0.05–0.09 at L18–30). This directly
supports the scope claim that *representation differences predict cross-lingual
safety gaps* — but only **after** alignment, not in the pretrained model.
(Corr2 — harmful↔refusal direction alignment vs refusal rate — remains noisy at
n=3–4 languages and is not reported as conclusive.)

---

## 4. Caveats
- **Refusal = keyword heuristic**, not the LLM-council judge (rates are
  approximate; absolute values low). LLM-judge pass (08) still recommended.
- **Small FT budget** (300 examples); conclusions about multilingual SFT are
  budget-specific.
- **Yoruba translation quality is rough** (low-resource MT); treat yo with caution.
- **sw/yo eval prompts** are a held-out tail of NLLB-translated prompts (no
  XSafety source exists); minor train/eval overlap is possible but does not
  affect probe CV.
- **Correlation power is low** (n=4–5 languages); p-values are indicative.

---

## 5. Artifacts & reproduction
- `scripts/04b_eval_splits.py` — held-out eval splits for translated-only langs
- `scripts/03c_translate_pku.py` — balanced multilingual PKU set (Command-A)
- `scripts/10_finetune.py --data-source multilingual --out-dir models/finetuned_ml`
  (now wandb-enabled: `FT_REPORT_TO=wandb`, seed fixed at 42)
- `scripts/05b_extract_reuse_prompts.py` — fair probe extraction on identical prompts
- `scripts/06b_regen_alignment.py` — alignment matrices incl. hi (canonical 6 langs)
- `scripts/14_correlation_analysis.py` — Corr1/Corr2
- Data: `results/behavioral/baseline_summary.csv`, `results/probes/summary.csv`,
  `results/probes/corr1_*.csv`, `results/probes/alignment_*_layer*.csv`

### Remaining (not blocking RQ2)
- [ ] LLM-judge (08) refusal/unsafe rates for sw/yo + base_sft_ml (replace heuristic)
- [ ] Larger-budget multilingual SFT to fairly test the multilingual hypothesis
- [ ] base_sft_ml hi probe (activations not extracted — base_sft has no hi prompt set)
