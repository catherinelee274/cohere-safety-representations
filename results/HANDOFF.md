# RQ2 Post-Training — Project Handoff

## 1. What this project is
Mechanistic-interpretability study of **safety representations across languages** in the
**TinyAya** multilingual model family (CohereLabs). This handoff covers **RQ2 (post-training /
safety-training evolution)**: how safety *behavior* and internal *representations* evolve across
**Base → SFT → DPO**, and whether **multilingual** safety fine-tuning helps **low-resource**
languages more than **English-only** FT.

- **Languages:** en, fr, ar, hi, sw, yo  (sw=Swahili, yo=Yoruba are low-resource). `zh` is legacy — ignore.
- **Base model:** `CohereLabs/tiny-aya-base` (~36-layer decoder). Adapters: QLoRA r16/α32, lr 2e-4, 3 epochs.
- **Metrics:** keyword refusal rate; LLM-judge unsafe rate; probe AUC (DiffMean + LogReg, layers 4/9/18/24/30/35);
  cross-lingual cosine alignment; Spearman correlations (rep-distance-to-English ↔ refusal gap).

## 2. Models trained (all from base + QLoRA)
| key | what | data |
|---|---|---|
| `base` / `global` / `earth`/`fire`/`water` | pretrained / instruction / regional variants | — |
| `base_sft` / `base_dpo` | English-only Safety-FT (original) | PKU-SafeRLHF (English) |
| `base_sft_ml` | **diluted** multilingual SFT | 50/lang × 6 = 300 |
| `base_sft_ml_big` | **fair-budget** multilingual SFT | 500/lang × 6 = 3000 |
| `base_sft_en3000` | English control (same 3000 prompts) | 3000 English |
| `base_dpo_ml` | fair-budget multilingual DPO | SFT 3000 + DPO 1200 pref pairs (200/lang) |
| `base_dpo_en3000` | English DPO control (same prompts) | SFT 3000 + DPO 1200 English |

Design principle: **budget-matched, same-prompt controls** so only *language coverage* differs.
Fair comparisons: `base_sft_ml_big` vs `base_sft_en3000`; `base_dpo_ml` vs `base_dpo_en3000`.

## 3. Key findings (see results/rq2_*.md)
1. **Low-resource behavioral gap confirmed:** base refuses ~0.4–0.7% in sw/yo vs ~15% in en.
2. **Diluted multilingual (50/lang) lost everywhere** — a *budget-dilution artifact*, not evidence against multilingual.
3. **Fair-budget SFT:** by the Command-R+Claude ensemble, `base_sft_ml_big` is **safer than `base_sft_en3000`
   in every language**; largest gains for low-resource sw and for fr.
4. **Fair-budget DPO (confound-free, seq 512):** multilingual helps **Swahili** on every axis —
   keyword refusal **0.090 vs 0.023 (~4×)**, ensemble unsafe **0.103 vs 0.123**, DiffMean AUC **+0.038**;
   fr/ar also safer; **English pays the tax** (unsafe 0.123 vs 0.047). Yoruba NOT helped.
5. **Representation↔behavior correlation emerges with alignment:** Spearman ρ ≈ 0 (base) → 0.5–0.6 (SFT)
   → 0.82–0.87 (DPO) between cosine-distance-to-English and refusal-rate gap.
6. **Inter-judge κ (Command-R vs Claude) is low:** ~0.5 for en/fr/ar, **~0 for Yoruba** → *safety
   evaluation itself degrades for low-resource languages*, a finding in its own right. κ never hits the 0.7 target.
7. **harmfulness ≠ refusal signal:** `base_dpo_ml`'s English DiffMean AUC is *higher* (+0.105) even though it
   refuses *less* in English — encodes harm more separably yet acts on it less.
8. **Three-seed replication confirms the Swahili DPO result:** multilingual vs English-only gives harmful
   refusal **13.9% ± 6.7% vs 5.2% ± 3.0%**, ensemble unsafe **9.4% ± 1.5% vs 11.1% ± 2.1%**, and layer-18
   DiffMean AUC **0.718 ± 0.068 vs 0.643 ± 0.029**. Swahili McNemar p<0.001 in every seed.
9. **Not blanket refusal, but not free:** Swahili DPO over-refusal is **2.7% ± 1.3% vs 1.2% ± 1.2%**.
   This is a modest helpfulness cost, far smaller than the harmful-prompt refusal gain.

## 4. Infrastructure notes / gotchas (IMPORTANT for re-running)
- **HF cache is broken:** `~/.cache/huggingface` is a dead symlink → `/workspace/huggingface` (from `.bashrc`),
  and `/workspace` doesn't exist in this container. Model loads/FT fail with `PermissionError` unless you
  override: `export HF_HOME=/home/catherine/hf_cache` (+ `HF_HUB_CACHE`, `HF_DATASETS_CACHE`). Do NOT edit `.bashrc`.
- **API keys in `.env`:** `HF_TOKEN`, `CO_API_KEY` (Cohere/Command-A), `WANDB_API_KEY`, `ANTHROPIC_API_KEY`.
  **No OpenAI key** (GPT-4o judge unavailable unless one is added).
- **GPU:** single RTX 3090 (24 GB). Multilingual DPO OOMs at batch 4 (long non-Latin sequences) —
  **`run_dpo` now self-heals** (catches OOM, halves batch + doubles grad-accum, retries to batch 1;
  sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`). No manual `--batch-size` needed.
- **wandb:** set `FT_REPORT_TO=wandb` to log training; runs under project `cohere-safety-representations`,
  entity `catherinelee274-stanford-university`. Seed fixed via `FT_SEED` (default 42).
- **Judge:** `src/judge.py::judge_ensemble` runs whichever judges have keys (GPT-4o / Command-R / Claude);
  currently Command-R + Claude (`claude-sonnet-4-6`). `08_llm_judge.py` supports
  `--limit/--workers/--models/--langs/--judge-tag`; ensemble outputs → `judge_summary_ens.csv`.
- **Probe-summary gotcha:** `06_train_probes.py` drops a model's rows per invocation, so running it per-lang
  leaves only the LAST lang in `summary.csv`. Run all langs in one process OR repopulate from the pkls after.

## 5. Scripts (added / modified this phase)
- `scripts/04b_eval_splits.py` — held-out `harmful_eval.jsonl` for translated-only langs (sw/yo).
- `scripts/03c_translate_pku.py` — multilingual SFT set via Command-A (`--tag`, `--emit-en-control`).
- `scripts/03d_translate_pku_dpo.py` — multilingual DPO preference pairs.
- `scripts/05b_extract_reuse_prompts.py` — extract activations on ANOTHER model's saved prompts (fair probes).
- `scripts/06b_regen_alignment.py` — recompute alignment matrices from pkls (CPU; canonical 6 langs incl hi).
- `scripts/backfill_wandb.py` — replay a captured training log into wandb.
- `scripts/10_finetune.py` — `--data-source multilingual` (SFT+DPO), `--skip-sft`, env-driven wandb+seed, DPO OOM self-heal.
- `scripts/08_llm_judge.py` — sampling/filtering/concurrency + inter-judge κ + tagged outputs.
- `scripts/07b_overrefusal_eval.py` — harmless-prompt wrongful-refusal evaluation with isolated outputs.
- `scripts/24_mcnemar_significance.py` — exact paired McNemar tests for all three seeds.
- `scripts/25_run_multiseed.py` — reproducible seed 43/44 fair-budget SFT+DPO orchestration.
- `scripts/27_evaluate_multiseed.py` — seeded behavioral, matched activation, and probe evaluation.
- `scripts/28_aggregate_multiseed.py` / `30_aggregate_multiseed_judge.py` — mean ± std aggregation.
- `src/judge.py` — `judge_claude` + generalized ensemble + κ.
- `config/models.py` — registered `base_sft_ml`, `base_sft_ml_big`, `base_sft_en3000`, `base_dpo_ml`, `base_dpo_en3000`.
- **Result docs:** `results/rq2_sft_results.md`, `rq2_dpo_results.md`, `rq2_multilingual_results.md`,
  `rq2_fairtest_results.md`, `rq2_dpo_fairtest_results.md`. Data: `results/behavioral/baseline_summary.csv`,
  `results/behavioral/judge_summary_ens.csv`, `results/probes/summary.csv`, `results/probes/corr1_stats.csv`.
- **Nothing is committed to git yet.**

## 6. Current limitations
- **Judge unreliable for low-resource** (κ≈0 for yo) — the languages the claims hinge on are least trustworthy.
- **Low statistical power** — unsafe rates on 300 sampled prompts; correlations at n=4–5 langs (spurious ρ=1.0).
- **Yoruba MT quality is rough** — contaminates both its training data and its eval.
- **Small model** (TinyAya) — effects inherently noisy; external validity limited.
- **Human validation is still missing:** `data/judge_gold_rq2_300.jsonl` has 50 examples/lang but no completed labels.
- **Hindi seeded probes are absent:** no matched 860-prompt reference activation set was available.

## 7. What to do next — to get STRONGER results (priority order)
**Tier 1 status:**
1. ✅ **Multiple seeds (×3)** with mean ± std.
2. ✅ **Over-refusal control** on benign prompts for every fair model/seed/language.
3. ✅ **Exact paired McNemar tests** per seed and language.
4. ⬜ **Validate judges vs human gold** — label `data/judge_gold_rq2_300.jsonl`, then report judge-vs-gold accuracy.

**Tier 2 (broaden evidence):**
5. Add **GPT-4o** as a 3rd judge (needs OpenAI key) for a real council + all-pairs κ.
6. Add **Filipino** (data exists) → n=6–7 for correlations; report **bootstrap CIs**.
7. **Fix or drop Yoruba** — native-reviewed MT, or swap for a better-resourced low-resource language.
8. Judge the **full 558/lang** (not 300 sampled) to tighten every rate.

**Tier 3 (depth / reconnect to core hypothesis):**
9. **Anchor against `global` + regional (`earth/fire/water`)** — the original RQ was "regional > global safety reps."
10. **SAE analysis (RQ3)** — which shared features carry cross-lingual safety transfer; steering experiments.
11. **Scale check** — replicate the Swahili result on one larger Aya model for external validity.

**Current final write-up:** `results/rq2_multiseed_final.md`.
