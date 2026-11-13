# RQ2 Final Multi-Seed Results

## Design

The fair-budget multilingual and English-only controls were trained with seeds 42, 43, and 44.
Each condition used 3,000 SFT examples; DPO used 1,200 preference pairs. Behavioral refusal was
measured on matched harmful prompts, wrongful refusal on harmless prompts, and unsafe rate on a
deterministic sample of 300 responses per model/language/seed using Command-R + Claude. Probes used
matched activation prompts for en/fr/ar/sw/yo; Hindi was excluded because no comparable reference
activation set exists.

## Headline result: Swahili DPO transfer is robust

| Metric | Multilingual DPO | English-only DPO | Difference |
|---|---:|---:|---:|
| Harmful-prompt refusal | 13.9% ± 6.7% | 5.2% ± 3.0% | +8.7 pp |
| Ensemble unsafe rate | 9.4% ± 1.5% | 11.1% ± 2.1% | -1.7 pp |
| Layer-18 DiffMean AUC | 0.718 ± 0.068 | 0.643 ± 0.029 | +0.075 |
| Harmless-prompt over-refusal | 2.7% ± 1.3% | 1.2% ± 1.2% | +1.5 pp |

Multilingual DPO increased Swahili harmful-prompt refusal in every seed. Exact paired McNemar tests
were significant for all three seeds: p=4.34e-7 (42), p=3.80e-13 (43), and p=2.24e-4 (44). The
behavioral gain therefore survives seed replication. The over-refusal increase is real but small in
absolute terms; the result is better described as improved safety with a modest helpfulness cost,
not cost-free transfer.

## Cross-language pattern

### DPO

| Language | Refusal: ML | Refusal: EN | Unsafe: ML | Unsafe: EN |
|---|---:|---:|---:|---:|
| en | 24.4% ± 6.7% | 20.8% ± 8.7% | 8.2% ± 3.7% | 6.7% ± 2.9% |
| fr | 15.5% ± 6.5% | 10.6% ± 3.8% | 6.3% ± 2.9% | 11.2% ± 2.8% |
| ar | 2.2% ± 1.1% | 2.6% ± 0.3% | 7.3% ± 3.5% | 9.8% ± 1.7% |
| hi | 0.06% ± 0.10% | 0.06% ± 0.10% | 10.8% ± 11.9% | 13.6% ± 6.2% |
| sw | 13.9% ± 6.7% | 5.2% ± 3.0% | 9.4% ± 1.5% | 11.1% ± 2.1% |
| yo | 0.06% ± 0.10% | 0.3% ± 0.3% | 3.8% ± 0.2% | 4.8% ± 0.4% |

French also benefits clearly by judge unsafe rate and probe AUC. Arabic is safer by the judge and
more linearly separable despite no refusal-rate increase, reinforcing that harmfulness encoding and
refusal behavior are distinct. English retains a safety tax: multilingual DPO has a 1.6 pp higher
unsafe rate on average, although the refusal direction varies by seed. Yoruba has no credible refusal
gain.

### SFT

Multilingual SFT has a lower mean ensemble unsafe rate in every language, including Swahili
(19.7% ± 1.7% vs 22.2% ± 2.6%). However, its Swahili refusal advantage is smaller and inconsistent:
+1.7 ± 1.5 pp, with only seed 42 significant. The strongest repeatable SFT refusal result is French,
where multilingual exceeds English-only in all three seeds and each paired test is significant.

## Over-refusal

Wrongful refusal is low overall. Across all stage/arm/language aggregates, the largest mean is 3.7%
(multilingual DPO, French). Swahili is 2.7% ± 1.3% for multilingual DPO versus 1.2% ± 1.2% for the
English control. These rates rule out blanket refusal as the main explanation, but they do show a
small helpfulness tradeoff that should be reported.

## Representation results

At layer 18, multilingual DPO exceeds English-only DPO in DiffMean AUC for en (+0.086), fr (+0.089),
ar (+0.089), and sw (+0.075), but not yo (-0.012). Swahili's representational improvement therefore
replicates alongside its behavioral improvement. SFT is less consistent: multilingual SFT is higher
for ar and yo but lower for en, fr, and sw at this layer.

## Judge reliability and remaining blocker

Judge agreement remains the central limitation. Mean Command-R/Claude kappa is only ~0.16-0.17 for
Swahili DPO and approximately zero or negative for Yoruba. Consequently, the Swahili conclusion is
strongest where heuristic refusal, judge rate, and probe AUC agree; Yoruba judge rates should not be
treated as reliable evidence. The prepared 300-response human packet has zero labels completed, so
judge-vs-human validation remains outstanding.

## Conclusion

The original Swahili result is statistically robust across three seeds: multilingual DPO produces
more refusal on harmful prompts, lower judged unsafe rate, and stronger harmfulness separation than
the matched English-only control. The effect is not explained by blanket refusal, though it carries a
small over-refusal cost. Transfer is language-specific rather than universal: French benefits,
Arabic improves on judge/probe metrics without more refusal, English pays a modest safety tax, and
Yoruba does not show reliable improvement.

## Result files

- `results/behavioral/multiseed_summary.csv`
- `results/behavioral/multiseed_differences.csv`
- `results/behavioral/multiseed_judge_summary.csv`
- `results/behavioral/multiseed_overrefusal_summary.csv`
- `results/behavioral/mcnemar_significance.csv`
- `results/probes/multiseed_summary.csv`
