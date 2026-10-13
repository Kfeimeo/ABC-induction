# ABC induction v0.2 — experiment report

Pipeline v0.1: raw corpus -> candidate spans -> multi-scale sparse contexts -> provisional classes (agglomerative, sparsity-corrected similarity) -> constituency score + MDL elimination with a predictive chart -> occurrence-level (legacy) ambiguity splits -> (E, R).

Pipeline v0.2 (on top of the v0.1 result): finite Bayesian latent grammar (sparse PCFG, Dirichlet posteriors) -> sparse Inside-Outside (semiring chart, numba) -> Dirichlet CAVI with a computable ELBO -> objective F = ELBO + log P(G) -> greedy class merge / class split / rule and lexicon proposals with checkpoint + rollback; lexical contexts as leaf potentials, punctuation as span potentials, adaptive constituent contexts as proposal features only. Every run reports both stages on the same gold (`v0.1` rows = v0.1 metrics, `v0.2` rows = metrics of the v0.2 posterior).

Metric columns: class pair F1 / ARI are computed over gold-constituent span types (type level), occurrence-level F1 over gold-constituent occurrences; constituent F1 is span-level over all gold spans (derivation level, i.e. the nodes of the induced grammar's Viterbi derivations); composition P/R/F1 compare R(A,B,C) after Hungarian class alignment; ambiguity F1 is over gold-ambiguous span types and occurrence accuracy is measured on their occurrences.

## Synthetic-1 (clean): v0.1 vs v0.2 ablations

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_v01 | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 14.380 |
| clean_v02_grammar_only | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 21.390 |
| clean_v02_ctx | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 22.000 |
| clean_v02_ms | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 197.070 |
| clean_v02_adaptive | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 207.730 |
| clean_v02_punct | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 194.750 |
| clean_v02_legacy_split | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 196.770 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_v02_grammar_only | 1.000 | 1.000 | 1.000 | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 15.696 | 9 | 0/0 | 8.810 |
| clean_v02_ctx | 1.000 | 1.000 | 1.000 | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 18.383 | 9 | 0/0 | 9.370 |
| clean_v02_ms | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 184.280 |
| clean_v02_adaptive | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/123 | 195.000 |
| clean_v02_punct | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/123 | 193.960 |
| clean_v02_legacy_split | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/123 | 195.910 |

- [clean_v01/report.md](outputs/v02/clean_v01/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.8 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_v02_grammar_only/report.md](outputs/v02/clean_v02_grammar_only/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 296.6 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_v02_ctx/report.md](outputs/v02/clean_v02_ctx/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 296.5 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_v02_ms/report.md](outputs/v02/clean_v02_ms/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 319.9 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_v02_adaptive/report.md](outputs/v02/clean_v02_adaptive/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 329.2 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/123 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_v02_punct/report.md](outputs/v02/clean_v02_punct/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 331.9 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/123 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_v02_legacy_split/report.md](outputs/v02/clean_v02_legacy_split/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 332.2 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/123 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP

## Synthetic-2 (ambiguity): v0.1 vs v0.2 ablations

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amb_v01 | 0.945 | 0.926 | 0.960 | 45 | 0.979 | 0.962 | 0.966 | 0.545 | 0.857 | 0.667 | 1.000 | 0.976 | 23.628 | 17.010 |
| amb_v02_grammar_only | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 21.900 |
| amb_v02_ctx | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 9.750 |
| amb_v02_ms | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 168.010 |
| amb_v02_adaptive | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 182.990 |
| amb_v02_legacy_split | 0.945 | 0.926 | 0.960 | 45 | 0.979 | 0.962 | 0.966 | 0.545 | 0.857 | 0.667 | 1.000 | 0.976 | 23.628 | 179.400 |
| amb_v02_grammar_only_legacy | 0.945 | 0.926 | 0.960 | 45 | 0.979 | 0.962 | 0.966 | 0.545 | 0.857 | 0.667 | 1.000 | 0.976 | 23.628 | 8.700 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amb_v02_grammar_only | 0.923 | 0.898 | 0.931 | 55 | 0.990 | 0.976 | 0.333 | 0.857 | 0.480 | 0.000 | 0.639 | 17.263 | 19 | 0/0 | 8.900 |
| amb_v02_ctx | 0.923 | 0.898 | 0.931 | 55 | 0.990 | 0.976 | 0.333 | 0.857 | 0.480 | 0.000 | 0.639 | 22.421 | 19 | 0/0 | 8.770 |
| amb_v02_ms | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 14/92 | 167.030 |
| amb_v02_adaptive | 0.993 | 0.991 | 0.911 | 18 | 0.995 | 0.988 | 0.778 | 1.000 | 0.875 | 0.615 | 0.948 | 19.455 | 9 | 17/126 | 182.050 |
| amb_v02_legacy_split | 0.990 | 0.987 | 0.886 | 18 | 0.997 | 0.993 | 0.778 | 1.000 | 0.875 | 0.500 | 0.948 | 18.927 | 9 | 14/124 | 178.580 |
| amb_v02_grammar_only_legacy | 0.943 | 0.924 | 0.997 | 22 | 1.000 | 1.000 | 0.600 | 0.857 | 0.706 | 1.000 | 1.000 | 16.592 | 19 | 0/0 | 7.520 |

- [amb_v01/report.md](outputs/v02/amb_v01/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 292.7 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []
- [amb_v02_grammar_only/report.md](outputs/v02/amb_v02_grammar_only/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 298.6 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed Nn + VP -> S; false ['unaligned', 'VP + NP -> VP', 'NP + Nn -> NP', 'VP + Adj -> VP']; v0.2 ambiguity: missed ['walks', 'duck', 'light', 'fast'], false [], occurrence accuracy 0.6394
- [amb_v02_ctx/report.md](outputs/v02/amb_v02_ctx/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 298.9 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed Nn + VP -> S; false ['unaligned', 'VP + NP -> VP', 'NP + Nn -> NP', 'VP + Adj -> VP']; v0.2 ambiguity: missed ['walks', 'duck', 'light', 'fast'], false [], occurrence accuracy 0.6394
- [amb_v02_ms/report.md](outputs/v02/amb_v02_ms/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 321.9 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 14/92 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331
- [amb_v02_adaptive/report.md](outputs/v02/amb_v02_adaptive/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 332.9 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 17/126 moves accepted (rule_prune(1), lex_add(walks), lex_add(light), merge(C4275 + C4276), rule_prune(5), lex_add(duck), lex_add(duck), rule_prune(3)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['unaligned', 'unaligned']; v0.2 ambiguity: missed [], false ['man', 'teacher', 'woman', 'horse', 'bird'], occurrence accuracy 0.948
- [amb_v02_legacy_split/report.md](outputs/v02/amb_v02_legacy_split/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 333.5 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []; v0.2: 14/124 moves accepted (merge(C4275 + C4276), rule_prune(10), lex_add(duck), lex_add(horse), lex_add(woman), lex_add(man), lex_add(teacher), lex_add(bird)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['unaligned', 'unaligned']; v0.2 ambiguity: missed [], false ['small', 'man', 'teacher', 'woman', 'horse', 'red', 'bird', 'happy'], occurrence accuracy 0.948
- [amb_v02_grammar_only_legacy/report.md](outputs/v02/amb_v02_grammar_only_legacy/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 299.2 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed Nn + VP -> S; false ['unaligned', 'unaligned', 'unaligned', 'unaligned']; v0.2 ambiguity: missed [], false [], occurrence accuracy 1.0

## Synthetic-3 (noisy): v0.1 vs v0.2 ablations

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noisy_v01 | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 14.310 |
| noisy_v02_grammar_only | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 23.270 |
| noisy_v02_ctx | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 15.660 |
| noisy_v02_ms | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 231.110 |
| noisy_v02_adaptive | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 292.940 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noisy_v02_grammar_only | 0.866 | 0.827 | 0.901 | 54 | 0.951 | 0.925 | 0.212 | 1.000 | 0.350 | 0.000 | - | 20.339 | 33 | 0/0 | 12.070 |
| noisy_v02_ctx | 0.866 | 0.826 | 0.901 | 55 | 0.951 | 0.924 | 0.212 | 1.000 | 0.350 | 0.000 | - | 30.323 | 33 | 0/0 | 14.790 |
| noisy_v02_ms | 0.906 | 0.878 | 0.982 | 81 | 0.952 | 0.927 | 0.179 | 1.000 | 0.304 | 0.000 | - | 26.474 | 41 | 25/108 | 230.110 |
| noisy_v02_adaptive | 0.824 | 0.777 | 0.949 | 132 | 0.931 | 0.878 | 0.163 | 1.000 | 0.280 | 0.000 | - | 26.262 | 45 | 28/144 | 291.960 |

- [noisy_v01/report.md](outputs/v02/noisy_v01/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 268.2 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn
- [noisy_v02_grammar_only/report.md](outputs/v02/noisy_v02_grammar_only/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 301.0 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Nn -> Nn', 'unaligned', 'Det + Det -> Det', 'Adv + NP -> NP']
- [noisy_v02_ctx/report.md](outputs/v02/noisy_v02_ctx/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 305.9 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 0/0 moves accepted (); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Nn -> Nn', 'unaligned', 'Det + Det -> Det', 'Adv + NP -> NP']
- [noisy_v02_ms/report.md](outputs/v02/noisy_v02_ms/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 336.3 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 25/108 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4063 + C4216.1 -> C4063)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'unaligned', 'S + Adv -> S', 'unaligned']
- [noisy_v02_adaptive/report.md](outputs/v02/noisy_v02_adaptive/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 347.1 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 28/144 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(12), rule_prune(1), lex_add(the), lex_add(big), rule_add(C4217 + C4063 -> C4217)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'S + Adv -> S', 'S + VP -> S', 'VP + VP -> VP']

## Seed stability of v0.2 (full) on Synthetic-1 / Synthetic-2 (v0.1 numbers of the same seeds in the same rows)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_v02_seed1 | 0.967 | 0.955 | 0.976 | 14 | 0.990 | 0.988 | 0.991 | 0.545 | 0.857 | 0.667 | 0.000 | - | 22.204 | 192.420 |
| clean_v02_seed2 | 0.927 | 0.902 | 0.985 | 15 | 0.994 | 0.990 | 0.992 | 0.545 | 0.857 | 0.667 | 0.000 | - | 22.056 | 184.120 |
| amb_v02_seed1 | 0.958 | 0.943 | 0.926 | 27 | 0.987 | 0.975 | 0.977 | 0.778 | 1.000 | 0.875 | 0.000 | 0.644 | 23.644 | 179.020 |
| amb_v02_seed2 | 0.936 | 0.913 | 0.865 | 16 | 0.983 | 0.970 | 0.975 | 0.500 | 0.714 | 0.588 | 0.000 | 0.592 | 24.507 | 150.320 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_v02_seed1 | 0.985 | 0.979 | 0.871 | 10 | 0.999 | 0.998 | 0.875 | 1.000 | 0.933 | 0.000 | - | 16.901 | 10 | 15/123 | 181.680 |
| clean_v02_seed2 | 0.996 | 0.994 | 0.873 | 10 | 1.000 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 16.966 | 9 | 15/123 | 172.460 |
| amb_v02_seed1 | 0.996 | 0.995 | 0.919 | 13 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.899 | 19.392 | 7 | 16/124 | 169.200 |
| amb_v02_seed2 | 0.966 | 0.953 | 0.828 | 30 | 0.992 | 0.982 | 0.368 | 1.000 | 0.538 | 0.250 | 0.625 | 19.797 | 19 | 11/90 | 138.730 |

- [clean_v02_seed1/report.md](outputs/v02/clean_v02_seed1/report.md); 3000 sentences, 2257 span types, 34555 contexts, 80828 edges, peak 331.9 MB, 2013 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; v0.2: 15/123 moves accepted (merge(C4263 + C4264), rule_prune(1), lex_add(teacher), lex_add(cat), lex_add(woman), lex_add(child), lex_add(man), lex_add(bird)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Adj + Adj -> Det']
- [clean_v02_seed2/report.md](outputs/v02/clean_v02_seed2/report.md); 3000 sentences, 2277 span types, 34509 contexts, 81089 edges, peak 333.3 MB, 2031 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; v0.2: 15/123 moves accepted (merge(C4292 + C4304), rule_prune(1), rule_prune(1), lex_add(man), lex_add(cat), lex_add(dog), lex_add(child), lex_add(woman)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['unaligned', 'unaligned']
- [amb_v02_seed1/report.md](outputs/v02/amb_v02_seed1/report.md); 3000 sentences, 2263 span types, 36834 contexts, 80843 edges, peak 332.8 MB, 1983 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; ambiguity: missed ['light', 'duck', 'walks', 'fast'], false []; v0.2: 16/124 moves accepted (lex_add(walks), lex_add(light), rule_prune(2), rule_prune(1), lex_add(duck), lex_add(light), rule_prune(3), lex_add(duck)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2 ambiguity: missed [], false ['man', 'woman'], occurrence accuracy 0.8986
- [amb_v02_seed2/report.md](outputs/v02/amb_v02_seed2/report.md); 3000 sentences, 2276 span types, 36820 contexts, 81318 edges, peak 334.3 MB, 1986 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, Nn + VP -> S; ambiguity: missed ['light', 'fast', 'walks', 'duck'], false []; v0.2: 11/90 moves accepted (merge(C4258 + C4260), merge(C4218 + C4251), split(C4218+C4251), split(C4246), rule_prune(21), lex_prune(), split(C4165), rule_prune(6)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'VP + NP -> VP', 'Adj + Nn -> NP', 'Nn + Nn -> Nn']; v0.2 ambiguity: missed ['light', 'walks', 'duck'], false ['cat', 'man', 'woman'], occurrence accuracy 0.6245

## Initialisation robustness: v0.1 warm start vs random initialisation (K = 8 / 12)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_v02_random_k8 | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 305.690 |
| clean_v02_random_k12 | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 415.790 |
| clean_v02_n10000 | 0.889 | 0.844 | 0.986 | 12 | 0.997 | 0.996 | 0.997 | 0.500 | 0.857 | 0.632 | 0.000 | - | 21.744 | 608.240 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_v02_random_k8 | 0.913 | 0.884 | 0.880 | 64 | 0.972 | 0.934 | 0.500 | 0.857 | 0.632 | 0.000 | - | 20.521 | 13 | 27/112 | 304.790 |
| clean_v02_random_k12 | 0.816 | 0.762 | 0.917 | 139 | 0.956 | 0.896 | 0.462 | 0.857 | 0.600 | 0.000 | - | 22.291 | 15 | 40/160 | 415.020 |
| clean_v02_n10000 | 0.972 | 0.958 | 0.855 | 96 | 0.990 | 0.977 | 0.625 | 0.714 | 0.667 | 0.000 | - | 16.010 | 9 | 19/137 | 574.490 |

- [clean_v02_random_k8/report.md](outputs/v02/clean_v02_random_k8/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 354.8 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 27/112 moves accepted (merge(C5 + C6), merge(C4 + C7), rule_prune(71), rule_prune(1), lex_add(cat), lex_add(dog), rule_add(C3 + C5+C6 -> C3), rule_add(C4+C7 + C3 -> C1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, Vt + NP -> VP, Vt + Nn -> VP; missed VP + Adv -> VP; false ['S + Adv -> S', 'Det + Adj -> Det', 'Det + Nn -> Nn', 'Adj + Nn -> NP']
- [clean_v02_random_k12/report.md](outputs/v02/clean_v02_random_k12/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 354.4 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 40/160 moves accepted (merge(C10 + C11), merge(C0 + C9), rule_prune(304), rule_prune(17), lex_add(the), lex_add(sees), rule_add(C2 + C1 -> C1), merge(C1 + C0+C9)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, Vt + NP -> VP, Vt + Nn -> VP; missed VP + Adv -> VP; false ['S + Adv -> S', 'Det + Adj -> Det', 'Nn + VP -> VP', 'VP + Nn -> VP']
- [clean_v02_n10000/report.md](outputs/v02/clean_v02_n10000/report.md); 10000 sentences, 5989 span types, 83459 contexts, 251726 edges, peak 782.2 MB, 5600 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; v0.2: 19/137 moves accepted (merge(C11581 + C11585), lex_add(child), lex_add(horse), split(C11588), rule_prune(5), lex_add(man), lex_add(bird), merge(C11579 + C11588.2)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP; missed Nn + VP -> S, Vt + Nn -> VP; false ['Det + Adj -> Det', 'unaligned', 'unaligned']

## Key failure cases (from the main runs)

# Natural-language study (TinyStories, raw text only)

No gold syntax exists for this corpus; the numbers below are intrinsic (stability, compression, held-out prediction, productivity, coverage) and are **not** syntax-recovery accuracies. v0.1 numbers come from the v0.1 stage of the same run, v0.2 numbers from the posterior of the v0.2 stage.

## 10000 sentences

- spans: 14754 types / 202971 occurrences; contexts 602037; edges 702499; merges 7488; accepted classes 1270; relations 132; splits 2; runtime 1601.37 s; peak memory 1991.2 MB
- MDL: 135.572 bits/sentence (grammar 47712.0 bits), fully derived sentences 0/2000 (MDL sample)
- productivity: 132 relations, lexical support mean 6.02 / median 4.0 / max 34; 18 relations with >= 10 distinct lexical realisations
- held-out context prediction (mean log2 P of k=1 context over 27118 held-out occurrences): class model -14.865, type model -14.623, background -14.423; rare types (freq<10, n=2887): class -13.802 vs type -12.729; contexts unseen for the type but seen for its class: 0.0595
- punctuation removed: 1217 classes, 118 relations, 123.339 bits/sentence

### v0.2 posterior (same corpus)

- coverage: fully parsed (single root) 0.0005, mean chunks/sentence 6.2897, unknown-token mass 19844
- complexity: 292 active classes, 367 active rules, 756 lexical entries, 110.852 bits/sentence (-ELBO/N); pruning {'cells_pruned': 34215, 'mean_retained_mass': 0.9997144157462409, 'max_dropped_mass': 0.057286329006235426, 'n_cells': 626971}
- structure moves: 21/67 accepted (merge C21610 + C21980, merge C21997 + C22174, split C14879, rule_prune , lex_add Lily, lex_add play, rule_add C22010 + C21285 -> C21285, rule_add C20901 + C20901 -> C20901); runtime {'init': 0.2, 'potentials': 48.3, 'cavi_init': 151.8, 'structure_learning': 1270.7, 'decoding': 9.3}
- stability across disjoint halves (v0.2): class pairwise F1 0.3271 / ARI 0.2577 on 1666 shared classified types; constituent-set agreement 0.6551; rule agreement (Jaccard after class alignment) 0.0389 (378 vs 365 rules); ambiguity agreement 0.1101
- occurrence ambiguity: 291 span types with a second class above the mass threshold; highest posterior entropy (not a linguistic claim): `juice` (15 occ, H=3.0983, {'C20901': 0.127, 'C21201+C22133': 0.11, 'C21610+C21980': 0.083, 'C22102': 0.071}); `walk` (35 occ, H=3.0314, {'C20901': 0.188, 'C21610+C21980': 0.102, 'C22236': 0.077, 'C21201+C22133': 0.067}); `disappear` (3 occ, H=3.0103, {'C21201+C22133': 0.131, 'C21982': 0.098, 'C20901': 0.096, 'C22218': 0.089}); `promise` (11 occ, H=2.9878, {'C20901': 0.198, 'C21610+C21980': 0.101, 'C22204': 0.097, 'C20172': 0.09}); `work` (26 occ, H=2.9153, {'C20901': 0.241, 'C22127': 0.146, 'C22218': 0.066, 'C20953': 0.052}); `stronger` (3 occ, H=2.891, {'C22034': 0.194, 'C20901': 0.147, 'C21264': 0.08, 'C19987': 0.074}); `kids` (43 occ, H=2.8884, {'C20901': 0.316, 'C22216': 0.068, 'C21958': 0.053, 'C22228': 0.05}); `won` (6 occ, H=2.8831, {'C20901': 0.257, 'C21264': 0.119, 'C21870': 0.09, 'C21970': 0.076})
- punctuation ablation (v0.2): as token: parsed 0.0005, 292 classes / 367 rules, 110.852 bits/sent; removed: parsed 0.0104, 292 classes / 346 rules, 108.629 bits/sent; structural prior: parsed 0.0, 293 classes / 254 rules, 149.115 bits/sent
- structural-prior stability across halves: class F1 0.3567 / ARI 0.2375; constituent agreement 0.7323; rule agreement 0.078

Top v0.2 classes (posterior view):

- C21834 (mass 10694.66): lexical ['. (9646.0)', '! (732.0)']; spans ['. (1.00 x9646)', '! (1.00 x732)', 'again . (1.00 x101)', 'again ! (1.00 x13)']; contexts ['it _ <EOS> (231.9)', 'said _ <EOS> (216.0)']; composition in ['C20970 + C21834 -> C21834 (165.1)', 'C19776 + C21834 -> C21834 (105.4)']
- C20901 (mass 7304.83): lexical ['<unk> (6459.1)', 'Before (3.0)']; spans ['old (0.55 x97)', 'we (0.60 x84)', 'until (0.63 x79)', 'if (0.75 x64)']; contexts ['<BOS> _ the (34.9)', '<BOS> _ that (29.0)']; composition in ['C20901 + C20901 -> C20901 (807.5)', 'C21102 + C19994 -> C20901 (35.3)']
- C22235 (mass 5933.86): lexical ['He (1353.0)', 'She (1323.0)', 'They (1214.0)', 'It (209.1)', 'Lily (147.2)', '<unk> (122.2)', 'Timmy (74.8)', 'John (51.0)', 'Everyone (44.0)', 'Jane (40.0)']; spans ['He (1.00 x1353)', 'She (1.00 x1323)', 'They (1.00 x1214)', 'Timmy (0.54 x139)']; contexts ['<BOS> _ was (657.3)', '<BOS> _ had (280.9)']; composition in ['C20914 + C22216 -> C22235 (565.4)', 'C22069 + C20738 -> C22235 (158.2)']
- C20593 (mass 4704.18): lexical ['and (4406.0)']; spans ['and (1.00 x4406)', ', and (1.00 x180)', 'that he (1.00 x70)', 'that she (1.00 x64)']; contexts ['mom _ dad (29.0)', ', _ the (24.0)']; composition in ['C18465 + C20738 -> C20593 (162.9)', 'C22137 + C20593 -> C20593 (135.3)']
- C20021 (mass 4575.9): lexical ['the (4057.0)']; spans ['the (1.00 x4057)', 'a big (1.00 x307)', 'the big (1.00 x38)', 'a loud (0.68 x22)']; contexts ['in _ park (42.0)', 'to _ park (34.0)']; composition in ['C21816 + C20153 -> C20021 (448.6)', 'C20021 + C20153 -> C20021 (70.3)']
- C22137 (mass 3265.0): lexical [', (3265.0)']; spans [', (1.00 x3265)']; contexts ['said _ " (189.0)', 'time _ there (125.0)']; composition in []
- C21760 (mass 3056.61): lexical ['was (2218.0)', 'is (383.0)', 'felt (242.0)', "wasn't (10.0)"]; spans ['was (1.00 x2218)', 'is (1.00 x383)', 'felt (1.00 x242)', 'was very (0.55 x253)']; contexts ['there _ a (200.0)', 'It _ a (79.0)']; composition in ['C21760 + C22027 -> C21760 (203.6)']
- C21285 (mass 2995.59): lexical ['to (2896.0)']; spans ['to (1.00 x2896)']; contexts ['went _ the (52.0)', 'ran _ the (40.6)']; composition in ['C22010 + C21285 -> C21285 (92.3)', 'C22204 + C20593 -> C21285 (7.3)']
- C21816 (mass 2635.26): lexical ['a (2565.0)']; spans ['a (1.00 x2565)', 'a brave (1.00 x9)', 'a scary (1.00 x4)', 'a sweet (1.00 x4)']; contexts ['upon _ time (199.0)', 'saw _ big (68.0)']; composition in ['C21816 + C22034 -> C21816 (53.6)', 'C21816 + C22212 -> C21816 (16.7)']
- C22148 (mass 2445.99): lexical ['Lily (393.8)', 'Tom (297.0)', 'Ben (284.0)', '<unk> (252.0)', 'Sam (150.0)', 'Tim (147.0)', 'Mom (135.0)', 'Anna (130.0)', 'Max (109.0)', 'Mia (88.0)']; spans ['Lily (0.73 x541)', 'Tom (1.00 x297)', 'Ben (1.00 x284)', 'Sam (1.00 x150)']; contexts ['<BOS> _ and (164.4)', '" _ said (135.2)']; composition in []
- C21928 (mass 2332.06): lexical ['her (1075.0)', 'his (681.0)', 'their (337.0)', 'its (68.0)', 'Rose (8.0)', '<unk> (3.9)']; spans ['her (1.00 x1075)', 'his (1.00 x681)', 'their (1.00 x337)', 'its (1.00 x68)']; contexts ['with _ friends (36.8)', 'asked _ mom (25.0)']; composition in ['C21928 + C21542 -> C21928 (74.3)', 'C21928 + C20960 -> C21928 (70.0)']
- C20738 (mass 1994.28): lexical ['he (797.0)', 'she (678.0)', 'they (503.0)', '<unk> (16.3)']; spans ['he (1.00 x797)', 'she (1.00 x678)', 'they (1.00 x503)']; contexts [', _ saw (69.7)', '" _ said (47.0)']; composition in []

Top inferred classes (human inspection):

- C21834 (9 members, 10498 occ): lexical ['.', '!']; multi-token ['again .', 'again !', 'voice .', 'at the park .', 'at the beach .', 'horn .']; contexts ['[k=1] it _ <EOS>', '[k=1] said _ <EOS>', '[k=1] again _ <EOS>']
- C20593 (8 members, 4656 occ): lexical ['and']; multi-token [', and', 'that he', 'that they', 'because they', 'because he', 'because she']; contexts ['[k=1] mom _ dad', '[k=1] , _ the', '[k=1] Anna _ Ben']
- C22235 (80 members, 4544 occ): lexical ['He', 'They', 'She', 'Sammy', 'Lisa', 'John', 'Sarah', 'Everyone', 'Ella', 'Pete', 'Molly', 'Billy']; multi-token ['The bird', 'He smiled and', 'The squirrel', 'The puppy', 'The little girl', 'He also']; contexts ['[k=1] <BOS> _ was', '[k=1] <BOS> _ said', '[k=1] <BOS> _ had']
- C20021 (4 members, 4409 occ): lexical ['the']; multi-token ['a big', 'the big', 'a tall']; contexts ['[k=1] in _ park', '[k=1] to _ park', '[k=1] on _ ground']
- C22137 (5 members, 3295 occ): lexical [',']; multi-token [', a little girl named', ', Lily and', ', a little boy named', 'to her ,']; contexts ['[k=1] said _ "', '[k=1] time _ there', '[k=2] a time _ there was']
- C21285 (7 members, 2924 occ): lexical ['to']; multi-token ['fun to', 'able to', 'to the park to', 'to go outside and', 'to be good and', 'his tail and']; contexts ['[k=1] went _ the', '[k=1] ran _ the', '[k=1] loved _ play']
- C21760 (7 members, 2872 occ): lexical ['was', 'is', 'felt', "wasn't"]; multi-token ['said they were', 'was only', 'were always']; contexts ['[k=1] there _ a', '[k=1] It _ a', '[k=1] He _ so']
- C21816 (7 members, 2581 occ): lexical ['a']; multi-token ['a sweet', 'a quiet', 'a brave', 'a warm', "the girl's", 'a cute']; contexts ['[k=1] upon _ time', '[k=2] Once upon _ time ,', '[k=3] <BOS> Once upon _ time , there']
- C22148 (31 members, 2441 occ): lexical ['Tim', 'Lily', 'Ben', 'Tom', 'Mom', 'Sam', 'Timmy', 'Lila', 'Max', 'Alice', 'Mia', 'Sara']; multi-token ['The turtle', 'the bunny', 'She looked around and', 'They turned around and', 'The crane', 'the baby']; contexts ['[k=1] <BOS> _ was', '[k=1] <BOS> _ and', '[k=1] " _ said']
- C21928 (16 members, 2208 occ): lexical ['their', 'her', 'its', 'his', 'Rose']; multi-token ['their parents', 'her own', 'her tight', 'her dad', 'his parents', 'her daddy']; contexts ['[k=1] with _ friends', '[k=1] asked _ mom', '[k=1] to _ mom']
- C20738 (3 members, 1978 occ): lexical ['he', 'they', 'she']; multi-token []; contexts ['[k=1] , _ saw', '[k=1] " _ said', '[k=1] , _ found']
- C22068 (11 members, 1879 occ): lexical ['"']; multi-token ['She says , "', 'He said , "', 'She said , "', 'He says , "', 'Tim said , "', 'Lily said , "']; contexts ['[k=1] <BOS> _ I', "[k=1] <BOS> _ Let's", '[k=1] , _ I']

Top relations A + B -> C:


Top ambiguity splits:

- `It` (489 occ): C21583 x217 ['[k=1] <BOS> _ was (x130)', '[k=2] <BOS> <BOS> _ was a (x63)'] vs C22235 x272 ['[k=1] <BOS> _ was (x62)', '[k=1] <BOS> _ had (x23)']; split gain 58.1 bits
- `grass` (30 occ): C19216 x22 ['[k=1] the _ . (x12)', '[k=2] on the _ . <EOS> (x11)'] vs C22195 x8 ['[k=1] the _ . (x7)', '[k=2] in the _ . <EOS> (x7)']; split gain 54.2 bits


## 研究問題的回答（v0.2 作者分析，依據上方自動產生的數字）

所有數字來自 `outputs/v02/summary.json`（seed 42、3000 句，除非另註明）與 `outputs/natural/natural_summary.json`。每個 run 同時報告 v0.1 stage（同一語料、同一 v0.1 pipeline）與 v0.2 posterior 的指標，所以表中 `v0.1` 行 = 該 run 的 v0.1 數字（legacy split 開啟的 `*_v01` run 才等於 v0.1 report 的主結果），`v0.2` 行 = 同一 gold 上的 v0.2 posterior。

### 與規格的一處已記錄偏離

生成參數是每個 class 的 **top-down expansion 分佈** `θ_C(e)`（Kurihara–Sato VB-PCFG），不是規格 §4/§5 的 bottom-up `θ_AB = P(C|A,B)`、`π_x = P(C|x)`。原因（`docs/v02_model.md` §1）：句子已觀測時，bottom-up labelling model 沒有 string likelihood——完整 rule inventory 下每棵樹的 labelling 總和恆為 1，不完整的 inventory 反而因「讓更多 labelling 合法」而得分，單一 class 的 `A+A→A` 文法最大化它；structure learning 就沒有訊號。v0.1 的 MDL data cost 本來就是 top-down（`N(C,e)·−log P(e|C)`），v0.2 是它的 Bayesian 版本。`P(C|A,B)` 與 `π_x` 由 expected counts 推導並照規格報告（`rule_conditional`、`type_mixture`，`Σ_C P(C|A,B)=1` 有測試）。其餘規格項目（finite K_max、sparse Inside–Outside、Dirichlet CAVI、ELBO + structure prior、greedy merge/split、ProbLog oracle、punctuation prior、adaptive constituent context 只作 proposal）均照做。

### RQ7（先答）：production inference 與 ProbLog / brute force 是否一致？

**一致。** `tests/test_chart_exact.py`：1 class / 2 classes / ambiguous lexical / multiple derivations / unknown token，含與不含 λ、ψ potentials，Python reference chart = brute-force enumerator（|Δ| < 1e-9：log Z、`P(Z_ij=C|s)`、`P(Q_ij=1|s)`、expected rule/lexical/root counts、Viterbi），numba backend = reference（1e-9），chunk layer = 分段枚舉。`tests/problog_reference/`（15 cases，含 shared-intermediate multi-hop）：ProbLog = chart = brute force（|ΔP| < 1e-8）。找到並修掉的真實 bug 全在 oracle 編碼側（root 必須是一個 annotated disjunction，否則不同 root 的 derivation 不互斥而少算；UNK disjunct 每 class 只能一個），production chart 沒有改過一行——這正是 oracle 的價值。ProbLog 不在 pipeline 任何路徑上（`problog_reference.enabled: false`，未安裝時測試 skip）。

### RQ1：hard/heuristic decisions → posterior marginal，class/constituency 是否更穩定？

**是，而且不靠任何 structure move。** Synthetic-1 `clean_v02_grammar_only`（只把 v0.1 文法當 warm start 跑 CAVI）：class F1 0.983 → **1.000**（8/8 class）、constituent F1 0.998 → 1.000、composition P 0.778 → **1.000**（R 1.0）。v0.1 的兩條 crutch relation `Det+Nn→Nn`（support 66）與 `Adj+Adj→Adj` 在 posterior expected count 下歸零（E[N] < 0.5，之後被 prune proposal 以 ΔF=+10.5 移除）；v0.1 的 Viterbi hard-EM 把它們鎖住，sum-product 沒有。seed 穩定性：v0.1 clean seed1/seed2 class F1 0.967/0.927、comp P 0.545/0.545 → v0.2 0.985/0.996、0.875/0.778；amb seed1/seed2 0.958/0.936 → 0.996/0.966。10k 句：v0.1 0.889（12 class，bare-S 類）→ v0.2 0.972（但 comp 0.625/0.714，見下）。

### RQ2：Inside–Outside 是否自然處理 structural ambiguity？

**是。** 不再有 heuristic confidence propagation：`structural_ambiguity.json` 的 hyperpaths 現在由 sum over derivations 吸收，constituency 是 chart marginal。證據：(i) constituent F1 在 clean/amb 達 1.000/0.995、noisy 0.902 → 0.951（+5 點，v0.1 的 Viterbi 分析在 filler 周圍常選錯括號）；(ii) 以 posterior > 0.5 解碼與以 MAP derivation 解碼的 constituent F1 在 clean 上相同（1.0），表示 posterior 很尖；(iii) 10k 句的 bare-S 類（`dog waits` 作 root）在 v0.2 一開始就沒有 extra root class（`extra_root_classes = []`）。

### RQ3：Dirichlet VB 是否能自然表示 lexical/category ambiguity？

**能表示，也能發現，但「發現」依賴 lexicon-add proposal，而 objective 也會接受 gold 不認的重新分析。**

- 表示：Synthetic-2 legacy split 關閉時，四個 planted 詞由 `lex_add` proposal 加入第二個 lexical entry（ΔF：walks +1623、light +1253、duck +881、fast +636 nats），posterior type mixture 變成 `walks = 0.55 Vt + 0.45 VP`、`fast = 0.62 Adj + 0.38 Adv`、`light = 0.65 Nn + 0.35 Adj`、`duck = 0.62 Nn + 0.21 VP + 0.17 NP`（gold 0.6/0.4、0.5/0.5、0.5/0.5、0.6/0.4）；occurrence 指派正確率 **0.948**（majority baseline 0.63；v0.1 legacy split 0.976）。這就是規格要的「ambiguity 由 q(Z_o=C) 自然表示」。
- 代價：同一個 objective 也以 ΔF ≈ +200 nats/詞把 bare noun（man、woman、horse…）加進 NP class——`Nn+VP→S` 與 `NP+VP→S` 在字串上 weakly equivalent，把 bare noun 當 NP 讓 S 的 expansion 分佈更尖。gold 把 bare noun 標為 Nn，所以 ambiguity type F1 只有 0.47–0.62（false: man/teacher/woman/horse/bird），clean 上也出現同樣的 6 個 false ambiguity。這是 model selection 的結果，不是 bug，我們**沒有**用加大 `lambda_lex` 把它壓掉（規格：不要盲目調參）；在自然語言裡 bare plural 本來就是 NP。
- `amb_v02_ms`（無 adaptive context）更保守：ambiguity F1 0.857、occ acc 0.933、0 false split，但 duck 沒切開（mixture Nn 1.0）。`amb_v02_grammar_only` / `ctx`（無 proposals）完全切不開（0.0 / 0.639）：**posterior 只能在 lexicon 已有兩個 entry 時表示 ambiguity，entry 本身必須由 proposal 提出**。legacy split 作初始化（`amb_v02_legacy_split`）結果與不用相近（occ acc 0.948），所以舊機制可以退場作 ablation。

### RQ4：class-level ELBO split/merge 是否減少 fragmentation / over-refinement？

**減少，且每一步都可審計。** 人工案例（`tests/test_cavi_structure.py`）：NP 被拆成 NP/NP2 → merge 接受（ΔF +38）；Adj+Adv 被合成一類 → split 接受（ΔF +468，兩半正好是 6 個形容詞 / 5 個副詞）；所有錯誤提案（split S 成 NP+VP / Nn+VP 兩半：ΔF −28；merge VP+Adv：−740；13 條 `rule_add`：各 −10）被拒；`monotonicity_violations = 0`。真實 run：Synthetic-2 的 duplicate S 類 `C4275+C4276` merge（+7.2）消掉 v0.1 的 bare-S 過細類；noisy 上 class F1 0.843 → **0.906**（`noisy_v02_ms`），fragmentation 清零；10k 句 0.889 → 0.972。沒做到的：noisy 的 composition precision 仍 0.18（filler 子串造成的 `S+Adv→S`、`Det+Adj→Det` 等 false relation，見失敗模式），而 10k 句少了 `Nn+VP→S`、`Vt+Nn→VP`（bare noun 被併入 NP 後這兩條被 prune）。

### RQ5：abstract constituent-context 作 proposal 是否緩解 sparsity 而不 self-reinforce？

**沒有可觀測的增益，也沒有 collapse。** `*_v02_ms`（無 adaptive）vs `*_v02_adaptive`：clean 完全相同（10 class、comp 1.0/1.0）；amb 0.995/0.5833 → 0.993/0.7778（comp P 升，但 ambiguity F1 0.857 → 0.615，因 adaptive 版接受了更多 bare-noun lex_add）；noisy 0.906 → 0.824（adaptive 版多接受一次 split，把 Nn 拆成兩個 fragment）。abstract feature 在 merge candidate 排名中確實出現（例如 VP vs Adv 的 raw similarity 0.96 但 abstract 0.58，所以該 merge 排到後面並被 ΔF 拒絕 −563），但決定權都在 ΔF；因此它既沒有 self-reinforcing（ELBO 不含它），也還沒證明有用。RQ5 的答案是「安全但目前多餘」。

### RQ6：punctuation structural prior 是否提高自然語料 stability / coverage？

見下方 TinyStories 段落（synthetic 文法沒有標點，`clean_v02_punct` 與 `clean_v02_adaptive` 結果相同，符合預期）。

### v0.1 已知 failure modes 逐項檢查（improved / unchanged / worse）

| failure mode | 結果 | 證據 |
|---|---|---|
| NP/Nn fragmentation | **improved** | clean 3k：v0.1 NP→2 fragments、Nn→2 → v0.2 每個 gold class 1 個 fragment（`diagnostics_v02.fragmentation`）；noisy：Nn 4 fragments → 1；10k：12 class → 8 主類 |
| bare-S over-refinement | **improved** | v0.1 10k 的 `Nn+VP` root class（1073 次）→ v0.2 `extra_root_classes = []`；Synthetic-2 的 duplicate S merge 接受 |
| Adj→Det contamination | **improved** | amb seed2：v0.1 Adj 併入 Det（fast 正確率 0.397）→ v0.2 所有 class purity ≥ 0.95 除一個 VP/Vt 混合類（0.863）；`fast = 0.60 Adj + 0.40 Adv` |
| private crutch rules | **improved on synthetic-1/2, visible elsewhere** | clean 3k 的 `Det+Nn→Nn` 歸零；private penalty 在 noisy/seed2 仍標出 self-paired rules（dominant share 0.63–1.0），這些 rule 都在 false relation 列表裡。已知瑕疵：報告的 run 對 `X+X→Y` 兩側各計一次（share 可到 2），已修正為每個 realisation 計一次，但 suite 未重跑 |
| filler-induced rules | **unchanged** | noisy：`uh/um/well` 仍自成一類（v0.1 併入 Adv），`filler_induced_rules` 4 條（filler share 0.51–0.71），comp P 0.27 → 0.18。posterior 不能解決「filler 是高頻規則子串」這個資料性質 |
| long-context sparsity | **unchanged（繞過）** | lexical context 只作 leaf potential；`grammar_only` 與 `ctx` 在三個語料上幾乎相同（clean 同、amb 同、noisy 0.866 vs 0.866），即 context 證據沒有超出 v0.1 已給的初始化；adaptive context 也沒補上（RQ5） |
| ambiguity pollution | **improved** | seed2 的鏈式失敗消失（purity 表）；但新的「bare noun → NP」false ambiguity 取而代之 |

### 新增的 failure modes（v0.2 特有）

1. **string-equivalent reanalysis**：objective 偏好把 bare noun 加進 NP（ΔF +200/詞），使 ambiguity precision 下降、`Nn+VP→S` 在 10k 被 prune。只有字串證據時這是 unidentifiable 的（v0.1 report 已指出）；v0.2 把它從「偶然」變成「系統性選擇」。
2. **random initialisation 不夠**：`random_k8/k12` class F1 0.91/0.82（64/139 class，大量 rule_prune 與 merge 之後仍碎），v0.1 warm start 是必要的（RQ1 的「posterior 更穩」是以 v0.1 初始化為前提）。
3. **adaptive split 在 noisy 上過度**：一次被接受的 Nn split（ΔF > 5）把 Nn 拆成兩個 fragment，class F1 0.906 → 0.824；`split_threshold` 5 nats 對 3000 句太低，但我們沒有為單一語料調它。
4. **效率**：structure learning 佔 v0.2 時間 80–90%（3k 句 160–300 s；10k 句 548 s；每個 proposal 要 5 次 E-step），`proposal_eval: affected` 在自然語料是必要的。

### 關鍵失敗案例（可在各 run 的 report.md「Structure moves」與「failure modes」段落逐條查到）

1. `amb_v02_adaptive`：`lex_add(man → NP)` ΔF +205.8（ELBO +207.8、prior −2.0），之後 `rule_prune` 把 `Nn+VP→S` 的 expected count 掃空。
2. `noisy_v02_ms`：filler 類 `C3138+C4171.2` 參與 4 條 filler-induced rule（`X + filler → X`），posterior 無法拒絕，因為它們確實壓縮語料。
3. `amb_v02_seed2`：Vt 碎成 2 個 class（0.80/0.20），merge candidate 被 ΔF 拒絕——posterior 認為它們的 composition-out 不同（其中一個是 `walks` 的 Vt sense 所在）。
4. `clean_v02_random_k12`：Det 碎成 2 類、`S+Adv→S` 等 false relation；merge/split 無法從隨機起點收斂到 8 類（40 個 accepted moves 後仍 139 類，多為 singleton 多詞 span）。

### 自然語言（TinyStories，10k 句，raw text only；RQ6 在此回答）

數字在上方「Natural-language study」段落與 `outputs/natural/`。v0.1 stage 完全相同（1270 accepted class、132 relation、135.6 bits/句），v0.2 由其 top-300 class（以 derivation usage 排序）warm start，chunk fallback 開啟、beam chart（`max_classes_per_cell 32`，mean retained mass 0.9997，max dropped 0.057）。總時間 1601 s（v0.2 部分 1480 s，其中 structure learning 1271 s）。

- **coverage**：只有 0.05% 的句子由單一 root 推導（MAP），posterior 平均 6.3 chunk/句（v0.1 Viterbi 5.6），unknown-token mass 19.8k/103k tokens（token 不在任何 class 的 lexicon 內，包含 v0.1 時 freq<3 的詞與 300 cap 之外的 class）。壓縮 110.9 bits/句（-ELBO/N）優於 v0.1 的 135.6 bits/句（MDL），但 fallback 的 chunk 序列本身便宜，兩者不可直接比。
- **complexity**：292 class、367 rule、756 lexical entry；21/67 個 structure move 被接受（2 merge、1 split、prune、`lex_add(Lily, play)`、`rule_add`）。
- **stability（兩個不相交 5k 子集）**：class pairwise F1 **0.327 / ARI 0.258**（v0.1 同一比較 0.224 / 0.222，1666 vs 2038 個共有分類 type），constituent-set agreement 0.655（v0.1 0.659），**rule agreement（class 對齊後 Jaccard）0.039**（378 vs 365 rule），ambiguity agreement 0.11。亦即 class 結構的可重現性略升，relation 幾乎完全不可重現——10k 句的自然語料下 (E,R) 中的 R 仍是 data-sparsity 的受害者（v0.1 RQ5 結論不變）。
- **punctuation prior（RQ6）**：三種模式——as token：parsed 0.05%、367 rule、110.9 bits；removed：parsed 1.0%、346 rule、108.6 bits；structural prior：parsed 0.0%、**254 rule**、149.1 bits。structural prior 把 rule 數壓低三成並提高子集 stability（class F1 0.357 / ARI 0.238、constituent agreement **0.732**、rule agreement 0.078，皆高於 as-token 的 0.327 / 0.258 / 0.655 / 0.039），但 bits/句變差 38 bits：ψ 是 virtual evidence，所有被懲罰的 span（跨逗號）付出 log-mass，所以目標值不可與其他模式比較；coverage 沒有提高。**結論：punctuation structural prior 提高 stability（尤其 constituent agreement），不提高 coverage。**
- **ambiguity**：291 個 type 的第二 class mass ≥ 0.2；最高 entropy 的 type（juice、walk、promise、work、kids…）全部是「主要 mass 在 `<unk>`-dominated class C20901」的低頻詞，這是 fallback class 的 artefact，不是語言學 ambiguity（規格要求不宣稱）。真正像 ambiguity 的例子在較高頻詞：`It`（v0.1 legacy split 也切過）在 v0.2 以 lexical entry 出現在主語代名詞類與 expletive 用法。
- **class 可讀性**：top class 與 v0.1 一致（句末標點、`and`、主語代名詞+人名、`the`、`,`、copula、`to`、`a`、人名、所有格、小寫代名詞），posterior 另外顯示 type mixture（`Lily` 0.73 在人名類、0.27 在 `He/She/They` 類；`was very` 0.55 在 copula 類）。
- **瓶頸**：(i) `<unk>` 類 C20901（mass 7.3k）吸收了所有 OOV 與被 cap 掉的詞並以 `C20901+C20901→C20901` 自我組合（807 次）——這是 v0.2 版的 filler 問題；(ii) 每個 proposal 的 affected-sentence E-step 在 10k 句仍需 10–30 s，4 輪 structure learning 已佔 21 分鐘；(iii) k_max=300 的 cap 是人為的（v0.1 有 1270 class），被 cap 的 class 的詞直接成為 `<unk>`。

### 下一步建議

1. **bare-noun / NP 的 identifiability**：加入「同一 token 在兩個 class 的 entry 必須有不同 composition 行為」的 prior（或直接把 `lambda_lex` 與 lexical reuse prior 綁定），否則 objective 會把每個名詞都做成 NP/Nn 雙 entry。
2. **`<unk>` 的分解**：以 character/morphology-free 的 frequency-tier 類（rare-word class per existing class，`C → <unk_C>`）取代單一 UNK expansion，避免 UNK 類自我組合成 filler。
3. **k_max 與 class cap**：改為把被 cap 的 v0.1 class 以 context similarity 併入最近的保留 class，而非丟棄。
4. **效率**：proposal 的 affected-sentence E-step 可以只在包含 moved class 的 *cell* 上重算（增量 inside-outside），預期 5–10×；expectation semiring 可省去 outside 的第二次掃描。
5. **adaptive context**：目前無益；若要保留，應只在 split proposal 缺 composition 證據時啟用，並以 ΔF 回饋選 feature。
6. **朝 CCG 走**：v0.2 的 posterior `P(C|A,B)` 與 rule expected counts 已是 CCG slash-category 因式分解需要的軟統計（modifier rule = 結果類等於一側；functor = 另一側），下一版可直接從 (E,R) 的 posterior 推 slash categories。
