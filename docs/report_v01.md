# ABC induction v0.1 — experiment report

Pipeline: raw corpus -> candidate spans -> multi-scale sparse contexts -> provisional classes (agglomerative, sparsity-corrected similarity) -> constituency score + MDL elimination with a predictive chart -> occurrence-level ambiguity splits -> (E, R).

Metric columns: class pair F1 / ARI are computed over gold-constituent span types (type level), occurrence-level F1 over gold-constituent occurrences; constituent F1 is span-level over all gold spans (derivation level, i.e. the nodes of the induced grammar's Viterbi derivations); composition P/R/F1 compare R(A,B,C) after Hungarian class alignment; ambiguity F1 is over gold-ambiguous span types and occurrence accuracy is measured on their occurrences.

## Main runs (full pipeline, ablation E)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic1_clean | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 10.170 |
| synthetic2_ambiguity | 0.945 | 0.926 | 0.960 | 45 | 0.979 | 0.962 | 0.966 | 0.545 | 0.857 | 0.667 | 1.000 | 0.976 | 23.628 | 12.650 |
| synthetic3_noisy | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 12.300 |

- [synthetic1_clean/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/synthetic1_clean/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.3 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [synthetic2_ambiguity/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/synthetic2_ambiguity/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 291.9 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []
- [synthetic3_noisy/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/synthetic3_noisy/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 267.8 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn

## Ablations A-E on Synthetic-1 (clean)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_A_context_only | 0.740 | 0.673 | 0.932 | 70 | 0.769 | 0.525 | 0.434 | 0.019 | 1.000 | 0.037 | 0.000 | - | - | 3.320 |
| clean_B_composition_only | 0.740 | 0.673 | 0.932 | 70 | 0.767 | 0.524 | 0.431 | 0.019 | 1.000 | 0.037 | 0.000 | - | - | 3.340 |
| clean_C_context_composition | 0.740 | 0.673 | 0.932 | 70 | 0.767 | 0.524 | 0.431 | 0.019 | 1.000 | 0.037 | 0.000 | - | - | 3.420 |
| clean_D_plus_complexity | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 8.190 |
| clean_E_plus_ambiguity | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 9.960 |

- [clean_A_context_only/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_A_context_only/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.3 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_B_composition_only/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_B_composition_only/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.5 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_C_context_composition/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_C_context_composition/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 283.0 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_D_plus_complexity/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_D_plus_complexity/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 281.6 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_E_plus_ambiguity/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_E_plus_ambiguity/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.8 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP

## Ablations D vs E on Synthetic-2 (ambiguity)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amb_D_no_splitting | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 10.210 |
| amb_E_full | 0.945 | 0.926 | 0.960 | 45 | 0.979 | 0.962 | 0.966 | 0.545 | 0.857 | 0.667 | 1.000 | 0.976 | 23.628 | 13.270 |
| amb_E_kmeans_proposal | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 17.020 |
| amb_E_no_mdl_criterion | 0.941 | 0.922 | 0.898 | 47 | 0.979 | 0.962 | 0.966 | 0.667 | 0.857 | 0.750 | 0.333 | 0.976 | 25.634 | 13.680 |

- [amb_D_no_splitting/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_D_no_splitting/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 292.4 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []
- [amb_E_full/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_E_full/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 291.9 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []
- [amb_E_kmeans_proposal/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_E_kmeans_proposal/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 292.7 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []
- [amb_E_no_mdl_criterion/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_E_no_mdl_criterion/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 292.0 MB, 1995 merges, 20 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false ['small', 'quickly', 'teacher', 'woman', 'young', 'horse', 'red', 'waits', 'quietly', 'bird', 'happy', 'big', 'old', 'slowly', 'today', 'often']

## Ablations A-E on Synthetic-3 (noisy)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noisy_A_context_only | 0.520 | 0.447 | 0.821 | 131 | 0.744 | 0.481 | 0.422 | 0.018 | 0.857 | 0.036 | 0.000 | - | - | 3.100 |
| noisy_C_context_composition | 0.520 | 0.447 | 0.821 | 131 | 0.741 | 0.477 | 0.414 | 0.018 | 0.857 | 0.036 | 0.000 | - | - | 2.950 |
| noisy_D_plus_complexity | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 9.010 |
| noisy_E_plus_ambiguity | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 12.410 |

- [noisy_A_context_only/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/noisy_A_context_only/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 268.4 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn
- [noisy_C_context_composition/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/noisy_C_context_composition/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 266.6 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn
- [noisy_D_plus_complexity/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/noisy_D_plus_complexity/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 267.4 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn
- [noisy_E_plus_ambiguity/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/noisy_E_plus_ambiguity/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 266.9 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn

## Context range k on Synthetic-1 and Synthetic-2 (full pipeline)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_k1 | 0.882 | 0.844 | 0.886 | 52 | 0.944 | 0.877 | 0.879 | 0.500 | 0.714 | 0.588 | 0.000 | - | 23.087 | 8.750 |
| clean_k12 | 0.937 | 0.916 | 0.971 | 13 | 0.990 | 0.990 | 0.995 | 0.667 | 0.857 | 0.750 | 0.000 | - | 22.267 | 10.360 |
| clean_k1234 | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 9.800 |
| amb_k1 | 0.894 | 0.859 | 0.855 | 122 | 0.930 | 0.850 | 0.855 | 0.364 | 0.571 | 0.444 | 0.857 | 0.819 | 24.496 | 10.040 |
| amb_k12 | 0.938 | 0.917 | 0.959 | 39 | 0.980 | 0.964 | 0.967 | 0.545 | 0.857 | 0.667 | 1.000 | 0.975 | 23.633 | 11.300 |
| amb_k1234 | 0.945 | 0.926 | 0.960 | 45 | 0.979 | 0.962 | 0.966 | 0.545 | 0.857 | 0.667 | 1.000 | 0.976 | 23.628 | 12.510 |
| noisy_k1 | 0.869 | 0.831 | 0.885 | 75 | 0.901 | 0.814 | 0.801 | 0.227 | 0.714 | 0.345 | 0.000 | - | 26.879 | 10.690 |
| noisy_k1234 | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 11.600 |

- [clean_k1/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_k1/report.md); 3000 sentences, 2242 span types, 1047 contexts, 14276 edges, peak 246.6 MB, 1841 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S, VP + Adv -> VP
- [clean_k12/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_k12/report.md); 3000 sentences, 2242 span types, 8945 contexts, 35312 edges, peak 273.6 MB, 2029 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [clean_k1234/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_k1234/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 281.0 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [amb_k1/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_k1/report.md); 3000 sentences, 2285 span types, 1276 contexts, 14875 edges, peak 257.3 MB, 1792 merges, 3 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, Nn + VP -> S, VP + Adv -> VP; ambiguity: missed ['light'], false []
- [amb_k12/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_k12/report.md); 3000 sentences, 2285 span types, 9919 contexts, 35978 edges, peak 283.8 MB, 2011 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []
- [amb_k1234/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_k1234/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 292.7 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []
- [noisy_k1/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/noisy_k1/report.md); 3000 sentences, 2314 span types, 2501 contexts, 15916 edges, peak 235.1 MB, 1749 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S, VP + Adv -> VP
- [noisy_k1234/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/noisy_k1234/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 267.9 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn

## Similarity baselines on Synthetic-1 (A: weighted Jaccard, B: Jensen-Shannon, Bhattacharyya; with/without sparsity correction)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_sim_jaccard | 0.551 | 0.472 | 0.746 | 267 | 0.859 | 0.675 | 0.685 | 0.071 | 0.286 | 0.114 | 0.000 | - | 23.091 | 40.080 |
| clean_sim_js | 0.844 | 0.797 | 0.751 | 277 | 0.822 | 0.598 | 0.601 | 0.333 | 0.571 | 0.421 | 0.000 | - | 22.211 | 49.570 |
| clean_sim_bc_uncorrected | 0.886 | 0.850 | 0.834 | 238 | 0.908 | 0.790 | 0.841 | 0.800 | 0.571 | 0.667 | 0.000 | - | 26.133 | 11.010 |

- [clean_sim_jaccard/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_sim_jaccard/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 265.7 MB, 1721 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S; missed: Adj + Nn -> Nn, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_sim_js/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_sim_js/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 269.6 MB, 1930 merges, 0 splits; relations recovered: NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, Det + Nn -> NP, Nn + VP -> S
- [clean_sim_bc_uncorrected/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_sim_bc_uncorrected/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 267.1 MB, 1326 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP; missed: Adj + Nn -> Nn, Vt + NP -> VP, Vt + Nn -> VP

## Sample complexity on Synthetic-1

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_n1000 | 0.895 | 0.860 | 0.871 | 29 | 0.953 | 0.903 | 0.910 | 0.333 | 0.714 | 0.455 | 0.000 | - | 22.815 | 3.340 |
| clean_n3000 | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 9.210 |
| clean_n10000 | 0.889 | 0.844 | 0.986 | 12 | 0.997 | 0.996 | 0.997 | 0.500 | 0.857 | 0.632 | 0.000 | - | 21.744 | 35.760 |

- [clean_n1000/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_n1000/report.md); 1000 sentences, 862 span types, 13663 contexts, 26194 edges, peak 179.2 MB, 722 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, Nn + VP -> S
- [clean_n3000/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_n3000/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.7 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_n10000/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_n10000/report.md); 10000 sentences, 5989 span types, 83459 contexts, 251726 edges, peak 781.8 MB, 5600 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S

## Strength of the complexity prior (MDL class/rule costs) on Synthetic-1 (3k / 10k) and Synthetic-2

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_prior_c12 | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 9.240 |
| clean_prior_c40 | 0.956 | 0.940 | 0.976 | 13 | 0.991 | 0.987 | 0.990 | 0.545 | 0.857 | 0.667 | 0.000 | - | 22.396 | 9.220 |
| clean_prior_c100 | 0.956 | 0.940 | 0.966 | 13 | 0.988 | 0.988 | 0.994 | 0.750 | 0.857 | 0.800 | 0.000 | - | 22.766 | 9.080 |
| clean10k_prior_c12 | 0.889 | 0.844 | 0.986 | 12 | 0.997 | 0.996 | 0.997 | 0.500 | 0.857 | 0.632 | 0.000 | - | 21.744 | 35.400 |
| clean10k_prior_c100 | 0.889 | 0.844 | 0.983 | 11 | 0.996 | 0.996 | 0.999 | 0.750 | 0.857 | 0.800 | 0.000 | - | 21.897 | 33.600 |
| amb_prior_c100 | 0.933 | 0.910 | 0.919 | 45 | 0.952 | 0.896 | 0.898 | 0.500 | 0.857 | 0.632 | 1.000 | 0.978 | 24.056 | 10.320 |

- [clean_prior_c12/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_prior_c12/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 282.5 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [clean_prior_c40/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_prior_c40/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 283.0 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [clean_prior_c100/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_prior_c100/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 283.5 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [clean10k_prior_c12/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean10k_prior_c12/report.md); 10000 sentences, 5989 span types, 83459 contexts, 251726 edges, peak 781.1 MB, 5600 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [clean10k_prior_c100/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean10k_prior_c100/report.md); 10000 sentences, 5989 span types, 83459 contexts, 251726 edges, peak 782.5 MB, 5600 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [amb_prior_c100/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_prior_c100/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 293.1 MB, 1995 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed [], false []

## Seed stability (Synthetic-1 and Synthetic-2)

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_seed1 | 0.967 | 0.955 | 0.976 | 14 | 0.990 | 0.988 | 0.991 | 0.545 | 0.857 | 0.667 | 0.000 | - | 22.204 | 10.560 |
| clean_seed2 | 0.927 | 0.902 | 0.985 | 15 | 0.994 | 0.990 | 0.992 | 0.545 | 0.857 | 0.667 | 0.000 | - | 22.056 | 10.700 |
| amb_seed1 | 0.959 | 0.944 | 0.975 | 31 | 0.987 | 0.975 | 0.977 | 0.875 | 1.000 | 0.933 | 1.000 | 0.979 | 23.620 | 9.670 |
| amb_seed2 | 0.936 | 0.914 | 0.899 | 20 | 0.983 | 0.970 | 0.975 | 0.625 | 0.714 | 0.667 | 1.000 | 0.788 | 24.739 | 9.140 |

- [clean_seed1/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_seed1/report.md); 3000 sentences, 2257 span types, 34555 contexts, 80828 edges, peak 281.9 MB, 2013 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [clean_seed2/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/clean_seed2/report.md); 3000 sentences, 2277 span types, 34509 contexts, 81089 edges, peak 299.8 MB, 2031 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S
- [amb_seed1/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_seed1/report.md); 3000 sentences, 2263 span types, 36834 contexts, 80843 edges, peak 297.3 MB, 1983 merges, 4 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; ambiguity: missed [], false []
- [amb_seed2/report.md](E:/dev/AI/NLP/CCG induction/ABC induction/outputs/amb_seed2/report.md); 3000 sentences, 2276 span types, 36820 contexts, 81318 edges, peak 295.4 MB, 1986 merges, 4 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, Nn + VP -> S; ambiguity: missed [], false []

## Key failure cases (from the main runs)

### synthetic1_clean

- false merges: 30 — e.g. NP['the bird often quickly', 'Tom quietly'] + Nn['happy small child', 'a old teacher'] (score 0.811, why: ['[k=1] the _ <EOS>', '[k=1] <BOS> _ knows']); Nn['young young bird', 'some young horse'] + NP['the old dog', 'a small cat'] (score 1.0, why: ['[k=1] <BOS> _ laughs', '[k=2] <BOS> <BOS> _ laughs <EOS>']); Nn['bird often quickly', 'red young cat'] + NP['the old bird', 'the young man'] (score 0.903, why: ['[k=1] the _ <EOS>', '[k=1] <BOS> _ knows'])
- fragmented gold classes: NP -> 4226(157), 4223(13); Nn -> 4223(117), 4226(2)
- false constituent types: 77 — e.g. `the old` (freq 63, in class ~Det), `child waits` (freq 26, in class ~S), `woman laughs` (freq 25, in class ~S), `teacher laughs` (freq 22, in class ~S), `man sleeps` (freq 21, in class ~S)
- missed constituent types: 0
- false relations: Det + Nn -> Nn (support 66, lexical 13); Adj + Adj -> Adj (support 2, lexical 2)

### synthetic2_ambiguity

- false merges: 33 — e.g. NP['every happy bird', 'the red cat'] + Nn['man quickly', 'fast teacher'] (score 0.972, why: ['[k=1] <BOS> _ jumps', '[k=2] <BOS> <BOS> _ jumps <EOS>']); NP['some light teacher', 'every happy man'] + Nn['light horse', 'old cat'] (score 0.737, why: ['[k=1] a _ <EOS>', '[k=1] knows _ <EOS>']); NP['the light small dog', 'every small duck'] + Nn['small teacher', 'big happy cat'] (score 1.0, why: ['[k=1] every _ <EOS>', '[k=1] <BOS> _ walks'])
- fragmented gold classes: NP -> 4271(178), 4242(4), pruned(1); Vt -> 3911(6), multi(1); VP -> 4278(357), pruned(1); Adj -> 4253(6), multi(1)
- false constituent types: 148 — e.g. `the young` (freq 54, in class ~Det), `the fast` (freq 42, in class ~Det), `the old` (freq 41, in class ~Det), `horse walks` (freq 38, in class ~None), `the big` (freq 38, in class ~Det)
- missed constituent types: 32 — e.g. `happy cat` (Nn), `light child` (Nn), `light light` (Nn), `light cat` (Nn), `small dog` (Nn)
- false relations: unaligned (support 210, lexical 116); NP + Nn -> NP (support 30, lexical 15); unaligned (support 9, lexical 7); Det + Adj -> Adj (support 9, lexical 6); unaligned (support 5, lexical 5)
- ambiguity per word: `walks` acc 0.9869 (split accepted, gold {'Vt': 208, 'VP': 173}); `duck` acc 0.9757 (split accepted, gold {'VP': 105, 'Nn': 388}); `light` acc 0.9792 (split accepted, gold {'Adj': 170, 'Nn': 311}); `fast` acc 0.9537 (split accepted, gold {'Adv': 99, 'Adj': 160})

### synthetic3_noisy

- false merges: 43 — e.g. Nn['adj_rare0 big nn_rare2', 'every big big horse'] + NP['det_rare2 small woman', 'some bird'] (score 1.0, why: ['[k=1] <BOS> _ laughs', '[k=2] <BOS> <BOS> _ laughs <EOS>']); NP['Tom well', 'a happy bird'] + Nn['adj_rare0 nn_rare1', 'the big bird'] (score 1.0, why: ['[k=1] <BOS> _ sleeps', '[k=2] <BOS> <BOS> _ sleeps <EOS>']); Nn['big teacher', 'Mary um'] + NP['every nn_rare0', 'the adj_rare1 woman'] (score 1.0, why: ['[k=1] <BOS> _ waits', '[k=2] <BOS> <BOS> _ waits <EOS>'])
- fragmented gold classes: Adj -> 4216(7), 3976(1), 4028(1), 3812(1); Nn -> 4206(66), pruned(42), 4217(40), 4218(1); VP -> 4218(300), pruned(8); NP -> 4206(186), 4217(6), pruned(5), 4216(1)
- false constituent types: 385 — e.g. `um` (freq 155, in class ~Adv), `uh` (freq 152, in class ~Adv), `well` (freq 145, in class ~Adv), `the small` (freq 85, in class ~Det), `dog hears` (freq 70, in class ~None)
- missed constituent types: 64 — e.g. `small bird` (Nn), `big woman` (Nn), `happy cat` (Nn), `big nn_rare3` (Nn), `happy horse` (Nn)
- false relations: Det + Det -> Det (support 695, lexical 111); unaligned (support 178, lexical 94); Adv + NP -> NP (support 77, lexical 22); Det + Adv -> Det (support 73, lexical 21); Det + Nn -> Nn (support 53, lexical 27)

## 研究問題的回答（作者分析，依據上方自動產生的數字）

所有數字來自 `outputs/summary.json`（seed 42，3000 句合成語料，除非另註明）。縮寫：class F1 = gold-constituent span types 的 pairwise F1；const F1 = span-level constituent F1（derivation level）；comp = composition relation R(A,B,C) 經 Hungarian alignment 後的 P/R/F1。

### 驗收狀態

| 門檻 | 結果 | 通過 |
|---|---|---|
| Synthetic-1 class pairwise F1 ≥ 0.95 | 0.983 (ARI 0.976, purity 0.984, 8/8 classes) | ✓ |
| Synthetic-1 constituent F1 ≥ 0.95 | 0.998 (len≥2: 0.999, precision 1.000) | ✓ |
| Synthetic-1 composition F1 ≥ 0.95 | 0.875 (recall 1.000 = 7/7 gold relations；precision 0.778：多出 `Det+Nn->Nn` (support 66, 來自 NP/Nn fragment 混淆) 與 `Adj+Adj->Adj` (support 2)) | ✗（差距來自 2 條低支持度 spurious relation，未調參壓掉） |
| Synthetic-2 ambiguous type F1 ≥ 0.90 | 1.000 (4/4, 0 false splits) | ✓ |
| Synthetic-2 occurrence assignment accuracy ≥ 0.90 | 0.978 (majority-sense baseline 0.628) | ✓ |

依規格「達不到不要盲目調參」：composition precision 沒有被門檻化掉，而是在下面交代原因。

### RQ1：只用 raw-text context P(context|span) 能否恢復 latent equivalence classes？

**能，但有三個前提，而且每個前提都是實驗逼出來的。**

1. **多尺度 context 必須經過 sparsity 校正。** 3000 句時 k=3,4 的 joint context 幾乎都是 unique（scale-1 可共享的 context mass 0.998，scale-4 只有 0.60）。直接把各尺度混成一個分佈做 Bhattacharyya，Vt / Adj / Adv 三類永遠合不起來（similarity 被稀疏尺度稀釋到 0.3 以下），而 `clean_sim_bc_uncorrected` 只到 class F1 0.886、238 個 class。用 split-half self-similarity 做 attenuation correction（sim = Σ_k sim_k / Σ_k √(self_k(x)·self_k(y))）之後是 0.983、8 個 class、0 個 big-class false merge。
2. **context range 要夠。** `clean_k1` 0.882 → `clean_k12` 0.937 → `clean_k1234` 0.983；Synthetic-2 上 ambiguity F1 0.857 → 1.000 → 1.000。k=1 時 VP 的 lexical 成員（sleeps）與 Adv（quickly）共享 `(noun _ EOS)` context（gold 不同類之間的最高 BC = 0.64，同類最低 0.71），要到 k≥3 才能分開（RQ5）。噪音語料例外：`noisy_k1` 0.869 > `noisy_k1234` 0.843——filler 使長 context 更稀疏。
3. **只有 clustering 不夠，稀有 type 要靠 grammar 自己貼標籤。** 不做 MDL（ablation A/B/C）時 class F1 只有 0.740、70 個 class：高頻 lexical 類完全正確，但稀有的多詞 span（`old man`、`young horse`）各自留成 singleton。開啟 MDL + predictive chart 後，這些 span 由 `Adj+Nn->Nn` 推導而得到類別，才到 0.983。

Baseline 比較（clean）：weighted Jaccard 0.590（對頻率差敏感）、Jensen-Shannon 0.844、Bhattacharyya+校正 0.983；JS 與 Jaccard 走 Python pair loop，慢 4 倍。

殘餘錯誤：(a) NP/Nn 小碎片的 false merge（30 件，都是 4 個以下 type 的小類，如 `the small dog`(NP) 與 `small dog`(Nn) 都只在主語位置出現）；(b) 噪音語料 filler `uh / well` 被併入 Adv、`big` 一度併入 Det（class purity 0.91）。

### RQ2：加入 A+B→C composition evidence 後，能否分開真正 constituent 與偶然 substring？

**作為加法式分數 (αS_ctx+βS_comp+γS_prod)：不能；作為 derivation（MDL elimination + predictive chart）：能。**

- Ablation A（只 context）、B（只 composition+productivity）、C（兩者相加）結果**完全相同**：class F1 0.740、70 個 class、len≥2 constituent precision 0.43、composition precision 0.02。原因：`NP+Vt`、`Vt+Det`、`Nn+Vt` 這類高頻非 constituent 的 context 一致性、composition coverage 與 productivity 都和真 constituent 一樣高（它們本來就是「規則的」字串），加法式分數分不開。
- Ablation D（+ complexity penalty = MDL）：len≥2 precision 0.43 → **1.000**、const F1 0.769 → 0.998、composition precision 0.02 → 0.78、class 數 70 → 8，所有 7 條 gold relation 全部恢復。機制不是「composition 分數高」，而是「有了真 class 與真 relation 之後，非 constituent class 對編碼語料沒有貢獻」：`NP+Vt->X` 一移除，所有句子仍由 `NP+VP->S`、`Vt+NP->VP` 推導，grammar bits 減少、data bits 不增。關鍵工程細節（都在失敗案例裡逼出來）：chart 必須允許 grammar 推導「不是 candidate span 的稀有 span」（否則 `sees the small cat` 不能當 VP 節點，`the dog sees` 反而靠覆蓋率勝出）；data cost 必須是由 derivation counts 計算的 closed form（否則 hard-EM 把先用到的分析鎖死）；rule elimination 只能在 predictive 階段做（否則 `VP+Adv->VP` 在稀有 span 無法使用的階段被誤刪）。
- 噪音語料：D 把 class F1 從 0.520 提到 0.843、const F1 0.744 → 0.903，但 composition precision 只有 0.27：`Det+Det->Det`(679)、`Adv+NP->NP`(77) 等 spurious relation 來自 filler（`uh the`、`the um`）與 fragment 混淆。
- **原則性限制：**(i) 只有字串證據時，`Adj+Nn->Nn` 與 `Det+Adj->Det` 在沒有 bare noun 的 grammar 裡是 weakly equivalent（生成同一字串集），MDL 任選其一（第一版 grammar 得到 `Det+Adj->Det`，composition F1 0.80）；加入 `Nn+VP->S`、`Vt+Nn->VP`（bare noun，對應自然語言的 bare plural）後才可辨識，但代價是 NP/Nn context 重疊造成上述碎片混淆。(ii) `dog waits` 這種 span 在 17% 出現是 bare-S constituent、83% 是 `the dog waits` 內的非 constituent：系統有時為它保留一個獨立 root class（10k 句時 1073 次使用），它是「過細但不錯」的分析（S_bare），卻算成 false relation；class prior 掃描顯示 c_class 12→100 對 3k 句反而變差（comp F1 0.875→0.80，13 class）、對 10k 句變好（0.63→0.80），**不是單調的**，所以不能靠調 prior 解決——這是 class identifiability 問題。

### RQ3：同一 surface expression 的 occurrence context 是否足以發現 category ambiguity？

**足以，前提是兩個 sense 所屬的 class 已經被其他詞建立起來。** Synthetic-2 的 4 個 planted 詞（duck Nn/VP、walks VP/Vt、fast Adj/Adv、light Adj/Nn）全部被切開且無 false split（type F1 1.000），occurrence 指派正確率 0.978（walks 0.987、duck 0.976、light 0.981、fast 0.965），majority baseline 0.628。
- 提案方式很關鍵：用「各 class 的 context profile（排除該詞自身）對每個 occurrence 算 likelihood」的 guided assignment 可行；無監督 2-means（`amb_E_kmeans_proposal`）**一個 split 都沒接受**（兩群對齊到同一 class）。
- context range：k=1 只有 0.857 / 0.819，k≤2 以上即 1.000 / 0.975。
- 失敗案例（seed 2）：`light`、`fast` 兩個 ambiguous 詞佔 Adj 8 個成員中的 2 個，Adj 的 context 分佈被 Nn/Adv context 污染，clustering 階段 Adj 被併入 Det；之後 `fast` 的 Adj sense 沒有 class 可對齊（accuracy 0.397）。這是「ambiguity 污染 class identifiability」的鏈式失敗，不是 split 機制本身的錯。

### RQ4：ambiguity split 能否被 MDL / grammar compression 客觀支持，而非 clustering artefact？

**能，而且 MDL 是唯一擋住 artefact 的東西。**
- `amb_E_no_mdl_criterion`（只用 context 準則）：接受 20 個 split，其中 16 個是 false（`small`、`quickly`… 因 topic-like context 變化被拆），type F1 0.333；加上 Δ_split = DL(H1) − DL(H2) − λ_split > 0 後：4/4、0 false。
- split 確實改善 grammar consistency：`amb_D_no_splitting` → `amb_E_full` 的 occurrence-level class F1 0.912 → 0.961，composition F1 0.632 → 0.667，bits/sentence 23.82 → 23.44（壓縮變好）。
- 必要的修正：若把 ambiguous 詞單一指派為 Nn，它的 VP-sense 句子會催生 `NP+Nn->S` 這種「只被該詞自己支撐」的 crutch relation，使 split 看起來沒有 DL 收益（修正前 seed 1/2 漏掉 duck/fast/light）。把「該詞佔用量 ≥ 50% 的 rule」從所有假設中排除後，seed 1 也達 1.000 / 0.979。這對應規格中「只被該 expression 使用的 branch 應受強 penalty」，只是作用在 relation 而非 class 上。

### RQ5：失敗時主要瓶頸是哪一個？

依證據排序：
1. **class identifiability（最主要）。** 所有 run 的殘餘錯誤都集中在 NP/Nn 碎片、bare-S class、以及 seed 2 的 Adj→Det 污染；composition precision 的損失幾乎全部可追溯到這裡。seed 間的變異（clean 0.927–0.983；amb occurrence accuracy 0.788–0.979）也來自同一來源。
2. **data sparsity。** n=1000：0.901（28 class）；n=3000：0.983；k≥3 context 在 3k 句時大多 unique；稀有多詞 span 只能靠 grammar 推導取得類別。但 n=10000 並未更好（0.889，12 class）：資料變多讓 bare-S class 的 data-term 優勢變大——所以 sparsity 不是唯一瓶頸。
3. **constituency criterion。** 字串 weak equivalence（attachment 方向）與「過細 class」問題無法由 raw text 解決；MDL 常數（c_class、c_rule）是明確的 prior，但實驗顯示它不是可靠槓桿。
4. **context representation。** VP/Adv 需要 k≥3（固定尺度）；自然語言中這類「只在更長距離才可分」的對立會更多——adaptive / unbounded context refinement 是下一步。
5. **ambiguity overfitting**：在 MDL 準則下已受控（0 false splits），但對上游 class 污染脆弱。

噪音語料（Synthetic-3）綜合了以上各點：class F1 0.843、const F1 0.903、relation recall 0.857 但 precision 0.273；filler 被視為 Adv、rare words 留成 singleton、topic 相關性本身沒有造成 false split。

### 自然語言（TinyStories，10k 句，raw text only）

本節的數字是 intrinsic 指標（stability、compression、held-out context prediction、productivity），**不是** syntax recovery accuracy；完整數字與 top classes / relations / splits 在上方自動產生的 "Natural-language study" 段落與 `outputs/natural/`。

- **規模**：10,000 句、103k tokens、vocab 4,747；14,754 個候選 span type（freq ≥ 3）、203k occurrences、602k unique contexts、702k span–context edges；7,266 個 provisional class → MDL 後 1,270 個 accepted class、132 條 relation；總時間 109 s（增量式 MDL closed form 之後；優化前 426 s，見 `docs/profiling.md`），peak memory 1.7 GB。剩下的瓶頸是 pure-Python CKY（約一半時間），50k/100k 句在 v0.1 仍未跑，留待 parser 加速後做 sample complexity。
- **lexical class 有明顯 signal**（人工檢查 top classes）：主語代名詞與人名 {He, They, She, Sammy, Lisa, John, …}、受詞代名詞 {them, him, it, everything}、小寫主語代名詞 {he, they, she}、所有格 {their, her, its, his}、冠詞 {the} / {a}、copula {was, is, felt, wasn't}、句末標點 {., !}、引號、連接詞 {and} 與 complementizer 片段 {that he, because they}、地點名詞 {kitchen, bowl, window, tree, sky, pond, park, …}。這些與 RQ1 的結論一致：k≤4 的 context 加上 sparsity 校正足以把高頻 lexical 類分開。
- **constituent 結構只恢復到局部**：MDL 樣本中 0 句被完整推導、每句 5.5 個 chunk、1,751 個 unk token；132 條 relation 多是兩三個 token 的局部組合（`was + so`、`tree + .`、`The + bird`、`She + felt`、`One + day ,`），而非完整的 NP/VP/S 層級。多詞成員混入了以相同邊界結尾的非 constituent（`He smiled and` 進入主語類、`to go outside and` 進入 `to` 類）：這正是 Synthetic-3 中 filler/substring 現象的自然語言版本，MDL 在 1,270 個 class 的規模下無法像合成語料那樣把它們全部清掉（RQ2 的機制依賴「真 class 與真 relation 先存在」）。
- **ambiguity**：722 個 type 被測試、2 個 split 被接受：`It`（`It was a …` 的 expletive 用法 vs `It had …` 的主語用法，split_gain +57 bits）與 `grass`（`on the grass .` vs `in the grass .`，+54 bits——後者看起來是地點類被切得太細的 artefact，而非 category ambiguity）。被拒絕的 top 候選（`.`、`They`、`he`、`she`、`so`）全部因「第二群對不上任何其他詞使用的 class」被擋下，符合設計。
- **stability**：同一語料換 seed → class pairwise F1 / ARI 1.0（pipeline 幾乎是 deterministic 的，seed 只影響 split-half 與 MDL 抽樣，所以這個數字沒有資訊量）；**兩個不相交的 5k 半份語料 → pairwise F1 0.22 / ARI 0.22**（2,038 個共有已分類 type），constituent-set agreement 0.66，split agreement 0。也就是說在 5k 句規模下，高頻 lexical 類以外的 class 結構不可重現——這是 RQ5 的 data sparsity 瓶頸在自然語言上的直接證據。
- **held-out context prediction（負結果）**：27,118 個 held-out occurrence 的 k=1 joint context，mean log2 P：class model −14.87、type model −14.62、background −14.42——背景分佈最好；rare types（freq<10）也是 type (−12.73) 優於 class (−13.80)。只有 6.0% 的 held-out context 是「type 沒見過、class 見過」。結論：以 (left, right) joint bigram 為單位的 context 在 10k 句下稀疏到任何條件化都比不上 unigram 背景；這個指標需要改成 one-sided / backoff context 才有鑑別力，v0.1 不據此宣稱 class 的泛化能力。
- **punctuation ablation**：去掉標點後 1,217 個 class、118 條 relation、123 bits/句（vs 136），結構相近；標點本身形成最高頻的類（`.` `!`）與最高支持度的 relation（`X + .`），是自然語料中最強的 boundary signal。

### 關鍵失敗案例（可在各 run 的 report.md 中逐條查到「why merged / why rejected / why split」）

1. **false merge（clean/amb/noisy 都有）**：`the small dog`(NP) ⟷ `small dog`(Nn)：兩者都只以 2–4 次出現在主語位置，k=1..4 context 完全相同 `(BOS _ sleeps)`；校正後 similarity 1.0。只有更多資料或更長 context 才能分。
2. **過細 class**：`dog waits / man runs`（Nn+VP）在 10k 句形成獨立 root class（1073 次使用），MDL 移除會多花 ~300 bits（data term），grammar 只省 22 bits。
3. **string-equivalent 分析**：無 bare noun 的 grammar 下 `Det+Adj->Det` 取代 `Adj+Nn->Nn`（兩者 DL 幾乎相同）。
4. **crutch relation 阻擋 split**：`NP+Nn->S`（只由 duck/light 的錯誤指派支撐）使 `duck` 的 split_gain = −102.8 bits；排除 private rules 後 +194.7 bits。
5. **class 污染**：seed 2 中 Adj 因 `light`+`fast` 的混合分佈被併入 Det，連帶 `Adj+Nn->Nn` 消失、`fast` 無法切對。
6. **噪音**：`uh the`、`the um` 等 filler 子串形成 `Det+Det->Det`(679)；`uh / well` 併入 Adv。

### 下一步建議

1. **class-level MDL merge/split**：目前 MDL 只刪 class/relation，不合併碎片；加入「合併兩個 class 是否縮短 DL」（含 parameter cost 的 Stolcke 式 prior）可直接處理 NP/Nn 碎片與 bare-S 過細 class。
2. **adaptive context refinement**：對 similarity 高但 composition 行為不一致的 class pair（VP vs Adv、NP vs Nn）自動擴展 context 直到可分，而非固定 k=1..4。
3. **多詞 span 的 occurrence-level ambiguity**，包括「constituent vs non-constituent」兩態（`dog waits` 案例），而不只 class A vs class B。
4. **把 ambiguous 詞的影響從 class profile 中隔離**（split 之後重算 class context profile 並回饋到 clustering），避免 seed 2 的污染鏈。
5. **效率**：CKY/MDL 用 numba 或 C++ 實作；predictive chart 加 beam/剪枝；這是 50k–500k 句自然語料的前提。
6. **朝 CCG 走**：(E,R) 已經給出 functor/argument 的候選（每條 A+B→C 中「結果類等於某一側」的 rule 是 modifier，否則其中一側是 functor），下一版可從 R 的結構因式分解 slash categories，而不需要任何 POS/treebank。
