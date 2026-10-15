# ABC induction v0.3 — identity-preserving constituent (modifier) evidence study

Pipeline v0.1: raw corpus -> candidate spans -> multi-scale sparse contexts -> provisional classes (agglomerative, sparsity-corrected similarity) -> constituency score + MDL elimination with a predictive chart -> occurrence-level (legacy) ambiguity splits -> (E, R).

Pipeline v0.2 (on top of the v0.1 result): finite Bayesian latent grammar (sparse PCFG, Dirichlet posteriors) -> sparse Inside-Outside (semiring chart, numba) -> Dirichlet CAVI with a computable ELBO -> objective F = ELBO + log P(G) -> greedy class merge / class split / rule and lexicon proposals with checkpoint + rollback; lexical contexts as leaf potentials, punctuation as span potentials, adaptive constituent contexts as proposal features only. Every run reports both stages on the same gold (`v0.1` rows = v0.1 metrics, `v0.2` rows = metrics of the v0.2 posterior).

Metric columns: class pair F1 / ARI are computed over gold-constituent span types (type level), occurrence-level F1 over gold-constituent occurrences; constituent F1 is span-level over all gold spans (derivation level, i.e. the nodes of the induced grammar's Viterbi derivations); composition P/R/F1 compare R(A,B,C) after Hungarian class alignment; ambiguity F1 is over gold-ambiguous span types and occurrence accuracy is measured on their occurrences.

## Synthetic-4 (modifier + controls): modifier evidence and lexicon distinguishability

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_A_baseline | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 14.420 |
| s4_B_diag | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 15.550 |
| s4_C_str | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 16.470 |
| s4_D_strcls | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 15.960 |
| s4_E_dist_comp | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 16.780 |
| s4_F_dist_full | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 17.490 |
| s4_G_dist_only | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 15.840 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_A_baseline | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/30 | 13.180 |
| s4_B_diag | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/30 | 14.330 |
| s4_C_str | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 14.980 |
| s4_D_strcls | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 14.680 |
| s4_E_dist_comp | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 15.920 |
| s4_F_dist_full | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 16.240 |
| s4_G_dist_only | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/30 | 14.810 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_diag | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_C_str | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_D_strcls | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_E_dist_comp | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_F_dist_full | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_G_dist_only | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |

- [s4_A_baseline/report.md](outputs/v03/s4_A_baseline/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 393.9 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/30 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_B_diag/report.md](outputs/v03/s4_B_diag/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 400.9 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/30 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_C_str/report.md](outputs/v03/s4_C_str/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 408.0 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_D_strcls/report.md](outputs/v03/s4_D_strcls/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 408.3 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_E_dist_comp/report.md](outputs/v03/s4_E_dist_comp/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 407.2 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_F_dist_full/report.md](outputs/v03/s4_F_dist_full/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 392.2 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_G_dist_only/report.md](outputs/v03/s4_G_dist_only/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 406.2 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/30 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP

## Synthetic-1 (clean): regression gate — no new false lexicon entries, no loss of class / constituent / composition recovery

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_A_baseline | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 22.300 |
| s1_B_diag | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 23.350 |
| s1_C_str | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 25.970 |
| s1_D_strcls | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 25.420 |
| s1_E_dist_comp | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 25.670 |
| s1_F_dist_full | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 25.340 |
| s1_G_dist_only | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 22.600 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_A_baseline | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 21.640 |
| s1_B_diag | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 22.530 |
| s1_C_str | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 25.310 |
| s1_D_strcls | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 24.610 |
| s1_E_dist_comp | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 24.780 |
| s1_F_dist_full | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 24.610 |
| s1_G_dist_only | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 21.930 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_B_diag | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_C_str | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_D_strcls | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_E_dist_comp | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_F_dist_full | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_G_dist_only | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |

- [s1_A_baseline/report.md](outputs/v03/s1_A_baseline/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 362.2 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_B_diag/report.md](outputs/v03/s1_B_diag/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 370.3 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_C_str/report.md](outputs/v03/s1_C_str/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 376.7 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_D_strcls/report.md](outputs/v03/s1_D_strcls/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 377.8 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_E_dist_comp/report.md](outputs/v03/s1_E_dist_comp/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 377.9 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_F_dist_full/report.md](outputs/v03/s1_F_dist_full/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 377.1 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_G_dist_only/report.md](outputs/v03/s1_G_dist_only/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 371.9 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP

## Synthetic-2 (ambiguity): true ambiguity must survive the duplicate-entry penalty

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_A_baseline | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 22.580 |
| s2_B_diag | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 21.480 |
| s2_C_str | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 24.820 |
| s2_D_strcls | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 24.080 |
| s2_E_dist_comp | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 25.270 |
| s2_F_dist_full | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 26.740 |
| s2_G_dist_only | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 24.200 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_A_baseline | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 14/94 | 22.000 |
| s2_B_diag | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 14/94 | 20.750 |
| s2_C_str | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 23.900 |
| s2_D_strcls | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 23.320 |
| s2_E_dist_comp | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 24.530 |
| s2_F_dist_full | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 26.040 |
| s2_G_dist_only | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 12/92 | 23.360 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_B_diag | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.979] | 0.218/1.000 [0.670] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 5/18 | 1 | 1 | 0 |
| s2_C_str | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_D_strcls | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_E_dist_comp | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_F_dist_full | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_G_dist_only | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.979] | 0.218/1.000 [0.670] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 2 |

- [s2_A_baseline/report.md](outputs/v03/s2_A_baseline/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 383.5 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 14/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331
- [s2_B_diag/report.md](outputs/v03/s2_B_diag/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 373.8 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 14/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331
- [s2_C_str/report.md](outputs/v03/s2_C_str/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 380.5 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_D_strcls/report.md](outputs/v03/s2_D_strcls/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 382.2 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_E_dist_comp/report.md](outputs/v03/s2_E_dist_comp/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 380.3 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_F_dist_full/report.md](outputs/v03/s2_F_dist_full/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 399.9 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_G_dist_only/report.md](outputs/v03/s2_G_dist_only/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 374.3 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 12/92 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331

## Synthetic-3 (noisy): class collapse vs class-generalized evidence

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3_B_diag | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 42.160 |
| s3_C_str | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 55.670 |
| s3_D_strcls | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 53.930 |
| s3_F_dist_full | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 43.110 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3_B_diag | 0.906 | 0.878 | 0.982 | 81 | 0.952 | 0.927 | 0.179 | 1.000 | 0.304 | 0.000 | - | 26.474 | 41 | 25/107 | 41.410 |
| s3_C_str | 0.836 | 0.791 | 0.692 | 222 | 0.841 | 0.667 | 0.167 | 0.429 | 0.240 | 0.000 | - | 26.048 | 18 | 37/134 | 54.610 |
| s3_D_strcls | 0.836 | 0.791 | 0.692 | 222 | 0.841 | 0.667 | 0.167 | 0.429 | 0.240 | 0.000 | - | 26.048 | 18 | 37/134 | 53.060 |
| s3_F_dist_full | 0.889 | 0.857 | 0.969 | 101 | 0.942 | 0.903 | 0.179 | 1.000 | 0.304 | 0.000 | - | 26.482 | 41 | 23/95 | 42.210 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3_B_diag | 0.500/0.737 [0.263] | 0.515/0.895 [0.435] | 0.500/0.579 [0.395] | 0.667/0.947 [0.792] | 0.439/0.947 [0.585] | 0.143/0.600 | 1.000 | 1.000 | 0.333/0.368 | 4/17 | 2 | 0 | 0 |
| s3_C_str | 0.518/0.737 [0.263] | 0.448/0.684 [0.230] | 0.444/0.632 [0.229] | 0.463/1.000 [0.375] | 0.422/1.000 [0.338] | 0.143/0.600 | 1.000 | 0.917 | 0.333/0.368 | 12/12 | 6 | 0 | 0 |
| s3_D_strcls | 0.518/0.737 [0.263] | 0.448/0.684 [0.230] | 0.444/0.632 [0.229] | 0.463/1.000 [0.375] | 0.422/1.000 [0.338] | 0.143/0.600 | 1.000 | 0.917 | 0.333/0.368 | 12/12 | 6 | 0 | 0 |
| s3_F_dist_full | 0.500/0.737 [0.263] | 0.457/0.842 [0.507] | 0.522/0.632 [0.482] | 0.591/0.684 [0.790] | 0.390/0.842 [0.609] | 0.143/0.600 | 1.000 | 0.833 | 0.333/0.368 | 5/21 | 3 | 0 | 1 |

- [s3_B_diag/report.md](outputs/v03/s3_B_diag/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 400.6 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 25/107 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4217 + C4063 -> C4217)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'unaligned', 'S + Adv -> S', 'unaligned']
- [s3_C_str/report.md](outputs/v03/s3_C_str/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 418.3 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 37/134 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4217 + C4063 -> C4217)); relations Nn + VP -> S, VP + Adv -> VP, Vt + Nn -> VP; missed Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP; false ['Det + Nn -> Nn', 'Det + S -> S', 'Vt + Det -> Vt', 'S + Adv -> S']
- [s3_D_strcls/report.md](outputs/v03/s3_D_strcls/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 417.9 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 37/134 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4217 + C4063 -> C4217)); relations Nn + VP -> S, VP + Adv -> VP, Vt + Nn -> VP; missed Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP; false ['Det + Nn -> Nn', 'Det + S -> S', 'Vt + Det -> Vt', 'S + Adv -> S']
- [s3_F_dist_full/report.md](outputs/v03/s3_F_dist_full/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 414.0 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 23/95 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(big), rule_add(C4217 + C4063 -> C4217)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'unaligned', 'unaligned', 'S + Adv -> S']

## Forced class collapse (random init, K = 8): does class-generalized evidence inherit false merges?

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_random_k8 | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 265.630 |
| s4_C_random_k8 | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 224.400 |
| s4_D_random_k8 | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 224.440 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_random_k8 | 0.753 | 0.691 | 0.667 | 231 | 0.916 | 0.807 | 0.261 | 0.667 | 0.375 | 0.000 | - | 28.555 | 25 | 39/165 | 264.850 |
| s4_C_random_k8 | 0.606 | 0.516 | 0.692 | 286 | 0.931 | 0.842 | 0.300 | 1.000 | 0.462 | 0.000 | - | 29.793 | 38 | 34/154 | 223.540 |
| s4_D_random_k8 | 0.606 | 0.516 | 0.692 | 286 | 0.931 | 0.842 | 0.300 | 1.000 | 0.462 | 0.000 | - | 29.793 | 38 | 34/154 | 223.650 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_random_k8 | 0.652/0.833 [0.458] | 0.562/1.000 [0.570] | 0.750/0.667 [0.635] | 0.667/0.667 [0.705] | 0.486/1.000 [0.624] | 0.196/0.385 | 0.778 | 0.417 | 0.524/0.611 | 7/42 | 1 | 0 | 0 |
| s4_C_random_k8 | 0.652/0.833 [0.458] | 0.367/1.000 [0.714] | 0.288/0.944 [0.724] | 0.210/0.889 [0.529] | 0.200/1.000 [0.616] | 0.196/0.385 | 0.944 | 0.647 | 0.524/0.611 | 5/38 | 1 | 0 | 0 |
| s4_D_random_k8 | 0.652/0.833 [0.458] | 0.367/1.000 [0.714] | 0.288/0.944 [0.724] | 0.210/0.889 [0.529] | 0.200/1.000 [0.616] | 0.196/0.385 | 0.944 | 0.647 | 0.524/0.611 | 5/38 | 1 | 0 | 0 |

- [s4_B_random_k8/report.md](outputs/v03/s4_B_random_k8/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 435.0 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 39/165 moves accepted (merge(C1 + C7), merge(C4 + C1+C7), split(C3), rule_prune(110), rule_prune(4), lex_add(the), lex_add(a), merge(C6 + C4+C1+C7)); relations AdvS + S -> S, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + Nn -> VP; missed Adj + Nn -> Nn, Deg + Vt -> Vt, Vt + NP -> VP; false ['Vt + Adv -> VP', 'Det + Nn -> Nn', 'Det + Nn -> Adv', 'unaligned']
- [s4_C_random_k8/report.md](outputs/v03/s4_C_random_k8/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 453.3 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 34/154 moves accepted (merge(C1 + C7), merge(C4 + C1+C7), split(C3), rule_prune(110), rule_prune(4), lex_add(the), lex_add(a), merge(C6 + C4+C1+C7)); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['S + Adv -> S', 'Det + Nn -> Adv', 'Det + Nn -> AdvS', 'Deg + VP -> Vt']
- [s4_D_random_k8/report.md](outputs/v03/s4_D_random_k8/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 452.5 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 34/154 moves accepted (merge(C1 + C7), merge(C4 + C1+C7), split(C3), rule_prune(110), rule_prune(4), lex_add(the), lex_add(a), merge(C6 + C4+C1+C7)); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['S + Adv -> S', 'Det + Nn -> Adv', 'Det + Nn -> AdvS', 'Deg + VP -> Vt']

## Key failure cases (from the main runs)
