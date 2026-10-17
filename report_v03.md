# ABC induction v0.3 — identity-preserving constituent (modifier) evidence study

Pipeline v0.1: raw corpus -> candidate spans -> multi-scale sparse contexts -> provisional classes (agglomerative, sparsity-corrected similarity) -> constituency score + MDL elimination with a predictive chart -> occurrence-level (legacy) ambiguity splits -> (E, R).

Pipeline v0.2 (on top of the v0.1 result): finite Bayesian latent grammar (sparse PCFG, Dirichlet posteriors) -> sparse Inside-Outside (semiring chart, numba) -> Dirichlet CAVI with a computable ELBO -> objective F = ELBO + log P(G) -> greedy class merge / class split / rule and lexicon proposals with checkpoint + rollback; lexical contexts as leaf potentials, punctuation as span potentials, adaptive constituent contexts as proposal features only. Every run reports both stages on the same gold (`v0.1` rows = v0.1 metrics, `v0.2` rows = metrics of the v0.2 posterior).

Metric columns: class pair F1 / ARI are computed over gold-constituent span types (type level), occurrence-level F1 over gold-constituent occurrences; constituent F1 is span-level over all gold spans (derivation level, i.e. the nodes of the induced grammar's Viterbi derivations); composition P/R/F1 compare R(A,B,C) after Hungarian class alignment; ambiguity F1 is over gold-ambiguous span types and occurrence accuracy is measured on their occurrences.

## Synthetic-4 (modifier + controls): modifier evidence and lexicon distinguishability

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_A_baseline | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 15.680 |
| s4_B_diag | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 16.340 |
| s4_C_str | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 17.940 |
| s4_D_strcls | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 16.980 |
| s4_E_dist_comp | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 17.890 |
| s4_F_dist_full | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 16.850 |
| s4_G_dist_only | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 17.880 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_A_baseline | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/30 | 14.600 |
| s4_B_diag | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/30 | 15.210 |
| s4_C_str | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 17.000 |
| s4_D_strcls | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 16.000 |
| s4_E_dist_comp | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 16.740 |
| s4_F_dist_full | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/34 | 15.640 |
| s4_G_dist_only | 0.997 | 0.996 | 0.866 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 19.172 | 12 | 6/30 | 16.780 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_diag | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_C_str | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_D_strcls | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_E_dist_comp | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_F_dist_full | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |
| s4_G_dist_only | 0.789/0.833 [0.458] | 0.818/1.000 [0.709] | 1.000/1.000 [0.730] | 1.000/1.000 [1.000] | 0.818/1.000 [0.762] | 0.196/0.385 | 1.000 | 1.000 | 0.524/0.611 | 0/6 | 0 | 0 | 0 |

- [s4_A_baseline/report.md](outputs/v03/s4_A_baseline/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 391.3 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/30 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_B_diag/report.md](outputs/v03/s4_B_diag/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 403.1 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/30 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_C_str/report.md](outputs/v03/s4_C_str/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 407.7 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_D_strcls/report.md](outputs/v03/s4_D_strcls/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 408.1 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_E_dist_comp/report.md](outputs/v03/s4_E_dist_comp/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 408.8 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_F_dist_full/report.md](outputs/v03/s4_F_dist_full/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 408.5 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/34 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s4_G_dist_only/report.md](outputs/v03/s4_G_dist_only/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 406.8 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 6/30 moves accepted (merge(C5087 + C5088), merge(C5083 + C5085), split(C5087+C5088), split(C5084), rule_prune(15), lex_prune()); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP

## Synthetic-1 (clean): regression gate — no new false lexicon entries, no loss of class / constituent / composition recovery

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_A_baseline | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 25.070 |
| s1_B_diag | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 27.240 |
| s1_C_str | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 26.690 |
| s1_D_strcls | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 28.800 |
| s1_E_dist_comp | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 27.480 |
| s1_F_dist_full | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 28.140 |
| s1_G_dist_only | 0.983 | 0.976 | 0.985 | 8 | 0.998 | 0.999 | 1.000 | 0.778 | 1.000 | 0.875 | 0.000 | - | 22.056 | 25.110 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_A_baseline | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 24.090 |
| s1_B_diag | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 26.420 |
| s1_C_str | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 25.960 |
| s1_D_strcls | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 27.680 |
| s1_E_dist_comp | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 26.610 |
| s1_F_dist_full | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 27.270 |
| s1_G_dist_only | 0.995 | 0.993 | 0.873 | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | - | 16.982 | 7 | 14/115 | 24.380 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_B_diag | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_C_str | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_D_strcls | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_E_dist_comp | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_F_dist_full | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |
| s1_G_dist_only | 0.733/1.000 [0.338] | 1.000/1.000 [0.660] | 1.000/1.000 [0.659] | 1.000/1.000 [1.000] | 0.733/1.000 [0.682] | 0.171/0.417 | 1.000 | 1.000 | 0.304/0.636 | 12/13 | 12 | 0 | 0 |

- [s1_A_baseline/report.md](outputs/v03/s1_A_baseline/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 362.6 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_B_diag/report.md](outputs/v03/s1_B_diag/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 390.4 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_C_str/report.md](outputs/v03/s1_C_str/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 376.9 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_D_strcls/report.md](outputs/v03/s1_D_strcls/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 397.3 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_E_dist_comp/report.md](outputs/v03/s1_E_dist_comp/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 377.7 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_F_dist_full/report.md](outputs/v03/s1_F_dist_full/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 377.5 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP
- [s1_G_dist_only/report.md](outputs/v03/s1_G_dist_only/report.md); 3000 sentences, 2242 span types, 34810 contexts, 81019 edges, peak 372.7 MB, 1985 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; v0.2: 14/115 moves accepted (rule_prune(1), lex_add(child), lex_add(horse), rule_prune(1), lex_add(woman), lex_add(man), lex_add(bird), lex_add(cat)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP

## Synthetic-2 (ambiguity): true ambiguity must survive the duplicate-entry penalty

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_A_baseline | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 24.640 |
| s2_B_diag | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 24.020 |
| s2_C_str | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 25.490 |
| s2_D_strcls | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 25.970 |
| s2_E_dist_comp | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 27.390 |
| s2_F_dist_full | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 28.100 |
| s2_G_dist_only | 0.944 | 0.925 | 0.912 | 41 | 0.979 | 0.962 | 0.966 | 0.462 | 0.857 | 0.600 | 0.000 | 0.627 | 23.842 | 25.500 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_A_baseline | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 14/94 | 23.980 |
| s2_B_diag | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 14/94 | 23.290 |
| s2_C_str | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 24.680 |
| s2_D_strcls | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 25.180 |
| s2_E_dist_comp | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 26.600 |
| s2_F_dist_full | 0.994 | 0.991 | 0.978 | 15 | 1.000 | 1.000 | 0.538 | 1.000 | 0.700 | 0.857 | 0.934 | 20.626 | 13 | 15/94 | 27.310 |
| s2_G_dist_only | 0.995 | 0.993 | 0.978 | 14 | 1.000 | 1.000 | 0.583 | 1.000 | 0.737 | 0.857 | 0.933 | 20.598 | 12 | 12/92 | 24.530 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_B_diag | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.979] | 0.218/1.000 [0.670] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 5/18 | 1 | 1 | 0 |
| s2_C_str | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_D_strcls | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_E_dist_comp | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_F_dist_full | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.984] | 0.218/1.000 [0.682] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 0 |
| s2_G_dist_only | 0.333/1.000 [0.294] | 0.324/1.000 [0.627] | 0.923/1.000 [0.663] | 0.857/1.000 [0.979] | 0.218/1.000 [0.670] | 0.175/0.172 | 1.000 | 1.000 | 0.210/0.667 | 3/16 | 1 | 1 | 2 |

- [s2_A_baseline/report.md](outputs/v03/s2_A_baseline/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 383.2 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 14/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331
- [s2_B_diag/report.md](outputs/v03/s2_B_diag/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 373.9 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 14/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331
- [s2_C_str/report.md](outputs/v03/s2_C_str/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 401.7 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_D_strcls/report.md](outputs/v03/s2_D_strcls/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 380.7 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_E_dist_comp/report.md](outputs/v03/s2_E_dist_comp/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 380.0 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_F_dist_full/report.md](outputs/v03/s2_F_dist_full/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 382.3 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 15/94 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), rule_add(C4275 + C4253.1 -> C4275), rule_add(C4275 + C4219 -> C4275), merge(C4275 + C4276)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'S + Adv -> S', 'Nn + Adv -> VP', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9343
- [s2_G_dist_only/report.md](outputs/v03/s2_G_dist_only/report.md); 3000 sentences, 2285 span types, 37086 contexts, 81084 edges, peak 392.9 MB, 1995 merges, 0 splits; relations recovered: Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Nn + VP -> S; ambiguity: missed ['walks', 'duck', 'light', 'fast'], false []; v0.2: 12/92 moves accepted (split(C4253), rule_prune(7), lex_prune(), lex_add(walks), lex_add(light), merge(C4275 + C4276), merge(C4219 + C4253.1), split(C4219+C4253.1)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['NP + Nn -> S', 'Nn + Adv -> VP', 'unaligned', 'unaligned']; v0.2 ambiguity: missed ['duck'], false [], occurrence accuracy 0.9331

## Synthetic-3 (noisy): class collapse vs class-generalized evidence

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3_B_diag | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 46.110 |
| s3_C_str | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 59.290 |
| s3_D_strcls | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 61.050 |
| s3_F_dist_full | 0.843 | 0.794 | 0.837 | 76 | 0.902 | 0.818 | 0.807 | 0.273 | 0.857 | 0.414 | 0.000 | - | 27.308 | 46.080 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3_B_diag | 0.906 | 0.878 | 0.982 | 81 | 0.952 | 0.927 | 0.179 | 1.000 | 0.304 | 0.000 | - | 26.474 | 41 | 25/107 | 45.160 |
| s3_C_str | 0.836 | 0.791 | 0.692 | 222 | 0.841 | 0.667 | 0.167 | 0.429 | 0.240 | 0.000 | - | 26.048 | 18 | 37/134 | 58.230 |
| s3_D_strcls | 0.836 | 0.791 | 0.692 | 222 | 0.841 | 0.667 | 0.167 | 0.429 | 0.240 | 0.000 | - | 26.048 | 18 | 37/134 | 59.780 |
| s3_F_dist_full | 0.889 | 0.857 | 0.969 | 101 | 0.942 | 0.903 | 0.179 | 1.000 | 0.304 | 0.000 | - | 26.482 | 41 | 23/95 | 45.100 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3_B_diag | 0.500/0.737 [0.263] | 0.515/0.895 [0.435] | 0.500/0.579 [0.395] | 0.667/0.947 [0.792] | 0.439/0.947 [0.585] | 0.143/0.600 | 1.000 | 1.000 | 0.333/0.368 | 4/17 | 2 | 0 | 0 |
| s3_C_str | 0.518/0.737 [0.263] | 0.448/0.684 [0.230] | 0.444/0.632 [0.229] | 0.463/1.000 [0.375] | 0.422/1.000 [0.338] | 0.143/0.600 | 1.000 | 0.917 | 0.333/0.368 | 12/12 | 6 | 0 | 0 |
| s3_D_strcls | 0.518/0.737 [0.263] | 0.448/0.684 [0.230] | 0.444/0.632 [0.229] | 0.463/1.000 [0.375] | 0.422/1.000 [0.338] | 0.143/0.600 | 1.000 | 0.917 | 0.333/0.368 | 12/12 | 6 | 0 | 0 |
| s3_F_dist_full | 0.500/0.737 [0.263] | 0.457/0.842 [0.507] | 0.522/0.632 [0.482] | 0.591/0.684 [0.790] | 0.390/0.842 [0.609] | 0.143/0.600 | 1.000 | 0.833 | 0.333/0.368 | 5/21 | 3 | 0 | 1 |

- [s3_B_diag/report.md](outputs/v03/s3_B_diag/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 400.5 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 25/107 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4217 + C4063 -> C4217)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'unaligned', 'S + Adv -> S', 'unaligned']
- [s3_C_str/report.md](outputs/v03/s3_C_str/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 419.0 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 37/134 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4217 + C4063 -> C4217)); relations Nn + VP -> S, VP + Adv -> VP, Vt + Nn -> VP; missed Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP; false ['Det + Nn -> Nn', 'Det + S -> S', 'Vt + Det -> Vt', 'S + Adv -> S']
- [s3_D_strcls/report.md](outputs/v03/s3_D_strcls/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 418.9 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 37/134 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(hears), rule_add(C4217 + C4063 -> C4217)); relations Nn + VP -> S, VP + Adv -> VP, Vt + Nn -> VP; missed Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Vt + NP -> VP; false ['Det + Nn -> Nn', 'Det + S -> S', 'Vt + Det -> Vt', 'S + Adv -> S']
- [s3_F_dist_full/report.md](outputs/v03/s3_F_dist_full/report.md); 3000 sentences, 2314 span types, 44349 contexts, 82332 edges, peak 414.6 MB, 1905 merges, 0 splits; relations recovered: Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn; v0.2: 23/95 moves accepted (merge(C3812 + C4028), merge(C3138 + C4171), split(C4216), rule_prune(13), rule_prune(1), lex_add(the), lex_add(big), rule_add(C4217 + C4063 -> C4217)); relations Adj + Nn -> Nn, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['Det + Adj -> Det', 'unaligned', 'unaligned', 'S + Adv -> S']

## Forced class collapse (random init, K = 8): does class-generalized evidence inherit false merges?

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | const P len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_random_k8 | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 282.270 |
| s4_C_random_k8 | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 242.110 |
| s4_D_random_k8 | 0.943 | 0.924 | 0.916 | 32 | 0.967 | 0.933 | 0.937 | 0.545 | 0.667 | 0.600 | 0.000 | - | 25.443 | 239.630 |

v0.2 posterior on the same runs:

| run | class pair F1 | class ARI | occ. class F1 | #classes | const F1 | const F1 len>=2 | comp P | comp R | comp F1 | amb F1 | amb occ acc | bits/sent (ELBO) | #rules | moves acc/prop | v0.2 runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_random_k8 | 0.753 | 0.691 | 0.667 | 231 | 0.916 | 0.807 | 0.261 | 0.667 | 0.375 | 0.000 | - | 28.555 | 25 | 39/165 | 281.390 |
| s4_C_random_k8 | 0.606 | 0.516 | 0.692 | 286 | 0.931 | 0.842 | 0.300 | 1.000 | 0.462 | 0.000 | - | 29.793 | 38 | 34/154 | 240.940 |
| s4_D_random_k8 | 0.606 | 0.516 | 0.692 | 286 | 0.931 | 0.842 | 0.300 | 1.000 | 0.462 | 0.000 | - | 29.793 | 38 | 34/154 | 238.690 |

Identity-preserving (modifier) evidence vs gold (candidate precision at support >= 2, constituent candidates; occurrence-level precision in brackets) and lexicon expansion:

| run | ROOT P/R | CONST P/R | CLASS P/R | RULE P/R | COMBINED P/R | root LEFT/RIGHT P | dir acc | host acc (class) | ins P/R (root) | lex_add acc/prop | false lex | true amb lex | rejected by penalty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_B_random_k8 | 0.652/0.833 [0.458] | 0.562/1.000 [0.570] | 0.750/0.667 [0.635] | 0.667/0.667 [0.705] | 0.486/1.000 [0.624] | 0.196/0.385 | 0.778 | 0.417 | 0.524/0.611 | 7/42 | 1 | 0 | 0 |
| s4_C_random_k8 | 0.652/0.833 [0.458] | 0.367/1.000 [0.714] | 0.288/0.944 [0.724] | 0.210/0.889 [0.529] | 0.200/1.000 [0.616] | 0.196/0.385 | 0.944 | 0.647 | 0.524/0.611 | 5/38 | 1 | 0 | 0 |
| s4_D_random_k8 | 0.652/0.833 [0.458] | 0.367/1.000 [0.714] | 0.288/0.944 [0.724] | 0.210/0.889 [0.529] | 0.200/1.000 [0.616] | 0.196/0.385 | 0.944 | 0.647 | 0.524/0.611 | 5/38 | 1 | 0 | 0 |

- [s4_B_random_k8/report.md](outputs/v03/s4_B_random_k8/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 433.7 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 39/165 moves accepted (merge(C1 + C7), merge(C4 + C1+C7), split(C3), rule_prune(110), rule_prune(4), lex_add(the), lex_add(a), merge(C6 + C4+C1+C7)); relations AdvS + S -> S, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + Nn -> VP; missed Adj + Nn -> Nn, Deg + Vt -> Vt, Vt + NP -> VP; false ['Vt + Adv -> VP', 'Det + Nn -> Nn', 'Det + Nn -> Adv', 'unaligned']
- [s4_C_random_k8/report.md](outputs/v03/s4_C_random_k8/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 453.5 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 34/154 moves accepted (merge(C1 + C7), merge(C4 + C1+C7), split(C3), rule_prune(110), rule_prune(4), lex_add(the), lex_add(a), merge(C6 + C4+C1+C7)); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['S + Adv -> S', 'Det + Nn -> Adv', 'Det + Nn -> AdvS', 'Deg + VP -> Vt']
- [s4_D_random_k8/report.md](outputs/v03/s4_D_random_k8/report.md); 3000 sentences, 2768 span types, 49511 contexts, 96393 edges, peak 423.6 MB, 2321 merges, 0 splits; relations recovered: Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; missed: Adj + Nn -> Nn, AdvS + S -> S, Nn + VP -> S; v0.2: 34/154 moves accepted (merge(C1 + C7), merge(C4 + C1+C7), split(C3), rule_prune(110), rule_prune(4), lex_add(the), lex_add(a), merge(C6 + C4+C1+C7)); relations Adj + Nn -> Nn, AdvS + S -> S, Deg + Vt -> Vt, Det + Nn -> NP, NP + VP -> S, Nn + VP -> S, VP + Adv -> VP, Vt + NP -> VP, Vt + Nn -> VP; false ['S + Adv -> S', 'Det + Nn -> Adv', 'Det + Nn -> AdvS', 'Deg + VP -> Vt']

## Key failure cases (from the main runs)

# Natural-language study (TinyStories, raw text only)

No gold syntax exists for this corpus; the numbers below are intrinsic (stability, compression, held-out prediction, productivity, coverage) and are **not** syntax-recovery accuracies. v0.1 numbers come from the v0.1 stage of the same run, v0.2 numbers from the posterior of the v0.2 stage.

## 10000 sentences

- spans: 14437 types / 200510 occurrences; contexts 592551; edges 692957; merges 7421; accepted classes 1183; relations 113; splits 0; runtime 722.66 s; peak memory 1678.7 MB
- MDL: 130.658 bits/sentence (grammar 46166.0 bits), fully derived sentences 1/2000 (MDL sample)
- productivity: 113 relations, lexical support mean 5.94 / median 4.0 / max 22; 16 relations with >= 10 distinct lexical realisations
- held-out context prediction (mean log2 P of k=1 context over 27227 held-out occurrences): class model -14.761, type model -14.511, background -14.358; rare types (freq<10, n=3016): class -13.888 vs type -13.02; contexts unseen for the type but seen for its class: 0.0616

### v0.2 posterior (same corpus)

- coverage: fully parsed (single root) 0.0023, mean chunks/sentence 6.0461, unknown-token mass 17409
- complexity: 295 active classes, 437 active rules, 868 lexical entries, 117.256 bits/sentence (-ELBO/N); pruning {'cells_pruned': 30718, 'mean_retained_mass': 0.9997697368954361, 'max_dropped_mass': 0.06357327051543338, 'n_cells': 612649}
- structure moves: 26/86 accepted (merge C21477 + C21818, merge C21713 + C21477+C21818, split C21748, split C21666, rule_prune , lex_prune , lex_add It, lex_add wanted); runtime {'init': 0.1, 'potentials': 40.5, 'cavi_init': 25.1, 'structure_learning': 534.8, 'decoding': 8.6, 'modifier_evidence': 6.8}
- stability across disjoint halves (v0.2): class pairwise F1 0.2414 / ARI 0.2064 on 1553 shared classified types; constituent-set agreement 0.6553; rule agreement (Jaccard after class alignment) 0.0662 (374 vs 344 rules); ambiguity agreement 0.2058
- occurrence ambiguity: 353 span types with a second class above the mass threshold; highest posterior entropy (not a linguistic claim): `scream` (5 occ, H=2.7555, {'C21413': 0.183, 'C21608': 0.162, 'C21558': 0.13, 'C21739': 0.08}); `talk` (23 occ, H=2.7181, {'C21556': 0.236, 'C18998': 0.184, 'C21505': 0.105, 'C19969': 0.049}); `turn` (9 occ, H=2.6859, {'C21556': 0.162, 'C21817+C21530+C21713+C21477+C21818': 0.139, 'C19969': 0.121, 'C21505': 0.12}); `circle` (5 occ, H=2.6397, {'C21817+C21530+C21713+C21477+C21818': 0.184, 'C21130': 0.174, 'C21556': 0.148, 'C21502': 0.109}); `smiling` (7 occ, H=2.636, {'C21556': 0.209, 'C21381': 0.145, 'C21455+C21642': 0.135, 'C19247': 0.128}); `treasure` (17 occ, H=2.6196, {'C21556': 0.234, 'C21502': 0.146, 'C21817+C21530+C21713+C21477+C21818': 0.097, 'C21321': 0.095}); `selfish` (9 occ, H=2.6175, {'C21556': 0.229, 'C21786': 0.195, 'C21664': 0.116, 'C21455+C21642': 0.104}); `mean` (33 occ, H=2.5983, {'C21556': 0.253, 'C20616': 0.184, 'C21455+C21642': 0.125, 'C21664': 0.083})
- identity-preserving (modifier) candidate agreement across disjoint halves vs general rule agreement (rule Jaccard 0.0662): ROOT_STRING: candidates 0.0 (0 shared of 0/0), direction None, host None; CONSTITUENT_STRING: candidates 0.24 (12 shared of 35/27), direction 0.5833, host 0.0833; CLASS_GENERALIZED: candidates 0.137 (10 shared of 54/29), direction 0.6, host 0.1; POSTERIOR_RULE: candidates 0.1017 (6 shared of 46/19), direction 1.0, host 0.1667; COMBINED: candidates 0.1481 (12 shared of 61/32), direction 0.5833, host 0.0833

Top identity-preserving candidates (no gold; strings, not linguistic claims):

- **ROOT_STRING/RIGHT**: 
- **ROOT_STRING/LEFT**: 
- **ROOT_STRING/INTERNAL**: 
- **CONSTITUENT_STRING**: `that` RIGHT of C21827 (cand. C20528, support 44.0, 25 hosts, mod 0.7179; e.g. They pretend  ->  They pretend that); `big` RIGHT of C21811 (cand. C21625+C21639, support 65.0, 24 hosts, mod 1.0; e.g. Once upon a time , there was a  ->  Once upon a time , there was a big); `little` LEFT of C21780 (cand. C21048, support 216.0, 17 hosts, mod 1.0; e.g. bird  ->  little bird); `so` INTERNAL of C21768 (cand. C21121, support 34.0, 15 hosts, mod None; e.g. Timmy was happy  ->  Timmy was so happy); `very` LEFT of C21833 (cand. C21121, support 23.0, 15 hosts, mod 1.0; e.g. happy .  ->  very happy .); `little` INTERNAL of C21851 (cand. C21048, support 68.0, 12 hosts, mod None; e.g. The girl  ->  The little girl); `so` RIGHT of C21854 (cand. C21121, support 27.0, 12 hosts, mod 1.0; e.g. Timmy was  ->  Timmy was so); `very` INTERNAL of C21768 (cand. C21121, support 16.0, 12 hosts, mod None; e.g. They were excited  ->  They were very excited)
- **CLASS_GENERALIZED**: `little` LEFT of C21780 (cand. C21048, support 225.0, 24 hosts, mod 1.0; e.g. bird  ->  little bird); `very` LEFT of C21833 (cand. C21121, support 23.0, 15 hosts, mod 1.0; e.g. happy .  ->  very happy .); `big` LEFT of C21846 (cand. C21625+C21639, support 28.0, 14 hosts, mod 1.0; e.g. tree  ->  big tree); `little` INTERNAL of C21851 (cand. C21048, support 67.0, 11 hosts, mod None; e.g. The girl  ->  The little girl); `so` INTERNAL of C21768 (cand. C21121, support 27.0, 11 hosts, mod None; e.g. Timmy was happy  ->  Timmy was so happy); `big` INTERNAL of C21793 (cand. C21625+C21639, support 25.0, 11 hosts, mod None; e.g. a truck  ->  a big truck); `very` INTERNAL of C21785 (cand. C21121, support 17.0, 10 hosts, mod None; e.g. was curious .  ->  was very curious .); `that` RIGHT of C21827 (cand. C20528, support 13.0, 10 hosts, mod 0.7179; e.g. They learned  ->  They learned that)
- **POSTERIOR_RULE**: `so` RIGHT of C21854 (cand. C21121, support 20.763, 87 hosts, mod 1.0; e.g. She was  ->  She was so); `very` RIGHT of C21854 (cand. C21121, support 10.899, 80 hosts, mod 1.0; e.g. It felt  ->  It felt very); `that` RIGHT of C21827 (cand. C20528, support 44.468, 78 hosts, mod 0.7179; e.g. They pretend  ->  They pretend that); `big` RIGHT of C21811 (cand. C21625+C21639, support 38.495, 73 hosts, mod 1.0; e.g. Once upon a time , there was a  ->  Once upon a time , there was a big); `very` LEFT of C21833 (cand. C21121, support 42.202, 63 hosts, mod 1.0; e.g. hungry .  ->  very hungry .); `big` LEFT of C21846 (cand. C21625+C21639, support 37.521, 56 hosts, mod 1.0; e.g. sign  ->  big sign); `little` LEFT of C21780 (cand. C21048, support 218.128, 45 hosts, mod 1.0; e.g. bird  ->  little bird); `so` LEFT of C21833 (cand. C21121, support 15.365, 34 hosts, mod 1.0; e.g. delicious .  ->  so delicious .)

Top v0.2 classes (posterior view):

- C21856 (mass 10418.89): lexical ['. (9590.0)', '! (747.0)']; spans ['. (1.00 x9590)', '! (1.00 x747)', 'with her family . (0.94 x4)', 'with his owner . (0.93 x4)']; contexts ['it _ <EOS> (226.9)', 'said _ <EOS> (188.0)']; composition in ['C19043 + C21802 -> C21856 (51.3)', 'C17001 + C21852 -> C21856 (30.6)']
- C21556 (mass 8449.35): lexical ['<unk> (7651.9)', 'meowed (3.0)']; spans ['as (0.83 x89)', 'always (0.80 x93)', 'old (0.73 x95)', 'my (0.94 x71)']; contexts ['" _ you (43.0)', '<BOS> _ that (26.7)']; composition in ['C21556 + C21556 -> C21556 (794.5)']
- C21851 (mass 5852.62): lexical ['He (1330.0)', 'She (1286.0)', 'They (1173.0)', 'It (220.4)', 'Timmy (104.3)', '<unk> (97.1)', 'John (68.0)', 'Bob (47.0)', 'Everyone (35.0)', 'Molly (29.0)']; spans ['He (1.00 x1330)', 'She (1.00 x1286)', 'They (1.00 x1173)', 'Timmy (0.70 x149)']; contexts ['<BOS> _ was (660.0)', '<BOS> _ had (259.2)']; composition in ['C21616 + C21780 -> C21851 (414.5)', 'C21796 + C21307 -> C21851 (212.0)']
- C21821 (mass 4542.01): lexical ['and (4235.0)', 'or (34.0)']; spans ['and (1.00 x4235)', ', and (1.00 x158)', 'and he (0.93 x90)', 'and they (0.91 x59)']; contexts ['Lily _ Ben (31.0)', 'Anna _ Ben (27.0)']; composition in ['C21821 + C20375 -> C21821 (138.7)', 'C19875 + C21821 -> C21821 (119.0)']
- C21385 (mass 3937.29): lexical ['the (3841.0)', 'this (73.0)', '<unk> (3.4)']; spans ['the (1.00 x3841)', 'this (1.00 x73)', 'the blue (1.00 x9)', 'the shiny (1.00 x5)']; contexts ['in _ park (59.0)', 'to _ park (47.0)']; composition in ['C21385 + C21508 -> C21385 (19.9)']
- C19875 (mass 3290.7): lexical [', (3264.0)', ': (22.0)', '<unk> (4.7)']; spans [', (1.00 x3264)', ': (1.00 x22)']; contexts ['said _ " (209.0)', 'time _ there (124.0)']; composition in []
- C21564 (mass 2919.68): lexical ['to (2869.0)']; spans ['to (1.00 x2869)', 'back to (1.00 x62)']; contexts ['went _ the (54.8)', 'loved _ play (54.0)']; composition in ['C20666 + C21564 -> C21564 (50.7)']
- C21814 (mass 2861.52): lexical ['a (2609.0)', '<unk> (2.1)']; spans ['a (1.00 x2609)', 'a beautiful (1.00 x30)', 'a nice (1.00 x15)', 'a bad (1.00 x15)']; contexts ['upon _ time (195.0)', 'was _ little (76.9)']; composition in ['C21814 + C21776 -> C21814 (76.9)', 'C21814 + C21048 -> C21814 (72.9)']
- C21817+C21530+C21713+C21477+C21818 (mass 2655.12): lexical ['<unk> (1540.8)', 'car (110.0)', 'water (81.0)', 'tree (80.2)', 'park (60.2)', 'ground (57.0)', 'slide (44.0)', 'door (44.0)', 'floor (31.0)', 'sky (29.9)']; spans ['car (1.00 x110)', 'water (1.00 x81)', 'tree (0.67 x120)', 'ground (1.00 x57)']; contexts ['the _ . (667.5)', 'the _ and (306.1)']; composition in []
- C21849 (mass 2549.44): lexical ['Lily (570.0)', 'Ben (272.0)', 'Tom (254.0)', 'Sam (138.0)', 'Tim (137.0)', '<unk> (130.1)', 'Mom (123.0)', 'Anna (120.0)', 'Max (102.0)', 'Sara (81.0)']; spans ['Lily (1.00 x570)', 'Ben (1.00 x272)', 'Tom (1.00 x254)', 'Sam (1.00 x138)']; contexts ['<BOS> _ and (170.0)', '<BOS> _ was (138.0)']; composition in ['C21385 + C21502 -> C21849 (85.6)']
- C21790 (mass 2323.12): lexical ['was (2271.0)', '<unk> (11.1)']; spans ['was (1.00 x2271)', 'was feeling (1.00 x18)', 'was once (0.61 x3)']; contexts ['there _ a (221.7)', 'It _ a (75.8)']; composition in ['C21790 + C19123 -> C21790 (41.0)']
- C21736 (mass 2243.82): lexical ['her (1006.0)', 'his (705.0)', 'their (295.0)', 'your (93.0)', 'its (46.0)', '<unk> (5.0)']; spans ['her (1.00 x1006)', 'his (1.00 x705)', 'their (1.00 x295)', 'your (1.00 x93)']; contexts ['with _ friends (35.0)', 'with _ toys (32.0)']; composition in ['C19366 + C21385 -> C21736 (47.6)', 'C21736 + C21625+C21639 -> C21736 (46.2)']

Top inferred classes (human inspection):

- C21856 (7 members, 10353 occ): lexical ['.', '!']; multi-token ['all day .', 'with her mom and dad .', 'and the slide .', 'with his family .', 'with her family .']; contexts ['[k=1] it _ <EOS>', '[k=1] said _ <EOS>', '[k=1] him _ <EOS>']
- C21851 (93 members, 4609 occ): lexical ['He', 'They', 'She', 'John', 'Tommy', 'Everyone', 'Charlie', 'Bob', 'Sally', 'Toby', 'Jimmy', 'Molly']; multi-token ['They all', 'His mom', 'Then she', 'The bird', 'the bear', 'The boy']; contexts ['[k=1] <BOS> _ was', '[k=1] <BOS> _ had', '[k=1] <BOS> _ said']
- C21821 (11 members, 4497 occ): lexical ['and', 'or']; multi-token [', and', 'and she', ', and Tim', 'back and', 'and Mia', 'and Lily']; contexts ['[k=1] Lily _ Ben', '[k=1] Anna _ Ben', '[k=1] Tom _ Lily']
- C21385 (6 members, 3930 occ): lexical ['the', 'this']; multi-token ['the blue', 'the new', 'the soft', 'some too']; contexts ['[k=1] in _ park', '[k=1] to _ park', '[k=1] on _ ground']
- C19875 (3 members, 3290 occ): lexical [',', ':']; multi-token [', a little boy named']; contexts ['[k=1] said _ "', '[k=1] time _ there', '[k=2] a time _ there was']
- C21564 (5 members, 2941 occ): lexical ['to']; multi-token ['to swim and', 'back to', 'to have some', 'to go outside and']; contexts ['[k=1] went _ the', '[k=1] loved _ play', '[k=1] liked _ play']
- C21814 (24 members, 2845 occ): lexical ['a']; multi-token ['the little', 'a little', 'a nice', 'an old', 'a creative', 'a deep']; contexts ['[k=1] upon _ time', '[k=2] Once upon _ time ,', '[k=3] <BOS> Once upon _ time , there']
- C21849 (40 members, 2485 occ): lexical ['Tim', 'Tom', 'Ben', 'Lily', 'Lucy', 'Sam', 'Timmy', 'Anna', 'Lila', 'Sara', 'Sue', 'Max']; multi-token ['the fireman', 'the monkey', 'the little girl', 'the squirrel', 'play in the forest', 'the little boy']; contexts ['[k=1] <BOS> _ and', '[k=1] <BOS> _ was', '[k=1] " _ said']
- C21790 (6 members, 2310 occ): lexical ['was']; multi-token ['was only', 'was feeling', 'was once', 'must be', 'was blue and']; contexts ['[k=1] there _ a', '[k=1] It _ a', '[k=1] She _ so']
- C21736 (11 members, 2235 occ): lexical ['his', 'her', 'their', 'your', 'its']; multi-token ['in the air', 'all the', 'the time', 'his new', 'her new', 'his special']; contexts ['[k=1] with _ friends', '[k=1] with _ toys', '[k=1] to _ mom']
- C20375 (4 members, 2075 occ): lexical ['they', 'he', 'she']; multi-token ['the family']; contexts ['[k=1] , _ saw', '[k=1] " _ said', '[k=1] , _ found']
- C21169 (12 members, 1876 occ): lexical ['them', 'it', 'him', 'scissors']; multi-token ['the problem', 'the bird', 'the bunny', 'the bike', 'the cat', 'the dragon']; contexts ['[k=1] thought _ was', '[k=1] with _ .', '[k=1] on _ .']

Top relations A + B -> C:




## 25000 sentences

- spans: 36595 types / 571005 occurrences; contexts 1551102; edges 1905314; merges 21054; accepted classes 1766; relations 152; splits 0; runtime 1905.51 s; peak memory 2808.5 MB
- MDL: 141.099 bits/sentence (grammar 68012.0 bits), fully derived sentences 0/2000 (MDL sample)
- productivity: 152 relations, lexical support mean 7.07 / median 4.0 / max 66; 22 relations with >= 10 distinct lexical realisations
- held-out context prediction (mean log2 P of k=1 context over 27276 held-out occurrences): class model -14.106, type model -13.998, background -14.647; rare types (freq<10, n=1953): class -13.782 vs type -12.945; contexts unseen for the type but seen for its class: 0.0599

### v0.2 posterior (same corpus)

- coverage: fully parsed (single root) 0.0, mean chunks/sentence 6.4198, unknown-token mass 47029
- complexity: 293 active classes, 342 active rules, 979 lexical entries, 114.704 bits/sentence (-ELBO/N); pruning {'cells_pruned': 67031, 'mean_retained_mass': 0.9995565474898798, 'max_dropped_mass': 0.09055089855296994, 'n_cells': 1539489}
- structure moves: 26/74 accepted (merge C56241 + C56990, merge C56563 + C56891, rule_prune , rule_prune , lex_add They, lex_add She, rule_add C56938 + C57264 -> C56938, rule_add C57429 + C48938 -> C48938); runtime {'init': 0.3, 'potentials': 94.2, 'cavi_init': 62.0, 'structure_learning': 1115.0, 'decoding': 17.4, 'modifier_evidence': 20.5}
- stability across disjoint halves (v0.2): class pairwise F1 0.3598 / ARI 0.2902 on 2899 shared classified types; constituent-set agreement 0.5903; rule agreement (Jaccard after class alignment) 0.127 (375 vs 354 rules); ambiguity agreement 0.1493
- occurrence ambiguity: 841 span types with a second class above the mass threshold; highest posterior entropy (not a linguistic claim): `lost` (55 occ, H=2.8912, {'C48938': 0.294, 'C51310': 0.107, 'C57575': 0.093, 'C57632': 0.054}); `pretended` (24 occ, H=2.8803, {'C57607': 0.156, 'C57141': 0.142, 'C48938': 0.124, 'C53795': 0.087}); `close` (42 occ, H=2.759, {'C48938': 0.342, 'C48776': 0.087, 'C57575': 0.086, 'C55441': 0.04}); `hug` (112 occ, H=2.7523, {'C57398+C56241+C56990': 0.248, 'C48938': 0.211, 'C51310': 0.079, 'C57642': 0.059}); `explored` (13 occ, H=2.7435, {'C48938': 0.215, 'C55711': 0.142, 'C57085': 0.089, 'C54352': 0.079}); `honey` (23 occ, H=2.7395, {'C48938': 0.324, 'C57484': 0.083, 'C57314': 0.069, 'C52744': 0.061}); `walk` (98 occ, H=2.7275, {'C48938': 0.241, 'C57398+C56241+C56990': 0.194, 'C57579+C57597': 0.116, 'C57642': 0.063}); `wave` (28 occ, H=2.7184, {'C48938': 0.283, 'C57642': 0.097, 'C57579+C57597': 0.079, 'C54352': 0.074})
- identity-preserving (modifier) candidate agreement across disjoint halves vs general rule agreement (rule Jaccard 0.127): ROOT_STRING: candidates 0.0 (0 shared of 0/0), direction None, host None; CONSTITUENT_STRING: candidates 0.1415 (15 shared of 63/58), direction 0.7333, host 0.4; CLASS_GENERALIZED: candidates 0.1284 (14 shared of 66/57), direction 0.7857, host 0.5; POSTERIOR_RULE: candidates 0.1532 (17 shared of 72/56), direction 0.9412, host 0.6471; COMBINED: candidates 0.1348 (19 shared of 86/74), direction 0.8421, host 0.5789

Top identity-preserving candidates (no gold; strings, not linguistic claims):

- **ROOT_STRING/RIGHT**: 
- **ROOT_STRING/LEFT**: 
- **ROOT_STRING/INTERNAL**: `so` INTERNAL of ROOT (cand. C52468, support 4.0, 4 hosts, mod None; e.g. It was happy  ->  It was so happy); `very` INTERNAL of ROOT (cand. C52468, support 4.0, 4 hosts, mod None; e.g. She was enthusiastic  ->  She was very enthusiastic); `little` INTERNAL of ROOT (cand. C54326, support 3.0, 3 hosts, mod None; e.g. The girl was very scared  ->  The little girl was very scared)
- **CONSTITUENT_STRING**: `very` INTERNAL of C57635 (cand. C52468, support 109.0, 58 hosts, mod None; e.g. She was excited  ->  She was very excited); `big` RIGHT of C56938 (cand. C57264, support 118.0, 50 hosts, mod 1.0; e.g. Once upon a time , there was a  ->  Once upon a time , there was a big); `so` INTERNAL of C57635 (cand. C52468, support 169.0, 48 hosts, mod None; e.g. She was happy  ->  She was so happy); `very` LEFT of C57616 (cand. C52468, support 94.0, 47 hosts, mod 1.0; e.g. happy .  ->  very happy .); `little` LEFT of C57560 (cand. C54326, support 544.0, 33 hosts, mod 1.0; e.g. bird  ->  little bird); `big` LEFT of C57641 (cand. C57264, support 34.0, 30 hosts, mod 1.0; e.g. loops .  ->  big loops .); `One day ,` LEFT of C57079 (cand. C57306, support 57.0, 27 hosts, mod 1.0; e.g. Lily went to  ->  One day , Lily went to); `big ,` RIGHT of C56938 (cand. C57264, support 39.0, 24 hosts, mod 1.0; e.g. One day , she saw a  ->  One day , she saw a big ,)
- **CLASS_GENERALIZED**: `little` LEFT of C57560 (cand. C54326, support 598.0, 74 hosts, mod 1.0; e.g. bird  ->  little bird); `very` INTERNAL of C57635 (cand. C52468, support 117.0, 63 hosts, mod None; e.g. The little girl was scared  ->  The little girl was very scared); `so` INTERNAL of C57635 (cand. C52468, support 180.0, 55 hosts, mod None; e.g. She was happy  ->  She was so happy); `big` RIGHT of C56938 (cand. C57264, support 67.0, 43 hosts, mod 1.0; e.g. One day , they saw a  ->  One day , they saw a big); `very` LEFT of C57616 (cand. C52468, support 84.0, 42 hosts, mod 1.0; e.g. happy .  ->  very happy .); `big` LEFT of C57641 (cand. C57264, support 32.0, 28 hosts, mod 1.0; e.g. boys .  ->  big boys .); `One day ,` LEFT of C57079 (cand. C57306, support 57.0, 27 hosts, mod 1.0; e.g. Lily went to  ->  One day , Lily went to); `little` INTERNAL of C57609 (cand. C54326, support 113.0, 24 hosts, mod None; e.g. Once upon a time there was a girl  ->  Once upon a time there was a little girl)
- **POSTERIOR_RULE**: `big` LEFT of C57641 (cand. C57264, support 64.063, 168 hosts, mod 1.0; e.g. park .  ->  big park .); `very` LEFT of C57616 (cand. C52468, support 109.032, 156 hosts, mod 1.0; e.g. hungry .  ->  very hungry .); `little` LEFT of C57560 (cand. C54326, support 590.002, 120 hosts, mod 1.0; e.g. bird  ->  little bird); `so` LEFT of C57616 (cand. C52468, support 56.541, 117 hosts, mod 1.0; e.g. delicious .  ->  so delicious .); `that` RIGHT of C57141 (cand. C52595, support 134.613, 114 hosts, mod 1.0; e.g. After  ->  After that); `big` RIGHT of C56938 (cand. C57264, support 147.37, 112 hosts, mod 1.0; e.g. Once upon a time , there was a  ->  Once upon a time , there was a big); `kept` LEFT of C48938 (cand. C48938, support 24.663, 83 hosts, mod 1.0; e.g. telling  ->  kept telling); `down` RIGHT of C48938 (cand. C48938, support 43.065, 78 hosts, mod 1.0; e.g. raindrops zigzag  ->  raindrops zigzag down)

Top v0.2 classes (posterior view):

- C56210 (mass 28033.9): lexical ['. (24056.0)', '! (1851.0)']; spans ['. (1.00 x24056)', '! (1.00 x1851)', 'again . (1.00 x263)', 'in the park . (0.95 x91)']; contexts ['it _ <EOS> (652.6)', 'said _ <EOS> (477.6)']; composition in ['C55195+C56999 + C56210 -> C56210 (1813.2)', 'C48938 + C56210 -> C56210 (313.7)']
- C48938 (mass 21978.24): lexical ['<unk> (18424.2)', 'Put (3.0)']; spans ['down (0.56 x315)', 'how (0.71 x207)', 'old (0.63 x220)', 'no (0.77 x164)']; contexts ['<BOS> _ that (81.8)', '<BOS> _ to (67.8)']; composition in ['C48938 + C48938 -> C48938 (2229.3)', 'C48938 + C56210 -> C48938 (1037.7)']
- C57647 (mass 16205.41): lexical ['He (2480.5)', 'She (2281.2)', 'Lily (1408.0)', 'They (1157.5)', 'Tom (679.0)', 'Ben (675.0)', 'It (521.0)', 'Timmy (382.0)', '<unk> (359.0)', 'Sam (340.0)']; spans ['He (0.73 x3420)', 'She (0.69 x3289)', 'Lily (1.00 x1408)', 'Tom (1.00 x679)']; contexts ['<BOS> _ was (2143.3)', '<BOS> _ and (531.4)']; composition in ['C57115 + C57560 -> C57647 (1098.8)', 'C55976 + C57560 -> C57647 (584.8)']
- C57424 (mass 11100.2): lexical ['and (10710.0)', 'or (94.0)']; spans ['and (1.00 x10710)', 'and he (0.75 x229)', 'and they (0.77 x163)', 'and she (0.73 x130)']; contexts ['Lily _ Ben (76.0)', 'mom _ dad (67.0)']; composition in ['C57424 + C53850 -> C57424 (266.3)', 'C48776 + C57424 -> C57424 (29.9)']
- C55976 (mass 9986.47): lexical ['the (9747.0)', '<unk> (6.0)']; spans ['the (1.00 x9747)', 'the big (1.00 x80)', 'the broken (1.00 x16)', 'the living (0.61 x25)']; contexts ['in _ park (126.0)', 'to _ park (111.0)']; composition in ['C55976 + C57264 -> C55976 (154.2)', 'C55976 + C55099+C57493 -> C55976 (79.3)']
- C57648 (mass 8225.51): lexical [', (8171.0)', ': (50.0)', '<unk> (4.5)']; spans [', (1.00 x8171)', ': (1.00 x50)']; contexts ['said _ " (512.6)', 'time _ there (283.0)']; composition in []
- C43167+C56563+C56891 (mass 7843.71): lexical ['was (5728.0)', 'is (1059.0)', 'felt (571.0)', 'got (334.0)', '<unk> (128.7)', 'returned (10.0)', 'gets (9.0)', 'splashes (4.0)']; spans ['was (1.00 x5728)', 'is (1.00 x1059)', 'felt (1.00 x571)', 'got (1.00 x334)']; contexts ['there _ a (537.0)', 'It _ a (211.0)']; composition in []
- C57066 (mass 7206.9): lexical ['to (7161.0)', '<unk> (8.9)']; spans ['to (1.00 x7161)', 'to sing and (1.00 x9)']; contexts ['went _ the (151.0)', 'loved _ play (115.1)']; composition in ['C57545 + C57424 -> C57066 (37.0)']
- C57606 (mass 6838.76): lexical ['a (6423.0)']; spans ['a (1.00 x6423)', 'a pretty (1.00 x25)', 'a happy (1.00 x23)', 'a deep (1.00 x15)']; contexts ['upon _ time (454.0)', 'saw _ big (171.0)']; composition in ['C57606 + C55099+C57493 -> C57606 (138.8)', 'C57606 + C54326 -> C57606 (122.0)']
- C56495 (mass 5516.85): lexical ['her (2592.0)', 'his (1718.0)', 'their (780.0)', 'your (210.0)', 'its (110.0)', '<unk> (9.9)']; spans ['her (1.00 x2592)', 'his (1.00 x1718)', 'their (1.00 x780)', 'your (1.00 x210)']; contexts ['with _ toys (77.0)', 'to _ mom (69.0)']; composition in ['C56002 + C55976 -> C56495 (97.0)']
- C53850 (mass 5259.72): lexical ['he (2074.0)', 'she (1714.0)', 'they (1349.0)', '<unk> (62.5)']; spans ['he (1.00 x2074)', 'she (1.00 x1714)', 'they (1.00 x1349)', 'they all (1.00 x39)']; contexts [', _ saw (183.8)', '" _ said (117.0)']; composition in ['C53850 + C56002 -> C53850 (60.2)']
- C57560 (mass 4935.97): lexical ['<unk> (953.2)', 'girl (516.0)', 'boy (366.0)', 'bird (341.0)', 'dog (303.0)', 'man (287.0)', 'bear (178.0)', 'cat (146.0)', 'fish (97.0)', 'mouse (87.0)']; spans ['girl (1.00 x516)', 'boy (1.00 x366)', 'bird (1.00 x341)', 'dog (1.00 x303)']; contexts ['The _ was (307.1)', 'the _ . (248.2)']; composition in ['C54326 + C57560 -> C57560 (876.7)']

Top inferred classes (human inspection):

- C56210 (14 members, 26244 occ): lexical ['.', '!']; multi-token ['again .', 'every day .', 'and started to cry .', 'again and again .', 'in the sunshine .', 'outside his house .']; contexts ['[k=1] it _ <EOS>', '[k=1] said _ <EOS>', '[k=1] him _ <EOS>']
- C57647 (170 members, 14887 occ): lexical ['He', 'Tim', 'Tom', 'She', 'Ben', 'Lily', 'John', 'Tommy', 'Lucy', 'Sam', 'Annie', 'Timmy']; multi-token ['He tried and', 'The bunny', 'His dad', 'The bird', 'The boy', 'The squirrel']; contexts ['[k=1] <BOS> _ was', '[k=1] <BOS> _ and', '[k=2] <BOS> <BOS> _ was so']
- C57424 (8 members, 11164 occ): lexical ['and', 'or']; multi-token ['and they', 'and she', 'back and', 'and they all', 'brightly and', ', and the bird']; contexts ['[k=1] Lily _ Ben', '[k=1] mom _ dad', '[k=1] smiled _ said']
- C55976 (8 members, 9855 occ): lexical ['the']; multi-token ['the big', 'the new', 'the dirty', 'the big ,', 'the small', 'the different']; contexts ['[k=1] in _ park', '[k=1] to _ park', '[k=1] on _ ground']
- C57648 (8 members, 8242 occ): lexical [',', ':']; multi-token [', while playing ,', ', Tim met', 'to Tom ,', 'to his mom ,', 'to Lily ,', ', bang ,']; contexts ['[k=1] said _ "', '[k=1] time _ there', '[k=2] a time _ there was']
- C57066 (8 members, 7211 occ): lexical ['to']; multi-token ['close to', 'safe in', 'for them to', 'to go outside and', 'to sing and', 'to climb the tree and']; contexts ['[k=1] went _ the', '[k=1] loved _ play', '[k=1] liked _ play']
- C57606 (31 members, 6938 occ): lexical ['a']; multi-token ['a little', 'a green', 'a cheap', 'a lovely', 'a pretty', 'a grumpy']; contexts ['[k=1] upon _ time', '[k=2] Once upon _ time ,', '[k=3] <BOS> Once upon _ time , there']
- C43167 (2 members, 6787 occ): lexical ['was', 'is']; multi-token []; contexts ['[k=1] there _ a', '[k=1] It _ a', '[k=1] He _ so']
- C56495 (8 members, 5558 occ): lexical ['his', 'her', 'their', 'your', 'its']; multi-token ['all the', 'all of her', 'her shoes and']; contexts ['[k=1] with _ toys', '[k=1] to _ mom', '[k=1] with _ friends']
- C53850 (5 members, 5176 occ): lexical ['they', 'he', 'she']; multi-token ['they all', 'they both']; contexts ['[k=1] , _ saw', '[k=1] " _ said', '[k=1] , _ found']
- C57564 (20 members, 4580 occ): lexical ['"']; multi-token ['He says , "', 'She said , "', 'He said , "', 'Lily said , "', 'She says , "', 'Timmy said , "']; contexts ['[k=1] <BOS> _ I', '[k=1] , _ said', '[k=1] <BOS> _ You']
- C54335 (3 members, 4405 occ): lexical ['them', 'it', 'him']; multi-token []; contexts ['[k=1] with _ .', '[k=1] thought _ was', '[k=1] on _ .']

Top relations A + B -> C:



## 50000 sentences

- spans: 73347 types / 1246340 occurrences; contexts 3143204; edges 4032281; merges 45898; accepted classes 2278; relations 168; splits 0; runtime 9871.74 s; peak memory 4573.2 MB
- MDL: 148.414 bits/sentence (grammar 84786.0 bits), fully derived sentences 27/2000 (MDL sample)
- productivity: 168 relations, lexical support mean 7.78 / median 5.0 / max 59; 32 relations with >= 10 distinct lexical realisations
- held-out context prediction (mean log2 P of k=1 context over 27723 held-out occurrences): class model -13.188, type model -13.248, background -14.73; rare types (freq<10, n=1367): class -12.658 vs type -12.091; contexts unseen for the type but seen for its class: 0.0619

### v0.2 posterior (same corpus)

- coverage: fully parsed (single root) 0.0217, mean chunks/sentence 6.5003, unknown-token mass 91908
- complexity: 299 active classes, 353 active rules, 1213 lexical entries, 107.867 bits/sentence (-ELBO/N); pruning {'cells_pruned': 128301, 'mean_retained_mass': 0.9970776059532744, 'max_dropped_mass': 0.11116206581970145, 'n_cells': 3079536}
- structure moves: 36/79 accepted (merge C111457 + C117922, merge C119152 + C119239, split C116790, split C81100, rule_prune , lex_add They, lex_add She, rule_add C102075 + C115283 -> C102075); runtime {'init': 0.8, 'potentials': 299.2, 'cavi_init': 205.3, 'structure_learning': 3632.6, 'decoding': 41.6, 'modifier_evidence': 98.5}
- stability across disjoint halves (v0.2): class pairwise F1 0.4172 / ARI 0.3354 on 4407 shared classified types; constituent-set agreement 0.5313; rule agreement (Jaccard after class alignment) 0.1099 (333 vs 337 rules); ambiguity agreement 0.1659
- occurrence ambiguity: 1164 span types with a second class above the mass threshold; highest posterior entropy (not a linguistic claim): `return` (22 occ, H=3.0092, {'C102075': 0.209, 'C119120': 0.11, 'C107308': 0.092, 'C118861': 0.054}); `honey` (37 occ, H=2.949, {'C102075': 0.257, 'C119152+C119239': 0.091, 'C119151+C119233': 0.089, 'C107308': 0.066}); `scream` (27 occ, H=2.9106, {'C119198': 0.159, 'C102075': 0.159, 'C119219': 0.091, 'C111457+C117922': 0.081}); `exercise` (7 occ, H=2.8979, {'C102075': 0.236, 'C119198': 0.114, 'C111457+C117922': 0.097, 'C107308': 0.081}); `stirred` (25 occ, H=2.8699, {'C102075': 0.239, 'C111743': 0.118, 'C107308': 0.082, 'C119219': 0.073}); `push` (31 occ, H=2.844, {'C102075': 0.204, 'C119190.1': 0.111, 'C118345': 0.091, 'C119120': 0.089}); `bed` (206 occ, H=2.8404, {'C111457+C117922': 0.186, 'C102075': 0.181, 'C119152+C119239': 0.113, 'C107308': 0.062}); `fixed` (38 occ, H=2.8388, {'C102075': 0.288, 'C107308': 0.12, 'C119171': 0.112, 'C114275': 0.056})
- identity-preserving (modifier) candidate agreement across disjoint halves vs general rule agreement (rule Jaccard 0.1099): ROOT_STRING: candidates 0.0 (0 shared of 0/0), direction None, host None; CONSTITUENT_STRING: candidates 0.2409 (33 shared of 100/70), direction 0.7273, host 0.3636; CLASS_GENERALIZED: candidates 0.1564 (33 shared of 171/73), direction 0.697, host 0.3939; POSTERIOR_RULE: candidates 0.1023 (18 shared of 133/61), direction 0.8889, host 0.5; COMBINED: candidates 0.137 (37 shared of 203/104), direction 0.7297, host 0.3784

Top identity-preserving candidates (no gold; strings, not linguistic claims):

- **ROOT_STRING/RIGHT**: `, Tom` RIGHT of ROOT (cand. C115283, support 4.0, 4 hosts, mod None; e.g. He said , " Thank you  ->  He said , " Thank you , Tom); `and excited` RIGHT of ROOT (cand. None, support 4.0, 4 hosts, mod None; e.g. They felt hungry  ->  They felt hungry and excited); `and scared` RIGHT of ROOT (cand. None, support 4.0, 4 hosts, mod None; e.g. The girl was very sad  ->  The girl was very sad and scared); `and smiled` RIGHT of ROOT (cand. None, support 4.0, 4 hosts, mod None; e.g. She bowed  ->  She bowed and smiled); `, mom` RIGHT of ROOT (cand. None, support 3.0, 3 hosts, mod None; e.g. " Of course  ->  " Of course , mom); `, my loves` RIGHT of ROOT (cand. None, support 3.0, 3 hosts, mod None; e.g. " Of course  ->  " Of course , my loves); `again` RIGHT of ROOT (cand. C99919, support 3.0, 3 hosts, mod None; e.g. Everyone was so happy to see it  ->  Everyone was so happy to see it again); `in the park` RIGHT of ROOT (cand. C119179, support 3.0, 3 hosts, mod None; e.g. They had a good day  ->  They had a good day in the park)
- **ROOT_STRING/LEFT**: `She said ,` LEFT of ROOT (cand. None, support 5.0, 5 hosts, mod None; e.g. " I'm sorry , Spot  ->  She said , " I'm sorry , Spot); `She says ,` LEFT of ROOT (cand. None, support 4.0, 4 hosts, mod None; e.g. " You are right  ->  She says , " You are right)
- **ROOT_STRING/INTERNAL**: `very` INTERNAL of ROOT (cand. C116522, support 17.0, 17 hosts, mod None; e.g. She was enthusiastic  ->  She was very enthusiastic); `so` INTERNAL of ROOT (cand. C116522, support 10.0, 10 hosts, mod None; e.g. It was happy  ->  It was so happy); `little` INTERNAL of ROOT (cand. C118263, support 6.0, 6 hosts, mod None; e.g. The girl was very scared  ->  The little girl was very scared); `and run around` INTERNAL of ROOT (cand. None, support 4.0, 4 hosts, mod None; e.g. Max loved to play in the park  ->  Max loved to play and run around in the park); `Lily ,` INTERNAL of ROOT (cand. C116790.1, support 4.0, 3 hosts, mod None; e.g. She said , " Oh , I'm sorry  ->  She said , " Oh , Lily , I'm sorry); `, Lily` INTERNAL of ROOT (cand. C115283, support 3.0, 3 hosts, mod None; e.g. She said , " Oh , I'm sorry  ->  She said , " Oh , Lily , I'm sorry); `all` INTERNAL of ROOT (cand. C107319, support 3.0, 3 hosts, mod None; e.g. They had so much fun together  ->  They all had so much fun together); `always` INTERNAL of ROOT (cand. C107815, support 3.0, 3 hosts, mod None; e.g. It made her happy  ->  It always made her happy)
- **CONSTITUENT_STRING**: `in` LEFT of C119170 (cand. C114365, support 1003.0, 221 hosts, mod 1.0; e.g. the truck .  ->  in the truck .); `very` LEFT of C119222 (cand. C116522, support 458.0, 119 hosts, mod 1.0; e.g. proud .  ->  very proud .); `to` LEFT of C102075 (cand. C110213, support 372.0, 117 hosts, mod None; e.g. paint  ->  to paint); `not` RIGHT of C119235 (cand. C99748, support 190.0, 93 hosts, mod 1.0; e.g. But the chair was  ->  But the chair was not); `very` RIGHT of C119235 (cand. C116522, support 149.0, 92 hosts, mod 1.0; e.g. She was  ->  She was very); `so` LEFT of C119222 (cand. C116522, support 203.0, 79 hosts, mod 1.0; e.g. scared and upset .  ->  so scared and upset .); `so` INTERNAL of C118856 (cand. C116522, support 383.0, 76 hosts, mod None; e.g. She was happy  ->  She was so happy); `so` RIGHT of C119235 (cand. C116522, support 142.0, 76 hosts, mod 1.0; e.g. She was  ->  She was so)
- **CLASS_GENERALIZED**: `in` LEFT of C119170 (cand. C114365, support 595.0, 153 hosts, mod 1.0; e.g. the truck .  ->  in the truck .); `little` LEFT of C119145 (cand. C118263, support 1231.0, 126 hosts, mod 1.0; e.g. bird  ->  little bird); `very` RIGHT of C119235 (cand. C116522, support 176.0, 86 hosts, mod 1.0; e.g. She was  ->  She was very); `not` RIGHT of C119235 (cand. C99748, support 172.0, 84 hosts, mod 1.0; e.g. But the chair was  ->  But the chair was not); `so` INTERNAL of C118856 (cand. C116522, support 383.0, 76 hosts, mod None; e.g. She was happy  ->  She was so happy); `so` RIGHT of C119235 (cand. C116522, support 129.0, 69 hosts, mod 1.0; e.g. She was  ->  She was so); `very` LEFT of C119236 (cand. C116522, support 365.0, 62 hosts, mod 1.0; e.g. happy  ->  very happy); `very` INTERNAL of C118856 (cand. C116522, support 162.0, 60 hosts, mod None; e.g. She was excited  ->  She was very excited)
- **POSTERIOR_RULE**: `so` RIGHT of C119235 (cand. C116522, support 601.546, 476 hosts, mod 1.0; e.g. She was  ->  She was so); `in` LEFT of C119170 (cand. C114365, support 961.816, 455 hosts, mod 1.0; e.g. the truck .  ->  in the truck .); `very` RIGHT of C119235 (cand. C116522, support 512.518, 443 hosts, mod 1.0; e.g. Sam is  ->  Sam is very); `very` LEFT of C119222 (cand. C116522, support 53.135, 292 hosts, mod 1.0; e.g. hungry .  ->  very hungry .); `very` LEFT of C119236 (cand. C116522, support 924.401, 229 hosts, mod 1.0; e.g. hungry  ->  very hungry); `so` LEFT of C119222 (cand. C116522, support 29.108, 221 hosts, mod 1.0; e.g. scared and upset .  ->  so scared and upset .); `not` RIGHT of C119235 (cand. C99748, support 220.456, 190 hosts, mod 1.0; e.g. It is  ->  It is not); `so` LEFT of C119236 (cand. C116522, support 900.436, 177 hosts, mod 1.0; e.g. gentle  ->  so gentle)

Top v0.2 classes (posterior view):

- C116592 (mass 53213.74): lexical ['. (48105.0)', '! (3719.0)']; spans ['. (1.00 x48105)', '! (1.00 x3719)', 'again . (1.00 x507)', 'again ! (1.00 x35)']; contexts ['it _ <EOS> (1249.2)', 'said _ <EOS> (1014.9)']; composition in ['C99919 + C116592 -> C116592 (1389.7)']
- C119151+C119233 (mass 39383.33): lexical ['He (6708.0)', 'She (5931.6)', 'They (4671.4)', 'Lily (2876.0)', 'It (1419.1)', 'Tom (1415.0)', 'Ben (1379.0)', '<unk> (1197.4)', 'Timmy (720.0)', 'Tim (679.0)']; spans ['He (1.00 x6708)', 'She (0.89 x6631)', 'They (0.81 x5745)', 'Lily (1.00 x2876)']; contexts ['<BOS> _ was (4678.8)', '<BOS> _ had (1379.4)']; composition in ['C117187 + C119145 -> C119151+C119233 (3057.2)', 'C117187 + C119152+C119239 -> C119151+C119233 (963.9)']
- C102075 (mass 35891.94): lexical ['<unk> (33570.0)', 'computers (3.0)']; spans ['out (0.59 x887)', 'playing (0.62 x653)', 'into (0.67 x440)', 'my (0.71 x388)']; contexts ['the _ . (205.7)', 'started _ . (199.6)']; composition in ['C110213 + C119198 -> C102075 (1515.9)', 'C102075 + C119179 -> C102075 (706.4)']
- C116324 (mass 21991.83): lexical ['and (21448.0)', '<unk> (1.1)']; spans ['and (1.00 x21448)', ', and (1.00 x795)']; contexts ['Lily _ Ben (144.0)', 'smiled _ said (139.0)']; composition in ['C119243 + C116324 -> C116324 (542.8)']
- C119152+C119239 (mass 20941.73): lexical ['<unk> (6378.9)', 'park (715.0)', 'tree (570.0)', 'car (514.0)', 'house (469.0)', 'ball (461.0)', 'box (408.0)', 'garden (314.0)', 'truck (229.0)', 'slide (195.0)']; spans ['park (1.00 x715)', 'tree (1.00 x570)', 'car (1.00 x514)', 'house (1.00 x469)']; contexts ['the _ . (4320.9)', 'the _ and (1116.8)']; composition in ['C119152+C119239 + C116592 -> C119152+C119239 (3316.8)']
- C96748 (mass 19649.74): lexical ['the (19639.0)', '<unk> (10.7)']; spans ['the (1.00 x19639)']; contexts ['in _ park (251.0)', 'to _ park (239.0)']; composition in []
- C119243 (mass 16483.54): lexical [', (16379.0)', ': (88.0)', '<unk> (16.5)']; spans [', (1.00 x16379)', ': (1.00 x88)']; contexts ['said _ " (1011.0)', 'time _ there (556.0)']; composition in []
- C110213 (mass 14264.01): lexical ['to (14240.0)', '<unk> (24.0)']; spans ['to (1.00 x14240)']; contexts ['went _ the (286.0)', 'loved _ play (225.0)']; composition in []
- C85545 (mass 13641.0): lexical ['was (11451.0)', 'is (2063.0)', '<unk> (127.0)']; spans ['was (1.00 x11451)', 'is (1.00 x2063)']; contexts ['there _ a (1074.5)', 'It _ a (434.9)']; composition in []
- C107308 (mass 13278.28): lexical ['<unk> (11244.8)', 'zips (3.0)']; spans ['a tree (1.00 x104)', 'a tree . (1.00 x65)', 'a ball (1.00 x55)', 'a hole (1.00 x37)']; contexts ['and _ it (281.4)', 'for _ <EOS> (63.1)']; composition in ['C69 + C119152+C119239 -> C107308 (2030.5)']
- C69 (mass 12942.0): lexical ['a (12942.0)']; spans ['a (1.00 x12942)']; contexts ['upon _ time (939.0)', 'saw _ big (314.0)']; composition in []
- C119145 (mass 11273.94): lexical ['<unk> (1981.9)', 'girl (997.0)', 'boy (723.0)', 'bird (670.0)', 'dog (611.0)', 'man (581.0)', 'bear (358.0)', 'cat (333.0)', 'fish (193.0)', 'lady (155.0)']; spans ['girl (1.00 x997)', 'boy (1.00 x723)', 'bird (1.00 x670)', 'dog (1.00 x611)']; contexts ['The _ was (770.8)', 'the _ . (580.5)']; composition in ['C118263 + C119145 -> C119145 (1563.8)', 'C116293 + C119145 -> C119145 (271.3)']

Top inferred classes (human inspection):

- C116592 (5 members, 52369 occ): lexical ['.', '!']; multi-token ['again .', 'again !', 'is !']; contexts ['[k=1] it _ <EOS>', '[k=1] said _ <EOS>', '[k=1] happy _ <EOS>']
- C116324 (6 members, 22256 occ): lexical ['and']; multi-token [', and', 'with his mouth and', 'brightly and', 'at Anna and', 'playing games and']; contexts ['[k=1] Lily _ Ben', '[k=1] smiled _ said', '[k=1] mom _ dad']
- C96748 (6 members, 19659 occ): lexical ['the']; multi-token ['the other side of the', 'strike the', 'the bottom of the', 'the nearby', 'get out of the']; contexts ['[k=1] in _ park', '[k=1] to _ park', '[k=1] on _ ground']
- C119243 (8 members, 16493 occ): lexical [',', ':']; multi-token ['to Tom ,', 'at the park ,', 'to Sam ,', 'to Lily ,', 'to him ,', ', bang ,']; contexts ['[k=1] said _ "', '[k=1] time _ there', '[k=2] a time _ there was']
- C119151 (70 members, 15171 occ): lexical ['He', 'She', 'Everyone', 'Kate', 'Lulu', 'Grace', 'True', 'Robert', 'Nick', 'Ella', 'Nina', 'Lilly']; multi-token ['The bunny', 'The bird', 'The boy', 'The squirrel', 'The little girl', 'The mouse']; contexts ['[k=1] <BOS> _ was', '[k=1] <BOS> _ had', '[k=1] <BOS> _ said']
- C119233 (137 members, 14592 occ): lexical ['Tim', 'Tom', 'Ben', 'John', 'Tommy', 'Lily', 'Lucy', 'Sam', 'Timmy', 'Anna', 'Lila', 'Sara']; multi-token ['The hunter', 'The little mouse', 'The dragon', 'After a while , he', 'The flower', 'The baby']; contexts ['[k=1] <BOS> _ was', '[k=1] <BOS> _ and', '[k=1] " _ said']
- C110213 (3 members, 14246 occ): lexical ['to']; multi-token ['to eat carrots and', 'to run fast and']; contexts ['[k=1] went _ the', '[k=1] loved _ play', '[k=1] liked _ play']
- C85545 (2 members, 13514 occ): lexical ['was', 'is']; multi-token []; contexts ['[k=1] there _ a', '[k=1] It _ a', '[k=1] He _ so']
- C69 (1 members, 12942 occ): lexical ['a']; multi-token []; contexts ['[k=1] upon _ time', '[k=2] Once upon _ time ,', '[k=3] <BOS> Once upon _ time , there']
- C116015 (7 members, 10419 occ): lexical ['his', 'her', 'their', 'its', "everyone's"]; multi-token ['him his', 'Ben and his']; contexts ['[k=1] with _ friends', '[k=1] with _ toys', '[k=1] to _ mom']
- C111181 (3 members, 10115 occ): lexical ['they', 'he', 'she']; multi-token []; contexts ['[k=1] , _ saw', '[k=1] " _ said', '[k=1] , _ found']
- C114121 (6 members, 9141 occ): lexical ['"']; multi-token ['He says , "', 'She said , "', 'He said , "', 'She says , "', 'She smiled and said , "']; contexts ['[k=1] <BOS> _ I', '[k=1] <BOS> _ You', '[k=1] ! _ Lily']

Top relations A + B -> C:



# Analysis v0.3 — is category-preserving insertion / deletion a usable raw-text signal?

Setting: the v0.2 core is unchanged; adaptive constituent contexts are off; every number below is
from `configs/experiments_v03.yaml` (`outputs/v03`, 28 runs, all green) and the TinyStories study
(`outputs/v03/natural`).  Design and definitions: `docs/modifier_study.md`.  Candidate metrics are
at support >= 2 unless stated; "const" restricts the predicted candidates to posterior constituents
(`P(Q) >= 0.5`), which is what an *identity-preserving constituent* has to be; occurrence-level
precision is the share of evidence positions that are gold modifier children.

## 1. Diagnostic phase (evidence only, objective untouched)

### 1.1 Exact-string evidence (RQ1, questions 1–3)

| grammar | ROOT RIGHT (`A.`/`AB.`) type P / R | ROOT LEFT (`A.`/`XA.`) type P / R | ROOT occurrence P: LEFT / RIGHT / INTERNAL | root insertion attachment |
|---|---|---|---|---|
| Synthetic-4 | 0.38 / 1.00 (1.00 / 1.00 at support >= 5) | 0.20 / 0.77 (0.40 / 0.77) | 0.30 / 0.88 / 0.71 | 92 host in right context, 1 left, 75 not a modifier |
| Synthetic-1 | 0.42 / 1.00 (1.00 / 1.00) | 0.17 / 1.00 (0.24 / 0.83) | 0.07 / 0.88 / 0.60 | 122 / 3 / 114 |
| Synthetic-2 | 0.17 / 1.00 (0.42 / 1.00) | 0.17 / 1.00 (0.21 / 0.43) | 0.07 / 0.60 / 0.49 | 93 / 4 / 181 |
| Synthetic-3 (noisy) | 0.60 / 1.00 (0.82 / 1.00) | 0.14 / 0.50 (0.15 / 0.20) | 0.08 / 0.74 / 0.33 | 45 / 2 / 146 |

1. **`A.` / `AB.` (right insertion) is high-precision evidence.**  Every right-deletable suffix seen
   at least five times is a gold modifier on all four grammars (type precision 1.00, 0.82 under
   noise), occurrence precision 0.88–1.00 on the clean grammars.  What it finds is sentence-level
   RIGHT modifiers (`quickly`, `often`, …); its recall is complete for `VP + Adv -> VP` because a
   sentence-final adverb is always deletable.  Its host is the *sentence*, not `VP`: the attachment
   level is unknowable from the string pair, so the row says `S + Adv` while gold says `VP + Adv`.
2. **`A.` / `XA.` (left insertion) is not safe on its own.**  Precision 0.17–0.40.  The controls do
   exactly what they were built for: `dog sleeps.` / `the dog sleeps.` gives `the`, `a`, `every`,
   `some` LEFT evidence with 95 / 82 / 45 / 38 support and 47 / 44 / 33 / 28 distinct hosts — more
   support and more productivity than the real sentence modifiers (`probably` 39, `maybe` 36).
   Sentencehood is preserved, the category of the host (`Nn -> NP`) is not.  String identity of the
   host cannot see this.  Multi-token suffixes (`probably the`, `the old`) add composite false
   positives.
3. **`AB.` / `AXB.` (internal insertion) is medium-reliable, and the attachment is one-sided.**
   Occurrence precision 0.60–0.79 on the clean grammars, 0.33 under noise; the false insertions are
   again determiners (`John likes dog.` / `John likes the dog.`).  Where the inserted span is a gold
   modifier its host is almost always the *right* context (92 : 1 on Synthetic-4, 122 : 3 on
   Synthetic-1): `the old dog` inserts `old` in front of its host, `really likes` in front of `likes`.
   That is a property of these grammars (left-attaching modifiers can be inserted internally, a
   sentence-final adverb cannot), not of the evidence, and it is why the record says
   "`AB ~> AXB` preserves sentencehood" and nothing about the attachment.

The constituent-level string test (`CONSTITUENT_STRING`) needs the class-equality condition to mean
anything: on Synthetic-4, 15,279 string-identical constituent pairs `(A, AB)` change their posterior
class against ~3,500 that keep it (12,721 vs ~2,800 on Synthetic-1).  Every child of every binary
rule is a constituent whose parent is a constituent; "`A` and `AB` are both constituents" is not
evidence of anything.  With `E(A) = E(AB)` required it reaches type precision 0.82–1.00 (const) and
recall 1.00 on the clean grammars, occurrence precision 0.71–0.78, and it is the only string source
that sees sub-sentential modifiers (`clearly likes`, `small child`).

### 1.2 Class-generalised evidence (RQ2, RQ3, question 5)

| grammar | CLASS const P / R | CLASS host acc | CLASS occurrence P (L / R) | POSTERIOR_RULE P / R |
|---|---|---|---|---|
| Synthetic-4 | 1.00 / 1.00 | 1.00 | 0.81 / 0.87 | 1.00 / 1.00 |
| Synthetic-1 | 1.00 / 1.00 | 1.00 | 0.80 / 0.84 | 1.00 / 1.00 |
| Synthetic-2 | 0.92 / 1.00 | 1.00 | 0.80 / 0.86 | 0.86 / 1.00 |
| Synthetic-3 (noisy) | 0.50 / 0.58 (0.75 / 0.47 at >= 5) | 1.00 | 0.06 / 0.44 | 0.67 / 0.95 |
| Synthetic-4, random init K = 8 (forced collapse) | 0.75 / 0.67 | 0.42 | 0.79 / 0.80 | 0.67 / 0.67, host acc 0.31 |

**RQ2 — recall.**  Yes: class generalisation reaches recall 1.00 on every clean grammar where the
root string test misses the internal-only modifiers (`really`, `almost`, `clearly` are never
sentence-final or sentence-initial, so `ROOT_STRING` recall stays at 0.83 on Synthetic-4).  On the
constituent-candidate set its precision is as high as the string sources *when the classes are
right*.  The type-level false positives of all sub-root sources are stacked modifiers
(`clearly almost`, `red young`): strings whose deletion preserves the category but which are not
constituents — correct identity-preserving strings, wrong granularity; the `const` columns remove
them.

**RQ3 — contamination.**  Yes, and it is measurable.  Under noise (Synthetic-3) the LEFT
class-generalised occurrence precision collapses to 0.06 (fillers `um`, `uh`, `well` and determiners
inherit the host class of whatever they were merged with) while the RIGHT side stays at 0.44–0.53
and the exact-string RIGHT side at 0.74–0.79.  Under forced class collapse (random init, K = 8) the
class-generalised host accuracy falls to 0.42 and the rule-posterior host accuracy to 0.31, while
`ROOT_STRING` is unchanged by construction (0.65 / 0.83 const, RIGHT 1.00 at support >= 5): the
string anchor is the only source that does not move with the class inventory.

**RQ4 — combination.**  COMBINED has the recall of the class sources and the precision of the worst
member: 0.82 / 1.00 on Synthetic-4 but 0.22 / 1.00 on Synthetic-2 and 0.44 / 0.95 under noise.
Merging the sources into one list throws away the information that distinguishes them.  The useful
combination is a hierarchy, not a union: the exact-string RIGHT evidence as an anchor, the
class-generalised evidence as recall with the string anchor as its check.

### 1.3 Answers to the first four questions of the diagnostic

1. Gold modifier-like structures do get high evidence: recall 1.00 from the class sources and
   0.83–1.00 from the root strings on every clean grammar, with direction accuracy 1.00 everywhere.
2. Non-modifiers also get high *left* string evidence (the determiner control) and high
   class evidence under collapse; neither is a weakness of the detector, both are what the
   corresponding condition (sentencehood / class identity) actually licenses.
3. String-level precision is higher than class-level precision exactly where the classes are
   wrong (noise, collapse); where the classes are right the two agree.
4. Class collapse pollutes the generalised evidence (host accuracy 1.00 -> 0.42) and the
   rule-posterior evidence (host accuracy 1.00 -> 0.31), not the root strings.

## 2. Intervention phase

### 2.1 Modifier rule proposals (ablations C, D)

On the clean grammars the proposals are made and rejected: `Det + S -> S` (`dF = -11`),
`S + Adv -> S` (`-13`), `Adj + S -> S` (`-11`), `Deg + VP -> VP` (`-19`); the grammar already has
`VP + Adv -> VP` and `Deg + Vt -> Vt` from the v0.1 initialisation, so an endomorphic rule at the
wrong attachment level only adds derivational ambiguity and `dF` says no.  Results identical to
the baseline (Synthetic-1, Synthetic-4).  On Synthetic-2 two root-hosted rules squeeze through
(`S + Adj -> S`, `S + Adv -> S`, `dF` just above 0): composition precision drops 0.58 -> 0.50 and
`F` ends lower (`-43,110 -> -43,173`), class and ambiguity metrics unchanged.  Under noise the
string proposals are harmful: 37 accepted moves, class pair F1 0.91 -> 0.84, occurrence F1 0.98 ->
0.69, composition F1 0.30 -> 0.24, while `F` *rises* (`-55,594 -> -54,513`): the endomorphic
rules let the merged filler / determiner classes absorb more, which the ELBO likes and the gold
does not.  Under forced collapse the same: `F` lower, class F1 0.75 -> 0.61, composition F1 0.38 ->
0.46, class-generalised precision 0.75 -> 0.29 afterwards.  Adding `CLASS_GENERALIZED` as a
proposal source (D) changes nothing over C on any grammar: the class-generalised candidates name the
rules the grammar already has.

Conclusion: as a *proposal* source the root-string evidence is unreliable because its host is the
sentence.  The host must come from the constituent-level evidence (which does carry the class) or
from a later factorisation; and the objective, not the evidence, has to stay the judge — which it
did on the clean grammars and did not under noise.

### 2.2 Lexicon expansion and structural distinguishability (RQ5–RQ8, questions 6–7)

The twelve "false" entries of the clean grammar (`child/horse/woman/man/bird/cat/dog/teacher -> NP`,
`young/small/happy/old -> Det`) are the target of the duplicate penalty.  The trial posteriors say
they are not duplicates in any structural sense:

| entry | dF | D_comp | D_modifier | derivability gain | lambda_dup needed to reject |
|---|---|---|---|---|---|
| `child: Nn -> NP` | +290 | 0.93 | 0.93 | 0.00 | 290 |
| `young: Adj -> Det` | +312 | 1.00 | 1.00 | 0.00 | 312 |
| `walks: VP -> Vt` (Synthetic-2, true) | +1623 | 0.98 | 1.00 | 0.00 | 1623 |
| `light: Nn -> Adj` (Synthetic-2, true) | +1196 | 1.00 | 1.00 | 0.00 | 1196 |
| `young: Adj -> Adj'` (split fragment, duplicate) | +0.7 | 0.01 | 0.01 | 0.00 | 0.7 (rejected) |
| `the: Det -> Adj` (Synthetic-2, false) | +12 | 1.00 | 1.00 | 0.00 | 12 |

The new `NP` entry of `child` is used only as a bare subject / object (`NP + VP -> S`, `Vt + NP -> VP`,
97 expected occurrences), the `Nn` entry only after determiners and adjectives; the composition
profiles are almost disjoint (JS 0.93) and so are the modifier profiles (the `NP` entry never hosts an
`Adj`).  The derivability gain is 0 — every one of those sentences was already derivable through
`Nn + VP -> S` — but it is also 0 for the *true* ambiguities, because the v0.1 initialisation
carries crutch rules (`VP + NP -> VP`, count 167) that already derive `walks the cat`.  What the
penalty *does* catch are the real duplicates: the class-split fragments (`young -> C4219+C4253.1.2`,
`dF +0.7`, `D = 0.01`) are rejected on Synthetic-2 (ablation G: 2 rejections, ambiguity F1 0.857
and occurrence accuracy 0.933 unchanged, `F` unchanged).

**RQ5 — can structural distinguishability reduce the false additions?**  Not these.  Every
accepted false entry on the clean grammar has `D_struct ≈ 1` under either distance and needs
`lambda_dup` between 186 and 312 to be rejected; the true second categories need 1,196 and 1,623;
the one false entry that resembles them (`the -> Adj`, `dF +12`) would go at `lambda_dup = 12`.
A single penalty scale therefore separates *these* data (any value in 312 < lambda < 1,196), but
only through the `dF` magnitude, not through the distance — which is the quantity the penalty was
supposed to measure.  Choosing it would be fitting the gold (spec §25), so the defaults stay at the
diagnostic scale (`lambda_dup = 10`) and the table reports what each entry would have needed.
Ablations E / F / G leave Synthetic-1 at 12 false entries and the clean metrics untouched.

**RQ6 — is true ambiguity preserved?**  Yes under every setting: `walks`, `light` are accepted with
margins two orders of magnitude above the penalty; Synthetic-2 ambiguity F1 0.857 and occurrence
accuracy 0.933–0.934 in A–G.

**RQ7 / RQ8 — bare nouns.**  The bare-noun `Nn / NP` issue is not reduced, and the trial posteriors
show why: the two entries have *different* composition and modifier behaviour.  The gold encodes
bare nouns with a 0.25-weight rule `Nn + VP -> S`; the finite PCFG prefers to pay two Dirichlet
pseudo-counts per noun and route the same sentences through the 1.0-weight `NP + VP -> S`
(about 3 nats per occurrence).  Under the observable grammar this is an alternative representation
of one construction, not a duplicate analysis of one word — the entries differ in exactly the
composition and modifier contexts the study measures.  It is the "gold representation and observable
grammar disagree" case of §25, and the data do not support forcing it.

## 3. Natural language (TinyStories 10k, two disjoint 5k halves)

Base run (10,000 sentences, 295 classes, 437 rules, 723 s) and two disjoint 5,000-sentence halves
(`outputs/v03/natural/natural_report.md`, candidate lists under "Top identity-preserving
candidates").

**The root anchor is empty on real text.**  Among 10,000 TinyStories sentences there are 19
`A.`/`AB.` pairs, one `A.`/`XA.` pair and 18 `AB.`/`AXB.` pairs; no root candidate reaches support 3,
and the halves have 8 / 3 right pairs.  Real sentences almost never recur with one token added, so
the source whose precision was 1.00 on the synthetic grammars contributes nothing at this corpus
size.  The exact-string evidence that survives is the constituent-level one (540 / 378 / 553
preserving pairs against 60,311 class-changing ones), and its candidates read like modifiers:
`little` LEFT of the noun class (`bird -> little bird`, 17–24 distinct hosts, mod 1.0), `very` / `so`
LEFT of the adjective class (`happy -> very happy`, 15 hosts), `big`, `that` RIGHT of the verb class
(`They pretend -> They pretend that`, 25 hosts), `down` (`fell -> fell down`), `back`, `all`.
The rule-posterior source is visibly contaminated: one `<unk>`-absorbing class with a
`C + C -> C` self-loop lists `me`, `as`, `my`, `will`, `kept` as `mod = 1.0` candidates at
`P(Q) = 0.3–0.4` — the natural-text version of question 5, and the string sources do not produce it.

**Stability across disjoint halves (RQ8, questions 8–9).**  General structure on the same halves:
class pairwise F1 0.24, rule agreement (Jaccard after class alignment) **0.066** (374 vs 344
rules), ambiguity agreement 0.21.  Identity-preserving candidates (constituent candidates with
support >= 3):

| source | candidate Jaccard (shared / A / B) | direction agreement | host agreement (after class alignment) |
|---|---|---|---|
| ROOT_STRING | 0 (0 / 0 / 0) | – | – |
| CONSTITUENT_STRING | **0.24** (12 / 35 / 27) | 0.58 | 0.08 |
| CLASS_GENERALIZED | 0.14 (10 / 54 / 29) | 0.60 | 0.10 |
| POSTERIOR_RULE | 0.10 (6 / 46 / 19) | 1.00 | 0.17 |
| COMBINED | 0.15 (12 / 61 / 32) | 0.58 | 0.08 |
| general rules `R(A, B, C)` | 0.066 | – | – |

Shared candidates: `little`, `very`, `so`, `small`, `all`, `back`, `both`, `outside`, `don't`, and the
pronouns `he`, `she`, `they`.  The candidate *sets* are 1.5–3.6 times more reproducible than the rule
sets, with the string source on top and the rule source at the bottom, as on the synthetic
collapse runs.  But the agreement is in the *strings*, not in the relations: direction agrees on
only 58–60 % of the shared string / class candidates (the pronouns flip between LEFT of a verb phrase
and RIGHT of a preceding clause depending on the half), and the host class agrees on 8–17 %, because
the class inventories of the two halves barely align (class F1 0.24) and the host is a class.  The
rule-posterior candidates agree on direction (1.00 on six candidates) and still not on host.

So the answer to question 8 is split.  *Which strings are identity-preserving* is recovered far more
reliably from 5k sentences than *which composition relations hold* (0.24 vs 0.066) — that part of the
hypothesis holds, and the ordering string > class > rule-posterior is the same as under synthetic
collapse.  *Which category they preserve* is not recovered reliably, because that answer is a class
and the classes themselves are the unstable object.  On question 9: the candidates are an anchor
for the *strings* that a CCG factorisation should treat as `X/X` or `X\X` (`little`, `very`, `so`,
`big`, `that`, `back`, `down`, `all` carry direction and productivity), not yet for the `X`.  That
makes the next step the reverse of what was planned: use the stable string candidates to constrain
the class alignment (a candidate is an `X`-modifier in both halves of whatever `X` hosts it), rather
than use the classes to type the candidates.

### 3.1 Corpus size: the root anchor needs 10–50 times more text than the grammar sees

The 10k study is a subset chosen for the grammar induction, not for the string evidence, and the
string evidence scales differently.  `scripts/root_string_scan.py` runs the `ROOT_STRING` tests
alone (no posterior) on the shuffled TinyStories prefix (917,854 sentences of 2–15 tokens; the same
order as `run_natural.py`, so every subset contains the smaller ones):

| N | distinct sentences | `A.`/`AB.` pairs | `A.`/`XA.` | `AB.`/`AXB.` | RIGHT candidates (prod. >= 3 / single-token) | LEFT | INTERNAL | halves Jaccard R / L / I |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 9,638 | 175 | 16 | 169 | 2 / 0 | 0 / 0 | 6 / 4 | 0.00 / 0.00 / 0.67 |
| 50,000 | 46,750 | 1,847 | 242 | 1,887 | 112 / 10 | 19 / 1 | 103 / 43 | 0.34 / 0.10 / 0.58 |
| 100,000 | 91,706 | 4,598 | 555 | 4,973 | 296 / 29 | 34 / 3 | 307 / 98 | 0.41 / 0.55 / 0.43 |
| 200,000 | 179,240 | 10,984 | 1,318 | 12,460 | 724 / 82 | 88 / 3 | 738 / 205 | 0.48 / 0.49 / 0.50 |
| 500,000 | 431,068 | 38,126 | 3,586 | 40,412 | 2,290 / 278 | 199 / 8 | 2,111 / 402 | 0.48 / 0.51 / 0.53 |
| 917,854 | 766,306 | 76,940 | 6,818 | 86,633 | 4,576 / 530 | 325 / 15 | 4,226 / 613 | 0.50 / 0.61 / 0.53 |

Pairs grow roughly quadratically (the chance that a sentence recurs with one span removed grows
with the corpus); the anchor becomes usable between 100k and 200k sentences, and from 200k on its
candidate sets agree across disjoint halves at Jaccard 0.48–0.61 — twice the constituent-level
string agreement of the 10k grammar runs (0.24) and seven to nine times the rule agreement (0.066).
What it finds at 200k–900k, by productivity, single tokens only:

* RIGHT (`A.` / `AB.`): `too` (931 distinct hosts), `again` (505), `together` (281), `back` (177),
  `outside` (159), `anymore` (158), `now` (131) — sentence-final adjuncts;
* INTERNAL (`AB.` / `AXB.`): `very` (2,413), `little` (1,293), `so` (1,229), `big` (864), `not` (670),
  `all` (404), `both` (336), `still` (314), `just` (302), `small` (272), `new` (219) — degree words,
  prenominal adjectives, quantifiers, negation;
* LEFT (`A.` / `XA.`): `But` (219), `And` (82), `Little` (32), and otherwise multi-token discourse
  frames — `She said ,` (570), `He said ,` (426), `One day ,` (178), `They say ,` (170): sentencehood
  is preserved by removing a quotation frame, which is a true fact about the string and not a
  category-preserving modifier.  Multi-token RIGHT candidates are of the same kind
  (`, Lily` vocatives, `and smiled` coordinations, `in the park` adjuncts).

So the earlier statement "the root anchor is empty on real text" is a statement about 10k
sentences.  At the scale the string test needs, it produces the cleanest candidate lists of the
study, with the LEFT side again dominated by things that preserve sentencehood without being
modifiers (here discourse frames instead of determiners).  The grammar induction scales worse than the string test.  A first 50k run was killed at 13.7 GB in
the v0.1 clustering stage; the cause was the dense stop-context term of the pair scores evaluated
for all candidate pairs at once (`n_pairs x n_stop_contexts` floats), which is now computed in
chunks with the candidate order unchanged (`docs/profiling.md`).  With that fix 50k sentences run
at a peak of 4.6 GB (10k: 2.0 GB, 25k: 2.8 GB) in 2.7 hours for the base run (1.0 h of it structure
learning, 1.0 h the MDL pass over 12,298 provisional classes), and the halves are 25k each.

### 3.2 Grammar runs at 25k and 50k

| | 10k (halves 5k) | 25k (halves 12.5k) | 50k (halves 25k) |
|---|---|---|---|
| class pairwise F1 / ARI across halves | 0.24 / 0.21 | 0.36 / 0.29 | 0.42 / 0.34 |
| general rule agreement (Jaccard, aligned) | 0.066 (374 / 344 rules) | 0.127 (375 / 354) | 0.110 (333 / 337) |
| CONSTITUENT_STRING candidates: Jaccard / direction / host | 0.24 / 0.58 / 0.08 | 0.14 / 0.73 / 0.40 | 0.24 / 0.73 / 0.36 (33 shared of 100 / 70) |
| CLASS_GENERALIZED | 0.14 / 0.60 / 0.10 | 0.13 / 0.79 / 0.50 | 0.16 / 0.70 / 0.39 |
| POSTERIOR_RULE | 0.10 / 1.00 / 0.17 | 0.15 / 0.94 / 0.65 | 0.10 / 0.89 / 0.50 |
| COMBINED | 0.15 / 0.58 / 0.08 | 0.13 / 0.84 / 0.58 | 0.14 / 0.73 / 0.38 |
| root pairs in the base run (R / L / I) | 19 / 1 / 18 | 104 / 9 / 92 | 399 / 29 / 406 (RIGHT `again`; INTERNAL `very` 17 hosts, `so` 10, `little`, `all`, `always`, `big`, `just`, `small`, `too`) |
| class-changing vs class-preserving constituent pairs | 60k vs 1.5k | 147k vs 3.0k | 288k vs 5.3k |

Across the three sizes the class alignment improves steadily (F1 0.24 -> 0.36 -> 0.42) and the
general rule agreement stays around 0.07–0.13.  The identity-preserving candidate sets of the
string source are at 0.24 at 10k and 50k and 0.14 at 25k — above the rule agreement at every size
(by 3.6x, 1.1x and 2.2x), but not growing with the corpus; the class-generalised and rule-posterior
sets stay at 0.10–0.16.  What *does* grow with the class alignment is what the shared candidates
say: direction agreement 0.58 -> 0.70–0.89 and host agreement 0.08 -> 0.36–0.50.  At 50k the shared
string candidates are `all`, `always`, `back`, `big`, `both`, `again`, `before`, `bit`, the colour
and property adjectives `blue`, `brave`, `curious`, `different`, `elderly`, and the frame
`One day ,` (33 strings); the base run's own lists add `very` (119 hosts), `so`, `little`, `still`,
`not` (93 hosts, RIGHT), and `in` / `to` (221 / 117 hosts, LEFT of a class that merges noun phrases
with their prepositional phrases — a class-generalised candidate that a false merge created, visible
as such because the string source lists it with a different host).  The rule-posterior source is
the most reproducible in direction (0.89–0.94) and the least in membership (0.10–0.15): the grammar
keeps few endomorphic rules and not the same ones in both halves.

So the 10k claim is partly confirmed and partly not: the string candidates are more reproducible
than the relations at every size, but the gap does not widen with data, and the relation-level
objects (direction, host) only become reliable as the class inventory does.  The advantage of
identity-preserving evidence at scale remains the string inventory with its direction, which the
root scan delivers without a grammar at all.

## 4. Answers (§29)

1. `A.` / `AB.` gives high-precision RIGHT modifier evidence (type precision 1.00 at support >= 5 on
   all clean grammars, 0.82 under noise; occurrence precision 0.88–1.00), with the sentence as host.
2. `A.` / `XA.` does not: precision 0.17–0.40; the determiner control (`dog sleeps.` / `the dog
   sleeps.`) is the most productive left insertion in the corpus and preserves sentencehood without
   preserving the category.
3. `AB.` / `AXB.` is medium-reliable (occurrence precision 0.60–0.79 clean, 0.33 noisy); when the
   insertion is a modifier its host is the right context (92 : 1); the determiner false insertions
   are the same control.
4. Exact strings: precision where the classes are wrong, recall limited to sentence-edge modifiers
   and the sentence as host.  Class generalisation: recall 1.00 and the right host, precision equal to
   the strings when the classes are right and 0.06–0.42 when they are not.
5. Yes — class collapse propagates into the class-generalised evidence (host accuracy 1.00 -> 0.42,
   LEFT occurrence precision 0.80 -> 0.06 under noise) and into the rule-posterior evidence
   (host accuracy 0.31); the exact-string evidence is unchanged by construction.
6. Modifier / composition distinguishability does not reduce the spurious clean-grammar entries,
   because they are structurally distinct; it rejects the genuine duplicates (split fragments).
7. No true lexical ambiguity is lost under any setting.
8. The string-anchored candidate sets are more reproducible across disjoint halves than the
   general relations at every size tried (Jaccard 0.24 / 0.14 / 0.24 at 10k / 25k / 50k against
   0.066 / 0.127 / 0.110 for `R(A, B, C)`), the class-generalised and rule-posterior sets only
   marginally (0.10–0.16); the gap does not grow with data.  Direction (0.58 -> 0.70–0.89) and host
   agreement (0.08 -> 0.36–0.50) of the shared candidates rise with the class alignment.  The root
   exact-string anchor needs 100k–200k sentences (19 pairs at 10k, 104 at 25k, 399 at 50k, 11k at
   200k); from 200k on its candidate sets agree across halves at 0.48–0.61, above every grammar-based
   figure.
9. As an anchor for *which strings* get `X/X` / `X\X` types: yes, with direction for the
   high-productivity ones.  As an anchor for *which X*: not until the class inventory is aligned
   through them rather than before them.
