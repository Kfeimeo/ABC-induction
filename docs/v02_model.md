# v0.2 probabilistic model — mathematics and implementation notes

This document specifies exactly what `src/abc_induction/{grammar_model,chart,chart_numba,cavi,structure_learning,adaptive_context,probabilistic,problog_reference}.py` compute.  Section numbers refer to the v0.2 specification.

## 1. Latent model (finite Bayesian latent grammar)

Given the active classes `E` (|E| ≤ `k_max`), a *sparse* rule inventory `R ⊆ E³` (`C → A B`, written `A + B → C` in v0.1 notation) and a sparse lexicon `L ⊆ E × V` (`C → w`, plus one unknown-token expansion `C → <unk>` per class), the latent variables of a sentence `s = w₁…wₙ` are a binary tree `T` over `[0, n)` and class labels `Z` of its nodes.  The joint score is

```
score(T, Z | s) = ρ(Z_root)
                · ∏_{internal (i,k,j)} θ_{Z_ij}(Z_ik, Z_kj) · ψ(i, j)
                · ∏_{leaves i}         θ_{Z_i}(w_i)         · λ_i(Z_i)
```

* `θ_C` is **one distribution per class over its expansions** `Exp(C)` = {rules producing `C`} ∪ {lexical entries of `C`} ∪ {`<unk>`}: `θ_C ~ Dir(α_C)` with `α_{C,e} = alpha_rule / alpha_lex (= gamma_type) / alpha_unk` by expansion kind.
* `ρ` is the root distribution over classes, `ρ ~ Dir(alpha_root)`.
* `ψ(i, j) ≤ 1` is the punctuation span potential (§18), `λ_i(C) ≤ 1` the lexical-context leaf potential (§6).  Both are *fixed virtual evidence*: `Σ_{T,Z} score` is then a proper likelihood of (sentence, virtual evidence), the ELBO is a true lower bound, and the ProbLog/brute-force oracles encode them as independent probabilistic facts.
* Natural text only: a sentence may be a sequence of `m ≥ 1` root chunks with probability `p_cont^{m-1}(1 − p_cont)`, `p_cont ~ Beta(beta_chunk)` (`chunks_enabled`).  With the chunk layer off (`chunks_enabled: false`) the whole sentence is a hard root constituent.  Single tokens are always constituents (leaves), composition is continuous and binary.

**Deviation from §4/§5 of the spec (bottom-up `θ_AB = P(C | A, B)`, `π_x = P(C | x)`).**  With the string observed, a model whose parameters are the bottom-up conditionals has no string likelihood: for a *complete* rule inventory `Σ_{Z} P(Z | T, s) = 1` for every tree, so `Σ_{T,Z} = #trees` (Catalan) for every grammar, and an incomplete inventory is rewarded for making *more* labellings legal — nothing would drive structure learning (the single-class grammar `A + A → A` maximises it).  The top-down expansion distributions give the compression signal that v0.1's MDL data cost already used (`N(C, e)·−log P(e | C)`).  The two bottom-up quantities the spec asks for are therefore **derived** from the posterior expected counts and reported everywhere v0.1 reported them:

```
P(C | A, B) = (E[N_{C←AB}] + η) / (Σ_{C'} E[N_{C'←AB}] + K_AB η)       (rule_conditional)
π_x(C)      ∝  γ + E[N_{C→x}]                                          (type_mixture)
```

`Σ_C E[P(C | A, B)] = 1` is tested (`tests/test_chart_exact.py::test_posterior_rule_conditional_sums_to_one`).  Occurrence ambiguity is `q(Z_o = C)` from the chart, type-level ambiguity is `π_x` with mass on ≥ 2 classes — exactly the semantics the spec wants (e.g. `fast = 0.62 Adj + 0.38 Adv` on Synthetic-2), without a one-shot split.

## 2. Constituency = chart marginal (§2)

No separate Bernoulli `Q_ij`.  `P(Q_ij = 1 | s) = Σ_C P(Z_ij = C | s)`; every token has `P(Q_{i,i+1} = 1) = 1` and the sentence has `P(Q_{0n} = 1) = 1` when the chunk layer is off (tested invariants).

## 3. Sparse Inside–Outside (§7–9)

Cells are sparse maps `class → log value`; rules are iterated through the index `left class → rules` (python reference) / CSR `rule_ptr[A]` with per-cell active-class lists (numba).  With `θ̃ = exp E_q[log θ]`:

```
inside:   α_{i,i+1,C} = θ̃_C(w_i) · λ_i(C) · ψ(i,i+1)
          α_{ijC}     = ψ(i,j) · Σ_{k} Σ_{(C→A B)∈R} α_{ikA} α_{kjB} θ̃_C(A,B)
          Z̃_s        = Σ_C ρ̃(C) α_{0nC}                      (chunk layer: prefix/suffix DP over chunk masses)
outside:  β_{0nC}    = ρ̃(C)
          β_{ikA}   += β_{ijC} ψ(i,j) θ̃_C(A,B) α_{kjB},   β_{kjB} += β_{ijC} ψ(i,j) θ̃_C(A,B) α_{ikA}
node:     P(Z_ij=C | s) = α_{ijC} β_{ijC} / Z̃_s
rule app: P((i,k):A,(k,j):B,(i,j):C | s) = β_{ijC} ψ(i,j) θ̃_C(A,B) α_{ikA} α_{kjB} / Z̃_s
```

All sums are `logsumexp`.  Semirings: `LogSemiring` (logsumexp, +) for marginals and `ViterbiSemiring` (max, +) with back-pointers for the MAP derivation; expected counts come from inside+outside (no separate expectation semiring in this version).  Beam / threshold pruning (`chart.exact: false`) records `cells pruned`, `mean retained mass`, `max dropped mass`.  Exact mode is enforced on the synthetic suite and in every oracle test.

## 4. Dirichlet CAVI / variational EM (§10–11)

```
E-step  E_q[log θ_{C,e}] = ψ(α'_{C,e}) − ψ(Σ_{e'} α'_{C,e'}),   E_q[log ρ_C] = ψ(α'_root,C) − ψ(Σ α'_root)
        Inside–Outside with these → E[N_{C,e}], E[N_root,C], E[#chunks]
M-step  α'_{C,e} = α_{C,e} + E[N_{C,e}]      (rules, lexical entries, <unk>, root, chunk Beta)
ELBO    = Σ_s log Z̃_s − Σ_d KL( Dir(α'_d) ‖ Dir(α_d) )
```

The ELBO is evaluated right after each E-step (then `q(Z)` is the exact chart posterior, so `E_q[log p(D,Z|Θ)] − E_q[log q(Z)] = Σ_s log Z̃_s`); the sequence is non-decreasing with the exact chart (`tests/test_cavi_structure.py`, tolerance `1e-8` relative).  Every iteration is logged to `objective_trace.csv` with `ELBO, delta_ELBO, log_z, kl, structure_prior, objective, expected_rule_count, active_classes, active_rules, active_lex, posterior_entropy (mean entropy of π_x), n_underivable, mean_retained_mass`.

## 5. Structure objective and prior (§11–12)

```
F(G) = ELBO(G) + log P(G)
log P(G) = − λ_C |E| − λ_R |R| − λ_L |L| − Σ_r λ_p · max(0, max_x N_{r,x}/N_r − τ_p)   [rules with N_r ≥ private_min_count]
```

`N_{r,x}` = posterior expected count of rule `r` realised with inventory span type `x` on either side (accumulated in the kernel from rule applications).  All constants live in `StructurePrior` (`structure_prior:` config) and are printed per component (`prior_class, prior_rule, prior_lex, prior_private`); no threshold elsewhere hides a structure cost.  Units: nats (v0.1's MDL used bits; `-ELBO/(N ln 2)` is reported as bits/sentence).

## 6. Class merge and class split (§13–15)

Block-coordinate loop (`StructureLearner.run`): CAVI → merge stage → split stage → rule/lexicon prune → lexicon add → rule add → (adaptive context refinement feeds the next round) → repeat until a full round accepts nothing (`max_rounds` is only a safety cap).

*Merge proposals* — signatures per class: raw lexical context (posterior-weighted `L1/R1/(L1,R1)` tokens), composition-in `(A,B)` of its rules, composition-out (parent rule + sibling), lexicon, abstract constituent context.  Candidates come from an inverted index over each class's top features (never all `K²` pairs); ranking by a weighted cosine over the blocks.

*Split proposals* — occurrences of `A` with `q(Z=A) ≥ 0.5`, features = raw context + parent-rule-and-side + child rule + abstract context (composition features weighted ×3), weighted 2-means (`sample_weight = q`); ranking by `(1 − cos(signature_1, signature_2)) · balance`.  Initialisation copies every rule / lexical entry / root count involving `A` to `A1, A2` with the **occurrence-partition shares** (fraction of `A` mass under each rule that fell into cluster 1); unsupported copies are later removed by the prune proposals.

*Acceptance* — checkpoint → apply → map the expected counts of unaffected sentences → `proposal_vb_iterations` short CAVI (warm start) → `ΔF = F(G') − F(G)`; accept iff `ΔF > τ_kind` (`merge_threshold 0 < split_threshold 5`, conservative bias), else exact rollback.  After every accepted move a full CAVI re-score must satisfy `F_{t+1} ≥ F_t − tol`; a violation rolls the move back and is counted (`monotonicity_violations`, asserted 0 by the tests).  Each move is logged to `structure_moves.jsonl` with `a, b, proposal_reason, old_objective, new_objective, delta, delta_ELBO, delta_structure_prior, delta_prior_components, accepted, partition_a1/a2, similarity_parts`.

*Proposal evaluation on affected sentences* (`proposal_eval: affected`, natural text): only sentences whose posterior touches the moved classes / rules / token are re-estimated; the other sentences' counts are kept and their `log Z̃` is assumed unchanged (documented approximation; the post-acceptance full re-score is exact).

## 7. Adaptive constituent context (§16–17)

For a centre node with posterior ≥ 0.5, the nearest left/right spans with `P(Q=1) ≥ 0.9` give the soft abstract context `w = q(Z_L=A) q(Z_R=B) q(Z_u=C)`; depth 2 expands along the constituent hierarchy (`L2 ⊃ L1`, `R2 ⊃ R1`) only for flagged classes (raw-similar but composition-different merge candidates); refined features are kept by information gain × support (`top_features`).  These features enter the merge similarity and the split clustering **only** — never the ELBO (no double counting of lexical evidence).

## 8. Punctuation structural prior (§18)

`punctuation.mode ∈ {punctuation_as_token, punctuation_removed, structural_prior}`.  In `structural_prior` mode: sentence-final `. ? !` → `+w_final` on the sentence body `[0, n−1)` and on `[0, n)`; paired `( ) [ ] " "` → `+w_paired` on the inside and the enclosing span; internal `, ; :` → `−w_comma` on every span that strictly contains the mark (root and body exempt).  Potentials are normalised to `≤ 0` per sentence (bonuses become relative penalties on competing spans).  `punctuation_removed` strips the marks from the corpus before everything.

## 9. Exact oracles (§19–22)

`problog_reference/oracle.py` encodes the compiled grammar with one annotated disjunction per (class, Gorn address) so that a world fixes exactly one top-down derivation; the root is one annotated disjunction; `λ`, `ψ` are independent probabilistic facts a derivation must "pay".  Queries: `sentence` (= `Σ_{T,Z} score`), `node(I,J,C)` and `const(I,J)` with evidence `sentence` (posteriors).  `chart.brute_force` enumerates every binary tree and labelling.  `tests/test_chart_exact.py` (brute force = python chart = numba chart, |Δ| < 1e-9, with and without potentials, chunk layer, probability invariants) and `tests/problog_reference/` (ProbLog = chart = brute force, |ΔP| < 1e-8, including a shared-intermediate multi-hop case) are the correctness gates; the ProbLog tests skip when `problog` is not installed and it is never imported by the pipeline.

## 10. Initialisation (§25)

`init: v01` builds the model from the v0.1 result: accepted classes (ranked by derivation usage, capped at `k_max`), the MDL grammar's rules with their Viterbi counts, lexical entries from the MDL lexicon plus the final v0.1 type membership (so legacy ambiguity splits, if enabled, are carried over as two entries), root counts from the Viterbi roots.  Multi-token "phrase leaves" of v0.1 are dropped (binary composition only).  `init: random` (ablation) draws random lexical entries and a random sparse rule set for `random_k` classes.
