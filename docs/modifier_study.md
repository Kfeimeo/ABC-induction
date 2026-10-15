# v0.3 — identity-preserving constituent evidence (design)

The v0.2 core (sparse Inside–Outside, Dirichlet CAVI, `F = ELBO + log P(G)`, class merge / split,
rule and lexicon proposals) is fixed.  This round asks one question:

> Is *category-preserving insertion / deletion* — `C + M -> C` or `M + C -> C` — a raw-text structural
> signal that is more reliable, and more robust to class collapse, than distributional similarity?

Nothing observed here is called a linguistic modifier.  The terms are *identity-preserving
constituent*, *modifier candidate*, *endomorphic constituent*.  Whether a candidate is a semantic
modifier is a question for a later stage with semantics.

## 1. Two kinds of evidence, kept apart

**String-preserving evidence** does not depend on the induced classes.

| source | test | gives |
|---|---|---|
| `ROOT_STRING` A1 | `A.` and `AB.` both complete sentences, `A` the identical token sequence | `B`: RIGHT evidence, host = *sentence* (`S + B -> S`); recorded, never hard-assigned |
| `ROOT_STRING` A2 | `A.` and `XA.` | `X`: LEFT evidence, host = sentence |
| `ROOT_STRING` A3 | `AB.` and `AXB.` (`A`, `B` identical) | `X`: INTERNAL evidence; attachment level (to `A`, to `B`, to `AB`) unknown |
| `CONSTITUENT_STRING` | the same deletion test on posterior constituents below the root, `P(Q_A) > tau_q`, `P(Q_AB) > tau_q`, host string unchanged, **and** `E(A) = E(AB)` | RIGHT / LEFT / INTERNAL evidence with a class host |

Without the class-equality condition the constituent-level test is vacuous: every child of every
binary rule passes it (`likes` and `likes Mary` are both constituents).  On Synthetic-4, 15,279
string-identical constituent pairs change category against ~3,500 that preserve it.  The
`class_changing_occurrences` statistic is reported for every run.

**Class-generalised evidence** uses the posterior equivalence classes; higher coverage, inherits
every false merge.

| source | test |
|---|---|
| `CLASS_GENERALIZED` B1 | `E(A') = E(A'B) = C` inside one sentence (node posteriors of both spans `>= tau_q`), any host `A'` of class `C` |
| `CLASS_GENERALIZED` B2 | `E(A') = E(XA') = C` |
| `CLASS_GENERALIZED` B3 | `E(AB) = E(AXB) = C`, the reduced string a constituent of class `C` elsewhere (attachment unknown) |
| `POSTERIOR_RULE` | expected counts `E_q[N_{C,M->C}]`, `E_q[N_{M,C->C}]` of the active endomorphic rules (rule applications above the output threshold) |

Posterior marginals are never multiplied: class-generalised evidence is thresholded and its
`posterior_mass` is the smaller of the two node posteriors; `POSTERIOR_RULE` uses the joint
rule-application posterior from Inside–Outside.

The evidence hierarchy `ROOT exact string > CONSTITUENT exact string > CLASS generalised` is
reported source by source and never collapsed into one score.

## 2. Statistics per (candidate, host, direction, source)

`support` (evidence occurrences), `posterior_mass`, `distinct_host_types` (= productivity:
`runs quickly / walks quickly / talks quickly` beats one accidental `A + B -> A`),
`distinct_host_occurrences`, `string_exact_support`, `class_generalized_support`,
`posterior_rule_support`, `insertion_support`, and for class-level hosts

    mod_R(M, C) = E[N_{C,M->C}] / sum_Y E[N_{C,M->Y}]        mod_L(M, C) = E[N_{M,C->C}] / sum_Y E[N_{M,C->Y}]

Punctuation (`. ! ? , ; :` …) is a sentence-boundary observation: stripped before the root
index, never a candidate.  Files per run: `modifier_evidence.json` (rows),
`modifier_candidates.json` (per candidate string, supports per source, dominant direction),
`modifier_string_pairs.json` (`A.`/`AB.` examples).

Indexing is linear: sentence string -> ids, deletion variants of every sentence
(`max_modifier_length` prefix / suffix tokens, `max_insert_span_length` internal tokens); the
constituent index is the same on posterior constituents.

## 3. Gold and metrics (synthetic)

Gold modifier structure is read off the gold derivations: rule `A + B -> C` with `C == A` makes the
right child a RIGHT modifier of host `A`; `C == B` a LEFT modifier of host `B`.  Synthetic-4
(`grammars/synthetic4_modifier.yaml`) adds `AdvS + S -> S` (sentence-level left modifier),
`Deg + Vt -> Vt` (internal insertion with a known attachment level) to `VP + Adv -> VP` and
`Adj + Nn -> Nn`, plus controls that give the same surface patterns without category preservation:
`Det + Nn -> NP` with bare-noun sentences (`dog sleeps.` / `the dog sleeps.`), `Vt + NP -> VP`,
`Nn + VP -> S`.

Per source and COMBINED, at support thresholds 1 / 2 / 3 / 5: candidate precision / recall / F1
(type level, and restricted to posterior-constituent candidates), LEFT / RIGHT separately,
occurrence-level precision (handles ambiguous words), direction accuracy, host-class accuracy
(class hosts, after the run's class alignment; a ROOT host is reported, not scored), insertion
precision / recall, and the gold attachment level of `AB.`/`AXB.` insertions.

## 4. Stage 2 — using the evidence (acceptance stays `dF > tau`)

* `modifier.mode = string_only | string_plus_class`: endomorphic rules the evidence asks for but the
  grammar lacks are proposed as `rule_add(C, C, M)` / `rule_add(C, M, C)` (host = the evidence host,
  the root class for ROOT evidence; candidate class = dominant posterior class of the candidate);
  ranked before the adjacency-based rule proposals.  `posterior_rule_only` and `off` are controls.
* Lexicon-add distinguishability (`modifier.distinguishability`): for a proposed entry `(B, w)`
  the trial posterior gives per-occurrence composition profiles (role, result, sibling) and
  modifier-role profiles (LEFT / RIGHT modifier of host X, host of a modifier X, category-changing
  composition) of `w` under `B` and under each existing `(A, w)`;
  `D_comp`, `D_modifier` = Jensen–Shannon divergence to the closest `A`;
  `D_cover` = share of the new entry's leaf mass in sentences the old grammar could not derive as one
  root (derivability gain).  `D_struct` = `l_c D_comp + l_m D_modifier` (`combine: sum`, the
  first-version form) or `D_cover * (l_c D_comp + l_m D_modifier) / (l_c + l_m)` (`gated`);
  `penalty = lambda_dup * exp(-D_struct / tau)`; acceptance `dF - penalty > tau_rule`.  Every proposal
  is logged in `lexical_distinguishability.json` with both distances, the penalty, the decision,
  and the `lambda_dup` that would have been needed to reject it.

Adaptive constituent contexts are off in the whole study (`adaptive_context.enabled: false`).

## 5. Suite

`configs/experiments_v03.yaml`: A baseline, B + diagnostics, C + exact-string proposals,
D + class-generalised proposals, E D + composition penalty, F D + composition + modifier penalty,
G penalty only; on Synthetic-4, Synthetic-1 (regression gate), Synthetic-2 (ambiguity must survive),
Synthetic-3 (noise), and Synthetic-4 from random initialisation with K = 8 (forced class collapse).
`report_v03.md` + `docs/analysis_v03.md` hold the results and the answers to RQ1–RQ8; the
natural-language part (`scripts/run_natural.py`, TinyStories 10k, disjoint 5k halves) reports
candidate / direction / host agreement per source next to the general rule agreement.
