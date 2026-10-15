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
8. Identity-preserving candidate sets are 1.5–3.6 times more stable across disjoint 5k halves than
   the general relations (candidate Jaccard 0.24 string / 0.14 class / 0.10 rule vs 0.066 for
   `R(A, B, C)`); their direction is stable only for the rule-posterior candidates (1.00) and not
   for the string / class ones (0.58–0.60); their host class is not stable (0.08–0.17), because the
   classes are not.  The root exact-string anchor does not exist at 10k real sentences (19 pairs).
9. As an anchor for *which strings* get `X/X` / `X\X` types: yes, with direction for the
   high-productivity ones.  As an anchor for *which X*: not until the class inventory is aligned
   through them rather than before them.
