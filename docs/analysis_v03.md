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

**Two sentence populations.**  The grammar runs count far fewer root pairs at the same N (10k:
19 / 1 / 18; 25k: 104 / 9 / 92; 50k: 399 / 29 / 406).  The `ROOT_STRING` definition is the same in
both — applying `root_string_scan.scan` to the pipeline's own sentence files reproduces the
pipeline's numbers exactly, and `ModifierAnalyzer` on those files gives the same — the difference
is the population.  `run_natural.load_sentences` removes every repeated token sequence *before*
shuffling (769,810 distinct sentences; `dataset.remove_duplicate_sentences: true`), so its N
sentences are a sample of sentence *types*; the table above keeps every line (917,854), so its N
sentences are a sample of sentence *occurrences*.  Frequent short sentences — the `A.` side of
`A.`/`AB.` — are almost surely in an occurrence sample and are drawn with probability N / 770k in
a type sample, hence the order of magnitude; at full size the two coincide (76,940 pairs either
way).  On the type population (`--dedup`, `docs/root_scan_tinystories_dedup.md`):

| N (types) | `A.`/`AB.` | `A.`/`XA.` | `AB.`/`AXB.` | RIGHT cand. (prod. >= 3 / single-token) | LEFT | INTERNAL | halves Jaccard R / L / I |
|---|---|---|---|---|---|---|---|
| 10,000 | 19 | 1 | 18 | 0 / 0 | 0 / 0 | 0 / 0 | – |
| 25,000 | 104 | 9 | 92 | 0 / 0 | 0 / 0 | 3 / 3 | – |
| 50,000 | 399 | 29 | 406 | 8 / 1 | 2 / 0 | 13 / 9 | 0 / 0 / 0.33 |
| 100,000 | 1,583 | 121 | 1,599 | 49 / 8 | 8 / 2 | 66 / 32 | 0.27 / 0.20 / 0.31 |
| 200,000 | 6,145 | 551 | 6,274 | 277 / 46 | 28 / 2 | 358 / 134 | 0.30 / 0.64 / 0.32 |
| 500,000 | 36,260 | 3,093 | 37,673 | 2,077 / 296 | 162 / 7 | 2,031 / 409 | 0.41 / 0.43 / 0.46 |
| 769,810 | 76,940 | 6,818 | 86,633 | 4,576 / 530 | 325 / 15 | 4,226 / 613 | 0.43 / 0.42 / 0.47 |

Pairs grow roughly quadratically in either population (the chance that a sentence recurs with one
span removed grows with the corpus).  On the occurrence population the anchor becomes usable
between 100k and 200k sentences and its candidate sets agree across disjoint halves at Jaccard
0.48–0.61 from 200k on; on the type population — the one the grammar runs use — it needs 200k–500k
and reaches 0.41–0.47.  Both are above the constituent-level string agreement of the grammar runs
(0.14–0.24) and the rule agreement (0.07–0.13).  Which population is the right one for an anchor is
a modelling choice: the occurrence population is what raw text offers; the grammar deduplicates
because repeated sentences would otherwise dominate its counts.  Either way the candidate lists
below are the same strings.
What it finds at 200k–900k (occurrence population; the type population gives the same strings
in a slightly different order), by productivity, single tokens only:

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
   exact-string anchor needs 100k–200k sentences on the occurrence population (175 pairs at 10k,
   11k at 200k) and 200k–500k on the deduplicated type population the grammar runs use (19 at 10k,
   399 at 50k, 6.1k at 200k); from there its candidate sets agree across halves at 0.41–0.61, above
   every grammar-based figure.
9. As an anchor for *which strings* get `X/X` / `X\X` types: yes, with direction for the
   high-productivity ones.  As an anchor for *which X*: not until the class inventory is aligned
   through them rather than before them.
