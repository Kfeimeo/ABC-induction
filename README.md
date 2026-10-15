# ABC induction v0.2 — raw text → (E, R), now as a finite Bayesian latent grammar

Research prototype for **unsupervised discovery of constituent equivalence
classes and binary composition relations** from raw text:

```
raw corpus  →  ( E , R )        E : constituent equivalence classes
                                R ⊆ E³ : A + B → C  (partial, non-associative, non-deterministic)
```

No CCG categories, POS tags, treebanks, supertaggers, pretrained parsers or
embeddings are used; gold synthetic labels are used for evaluation only.

v0.1 (`docs/report_v01.md`) answered *does the abstraction carry signal?* with
context clustering + heuristic constituency + MDL elimination + one-shot
ambiguity splits.  **v0.2 keeps the whole v0.1 pipeline as initialisation and
regression baseline** and replaces its hard decisions by posterior marginals:

```
finite Bayesian latent grammar (sparse PCFG over E, Dirichlet posteriors)
+ sparse semiring Inside–Outside (python reference + numba backend)
+ Dirichlet CAVI with a computable ELBO
+ objective F(G) = ELBO + log P(G)   (centralised structure prior, private-rule penalty)
+ greedy class merge / class split / rule & lexicon proposals with checkpoint + rollback
+ lexical contexts as leaf potentials, punctuation as span potentials,
  adaptive constituent contexts as proposal features only
+ ProbLog and brute-force exact oracles on tiny grammars (tests only)
```

The mathematics (model, recurrences, CAVI updates, ELBO, prior, merge/split,
oracle encoding, and the one documented deviation from the spec — top-down
expansion distributions instead of bottom-up `P(C | A, B)` as the generative
parameters) is in [docs/v02_model.md](docs/v02_model.md).  The experiment
report with the v0.1 vs v0.2 tables, failure-mode checklist and RQ answers is
`report.md` (authored analysis in `docs/analysis_v02.md`).  Where the time
goes and what was optimised is in `docs/profiling.md`.  The v0.3 study of
identity-preserving (modifier) evidence — design in `docs/modifier_study.md`,
results and RQ answers in `report_v03.md` / `docs/analysis_v03.md` — sits on
top of the unchanged v0.2 core.

## Pipeline

| stage | module | what it does |
|---|---|---|
| v0.1 stages | `spans, contexts, clustering, constituency, mdl, composition, ambiguity` | unchanged (see `docs/report_v01.md`); `ambiguity` = `legacy_ambiguity_split`, off by default in v0.2 runs, on in the v0.1 rows |
| grammar model | `grammar_model.py` | `GrammarModel`: active classes, sparse rules `(C, A, B)`, sparse lexicon `(C, w)` + `<unk>`, root, chunk Beta; Dirichlet posteriors `α' = α + E[N]`; `E_q[log θ]`; derived `P(C \| A, B)` and type mixtures `π_x`; `StructurePrior` (`log P(G)`); merge / split operations with lineage; checkpoint / restore |
| chart | `chart.py` | sparse semiring chart (reference): inside, outside, node / rule-application posteriors, expected counts, Viterbi, chunk-sequence layer, beam / threshold pruning with dropped-mass accounting, `brute_force` enumerator |
| accelerated chart | `chart_numba.py` | same recurrences on dense `(n, n, K)` log tables with per-cell active lists and CSR rule index; in-kernel expected counts; identical results (tested to 1e-9) |
| inference | `cavi.py` | `VariationalInference`: E-step (either backend), M-step, ELBO components, trace rows; `Potentials` (leaf `λ`, span `ψ`) |
| structure learning | `structure_learning.py` | signatures (raw context, composition in/out, lexicon, abstract context), inverted-index merge candidates, weighted 2-means split candidates, rule / lexicon prune & add proposals, `ΔF` acceptance with rollback, monotonicity check, `structure_moves.jsonl` |
| adaptive context | `adaptive_context.py` | soft abstract constituent contexts `(L, _, R)` along the constituent hierarchy; depth-2 refinement for flagged classes; information-gain feature selection |
| v0.2 stage | `probabilistic.py`, `v02_stage.py` | init from v0.1 / random, lexical-context potentials (v0.1 scorer, leave-one-out), punctuation potentials, posterior decoding (occurrence classes, constituents, type mixtures, relations, occurrence ambiguity), `posterior_stats.json` |
| evaluation | `evaluation_v02.py` | v0.1 metric definitions on the v0.2 posterior + the failure-mode checklist (fragmentation, contamination, extra root classes, private crutch rules, filler-induced rules, ambiguity pollution) |
| oracle | `problog_reference/` | ProbLog encoding (Gorn-address annotated disjunctions) and exact queries; optional dependency, tests skip without it |
| v0.3 modifier evidence | `modifier_evidence.py`, `modifier_eval.py` | identity-preserving (endomorphic) constituent evidence `C + M -> C` / `M + C -> C` from exact sentence strings (`A.`/`AB.`, `A.`/`XA.`, `AB.`/`AXB.`), exact constituent strings, class-generalised spans and the endomorphic-rule expected counts, reported per source; gold modifier evaluation; optional modifier rule proposals and a duplicate-lexical-entry penalty (`docs/modifier_study.md`) |

## Layout

```
src/abc_induction/   package
grammars/            synthetic grammar definitions (clean / ambiguity / noisy)
configs/             experiments.yaml (v0.1 suite), experiments_v02.yaml (v0.2 suite), per-dataset configs
scripts/             run_experiment.py, run_all.py, run_natural.py, prepare_tinystories.py, debug_run.py, profile_run.py
tests/               exact-inference, backend-equivalence, CAVI, structure-learning and pipeline tests; tests/problog_reference/
outputs/<run>/       v0.1 artefacts + objective_trace.csv, structure_moves.jsonl, posterior_stats.json, classes_v02.json, relations_v02.json, ambiguity_v02.json
outputs/v02/         the v0.2 synthetic suite; outputs/natural/ the TinyStories study
docs/                v02_model.md (maths), analysis_v02.md (authored analysis), report_v01.md (v0.1 report), analysis.md, profiling.md
report.md            project report (v0.1 vs v0.2)
```

## Reproduce everything

```bash
# Python 3.11+, numpy, scipy, scikit-learn, pandas, pyyaml, psutil, numba, pytest;  optional: problog (oracle tests only)
python -m pytest tests -q -p no:cacheprovider          # ~2 min incl. ProbLog oracle (skipped if problog missing)
python scripts/run_all.py --suite configs/experiments_v02.yaml --out outputs/v02 --analysis docs/analysis_v02.md   # v0.2 suite -> report.md
python scripts/run_all.py --suite configs/experiments_v03.yaml --out outputs/v03 --analysis docs/analysis_v03.md --report report_v03.md   # v0.3 modifier study -> report_v03.md
python scripts/run_all.py                                                     # v0.1 regression suite (v0.2 stage off) -> report.md
python scripts/prepare_tinystories.py                                         # raw TinyStories prefix (text only)
python scripts/run_natural.py --sizes 10000 --punct-ablation --no-seed-run    # TinyStories 10k: stability halves, coverage, punctuation modes
python scripts/run_experiment.py configs/synthetic2_ambiguity.yaml --set structure_learning.proposal_eval=affected   # single run with overrides
```

Every run records `seed, config, git commit (None: not a git repository), runtime, peak memory`
in `metrics.json`; the v0.1 stage is cached per v0.1-relevant config (`outputs/_cache_v01`).

## Configuration (v0.2 keys, defaults in `experiment.DEFAULT_CONFIG`)

```yaml
probabilistic:      {enabled, init: v01|random, k_max, alpha_rule, gamma_type, alpha_unk, alpha_root, chunks_enabled,
                     max_cavi_iterations, cavi_tolerance, constituent_threshold, ambiguity_mass, ambiguity_min_occurrences}
structure_prior:    {lambda_class, lambda_rule, lambda_lex, lambda_private, private_tau, private_min_count}
structure_learning: {enabled, merge_enabled, split_enabled, rule_proposals_enabled, merge_threshold, split_threshold,
                     rule_threshold, proposal_vb_iterations, proposal_eval: full|affected, max_rounds,
                     rejection_memory (off; see docs/profiling.md), rejection_margin, ...}
lexical_context:    {enabled (ELBO factor), proposals (lexicon-add source), max_scale, temperature, top_k, floor}
adaptive_context:   {enabled, constituent_posterior_threshold, max_refinement_depth, top_features}
punctuation:        {mode: punctuation_as_token|punctuation_removed|structural_prior, sentence_final_weight, paired_weight, comma_weight}
chart:              {backend: python|numba, exact, max_classes_per_cell, posterior_prune_threshold}
modifier:           {mode: off|string_only|string_plus_class|posterior_rule_only, diagnostics, max_modifier_length, max_insert_span_length,
                     constituent_threshold, min_support, min_mod, max_rule_proposals, proposal_min_support,
                     distinguishability: {enabled, combine: sum|gated|cover, lambda_comp, lambda_modifier, lambda_dup, tau}}
problog_reference:  {enabled: false}      # never used by the pipeline
legacy_ambiguity_split: alias of the v0.1 `ambiguity` block
```

## Reading a run

`outputs/<run>/report.md` has the v0.1 sections (why merged / why rejected /
why split) followed by the v0.2 section: objective table (ELBO, KL, prior,
classes, rules), the **v0.1 vs v0.2 metric table on the same gold**, posterior
class views (lexical entries, span types, occurrences, contexts, composition
in/out, abstract contexts), every structure move with `why proposed`,
occurrence partition, `ΔELBO`, `Δprior`, `ΔF`, accepted/rejected, the
highest-entropy span types (occurrence-ambiguity candidates, not linguistic
claims), private rules, and the failure-mode checklist.  Three terms are kept
apart throughout: *occurrence ambiguity* (one type, several posterior
classes), *class split* (`A → A1, A2`, inventory change) and *structural
ambiguity* (several derivations, marginalised).

## Scope limits of v0.2

* finite `K_max`; Bayesian nonparametrics (HDP / Pitman–Yor / adaptor grammars) are future work;
* expected counts via inside+outside (no expectation semiring yet); no C++ backend (numba is sufficient up to the sizes reported);
* proposal evaluation on *affected* sentences is an approximation (exact re-score after acceptance);
* the chunk-sequence fallback is the only way an incomplete grammar can score a sentence it cannot derive as one root; with it off, such sentences have `log Z = −∞` and must be avoided by construction.
