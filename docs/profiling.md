# Profiling the induction stage

`python scripts/profile_run.py <config> --out outputs/profile/<name> [--set k=v ...]` runs one
experiment under `cProfile` and writes `profile.txt` (per-stage wall time, top functions by own
and cumulative time) and `profile.prof`.

## Where the time goes (before / after the incremental-MDL optimisation)

TinyStories, 4,000 sentences (5,783 span types, 3,233 provisional → ~800 accepted classes):

| stage | before | after |
|---|---|---|
| spans + contexts + clustering | 4.2 s | 4.2 s |
| constituency (MDL elimination) | 131.4 s | 43.1 s |
| ambiguity (MDL hypothesis tests) | 65.5 s | 27.1 s |
| **total** | **201.3 s** | **74.6 s** (2.7×) |

Synthetic-1, 10,000 sentences (389 provisional → 10 classes): 62.4 s → ~50 s (grammar is small, so
the closed form was never the bottleneck there).

Before the change 70% of the MDL time was `Grammar.refresh` (14,930 calls rebuilding all class
totals) and the full-table closed form (`data_bits_closed_form`, 8,373 calls over every (C, e)
count).  Now the data cost is maintained per class: a removal / split test re-parses the affected
sentences, applies the (scaled) count deltas, re-evaluates only the touched classes and undoes
the change exactly (`Grammar.apply_delta` / `undo`); rule, leaf and root costs are cached while
the counts are unchanged.

Remaining hot spots (after):

| function | own time | share |
|---|---|---|
| `viterbi_sentence` (91k parses, 0.43 ms each) | 24.5 s | 32% |
| dict lookups inside the parser | 13.1 s | 17% |
| `Grammar.refresh` (706 structural changes) | 3.8 s | 5% |
| `class_bits` / `root_bits` (closed form) | 6.0 s | 8% |

The parser is now the dominant cost; the next step is a compiled CKY (numba / C++) or a
per-sentence pre-tabulation of rule costs by (A, B) pair.

## Semantics note

Class / rule removal and split application now re-parse the affected sentences with *frozen*
tables and commit all count changes afterwards.  The previous implementation updated the count
numerators while iterating (with stale denominators), so description lengths differ by a few
tenths of a percent (TinyStories 4k: 256,987 → 257,327 bits, 799 → 795 classes); synthetic
metrics are unchanged.

# Profiling the v0.2 stage (CAVI + structure learning)

`python scripts/profile_run.py configs/synthetic1_clean.yaml --out outputs/profile/clean` profiles a
full v0.2 run.  Note that cProfile inflates Python-heavy code 2–3× (an E-step took 0.19 s under the
profiler and 0.07 s without), so decide from the `runtime_sec` in `metrics.json`, not from the
profiler's absolute numbers.

## Where the time went (synthetic suite, full v0.2 = merge/split/rule proposals + adaptive contexts)

Wall times of single runs, four runs sharing a 4-core machine; v0.1 stage from cache.

| run | before | after | v0.2 structure learning before → after |
|---|---|---|---|
| Synthetic-1 (3,000 sentences) | 133 s | 22 s | 121 s → 18 s |
| Synthetic-2 (3,000) | 132 s | 21 s | 116 s → 16 s |
| Synthetic-3 (3,000) | 100 s | 31 s | 91 s → 25 s |
| Synthetic-1, 10,000 sentences | 437 s | 85 s | 381 s → 69 s |

Before, 93–97% of a run was `StructureLearner.run`; v0.1 took 7–10 s (39 s at 10k).  The objective
trace, every structure move (`dF`, accepted/rejected) and all v0.2 metrics are bit-identical before
and after on all three grammars (same v0.1 cache), see below for the one caveat.

### 1. Proposal evaluation = `(proposal_vb_iterations + 1)` full-corpus E-steps per proposal

With `proposal_eval: full`, `StructureLearner.evaluate` runs 5 short CAVI iterations plus one
scoring E-step over *all* sentences for every proposal: 110–140 proposals per run → ~790 full
E-steps, ~88% of them on proposals that are rejected.  Every round re-proposes and re-evaluates the
same rejected merges / splits / rule additions (Synthetic-1: 72 `rule_add` evaluations are 12
distinct rules × 6 rounds, with the same `dF ≈ -10` each time).

`structure_learning.rejection_memory: true` skips a proposal rejected by more than
`rejection_margin` (5 nats) until an accepted merge / split / rule move touches one of its classes
(Synthetic-1: 93 of 123 evaluations skipped).  It is **off by default** because it is not exact:
a merge rejected with `dF = -42` in round 0 was accepted with `dF = +7` in round 1 after lexicon
additions in *unrelated* classes (Synthetic-2), and the same happened to splits on Synthetic-1 at
10k sentences, so with the memory on the accepted-move sequence (and the final metrics) changes.
`proposal_eval: affected` is the other lever; on the synthetic grammars the affected sets are 1/3 to
all of the corpus, so it only saved 20–25% and also changed the results.  Both stay opt-in; after
the two changes below the E-steps are no longer the bottleneck on this suite, so neither is needed.

### 2 + 3. The E-step: 70% Python glue around a 7 µs kernel

Per sentence (Synthetic-1, K = 8, mean length 4, no profiler): numba kernel 7 µs,
`Potentials.leaf_array` 7 µs (pure-Python `n × K` loop with `dict.get`, 117 M calls per run),
`run_sentence` wrapper 8 µs (`np.where/isfinite/astype` per sentence, ~30 arguments through the
dispatcher, 15-tuple return), the `e_step` loop ~10 µs (Python loop over the realisation buffer,
`vocab_size` rescanned every call) — 24 µs in total, ≈ 70 ms per E-step.

Now `chart_numba.e_step` is one jitted call (`_batch`) over all requested sentences: tokens, span
potentials and span-type ids are concatenated once per corpus (`_StaticArrays`), the leaf
potentials are materialised once per `(class_ids, class_map)` as an `(n_tokens, K)` slab
(`_leaf_slab`, rebuilt only when a proposal changes the class layout), and node posteriors / rule
applications / realisations come back as flat arrays with per-sentence offsets (buffers grow on
demand).  Sums are accumulated in the same order as the per-sentence reference, so the two
backends stay bit-identical: 70 ms → 14 ms per E-step (5×), 21 ms with posterior collection.

### 4. Split proposals: scipy row slicing and KMeans threads

After 2 + 3, `propose_splits` was 80% of the remaining structure-learning time:
`_weighted_two_means` computed the within / total sums of squares by slicing one sparse row at a
time (300k `csr[r]` calls per run), and `KMeans(n_clusters=2)` on a few thousand sparse rows took
0.47 s per call with the default OpenMP thread pool against 9 ms single-threaded (thread
contention; worse with `run_all.py`'s parallel workers).  The sums are now computed from
`||x||² - 2 x·m + ||m||²` on the whole matrix (equal to 2e-15) and KMeans runs under
`threadpool_limits(1)`.  Split quality / balance and the chosen labels are unchanged.

### Reproducibility note

The v0.2 trajectory is chaotic with respect to the last bits of floating point: Synthetic-3 run
with the v0.1 stage computed in-process vs. loaded from `outputs/_cache_v01` (the same numbers
through a pickle round trip) differs in the initial KL at 1e-14 and ends with 83 instead of 82
structure moves; Synthetic-2 on this machine ends on a different fixed point (F = -42072,
ambiguity F1 1.0) than the one in `report.md` (F = -40701, ambiguity F1 0.62), with the unmodified
code.  Comparisons between two runs should therefore use the same cache state and the same
machine / library versions.

Remaining hot spots after these changes (Synthetic-1): `context_potentials` 3–4 s (per-token
sparse column slicing, 14 s at 10k sentences), the numba JIT compile ~13 s on the first run after a
code change (cached in `__pycache__` afterwards), and KMeans in `propose_splits` (~1 s per round).
