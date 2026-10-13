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
