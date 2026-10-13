"""Composition evidence N(A, B, C) and the relation set R <= E^3.

For every classified occurrence ``x = s[i:j]`` and every binary split
``x = u v`` with ``u = s[i:k]`` in class A and ``v = s[k:j]`` in class B we
accumulate ``N(A, B, C)``.  Composition is a partial relation: nothing forces
``(A B) C = A (B C)``, and ``A + B`` may map to several result classes.

The module also records, per occurrence, every licensed split (derivational
hyperpaths) so that structural ambiguity can be reported without being
resolved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spans import SpanInventory


@dataclass
class CompositionRelation:
    """``left_class + right_class -> result_class`` with its evidence."""

    left_class: int
    right_class: int
    result_class: int
    support: int
    distinct_lexical_support: int
    probability: float
    examples: list[tuple[str, str, str]] = field(default_factory=list)
    distinct_left: int = 0
    distinct_right: int = 0

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.left_class, self.right_class, self.result_class)

    def to_json(self, names: dict[int, str] | None = None) -> dict[str, Any]:
        nm = names or {}
        return {
            "left_class": self.left_class,
            "right_class": self.right_class,
            "result_class": self.result_class,
            "label": f"{nm.get(self.left_class, self.left_class)} + {nm.get(self.right_class, self.right_class)} -> {nm.get(self.result_class, self.result_class)}",
            "support": self.support,
            "distinct_lexical_support": self.distinct_lexical_support,
            "distinct_left_realisations": self.distinct_left,
            "distinct_right_realisations": self.distinct_right,
            "probability": round(self.probability, 4),
            "examples": [f"{u} | {v} => {x}" for u, v, x in self.examples[:5]],
        }


@dataclass
class CompositionEvidence:
    """Aggregated composition statistics over a class assignment."""

    counts: dict[tuple[int, int, int], int]
    lexical_pairs: dict[tuple[int, int, int], set[tuple[int, int]]]
    examples: dict[tuple[int, int, int], list[tuple[str, str, str]]]
    occ_splits: dict[int, list[tuple[int, int, int]]]  # oid -> [(k, A, B)]
    n_occurrences_with_split: int
    n_occurrences_multi: int

    def side_counts(self, key: tuple[int, int, int]) -> tuple[int, int]:
        """Distinct left and right span types realising a relation."""
        pairs = self.lexical_pairs.get(key, set())
        return len({u for u, _ in pairs}), len({v for _, v in pairs})

    def relations(self, eta: float, n_classes: int, min_support: int = 1, min_lexical: int = 1, min_side: int = 1) -> list[CompositionRelation]:
        """Build the relation list with ``P(C | A, B) = (N_ABC + eta) / (sum_C' N_ABC' + K eta)``.

        A relation is productive only if it is realised by >= ``min_lexical``
        distinct (left, right) pairs and by >= ``min_side`` distinct span
        types on *each* side (a rule whose right-hand side is always the same
        one or two expressions is a sign of mis-assigned members, not of a
        general composition).
        """
        by_pair: dict[tuple[int, int], float] = {}
        for (a, b, c), n in self.counts.items():
            by_pair[(a, b)] = by_pair.get((a, b), 0.0) + n
        out = []
        for (a, b, c), n in self.counts.items():
            lex = len(self.lexical_pairs[(a, b, c)])
            nl, nr = self.side_counts((a, b, c))
            if n < min_support or lex < min_lexical or nl < min_side or nr < min_side:
                continue
            p = (n + eta) / (by_pair[(a, b)] + n_classes * eta)
            out.append(CompositionRelation(a, b, c, int(n), lex, p, self.examples.get((a, b, c), []), nl, nr))
        out.sort(key=lambda r: (-r.support, r.key))
        return out

    def restricted(self, classes: set[int]) -> "CompositionEvidence":
        keep = {k for k in self.counts if k[0] in classes and k[1] in classes and k[2] in classes}
        return CompositionEvidence(
            {k: self.counts[k] for k in keep},
            {k: self.lexical_pairs[k] for k in keep},
            {k: self.examples.get(k, []) for k in keep},
            {o: [s for s in v if (s[1], s[2]) and s[1] in classes and s[2] in classes] for o, v in self.occ_splits.items()},
            self.n_occurrences_with_split,
            self.n_occurrences_multi,
        )


def collect_composition(inv: SpanInventory, occ_class: np.ndarray, max_examples: int = 5) -> CompositionEvidence:
    """Accumulate ``N(A, B, C)`` over all binary splits of all classified occurrences."""
    counts: dict[tuple[int, int, int], int] = {}
    lex: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    examples: dict[tuple[int, int, int], list[tuple[str, str, str]]] = {}
    occ_splits: dict[int, list[tuple[int, int, int]]] = {}
    n_multi = 0
    n_with = 0
    occ_sid, occ_start, occ_end, occ_tid = inv.occ_sid, inv.occ_start, inv.occ_end, inv.occ_tid
    for oid in np.nonzero((occ_end - occ_start) >= 2)[0].tolist():
        c = int(occ_class[oid])
        if c < 0:
            continue
        n_multi += 1
        sid, i, j = int(occ_sid[oid]), int(occ_start[oid]), int(occ_end[oid])
        splits = []
        for k in range(i + 1, j):
            lo = inv.lookup(sid, i, k)
            ro = inv.lookup(sid, k, j)
            if lo < 0 or ro < 0:
                continue
            a, b = int(occ_class[lo]), int(occ_class[ro])
            if a < 0 or b < 0:
                continue
            key = (a, b, c)
            counts[key] = counts.get(key, 0) + 1
            pair = (int(occ_tid[lo]), int(occ_tid[ro]))
            s = lex.setdefault(key, set())
            if pair not in s:
                s.add(pair)
                ex = examples.setdefault(key, [])
                if len(ex) < max_examples:
                    ex.append((inv.type_str(pair[0]), inv.type_str(pair[1]), inv.occ_str(oid)))
            splits.append((k, a, b))
        if splits:
            n_with += 1
            occ_splits[oid] = splits
    return CompositionEvidence(counts, lex, examples, occ_splits, n_with, n_multi)


def structural_ambiguity_report(inv: SpanInventory, ev: CompositionEvidence, limit: int = 30) -> dict[str, Any]:
    """Occurrences licensed by >= 2 different splits (derivational hyperpaths)."""
    multi = {o: s for o, s in ev.occ_splits.items() if len(s) >= 2}
    examples = []
    for oid, splits in list(multi.items())[:limit]:
        sid, i, j = int(inv.occ_sid[oid]), int(inv.occ_start[oid]), int(inv.occ_end[oid])
        toks = inv.corpus.sentences[sid].tokens
        examples.append(
            {
                "span": inv.occ_str(oid),
                "sentence": " ".join(toks),
                "hyperpaths": [
                    {"split": " ".join(toks[i:k]) + " | " + " ".join(toks[k:j]), "left_class": a, "right_class": b} for k, a, b in splits
                ],
            }
        )
    return {
        "n_occurrences_with_multiple_splits": len(multi),
        "n_occurrences_with_split": ev.n_occurrences_with_split,
        "n_multi_token_classified_occurrences": ev.n_occurrences_multi,
        "examples": examples,
    }


def productivity_score(n_lexical_pairs: int, scale: float = 10.0) -> float:
    """Saturating score in [0, 1): 1 - exp(-pairs / scale)."""
    return 1.0 - math.exp(-n_lexical_pairs / scale)
