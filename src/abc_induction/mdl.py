"""Minimum-description-length scoring of a grammar (E, R) against the corpus.

The grammar is a set of accepted classes ``E``, allowed top-down expansions
``C -> A B`` (from the composition relations) and lexical entries
``C -> token``.  A sentence is encoded as a sequence of chunks; every chunk is
a binary derivation over the *classified* candidate spans (the class of every
occurrence is fixed; only the bracketing is chosen).  Tokens that cannot be
covered by any accepted span cost ``c_unk`` bits.

Two-part description length with maximum-likelihood parameters
---------------------------------------------------------------

    DL = DL(grammar) + DL(derivations | grammar)

    DL(grammar) = c_class |E| + c_rule |R| + c_lex |lexical entries|
                  + c_phrase_entry |multi-token leaves in use|

    DL(derivations) = sum_(C,e) N(C,e) * -log2 P(e | C)
                      + sum_C N_root(C) * -log2 P_root(C)
                      + c_chunk * #chunks + c_unk * #uncovered tokens

where ``N(C, e)`` counts how often class ``C`` expands as ``e`` (a rule
``A B``, a lexical token or a multi-token phrase leaf) in the current Viterbi
derivations and ``P(e | C) = (N(C,e) + eta) / (N_C + eta K_C)``.

Because the data cost is a closed-form function of the derivation counts,
removing a class (or a rule, or re-assigning the occurrences of an ambiguous
expression) is evaluated by re-parsing only the sentences whose derivation
used the removed item, updating the counts and re-evaluating the closed form:
all probabilities are re-normalised consistently and no single analysis is
locked in by the order of greedy decisions.

Efficiency: the data cost decomposes per class, so a hypothesis only
re-evaluates the classes whose counts changed (``apply_delta`` / ``undo``),
and the bit costs of rules, leaves and roots are cached while the counts are
unchanged (the cache is invalidated by every count or structure change).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .spans import SpanInventory

log = logging.getLogger(__name__)
INF = float("inf")

Expansion = tuple  # ('r', A, B) | ('l', tid) | ('p', tid)


@dataclass
class MDLCosts:
    """Grammar-size costs (bits) and data-cost constants."""

    c_class: float = 12.0
    c_rule: float = 10.0
    c_lex: float = 8.0
    c_phrase_entry: float = 12.0
    c_phrase_use: float = 8.0
    c_unk: float = 20.0
    c_chunk: float = 6.0
    eta: float = 0.5

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> "MDLCosts":
        cfg = cfg or {}
        return cls(**{k: float(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class Node:
    """A node of a Viterbi derivation (``split < 0`` = leaf; ``oid < 0`` = span
    predicted by the grammar, i.e. not a candidate occurrence)."""

    start: int
    end: int
    cls: int
    split: int
    oid: int
    left_cls: int = -1
    right_cls: int = -1


@dataclass
class SentenceParse:
    cost: float
    nodes: list[Node]
    chunks: int
    unk_tokens: int
    roots: list[tuple[int, int, int]] = field(default_factory=list)  # (start, end, cls)
    keys: list[Expansion] = field(default_factory=list)  # (C, expansion) per node


@dataclass
class DeltaRecord:
    """Exact undo information for ``Grammar.apply_delta``."""

    saved_counts: dict[tuple[int, Expansion], float]
    saved_total: dict[int, float]
    saved_k: dict[int, int]
    saved_roots: dict[int, float]
    saved_root_total: float
    touched: set[int]
    saved_caches: tuple[dict, dict, dict]
    saved_version: int


class Grammar:
    """Accepted classes, allowed expansions and usage counts -> bit costs.

    ``_total[c]`` (sum of positive counts of ``c``), ``_k[c]`` (number of
    expansions in the smoothing denominator) and ``exp_index[c]`` (expansions
    ever counted for ``c``) are maintained incrementally; ``refresh`` rebuilds
    them from scratch after structural changes (classes / rules / lexicon).
    """

    def __init__(self, costs: MDLCosts) -> None:
        self.costs = costs
        self.classes: set[int] = set()
        self.rules: set[tuple[int, int, int]] = set()  # allowed (C, A, B)
        self.lex: set[tuple[int, int]] = set()  # allowed (C, tid) single token
        self.counts: dict[tuple[int, Expansion], float] = {}
        self.root_counts: dict[int, float] = {}
        self._total: dict[int, float] = {}
        self._k: dict[int, int] = {}
        self._root_total = 0.0
        self.rules_by_children: dict[tuple[int, int], list[int]] = {}
        self.exp_index: dict[int, set[Expansion]] = {}
        self._rule_cache: dict[tuple[int, int, int], float] = {}
        self._leaf_cache: dict[tuple[int, int, int], float] = {}
        self._root_cache: dict[int, float] = {}
        self.version = 0

    # -------------------------------------------------------------- tables
    def invalidate(self) -> None:
        """Drop the cost caches (counts or structure changed)."""
        self._rule_cache = {}
        self._leaf_cache = {}
        self._root_cache = {}
        self.version += 1

    def refresh(self) -> None:
        """Rebuild every derived table from ``classes / rules / lex / counts``."""
        tot: dict[int, float] = {}
        k: dict[int, int] = {}
        idx: dict[int, set[Expansion]] = {}
        for (c, a, b) in self.rules:
            if c in self.classes and a in self.classes and b in self.classes:
                k[c] = k.get(c, 0) + 1
        for (c, t) in self.lex:
            if c in self.classes:
                k[c] = k.get(c, 0) + 1
        for (c, e), n in self.counts.items():
            idx.setdefault(c, set()).add(e)
            if c not in self.classes or n <= 0:
                continue
            tot[c] = tot.get(c, 0.0) + n
            if e[0] == "p":
                k[c] = k.get(c, 0) + 1
        self._total = tot
        self._k = k
        self.exp_index = idx
        self._root_total = sum(v for c, v in self.root_counts.items() if c in self.classes)
        by_children: dict[tuple[int, int], list[int]] = {}
        for (c, a, b) in self.rules:
            if c in self.classes and a in self.classes and b in self.classes:
                by_children.setdefault((a, b), []).append(c)
        self.rules_by_children = by_children
        self.invalidate()

    def add_count(self, key: tuple[int, Expansion], delta: float) -> None:
        """Incrementally add ``delta`` to one count (keeps totals / K consistent)."""
        c, e = key
        old = self.counts.get(key, 0.0)
        new = old + delta
        self.counts[key] = new
        self.exp_index.setdefault(c, set()).add(e)
        if c in self.classes:
            self._total[c] = self._total.get(c, 0.0) + max(new, 0.0) - max(old, 0.0)
            if e[0] == "p":
                self._k[c] = self._k.get(c, 0) + int(new > 0) - int(old > 0)

    def add_root(self, c: int, delta: float) -> None:
        self.root_counts[c] = self.root_counts.get(c, 0.0) + delta
        if c in self.classes:
            self._root_total += delta

    def add_lex(self, key: tuple[int, int]) -> bool:
        """Add an allowed lexical entry; returns True if it was new."""
        if key in self.lex:
            return False
        self.lex.add(key)
        if key[0] in self.classes:
            self._k[key[0]] = self._k.get(key[0], 0) + 1
        self.invalidate()
        return True

    def remove_lex(self, key: tuple[int, int]) -> None:
        if key in self.lex:
            self.lex.discard(key)
            if key[0] in self.classes:
                self._k[key[0]] = self._k.get(key[0], 0) - 1
            self.invalidate()

    # -------------------------------------------------------------- costs
    def _logp(self, c: int, n: float) -> float:
        eta = self.costs.eta
        denom = self._total.get(c, 0.0) + eta * (self._k.get(c, 0) + 1)
        return -math.log2((n + eta) / denom)

    def exp_cost(self, c: int, e: Expansion) -> float:
        n = self.counts.get((c, e), 0.0)
        bits = self._logp(c, n)
        if e[0] == "p":
            bits += self.costs.c_phrase_use
        return bits

    def rule_cost(self, c: int, a: int, b: int) -> float:
        key = (c, a, b)
        v = self._rule_cache.get(key)
        if v is None:
            if key not in self.rules or c not in self.classes or a not in self.classes or b not in self.classes:
                v = INF
            else:
                v = self.exp_cost(c, ("r", a, b))
            self._rule_cache[key] = v
        return v

    def leaf_cost(self, c: int, tid: int, length: int) -> float:
        key = (c, tid, length)
        v = self._leaf_cache.get(key)
        if v is None:
            if c not in self.classes:
                v = INF
            elif length == 1:
                v = self.exp_cost(c, ("l", tid)) if (c, tid) in self.lex else INF
            else:
                v = self.exp_cost(c, ("p", tid))
            self._leaf_cache[key] = v
        return v

    def root_cost(self, c: int) -> float:
        v = self._root_cache.get(c)
        if v is None:
            eta = self.costs.eta
            n = self.root_counts.get(c, 0.0)
            denom = self._root_total + eta * (len(self.classes) + 1)
            v = -math.log2((n + eta) / denom)
            self._root_cache[c] = v
        return v

    # ---------------------------------------------------------- grammar DL
    def active_rules(self) -> list[tuple[int, int, int]]:
        return [k for k in self.rules if k[0] in self.classes and k[1] in self.classes and k[2] in self.classes]

    def active_lex(self) -> list[tuple[int, int]]:
        return [k for k in self.lex if k[0] in self.classes]

    def active_phrases(self) -> list[tuple[int, int]]:
        return [(c, e[1]) for (c, e), n in self.counts.items() if e[0] == "p" and n > 0 and c in self.classes]

    def grammar_bits(self) -> float:
        c = self.costs
        return (
            c.c_class * len(self.classes)
            + c.c_rule * len(self.active_rules())
            + c.c_lex * len(self.active_lex())
            + c.c_phrase_entry * len(self.active_phrases())
        )

    def class_grammar_bits(self, cls: int) -> float:
        """Grammar bits attributable to one class (removed together with it)."""
        c = self.costs
        bits = c.c_class
        bits += c.c_rule * sum(1 for k in self.active_rules() if cls in k)
        bits += c.c_lex * sum(1 for k in self.active_lex() if k[0] == cls)
        bits += c.c_phrase_entry * sum(1 for (cc, e), n in self.counts.items() if cc == cls and e[0] == "p" and n > 0)
        return bits

    # ------------------------------------------------------------ data DL
    def class_bits(self, c: int) -> float:
        """``sum_e N(c,e) * -log2 P(e|c)`` for one class (0 if not accepted)."""
        if c not in self.classes:
            return 0.0
        exps = self.exp_index.get(c)
        if not exps:
            return 0.0
        eta = self.costs.eta
        denom = self._total.get(c, 0.0) + eta * (self._k.get(c, 0) + 1)
        phrase_use = self.costs.c_phrase_use
        counts = self.counts
        bits = 0.0
        for e in exps:
            n = counts.get((c, e), 0.0)
            if n > 0:
                bits += n * (-math.log2((n + eta) / denom) + (phrase_use if e[0] == "p" else 0.0))
        return bits

    def root_bits(self) -> float:
        eta = self.costs.eta
        denom = self._root_total + eta * (len(self.classes) + 1)
        bits = 0.0
        for c, n in self.root_counts.items():
            if n > 0 and c in self.classes:
                bits += n * -math.log2((n + eta) / denom)
        return bits

    def data_bits_closed_form(self, n_chunks: float, n_unk: float) -> float:
        """``sum N(C,e) * -log P(e|C) + roots + chunk/unk costs`` from the counts."""
        bits = sum(self.class_bits(c) for c in self.classes)
        return bits + self.root_bits() + self.costs.c_chunk * n_chunks + self.costs.c_unk * n_unk

    # ------------------------------------------------------------ deltas
    def apply_delta(self, d_counts: dict[tuple[int, Expansion], float], d_roots: dict[int, float], scale: float) -> DeltaRecord:
        """Apply scaled count changes; returns the record needed to undo them exactly."""
        touched = {c for (c, _) in d_counts} | set(d_roots)
        rec = DeltaRecord(
            saved_counts={k: self.counts.get(k, 0.0) for k in d_counts},
            saved_total={c: self._total.get(c, 0.0) for c in touched},
            saved_k={c: self._k.get(c, 0) for c in touched},
            saved_roots={c: self.root_counts.get(c, 0.0) for c in d_roots},
            saved_root_total=self._root_total,
            touched=touched,
            saved_caches=(self._rule_cache, self._leaf_cache, self._root_cache),
            saved_version=self.version,
        )
        for key, d in d_counts.items():
            self.add_count(key, scale * d)
        for c, d in d_roots.items():
            self.add_root(c, scale * d)
        self.invalidate()
        return rec

    def undo(self, rec: DeltaRecord) -> None:
        for k, v in rec.saved_counts.items():
            self.counts[k] = v
        for c, v in rec.saved_total.items():
            self._total[c] = v
        for c, v in rec.saved_k.items():
            self._k[c] = v
        for c, v in rec.saved_roots.items():
            self.root_counts[c] = v
        self._root_total = rec.saved_root_total
        self._rule_cache, self._leaf_cache, self._root_cache = rec.saved_caches
        self.version = rec.saved_version


# ------------------------------------------------------------------ parsing
def viterbi_sentence(
    n: int,
    spans: list[tuple[int, int, int, int, int]],
    grammar: Grammar,
    excluded: int = -1,
    occ_override: dict[int, int] | None = None,
    excluded_rule: tuple[int, int, int] | None = None,
    predict: bool = False,
    beam: int = 3,
    excluded_rules: set[tuple[int, int, int]] | None = None,
) -> SentenceParse:
    """Cheapest derivation of one sentence under the current tables.

    Candidate occurrences keep their assigned class (only the bracketing is
    chosen).  With ``predict=True`` the grammar may additionally derive spans
    that are *not* candidate occurrences (rare spans) from classified parts
    through its rules, keeping at most ``beam`` classes per predicted span.

    Args:
        n: sentence length in tokens.
        spans: ``(start, end, oid, tid, cls)`` for the classified occurrences.
        excluded: a class id to treat as absent (for elimination tests).
        occ_override: occurrence -> class overrides (for ambiguity tests).
        excluded_rule: a rule ``(C, A, B)`` to treat as absent.
        excluded_rules: a set of rules to treat as absent.
    """
    costs = grammar.costs
    classes = grammar.classes
    cls_of: dict[tuple[int, int], tuple[int, int, int]] = {}
    pos_oid: dict[tuple[int, int], int] = {}
    for s, e, oid, tid, c in spans:
        pos_oid[(s, e)] = oid
        if occ_override is not None and oid in occ_override:
            c = occ_override[oid]
        if c < 0 or c == excluded or c not in classes:
            continue
        cls_of[(s, e)] = (oid, tid, c)
    by_children = grammar.rules_by_children
    rules = grammar.rules
    rule_cache = grammar._rule_cache
    rule_cost = grammar.rule_cost
    leaf_cost = grammar.leaf_cost
    root_cost = grammar.root_cost
    if excluded_rules and excluded_rule is not None:
        excluded_rules = set(excluded_rules) | {excluded_rule}
    elif excluded_rule is not None:
        excluded_rules = {excluded_rule}
    c_chunk = costs.c_chunk
    c_unk = costs.c_unk
    # best[(i, j)] : class -> (cost, split, left_cls, right_cls)
    best: dict[tuple[int, int], dict[int, tuple[float, int, int, int]]] = {}
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            info = cls_of.get((i, j))
            entries: dict[int, tuple[float, int, int, int]] = {}
            fixed_c = -1
            if info is not None:
                oid, tid, fixed_c = info
                lc = leaf_cost(fixed_c, tid, length)
                if lc < INF:
                    entries[fixed_c] = (lc, -1, -1, -1)
            elif not predict or length < 2:
                continue
            for k in range(i + 1, j):
                left = best.get((i, k))
                if left is None:
                    continue
                right = best.get((k, j))
                if right is None:
                    continue
                for a, (ca, _, _, _) in left.items():
                    for b, (cb, _, _, _) in right.items():
                        if info is not None:
                            if (fixed_c, a, b) not in rules:
                                continue
                            cands = (fixed_c,)
                        else:
                            cands = by_children.get((a, b))
                            if cands is None:
                                continue
                        for c in cands:
                            if c == excluded:
                                continue
                            key = (c, a, b)
                            if excluded_rules is not None and key in excluded_rules:
                                continue
                            rc = rule_cache.get(key)
                            if rc is None:
                                rc = rule_cost(c, a, b)
                            if rc == INF:
                                continue
                            cand = ca + cb + rc
                            cur = entries.get(c)
                            if cur is None or cand < cur[0]:
                                entries[c] = (cand, k, a, b)
            if entries:
                if info is None and len(entries) > beam:
                    entries = dict(sorted(entries.items(), key=lambda kv: kv[1][0])[:beam])
                best[(i, j)] = entries
    seq: dict[tuple[int, int], tuple[float, int, int]] = {}  # (cost, split, root_cls); split=-1 root chunk, -2 unk
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            cost, bp, rc_ = INF, -1, -1
            b = best.get((i, j))
            if b is not None:
                for c, (cc, _, _, _) in b.items():
                    cand = cc + root_cost(c) + c_chunk
                    if cand < cost:
                        cost, rc_ = cand, c
            if length == 1 and c_unk < cost:
                cost, bp = c_unk, -2
            for k in range(i + 1, j):
                cand = seq[(i, k)][0] + seq[(k, j)][0]
                if cand < cost:
                    cost, bp, rc_ = cand, k, -1
            seq[(i, j)] = (cost, bp, rc_)
    nodes: list[Node] = []
    keys: list[Expansion] = []
    roots: list[tuple[int, int, int]] = []
    unk = 0
    chunks = 0

    def trace_tree(i: int, j: int, c: int) -> None:
        cost, k, a, b = best[(i, j)][c]
        info = cls_of.get((i, j))
        oid = pos_oid.get((i, j), -1)  # also set for unclassified candidate spans derived by the grammar
        nodes.append(Node(i, j, c, k, oid, a, b))
        if k >= 0:
            keys.append((c, ("r", a, b)))
            trace_tree(i, k, a)
            trace_tree(k, j, b)
        elif j - i == 1:
            keys.append((c, ("l", info[1])))
        else:
            keys.append((c, ("p", info[1])))

    stack = [(0, n)]
    while stack:
        i, j = stack.pop()
        _, bp, rc_ = seq[(i, j)]
        if bp == -2:
            unk += 1
        elif bp == -1:
            chunks += 1
            roots.append((i, j, rc_))
            trace_tree(i, j, rc_)
        else:
            stack.append((bp, j))
            stack.append((i, bp))
    return SentenceParse(seq[(0, n)][0], nodes, chunks, unk, roots, keys)


class MDLModel:
    """Holds per-sentence derivations and evaluates description lengths."""

    def __init__(
        self,
        inv: SpanInventory,
        occ_class: np.ndarray,
        costs: MDLCosts,
        sentence_ids: Iterable[int] | None = None,
        min_rule_support: float = 1.0,
    ) -> None:
        self.inv = inv
        self.occ_class = occ_class
        self.costs = costs
        self.sids = list(sentence_ids) if sentence_ids is not None else list(range(len(inv.corpus)))
        self.min_rule_support = min_rule_support
        self.grammar = Grammar(costs)
        self.parses: dict[int, SentenceParse] = {}
        self.class_sents: dict[int, set[int]] = {}
        self.rule_sents: dict[tuple[int, int, int], set[int]] = {}
        self.n_chunks = 0
        self.n_unk = 0
        self.predict = False
        self.beam = 3
        self._data_cache: tuple[int, float] | None = None

    # ------------------------------------------------------------ helpers
    def sentence_spans(self, sid: int) -> list[tuple[int, int, int, int, int]]:
        inv = self.inv
        return [
            (int(inv.occ_start[o]), int(inv.occ_end[o]), o, int(inv.occ_tid[o]), int(self.occ_class[o]))
            for o in inv.sent_occ[sid].tolist()
        ]

    def parse(self, sid: int, **kw: Any) -> SentenceParse:
        return viterbi_sentence(
            len(self.inv.corpus.token_ids[sid]), self.sentence_spans(sid), self.grammar, predict=self.predict, beam=self.beam, **kw
        )

    # ------------------------------------------------------- bookkeeping
    def _add_parse(self, sid: int, p: SentenceParse, sign: float) -> None:
        g = self.grammar
        for key in p.keys:
            g.add_count(key, sign)
        for _, _, c in p.roots:
            g.add_root(c, sign)
        self.n_chunks += int(sign) * p.chunks
        self.n_unk += int(sign) * p.unk_tokens
        for nd in p.nodes:
            s = self.class_sents.setdefault(nd.cls, set())
            if sign > 0:
                s.add(sid)
            else:
                s.discard(sid)
        for c, e in p.keys:
            if e[0] == "r":
                s = self.rule_sents.setdefault((c, e[1], e[2]), set())
                if sign > 0:
                    s.add(sid)
                else:
                    s.discard(sid)
        self._data_cache = None

    def _replace_parses(self, sids: Iterable[int], **kw: Any) -> None:
        """Re-parse ``sids`` with the *current* (frozen) tables, then commit all of them."""
        new = {sid: self.parse(sid, **kw) for sid in sids}
        for sid, p in new.items():
            self._add_parse(sid, self.parses[sid], -1.0)
            self.parses[sid] = p
            self._add_parse(sid, p, +1.0)
        self.grammar.invalidate()

    def _rebuild_indexes(self) -> None:
        self.class_sents = {}
        self.rule_sents = {}
        self.grammar.counts = {}
        self.grammar.root_counts = {}
        self.grammar.refresh()
        self.n_chunks = 0
        self.n_unk = 0
        for sid, p in self.parses.items():
            self._add_parse(sid, p, +1.0)
        self.grammar.refresh()
        self._data_cache = None

    # ------------------------------------------------------- estimation
    def initialise(self, classes: set[int], rule_support: dict[tuple[int, int, int], float]) -> None:
        """Define the allowed grammar and seed the tables from composition evidence."""
        g = self.grammar
        g.classes = set(classes)
        g.rules = {k for k, n in rule_support.items() if n >= self.min_rule_support and k[0] in classes and k[1] in classes and k[2] in classes}
        g.lex = set()
        g.counts = {}
        g.root_counts = {}
        inv = self.inv
        lengths = inv.occ_end - inv.occ_start
        sent_len = np.array([len(t) for t in inv.corpus.token_ids])
        for oid in np.nonzero(self.occ_class >= 0)[0].tolist():
            c = int(self.occ_class[oid])
            if c not in g.classes:
                continue
            if lengths[oid] == 1:
                key = (c, int(inv.occ_tid[oid]))
                g.lex.add(key)
                g.counts[(c, ("l", key[1]))] = g.counts.get((c, ("l", key[1])), 0.0) + 1.0
            if lengths[oid] == sent_len[inv.occ_sid[oid]]:
                g.root_counts[c] = g.root_counts.get(c, 0.0) + 1.0
        for (c, a, b) in g.rules:
            g.counts[(c, ("r", a, b))] = float(rule_support[(c, a, b)])
        g.refresh()
        self._data_cache = None

    def fit(self, em_iters: int = 1) -> float:
        """Viterbi-parse everything and set the counts to the usage counts."""
        for it in range(max(1, em_iters)):
            self.parses = {sid: self.parse(sid) for sid in self.sids}
            self._rebuild_indexes()
        return self.total_bits()

    def data_bits(self) -> float:
        g = self.grammar
        if self._data_cache is not None and self._data_cache[0] == g.version:
            return self._data_cache[1]
        v = g.data_bits_closed_form(self.n_chunks, self.n_unk)
        self._data_cache = (g.version, v)
        return v

    def total_bits(self) -> float:
        return self.grammar.grammar_bits() + self.data_bits()

    # ------------------------------------------------------- hypotheses
    def _delta_with_reparse(self, sids: Iterable[int], scale: float = 1.0, **kw: Any) -> tuple[float, dict[int, SentenceParse]]:
        """Re-parse ``sids`` under ``kw`` and return the data-bits delta (closed form) plus the new parses.

        Only the classes whose counts change are re-evaluated.  When ``sids``
        is a sample of the affected sentences, ``scale`` multiplies the *count
        changes* (not the bits: the closed form is concave in the counts, so
        extrapolating bits from a partially switched state would be biased).
        """
        g = self.grammar
        new: dict[int, SentenceParse] = {}
        d_counts: dict[tuple[int, Expansion], float] = {}
        d_roots: dict[int, float] = {}
        d_chunks = 0.0
        d_unk = 0.0
        for sid in sids:
            old = self.parses[sid]
            p = self.parse(sid, **kw)
            new[sid] = p
            for key in old.keys:
                d_counts[key] = d_counts.get(key, 0.0) - 1.0
            for key in p.keys:
                d_counts[key] = d_counts.get(key, 0.0) + 1.0
            for _, _, c in old.roots:
                d_roots[c] = d_roots.get(c, 0.0) - 1.0
            for _, _, c in p.roots:
                d_roots[c] = d_roots.get(c, 0.0) + 1.0
            d_chunks += p.chunks - old.chunks
            d_unk += p.unk_tokens - old.unk_tokens
        touched = {c for (c, _) in d_counts}
        before = sum(g.class_bits(c) for c in touched) + g.root_bits()
        rec = g.apply_delta(d_counts, d_roots, scale)
        after = sum(g.class_bits(c) for c in touched) + g.root_bits()
        g.undo(rec)
        dd = (after - before) + self.costs.c_chunk * scale * d_chunks + self.costs.c_unk * scale * d_unk
        return dd, new

    def removal_delta(self, cls: int, max_sentences: int = 400, rng: np.random.Generator | None = None) -> tuple[float, float, float, int]:
        """DL change if class ``cls`` is removed: ``(delta_total, delta_grammar, delta_data, n_affected)``."""
        affected = sorted(self.class_sents.get(cls, set()))
        g = self.grammar
        dg = -g.class_grammar_bits(cls)
        if not affected:
            return dg, dg, 0.0, 0
        scale = 1.0
        sample = affected
        if len(affected) > max_sentences:
            rng = rng or np.random.default_rng(0)
            sample = sorted(affected[i] for i in rng.choice(len(affected), size=max_sentences, replace=False))
            scale = len(affected) / max_sentences
        # the grammar is left intact; the parser alone excludes the class
        dd, _ = self._delta_with_reparse(sample, scale=scale, excluded=cls)
        return dg + dd, dg, dd, len(affected)

    def remove_class(self, cls: int) -> None:
        affected = sorted(self.class_sents.get(cls, set()))
        g = self.grammar
        g.classes.discard(cls)
        g.refresh()  # rules_by_children / K_C change; counts are then maintained incrementally
        self._replace_parses(affected)

    def rule_removal_delta(self, key: tuple[int, int, int]) -> tuple[float, float, float, int]:
        affected = sorted(self.rule_sents.get(key, set()))
        dg = -self.costs.c_rule
        if affected:
            dd, _ = self._delta_with_reparse(affected, excluded_rule=key)
        else:
            dd = 0.0
        return dg + dd, dg, dd, len(affected)

    def remove_rule(self, key: tuple[int, int, int]) -> None:
        affected = sorted(self.rule_sents.get(key, set()))
        g = self.grammar
        g.rules.discard(key)
        g.refresh()
        self._replace_parses(affected)

    def hypothesis_bits(
        self,
        sids: list[int],
        occ_override: dict[int, int],
        temp_lex: Iterable[tuple[int, int]] = (),
        excluded_rules: set[tuple[int, int, int]] | None = None,
    ) -> tuple[float, dict[int, SentenceParse]]:
        """Total DL if the given occurrences take other classes (temporary lexical
        entries added; ``excluded_rules`` are unavailable to the derivations)."""
        g = self.grammar
        base = self.data_bits()
        added = [k for k in temp_lex if k not in g.lex]
        lex_classes = {c for (c, _) in added if c in g.classes}
        pre = sum(g.class_bits(c) for c in lex_classes)
        for k in added:
            g.add_lex(k)
        post = sum(g.class_bits(c) for c in lex_classes)  # effect of the extra smoothing slots
        dd, new = self._delta_with_reparse(sids, occ_override=occ_override, excluded_rules=excluded_rules)
        total = g.grammar_bits() + base + (post - pre) + dd
        for k in added:
            g.remove_lex(k)
        return total, new

    def apply_override(self, sids: list[int], occ_override: dict[int, int], lex_entries: Iterable[tuple[int, int]]) -> None:
        """Make an occurrence re-assignment permanent."""
        g = self.grammar
        for o, c in occ_override.items():
            self.occ_class[o] = c
        for k in lex_entries:
            g.add_lex(k)
        self._replace_parses(sids)

    # ------------------------------------------------------- reporting
    def usage_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for p in self.parses.values():
            for nd in p.nodes:
                out[nd.cls] = out.get(nd.cls, 0) + 1
        return out

    def rule_usage(self) -> dict[tuple[int, int, int], int]:
        out: dict[tuple[int, int, int], int] = {}
        for (c, e), n in self.grammar.counts.items():
            if e[0] == "r" and n > 0:
                out[(c, e[1], e[2])] = int(n)
        return out

    def summary(self) -> dict[str, Any]:
        g = self.grammar
        return {
            "n_sentences": len(self.sids),
            "grammar_bits": round(g.grammar_bits(), 2),
            "data_bits": round(self.data_bits(), 2),
            "total_bits": round(self.total_bits(), 2),
            "bits_per_sentence": round(self.total_bits() / max(1, len(self.sids)), 3),
            "n_classes": len(g.classes),
            "n_rules": len(g.active_rules()),
            "n_rules_used": len(self.rule_usage()),
            "n_lexical_entries": len(g.active_lex()),
            "n_phrase_entries": len(g.active_phrases()),
            "unk_tokens": self.n_unk,
            "chunks_per_sentence": round(self.n_chunks / max(1, len(self.sids)), 3),
            "fully_derived_sentences": sum(1 for p in self.parses.values() if p.chunks == 1 and p.unk_tokens == 0),
            "predicted_nodes": sum(1 for p in self.parses.values() for nd in p.nodes if nd.split >= 0 and (nd.oid < 0 or self.occ_class[nd.oid] != nd.cls)),
            "predictive_chart": self.predict,
        }
