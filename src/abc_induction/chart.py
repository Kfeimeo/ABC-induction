"""Semiring-compatible sparse Inside-Outside chart (reference Python backend).

The latent model is a *generative* binary grammar (Chomsky normal form) over
the active classes ``E`` with a sparse rule inventory ``R``:

    score(T, Z | s) = rho(Z_root)
                      * prod_{internal (i,k,j)} theta_{Z_ij}(Z_ik, Z_kj) * psi(i, j)
                      * prod_{leaves i}         theta_{Z_i}(w_i)         * lambda_i(Z_i)

``theta_C`` is one distribution per class over its *expansions* (binary rules
``C -> A B`` and lexical entries ``C -> w``), ``rho`` the root distribution,
``psi`` an optional span potential (punctuation structural prior) and
``lambda`` an optional per-leaf class potential (lexical context evidence,
"virtual evidence").  Both potentials are fixed, so ``sum_{T,Z} score`` is a
proper likelihood of (sentence, virtual evidence) and every derivation of the
sentence is summed over - structural ambiguity is marginalised, never resolved
by a heuristic.

Everything is computed in log space.  A cell ``chart[i, j]`` is a sparse
``dict`` ``class -> value``; rules are iterated through the rule index
(``left class -> rules``), never over ``K^3``.

Semirings
---------
* :class:`LogSemiring`      (logsumexp, +): inside mass / marginals.
* :class:`ViterbiSemiring`  (max, +): best derivation (with back-pointers).

The accelerated backend (:mod:`abc_induction.chart_numba`) implements the same
recurrences on dense ``(n, n, K)`` arrays; this module is the correctness
reference and the fallback, and :func:`brute_force` enumerates every binary
tree and labelling for tiny oracle tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

NEG_INF = float("-inf")


def logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class LogSemiring:
    """(logsumexp, +, -inf, 0)."""

    name = "log"
    zero = NEG_INF
    one = 0.0

    @staticmethod
    def plus(a: float, b: float) -> float:
        return logaddexp(a, b)

    @staticmethod
    def times(a: float, b: float) -> float:
        return a + b


class ViterbiSemiring:
    """(max, +, -inf, 0)."""

    name = "viterbi"
    zero = NEG_INF
    one = 0.0

    @staticmethod
    def plus(a: float, b: float) -> float:
        return a if a >= b else b

    @staticmethod
    def times(a: float, b: float) -> float:
        return a + b


# --------------------------------------------------------------------------- grammar view
@dataclass
class CompiledGrammar:
    """Grammar tables with expected-log parameters, indexed for the chart.

    Classes are compact indices ``0..K-1``.  ``rules`` lists ``(C, A, B, logp)``;
    ``by_left[A]`` gives the rules with left child ``A`` (rule indices);
    ``lex[w]`` maps a token id to ``[(C, logp), ...]``; ``root[C]`` is the
    root log-probability.  ``unk`` holds the per-class log-probability of the
    unknown-token expansion (``-inf`` = none).
    """

    n_classes: int
    rules: list[tuple[int, int, int, float]]
    lex: dict[int, list[tuple[int, float]]]
    root: np.ndarray
    unk: np.ndarray
    by_left: list[list[int]] = field(default_factory=list)
    by_children: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    by_result: list[list[int]] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)  # compact index -> model class id
    lex_entry_index: dict[tuple[int, int], int] = field(default_factory=dict)  # (C, w) -> entry index

    def __post_init__(self) -> None:
        K = self.n_classes
        self.by_left = [[] for _ in range(K)]
        self.by_result = [[] for _ in range(K)]
        self.by_children = {}
        for r, (c, a, b, _) in enumerate(self.rules):
            self.by_left[a].append(r)
            self.by_result[c].append(r)
            self.by_children.setdefault((a, b), []).append(r)

    def leaf_entries(self, w: int) -> list[tuple[int, float]]:
        """``[(C, logp)]`` for token ``w`` (falls back to the UNK expansions)."""
        e = self.lex.get(w)
        if e:
            return e
        return [(c, float(self.unk[c])) for c in range(self.n_classes) if self.unk[c] > NEG_INF]


# --------------------------------------------------------------------------- inside
@dataclass
class ChartResult:
    """Inside/outside tables of one sentence.

    ``inside[(i, j)]`` / ``outside[(i, j)]`` are sparse dicts ``C -> log value``.
    ``log_z`` is the total (log) mass of all derivations; ``-inf`` when the
    sentence has no derivation.  ``back`` holds Viterbi back-pointers
    ``(i, j, C) -> (k, rule_idx)`` (``k < 0``: leaf).
    """

    n: int
    inside: dict[tuple[int, int], dict[int, float]]
    outside: dict[tuple[int, int], dict[int, float]] | None
    log_z: float
    back: dict[tuple[int, int, int], tuple[int, int]] | None = None
    pruned_cells: int = 0
    dropped_mass: list[float] = field(default_factory=list)
    chunk_logp: tuple[float, float] | None = None  # (log p_cont, log (1 - p_cont)) or None = single root
    prefix: list[float] | None = None  # log mass of [0, i) as a chunk sequence (chunk layer only)
    suffix: list[float] | None = None  # log mass of [j, n) as the rest of the sequence
    chunk_mass: dict[tuple[int, int], float] | None = None  # log sum_C inside[i,j][C] + root[C]

    # ---------------------------------------------------------------- posteriors
    def node_posterior(self, i: int, j: int, c: int) -> float:
        """``P(Z_ij = C | s)``."""
        if self.outside is None or self.log_z == NEG_INF:
            return 0.0
        a = self.inside.get((i, j), {}).get(c, NEG_INF)
        b = self.outside.get((i, j), {}).get(c, NEG_INF)
        if a == NEG_INF or b == NEG_INF:
            return 0.0
        return math.exp(a + b - self.log_z)

    def span_posteriors(self) -> dict[tuple[int, int], dict[int, float]]:
        """All ``P(Z_ij = C | s)`` as sparse dicts."""
        out: dict[tuple[int, int], dict[int, float]] = {}
        if self.outside is None or self.log_z == NEG_INF:
            return out
        for key, cell in self.inside.items():
            ocell = self.outside.get(key)
            if not ocell:
                continue
            d = {}
            for c, a in cell.items():
                b = ocell.get(c)
                if b is not None and b > NEG_INF and a > NEG_INF:
                    d[c] = math.exp(a + b - self.log_z)
            if d:
                out[key] = d
        return out

    def constituency_posterior(self) -> dict[tuple[int, int], float]:
        """``P(Q_ij = 1 | s) = sum_C P(Z_ij = C | s)``."""
        return {k: sum(d.values()) for k, d in self.span_posteriors().items()}


def inside_chart(
    tokens: list[int],
    g: CompiledGrammar,
    semiring=LogSemiring,
    leaf_potential: np.ndarray | None = None,
    span_potential: np.ndarray | None = None,
    max_classes_per_cell: int = 0,
    prune_threshold: float = 0.0,
    keep_back: bool = False,
    chunk_logp: tuple[float, float] | None = None,
) -> ChartResult:
    """Inside pass over all binary derivations of ``tokens``.

    Args:
        leaf_potential: ``(n, K)`` log potentials ``log lambda_i(C)`` or None.
        span_potential: ``(n+1, n+1)`` log potentials ``log psi(i, j)`` or None.
        max_classes_per_cell: beam (0 = exact).
        prune_threshold: drop classes whose cell mass is below this fraction of
            the cell maximum (0 = exact).  Dropped mass is recorded.
        keep_back: record Viterbi back-pointers (Viterbi semiring only).
        chunk_logp: ``(log p_cont, log(1 - p_cont))`` enables the chunk-sequence
            fallback: the sentence is a sequence of ``m >= 1`` root chunks with
            probability ``p_cont^(m-1) (1 - p_cont)``.  None = exactly one root.
    """
    n = len(tokens)
    plus, times, zero = semiring.plus, semiring.times, semiring.zero
    inside: dict[tuple[int, int], dict[int, float]] = {}
    back: dict[tuple[int, int, int], tuple[int, int]] | None = {} if keep_back else None
    pruned_cells = 0
    dropped: list[float] = []
    log_thr = math.log(prune_threshold) if prune_threshold > 0 else NEG_INF
    rules = g.rules
    by_left = g.by_left
    # ---- leaves
    for i, w in enumerate(tokens):
        cell: dict[int, float] = {}
        for c, lp in g.leaf_entries(w):
            v = lp
            if leaf_potential is not None:
                v = times(v, float(leaf_potential[i, c]))
            if v > NEG_INF:
                cell[c] = v
                if back is not None:
                    back[(i, i + 1, c)] = (-1, -1)
        if span_potential is not None and cell:
            sp = float(span_potential[i, i + 1])
            cell = {c: times(v, sp) for c, v in cell.items()}
        if cell:
            inside[(i, i + 1)] = cell
    # ---- internal spans
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            cell = {}
            bp: dict[int, tuple[int, int]] = {}
            for k in range(i + 1, j):
                left = inside.get((i, k))
                if not left:
                    continue
                right = inside.get((k, j))
                if not right:
                    continue
                for a, va in left.items():
                    for r in by_left[a]:
                        c, _, b, lp = rules[r]
                        vb = right.get(b)
                        if vb is None:
                            continue
                        v = times(times(va, vb), lp)
                        cur = cell.get(c)
                        if cur is None:
                            cell[c] = v
                            if back is not None:
                                bp[c] = (k, r)
                        else:
                            nv = plus(cur, v)
                            if back is not None and v > cur:
                                bp[c] = (k, r)
                            cell[c] = nv
            if not cell:
                continue
            if span_potential is not None:
                sp = float(span_potential[i, j])
                cell = {c: times(v, sp) for c, v in cell.items()}
            # ---- pruning (log semiring only; records the dropped posterior-free mass)
            if (max_classes_per_cell > 0 and len(cell) > max_classes_per_cell) or log_thr > NEG_INF:
                items = sorted(cell.items(), key=lambda kv: -kv[1])
                vmax = items[0][1]
                keep = [(c, v) for c, v in items if v - vmax >= log_thr]
                if max_classes_per_cell > 0:
                    keep = keep[:max_classes_per_cell]
                if len(keep) < len(items):
                    tot = vmax + math.log(sum(math.exp(v - vmax) for _, v in items))
                    kept = vmax + math.log(sum(math.exp(v - vmax) for _, v in keep))
                    dropped.append(1.0 - math.exp(kept - tot))
                    pruned_cells += 1
                    cell = dict(keep)
            inside[(i, j)] = cell
            if back is not None:
                for c, kv in bp.items():
                    if c in cell:
                        back[(i, j, c)] = kv
    # ---- root / chunk layer
    chunk_mass: dict[tuple[int, int], float] = {}
    for (i, j), cell in inside.items():
        m = zero
        for c, v in cell.items():
            rv = float(g.root[c])
            if rv > NEG_INF:
                m = plus(m, times(v, rv))
        if m > zero:
            chunk_mass[(i, j)] = m
    if chunk_logp is None:
        log_z = chunk_mass.get((0, n), zero)
        return ChartResult(n, inside, None, log_z, back, pruned_cells, dropped, None, None, None, chunk_mass)
    lc, ls = chunk_logp
    prefix = [zero] * (n + 1)  # prefix[i]: [0, i) as >= 1 chunks, one p_cont per chunk boundary
    prefix[0] = semiring.one
    for j in range(1, n + 1):
        acc = zero
        for i in range(0, j):
            m = chunk_mass.get((i, j))
            if m is None or prefix[i] == zero:
                continue
            acc = plus(acc, times(times(prefix[i], m), lc if i > 0 else semiring.one))
        prefix[j] = acc
    suffix = [zero] * (n + 1)  # suffix[j]: [j, n) as the rest (p_cont before every chunk, 1-p_cont at the end)
    suffix[n] = ls
    for i in range(n - 1, -1, -1):
        acc = zero
        for j in range(i + 1, n + 1):
            m = chunk_mass.get((i, j))
            if m is None or suffix[j] == zero:
                continue
            acc = plus(acc, times(times(lc, m), suffix[j]))
        suffix[i] = acc
    log_z = times(prefix[n], ls)
    return ChartResult(n, inside, None, log_z, back, pruned_cells, dropped, chunk_logp, prefix, suffix, chunk_mass)


# --------------------------------------------------------------------------- outside
def outside_chart(res: ChartResult, g: CompiledGrammar, span_potential: np.ndarray | None = None) -> None:
    """Outside pass (log semiring); fills ``res.outside`` in place.

    ``outside[i,j][C]`` is the log mass of everything *outside* node (i,j,C),
    so that ``inside + outside - log_z`` is the node posterior.  The span
    potential of a node is already inside its inside value, so a parent
    passes ``outside[parent] + psi(parent) + logp(rule) + inside[sibling]``
    down to its children.
    """
    n = res.n
    inside = res.inside
    rules = g.rules
    by_left = g.by_left
    outside: dict[tuple[int, int], dict[int, float]] = {}
    if res.chunk_logp is None:
        root = inside.get((0, n), {})
        outside[(0, n)] = {c: float(g.root[c]) for c in root if g.root[c] > NEG_INF}
    else:
        # every cell may be a chunk root: outside = root[C] + prefix[i] (+ p_cont if i > 0) + suffix[j]
        lc, _ = res.chunk_logp
        assert res.prefix is not None and res.suffix is not None
        for (i, j), cell in inside.items():
            base = res.prefix[i] + (lc if i > 0 else 0.0) + res.suffix[j]
            if base == NEG_INF:
                continue
            oc = outside.setdefault((i, j), {})
            for c in cell:
                rv = float(g.root[c])
                if rv > NEG_INF:
                    oc[c] = logaddexp(oc.get(c, NEG_INF), rv + base)
    for length in range(n, 1, -1):
        for i in range(0, n - length + 1):
            j = i + length
            ocell = outside.get((i, j))
            if not ocell:
                continue
            sp = float(span_potential[i, j]) if span_potential is not None else 0.0
            for k in range(i + 1, j):
                left = inside.get((i, k))
                if not left:
                    continue
                right = inside.get((k, j))
                if not right:
                    continue
                oleft = outside.setdefault((i, k), {})
                oright = outside.setdefault((k, j), {})
                for a, va in left.items():
                    for r in by_left[a]:
                        c, _, b, lp = rules[r]
                        oc = ocell.get(c)
                        if oc is None:
                            continue
                        vb = right.get(b)
                        if vb is None:
                            continue
                        base = oc + sp + lp
                        oleft[a] = logaddexp(oleft.get(a, NEG_INF), base + vb)
                        oright[b] = logaddexp(oright.get(b, NEG_INF), base + va)
    res.outside = outside


# --------------------------------------------------------------------------- expected counts
@dataclass
class SentenceExpectations:
    """Posterior expectations of one sentence."""

    log_z: float
    rule_counts: dict[int, float]  # rule index -> E[N]
    lex_counts: dict[tuple[int, int], float]  # (C, w) -> E[N]   (w = token id or -1 for UNK)
    root_counts: dict[int, float]
    node_posteriors: dict[tuple[int, int], dict[int, float]]
    rule_apps: list[tuple[int, int, int, int, float]]  # (rule, i, k, j, posterior)
    n_chunks: float = 1.0  # expected number of root chunks


def expectations(
    tokens: list[int],
    g: CompiledGrammar,
    res: ChartResult,
    span_potential: np.ndarray | None = None,
    min_posterior: float = 0.0,
) -> SentenceExpectations:
    """Expected rule / lexical / root counts and node posteriors from inside+outside."""
    n = res.n
    if res.outside is None:
        outside_chart(res, g, span_potential)
    assert res.outside is not None
    lz = res.log_z
    rule_counts: dict[int, float] = {}
    lex_counts: dict[tuple[int, int], float] = {}
    root_counts: dict[int, float] = {}
    apps: list[tuple[int, int, int, int, float]] = []
    if lz == NEG_INF:
        return SentenceExpectations(lz, rule_counts, lex_counts, root_counts, {}, apps)
    inside, outside = res.inside, res.outside
    rules, by_left = g.rules, g.by_left
    n_chunks = 0.0
    if res.chunk_logp is None:
        for c, v in inside.get((0, n), {}).items():
            rv = float(g.root[c])
            if rv > NEG_INF:
                root_counts[c] = math.exp(v + rv - lz)
                n_chunks += root_counts[c]
    else:
        lc, _ = res.chunk_logp
        assert res.prefix is not None and res.suffix is not None
        for (i, j), cell in inside.items():
            base = res.prefix[i] + (lc if i > 0 else 0.0) + res.suffix[j]
            if base == NEG_INF:
                continue
            for c, v in cell.items():
                rv = float(g.root[c])
                if rv > NEG_INF:
                    p = math.exp(v + rv + base - lz)
                    root_counts[c] = root_counts.get(c, 0.0) + p
                    n_chunks += p
    for i, w in enumerate(tokens):
        cell = inside.get((i, i + 1), {})
        ocell = outside.get((i, i + 1), {})
        has_lex = bool(g.lex.get(w))
        for c, v in cell.items():
            o = ocell.get(c)
            if o is None:
                continue
            p = math.exp(v + o - lz)
            key = (c, w if has_lex else -1)
            lex_counts[key] = lex_counts.get(key, 0.0) + p
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            ocell = outside.get((i, j))
            if not ocell:
                continue
            sp = float(span_potential[i, j]) if span_potential is not None else 0.0
            for k in range(i + 1, j):
                left = inside.get((i, k))
                if not left:
                    continue
                right = inside.get((k, j))
                if not right:
                    continue
                for a, va in left.items():
                    for r in by_left[a]:
                        c, _, b, lp = rules[r]
                        oc = ocell.get(c)
                        if oc is None:
                            continue
                        vb = right.get(b)
                        if vb is None:
                            continue
                        p = math.exp(oc + sp + lp + va + vb - lz)
                        rule_counts[r] = rule_counts.get(r, 0.0) + p
                        if p >= min_posterior:
                            apps.append((r, i, k, j, p))
    return SentenceExpectations(lz, rule_counts, lex_counts, root_counts, res.span_posteriors(), apps, n_chunks)


# --------------------------------------------------------------------------- Viterbi
@dataclass
class Derivation:
    """One derivation: ``nodes`` = ``(i, j, C, k, rule)`` (``k < 0`` leaf) and its log score."""

    score: float
    nodes: list[tuple[int, int, int, int, int]]
    root_class: int
    chunks: list[tuple[int, int, int]] = field(default_factory=list)  # (i, j, root class)


def viterbi(
    tokens: list[int],
    g: CompiledGrammar,
    leaf_potential: np.ndarray | None = None,
    span_potential: np.ndarray | None = None,
    chunk_logp: tuple[float, float] | None = None,
) -> Derivation | None:
    """Best derivation under the Viterbi semiring (None if underivable)."""
    res = inside_chart(tokens, g, ViterbiSemiring, leaf_potential, span_potential, keep_back=True, chunk_logp=chunk_logp)
    n = len(tokens)
    assert res.back is not None

    def best_class(i: int, j: int) -> tuple[float, int]:
        b, bc = NEG_INF, -1
        for c, v in res.inside.get((i, j), {}).items():
            rv = float(g.root[c])
            if rv > NEG_INF and v + rv > b:
                b, bc = v + rv, c
        return b, bc

    chunks: list[tuple[int, int, int]] = []
    if chunk_logp is None:
        best, bc = best_class(0, n)
        if bc < 0:
            return None
        chunks.append((0, n, bc))
    else:
        # best segmentation: prefix table with back-pointers (Viterbi semiring)
        assert res.prefix is not None
        lc, ls = chunk_logp
        bp = [-1] * (n + 1)
        bestp = [NEG_INF] * (n + 1)
        bestp[0] = 0.0
        for j in range(1, n + 1):
            for i in range(0, j):
                m = res.chunk_mass.get((i, j)) if res.chunk_mass else None
                if m is None or bestp[i] == NEG_INF:
                    continue
                v = bestp[i] + m + (lc if i > 0 else 0.0)
                if v > bestp[j]:
                    bestp[j], bp[j] = v, i
        if bestp[n] == NEG_INF:
            return None
        best = bestp[n] + ls
        j = n
        while j > 0:
            i = bp[j]
            _, bc = best_class(i, j)
            chunks.append((i, j, bc))
            j = i
        chunks.reverse()
    nodes: list[tuple[int, int, int, int, int]] = []
    stack = list(reversed(chunks))
    while stack:
        i, j, c = stack.pop()
        k, r = res.back[(i, j, c)]
        nodes.append((i, j, c, k, r))
        if k >= 0:
            _, a, b, _ = g.rules[r]
            stack.append((k, j, b))
            stack.append((i, k, a))
    return Derivation(best, nodes, chunks[0][2], chunks)


# --------------------------------------------------------------------------- brute force oracle
def _trees(i: int, j: int):
    """All binary bracketings of [i, j) as nested tuples ((i,j), left, right)."""
    if j - i == 1:
        yield (i, j, None, None)
        return
    for k in range(i + 1, j):
        for lt in _trees(i, k):
            for rt in _trees(k, j):
                yield (i, j, lt, rt)


def brute_force(
    tokens: list[int],
    g: CompiledGrammar,
    leaf_potential: np.ndarray | None = None,
    span_potential: np.ndarray | None = None,
) -> dict[str, Any]:
    """Enumerate every binary tree and every labelling; sum exact scores.

    Returns ``log_z``, node posteriors ``P(Z_ij = C | s)``, constituency
    posteriors, expected rule / lexical / root counts and the best
    derivation score.  Only for tiny inputs (n <= 6, K <= 4).
    """
    n = len(tokens)
    K = g.n_classes
    total = 0.0
    best = NEG_INF
    node_mass: dict[tuple[int, int, int], float] = {}
    rule_mass: dict[int, float] = {}
    lex_mass: dict[tuple[int, int], float] = {}
    root_mass: dict[int, float] = {}
    for tree in _trees(0, n):
        # collect internal nodes and leaves
        internal: list[tuple[int, int, int]] = []
        leaves: list[int] = []
        stack = [tree]
        order = []
        while stack:
            t = stack.pop()
            order.append(t)
            if t[2] is not None:
                stack.append(t[2])
                stack.append(t[3])
        for t in order:
            if t[2] is None:
                leaves.append(t[0])
            else:
                internal.append((t[0], t[1], t[2][1]))
        spans = [(t[0], t[1]) for t in order]
        # enumerate labellings of all nodes
        for labels in _labelings(len(spans), K):
            lab = dict(zip(spans, labels))
            w = 0.0
            ok = True
            used_rules: list[int] = []
            used_lex: list[tuple[int, int]] = []
            for t in order:
                i, j = t[0], t[1]
                c = lab[(i, j)]
                if t[2] is None:
                    ent = {cc: lp for cc, lp in g.leaf_entries(tokens[i])}
                    lp = ent.get(c, NEG_INF)
                    if lp == NEG_INF:
                        ok = False
                        break
                    w += lp
                    if leaf_potential is not None:
                        w += float(leaf_potential[i, c])
                    used_lex.append((c, tokens[i] if g.lex.get(tokens[i]) else -1))
                else:
                    a = lab[(t[2][0], t[2][1])]
                    b = lab[(t[3][0], t[3][1])]
                    cands = [r for r in g.by_children.get((a, b), []) if g.rules[r][0] == c]
                    if not cands:
                        ok = False
                        break
                    r = cands[0]
                    w += g.rules[r][3]
                    used_rules.append(r)
                if span_potential is not None:
                    w += float(span_potential[i, j])
            if not ok:
                continue
            rc = lab[(0, n)]
            rv = float(g.root[rc])
            if rv == NEG_INF:
                continue
            w += rv
            p = math.exp(w)
            total += p
            best = max(best, w)
            for (i, j), c in lab.items():
                node_mass[(i, j, c)] = node_mass.get((i, j, c), 0.0) + p
            for r in used_rules:
                rule_mass[r] = rule_mass.get(r, 0.0) + p
            for key in used_lex:
                lex_mass[key] = lex_mass.get(key, 0.0) + p
            root_mass[rc] = root_mass.get(rc, 0.0) + p
    log_z = math.log(total) if total > 0 else NEG_INF
    node_post = {k: v / total for k, v in node_mass.items()} if total > 0 else {}
    const: dict[tuple[int, int], float] = {}
    for (i, j, c), p in node_post.items():
        const[(i, j)] = const.get((i, j), 0.0) + p
    return {
        "log_z": log_z,
        "node_posteriors": node_post,
        "constituency": const,
        "rule_counts": {k: v / total for k, v in rule_mass.items()} if total > 0 else {},
        "lex_counts": {k: v / total for k, v in lex_mass.items()} if total > 0 else {},
        "root_counts": {k: v / total for k, v in root_mass.items()} if total > 0 else {},
        "best": best,
    }


def _labelings(m: int, K: int):
    if m == 0:
        yield ()
        return
    for rest in _labelings(m - 1, K):
        for c in range(K):
            yield rest + (c,)
