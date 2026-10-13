"""Exact-inference tests: Inside-Outside (both backends) == brute force on tiny grammars.

Cases: 1 class, 2 classes, ambiguous lexical type, multiple derivations of the
same sentence, virtual-evidence potentials, chunk layer, probability
invariants.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abc_induction.chart import (  # noqa: E402
    NEG_INF,
    CompiledGrammar,
    LogSemiring,
    brute_force,
    expectations,
    inside_chart,
    outside_chart,
    viterbi,
)
from abc_induction.grammar_model import GrammarModel  # noqa: E402

LOG = math.log


def tiny_grammars() -> list[tuple[str, CompiledGrammar]]:
    out = []
    # 1 class: A -> A A | a | b
    g1 = CompiledGrammar(1, [(0, 0, 0, LOG(0.4))], {10: [(0, LOG(0.3))], 11: [(0, LOG(0.3))]}, np.array([0.0]), np.full(1, NEG_INF))
    out.append(("one_class", g1))
    # 2 classes: S -> A S | A A ; A -> a | b   (several derivations per sentence)
    g2 = CompiledGrammar(
        2,
        [(1, 0, 1, LOG(0.5)), (1, 0, 0, LOG(0.5)), (0, 0, 0, LOG(0.2))],
        {10: [(0, LOG(0.5))], 11: [(0, LOG(0.3))]},
        np.log(np.array([0.2, 0.8])),
        np.full(2, NEG_INF),
    )
    out.append(("two_classes", g2))
    # ambiguous lexical type: token 10 is A or B; C -> A B | B A | C C
    g3 = CompiledGrammar(
        3,
        [(2, 0, 1, LOG(0.4)), (2, 1, 0, LOG(0.3)), (2, 2, 2, LOG(0.2)), (1, 1, 1, LOG(0.3))],
        {10: [(0, LOG(0.6)), (1, LOG(0.2))], 11: [(1, LOG(0.5))], 12: [(0, LOG(0.4))]},
        np.log(np.array([0.1, 0.1, 0.8])),
        np.full(3, NEG_INF),
    )
    out.append(("ambiguous_lexical", g3))
    # unknown tokens: class 1 has an UNK expansion
    g4 = CompiledGrammar(
        2,
        [(1, 0, 1, LOG(0.5)), (1, 0, 0, LOG(0.3)), (0, 1, 0, LOG(0.1))],
        {10: [(0, LOG(0.5))]},
        np.log(np.array([0.3, 0.7])),
        np.array([NEG_INF, LOG(0.1)]),
    )
    out.append(("unk", g4))
    return out


SENTENCES = {
    "one_class": [[10], [10, 11], [10, 11, 10], [11, 10, 10, 11]],
    "two_classes": [[10, 11], [10, 10, 11], [11, 10, 10, 10], [10, 11, 10, 11, 10]],
    "ambiguous_lexical": [[10, 11], [10, 10], [12, 10, 11], [10, 11, 10, 12], [11, 10, 10, 12, 10]],
    "unk": [[10, 99], [99, 10, 10], [10, 99, 98, 10]],
}


def _potentials(n: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    lam = np.log(rng.uniform(0.2, 1.0, size=(n, K)))
    psi = np.log(rng.uniform(0.2, 1.0, size=(n + 1, n + 1)))
    return lam, psi


@pytest.mark.parametrize("name,g", tiny_grammars())
def test_inside_outside_equals_brute_force(name: str, g: CompiledGrammar) -> None:
    for toks in SENTENCES[name]:
        n, K = len(toks), g.n_classes
        lam, psi = _potentials(n, K, 1)
        for lp, sp in ((None, None), (lam, psi)):
            res = inside_chart(toks, g, LogSemiring, lp, sp)
            outside_chart(res, g, sp)
            ex = expectations(toks, g, res, sp)
            bf = brute_force(toks, g, lp, sp)
            assert abs(res.log_z - bf["log_z"]) < 1e-9
            if res.log_z == NEG_INF:
                continue
            for (i, j, c), p in bf["node_posteriors"].items():
                assert abs(res.node_posterior(i, j, c) - p) < 1e-9
            const = res.constituency_posterior()
            for key, p in bf["constituency"].items():
                assert abs(const.get(key, 0.0) - p) < 1e-9
            for r, p in bf["rule_counts"].items():
                assert abs(ex.rule_counts.get(r, 0.0) - p) < 1e-9
            for key, p in bf["lex_counts"].items():
                assert abs(ex.lex_counts.get(key, 0.0) - p) < 1e-9
            for c, p in bf["root_counts"].items():
                assert abs(ex.root_counts.get(c, 0.0) - p) < 1e-9
            v = viterbi(toks, g, lp, sp)
            assert v is not None and abs(v.score - bf["best"]) < 1e-9
            # ---- probability invariants
            for key, p in const.items():
                assert -1e-12 <= p <= 1 + 1e-12
            for i in range(n):  # every token is a constituent
                assert abs(const[(i, i + 1)] - 1.0) < 1e-9
            assert abs(const[(0, n)] - 1.0) < 1e-9  # the sentence is a root constituent
            # sum_C P(Z_ij = C | Q_ij = 1) = 1
            for (i, j), d in res.span_posteriors().items():
                tot = sum(d.values())
                if tot > 1e-12:
                    assert abs(sum(p / tot for p in d.values()) - 1.0) < 1e-9
            # rule posteriors at (i,j,C) sum to the node posterior
            agg: dict = {}
            for r, i, k, j, p in ex.rule_apps:
                c = g.rules[r][0]
                agg[(i, j, c)] = agg.get((i, j, c), 0.0) + p
            for key, p in agg.items():
                assert abs(p - res.node_posterior(*key)) < 1e-9
            # expected counts: sum of rule counts = n - 1 (internal nodes); sum of lexical counts = n
            assert abs(sum(ex.rule_counts.values()) - (n - 1)) < 1e-9
            assert abs(sum(ex.lex_counts.values()) - n) < 1e-9


@pytest.mark.parametrize("name,g", tiny_grammars())
def test_numba_backend_matches_reference(name: str, g: CompiledGrammar) -> None:
    pytest.importorskip("numba")
    from abc_induction import chart_numba

    arrs = chart_numba.CompiledArrays(g, 100)
    for toks in SENTENCES[name]:
        n, K = len(toks), g.n_classes
        lam, psi = _potentials(n, K, 2)
        for lp, sp in ((None, None), (lam, psi)):
            res = inside_chart(toks, g, LogSemiring, lp, sp)
            outside_chart(res, g, sp)
            ex = expectations(toks, g, res, sp)
            n_rules = len(g.rules)
            counts = (np.zeros(n_rules), np.zeros(max(1, len(arrs.entry_keys))), np.zeros(K), np.zeros(K))
            bufs = (np.zeros((1000, 4)), np.zeros((1000, 5)), np.zeros((1, 3)))
            out = chart_numba.run_sentence(arrs, np.asarray(toks), lp, sp, None, 0, 0.0, 0, 0.0, 0.0, counts, bufs)
            log_z = out[0]
            if res.log_z == NEG_INF:
                assert log_z <= chart_numba.NEG
                continue
            assert abs(log_z - res.log_z) < 1e-9
            for r in range(n_rules):
                rr = int(arrs.rule_order[r])
                assert abs(counts[0][r] - ex.rule_counts.get(rr, 0.0)) < 1e-9
            for e, (c, w) in enumerate(arrs.entry_keys):
                assert abs(counts[1][e] - ex.lex_counts.get((c, w), 0.0)) < 1e-9
            for c in range(K):
                assert abs(counts[2][c] - ex.lex_counts.get((c, -1), 0.0)) < 1e-9
                assert abs(counts[3][c] - ex.root_counts.get(c, 0.0)) < 1e-9
            n_nodes = int(out[2])
            node_buf = bufs[0]
            for q in range(n_nodes):
                i, j, c, p = int(node_buf[q, 0]), int(node_buf[q, 1]), int(node_buf[q, 2]), node_buf[q, 3]
                assert abs(res.node_posterior(i, j, c) - p) < 1e-9


def test_chunk_layer_matches_segmentation_enumeration() -> None:
    _, g = tiny_grammars()[2]
    toks = [10, 11, 11, 10]
    pc = 0.3
    chunk = (LOG(pc), LOG(1 - pc))
    res = inside_chart(toks, g, LogSemiring, chunk_logp=chunk)
    outside_chart(res, g)
    ex = expectations(toks, g, res)
    n = len(toks)

    def segs(i):
        if i == n:
            yield []
            return
        for j in range(i + 1, n + 1):
            for rest in segs(j):
                yield [(i, j)] + rest

    total = 0.0
    nodes: dict = {}
    nch = 0.0
    for sg in segs(0):
        w = (1 - pc) * pc ** (len(sg) - 1)
        parts = [(brute_force(toks[i:j], g), i) for (i, j) in sg]
        z = w
        for bf, _ in parts:
            z *= math.exp(bf["log_z"]) if bf["log_z"] > NEG_INF else 0.0
        if z == 0:
            continue
        total += z
        nch += z * len(sg)
        for bf, off in parts:
            for (a, b, c), p in bf["node_posteriors"].items():
                nodes[(a + off, b + off, c)] = nodes.get((a + off, b + off, c), 0.0) + z * p
    assert abs(res.log_z - math.log(total)) < 1e-9
    for key, v in nodes.items():
        assert abs(res.node_posterior(*key) - v / total) < 1e-9
    assert abs(ex.n_chunks - nch / total) < 1e-9


def test_posterior_rule_conditional_sums_to_one() -> None:
    m = GrammarModel()
    a, b, c, d = (m.new_class(x) for x in "ABCD")
    m.add_rule(c, a, b, 3.0)
    m.add_rule(d, a, b, 1.0)
    m.add_rule(c, b, a, 2.0)
    tot = m.rule_conditional((c, a, b)) + m.rule_conditional((d, a, b))
    assert abs(tot - 1.0) < 1e-12
    assert abs(m.rule_conditional((c, b, a)) - 1.0) < 1e-12
