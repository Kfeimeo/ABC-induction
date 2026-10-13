"""ProbLog oracle == Inside-Outside == brute force on tiny cases (skipped without problog)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from abc_induction.chart import NEG_INF, LogSemiring, brute_force, inside_chart, outside_chart  # noqa: E402
from abc_induction.problog_reference import ProbLogOracle, problog_available  # noqa: E402
from test_chart_exact import SENTENCES, _potentials, tiny_grammars  # noqa: E402

pytestmark = pytest.mark.skipif(not problog_available(), reason="problog not installed (optional oracle dependency)")

CASES = [(name, g, toks) for name, g in tiny_grammars() for toks in SENTENCES[name] if len(toks) <= 4]


@pytest.mark.parametrize("name,g,toks", CASES, ids=[f"{n}-{len(t)}" for n, _, t in CASES])
def test_problog_marginal_and_posteriors(name, g, toks) -> None:
    n, K = len(toks), g.n_classes
    lam, psi = _potentials(n, K, 7)
    for lp, sp in ((None, None), (lam, psi)):
        res = inside_chart(toks, g, LogSemiring, lp, sp)
        outside_chart(res, g, sp)
        bf = brute_force(toks, g, lp, sp)
        oracle = ProbLogOracle(toks, g, lp, sp)
        p_problog = oracle.marginal()
        p_chart = math.exp(res.log_z) if res.log_z > NEG_INF else 0.0
        assert abs(p_chart - p_problog) < 1e-8, (name, toks, p_chart, p_problog)
        assert abs(math.exp(bf["log_z"]) - p_problog) < 1e-8 if bf["log_z"] > NEG_INF else p_problog < 1e-12
        if p_chart == 0.0:
            continue
        nodes, const = oracle.posteriors()
        for (i, j, c), p in nodes.items():
            assert abs(res.node_posterior(i, j, c) - p) < 1e-8, (name, toks, (i, j, c), res.node_posterior(i, j, c), p)
        chart_const = res.constituency_posterior()
        for (i, j), p in const.items():
            assert abs(chart_const.get((i, j), 0.0) - p) < 1e-8


def test_multi_hop_uncertainty_shared_intermediate() -> None:
    """Uncertain classes + uncertain compositions sharing intermediate nodes: no
    independence / double-counting bug may survive the three-way comparison."""
    from abc_induction.chart import CompiledGrammar

    L = math.log
    # tokens: 10 in {A, B}; 11 in {B, C}; rules produce C from several routes, C -> C C recursive
    g = CompiledGrammar(
        3,
        [(2, 0, 1, L(0.3)), (2, 1, 1, L(0.2)), (2, 2, 2, L(0.25)), (2, 0, 2, L(0.15)), (0, 0, 0, L(0.1)), (1, 2, 0, L(0.2))],
        {10: [(0, L(0.5)), (1, L(0.3))], 11: [(1, L(0.3)), (2, L(0.1))]},
        np.log(np.array([0.2, 0.2, 0.6])),
        np.full(3, NEG_INF),
    )
    for toks, with_posteriors in (([10, 11, 10], True), ([10, 11, 10, 11], False)):
        n, K = len(toks), 3
        lam, psi = _potentials(n, K, 11)
        res = inside_chart(toks, g, LogSemiring, lam, psi)
        outside_chart(res, g, psi)
        bf = brute_force(toks, g, lam, psi)
        oracle = ProbLogOracle(toks, g, lam, psi)
        p = oracle.marginal()
        assert abs(math.exp(res.log_z) - p) < 1e-8
        assert abs(math.exp(bf["log_z"]) - p) < 1e-8
        if not with_posteriors:
            continue  # posterior grounding is slow for n = 4 with recursive rules
        nodes, const = oracle.posteriors()
        for key, q in nodes.items():
            assert abs(res.node_posterior(*key) - q) < 1e-8
            assert abs(bf["node_posteriors"].get(key, 0.0) - q) < 1e-8
