"""CAVI (ELBO monotonicity, invariants) and structure-learning (over-split / over-merged) tests."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abc_induction.cavi import Potentials, VariationalInference  # noqa: E402
from abc_induction.corpus import Corpus  # noqa: E402
from abc_induction.grammar_model import UNK, GrammarModel, StructurePrior  # noqa: E402
from abc_induction.probabilistic import sentence_arrays, span_type_arrays  # noqa: E402
from abc_induction.spans import enumerate_spans  # noqa: E402
from abc_induction.structure_learning import StructureLearner  # noqa: E402
from abc_induction.synthetic import load_grammar  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ELBO_TOL = 1e-8


def _corpus(n: int = 600, seed: int = 5):
    g = load_grammar(os.path.join(ROOT, "grammars", "synthetic1_clean.yaml"))
    ds = g.generate(n, seed=seed)
    corpus = Corpus.from_token_lists(ds.token_lists(), max_sentence_length=10**9)
    inv = enumerate_spans(corpus, 8, 2)
    return g, ds, corpus, inv


def _model(g, ds, corpus, mode: str) -> tuple[GrammarModel, dict[str, int]]:
    """Gold grammar, optionally with NP over-split into NP / NP2 or Adj+Adv over-merged."""
    m = GrammarModel(alpha_rule=0.5, alpha_lex=0.5, chunks_enabled=False, prior=StructurePrior())
    alias = {"Adj": "AdjAdv", "Adv": "AdjAdv"} if mode == "overmerged" else {}
    names = sorted({alias.get(c, c) for c in ds.classes}) + (["NP2"] if mode == "oversplit" else [])
    cid = {c: m.new_class(c) for c in names}
    for r in ds.rules:
        m.add_rule(cid[alias.get(r.result, r.result)], cid[alias.get(r.left, r.left)], cid[alias.get(r.right, r.right)], 10.0)
        if mode == "oversplit":
            for side in ("left", "right", "result"):
                if getattr(r, side) == "NP":
                    k = {"left": r.left, "right": r.right, "result": r.result}
                    k[side] = "NP2"
                    m.add_rule(cid[k["result"]], cid[k["left"]], cid[k["right"]], 10.0)
    for c, lex in g.lexicon.items():
        for n, w in enumerate(sorted(lex)):
            if w not in corpus.vocab:
                continue
            target = alias.get(c, c)
            if mode == "oversplit" and c == "NP":
                target = "NP" if n % 2 == 0 else "NP2"
            m.add_lex(cid[target], corpus.vocab[w], 5.0)
    m.root = {cid["S"]: 10.0}
    return m, cid


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_elbo_non_decreasing_and_invariants(backend: str) -> None:
    if backend == "numba":
        pytest.importorskip("numba")
    g, ds, corpus, inv = _corpus(300)
    m, cid = _model(g, ds, corpus, "gold")
    m.add_rule(cid["NP"], cid["Det"], cid["Adj"], 1.0)  # junk rule
    m.add_lex(cid["Adv"], corpus.vocab["big"], 1.0)  # spurious ambiguity
    vi = VariationalInference(m, sentence_arrays(corpus), {}, Potentials(), span_type_arrays(inv), backend=backend, chart_cfg={"exact": True})
    rows = vi.run(10, tolerance=0.0)
    elbo = [r["ELBO"] for r in rows]
    for a, b in zip(elbo, elbo[1:]):
        assert b >= a - ELBO_TOL * abs(a), elbo
    assert all(not r.get("elbo_decreased") for r in rows)
    # invariants: posterior-mean P(C | A, B) sums to one; type mixtures sum to one; junk rule dies
    pairs = {(a, b) for (c, a, b) in m.rules}
    for a, b in pairs:
        tot = sum(m.rule_conditional((c, a, b)) for (c, aa, bb) in m.rules if (aa, bb) == (a, b))
        assert abs(tot - 1.0) < 1e-9
    for w in (corpus.vocab["big"], corpus.vocab["the"]):
        mix = m.type_mixture(w)
        assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert m.rules[(cid["NP"], cid["Det"], cid["Adj"])] < 1e-3
    assert m.type_mixture(corpus.vocab["big"])[cid["Adj"]] > 0.99
    ec, out = vi.e_step(collect=True)
    for sid, nodes in list(out.node_posteriors.items())[:50]:
        n = len(corpus.token_ids[sid])
        const: dict = {}
        for i, j, c, p in nodes:
            assert -1e-9 <= p <= 1 + 1e-9
            const[(i, j)] = const.get((i, j), 0.0) + p
        for i in range(n):  # every token is a constituent, the sentence is the root
            assert abs(const.get((i, i + 1), 0.0) - 1.0) < 1e-6
        assert abs(const.get((0, n), 0.0) - 1.0) < 1e-6


def _learn(mode: str):
    pytest.importorskip("numba")
    g, ds, corpus, inv = _corpus(900)
    m, cid = _model(g, ds, corpus, mode)
    vi = VariationalInference(m, sentence_arrays(corpus), {}, Potentials(), span_type_arrays(inv), backend="numba", chart_cfg={"exact": True})
    cfg = {"merge_threshold": 0.0, "split_threshold": 5.0, "rule_threshold": 0.0, "proposal_vb_iterations": 5, "proposal_eval": "full", "max_rounds": 3, "split_min_occurrences": 20, "max_cavi_iterations": 30, "max_proposals_per_stage": 8}
    sl = StructureLearner(vi, sentence_arrays(corpus), inv, cfg, {"enabled": True}, None, m.names)
    F0 = sl.full_cavi("init")
    moves = sl.run()
    return m, cid, corpus, sl, F0, moves


def test_oversplit_classes_are_merged() -> None:
    m, cid, corpus, sl, F0, moves = _learn("oversplit")
    acc = [mv for mv in moves if mv.get("accepted")]
    assert any(mv["kind"] == "merge" and {mv["a_name"], mv["b_name"]} == {"NP", "NP2"} for mv in acc)
    assert sl.F > F0 and sl.monotonicity_violations == 0
    assert not any(mv["kind"] == "split" for mv in acc)


def test_overmerged_class_is_split() -> None:
    m, cid, corpus, sl, F0, moves = _learn("overmerged")
    acc = [mv for mv in moves if mv.get("accepted")]
    assert any(mv["kind"] == "split" and mv["a_name"] == "AdjAdv" for mv in acc)
    assert sl.F > F0 and sl.monotonicity_violations == 0
    # the halves separate adjectives from adverbs
    halves = [c for c in m.classes if m.names[c].startswith("AdjAdv.")]
    assert len(halves) == 2
    words = {c: {corpus.itos[w] for (cc, w), n in m.lex.items() if cc == c and w != UNK and n > 1.0} for c in halves}
    adj = {"big", "small", "old", "young", "red", "happy"}
    adv = {"quickly", "slowly", "quietly", "today", "often"}
    assert any(words[c] <= adj and words[c] for c in halves) and any(words[c] <= adv and words[c] for c in halves)


def test_accepted_moves_never_decrease_objective() -> None:
    m, cid, corpus, sl, F0, moves = _learn("oversplit")
    objs = [mv["new_objective"] for mv in moves if mv.get("accepted")]
    prev = F0
    for mv in moves:
        if mv.get("accepted"):
            assert mv["new_objective"] > mv["old_objective"]
    assert sl.monotonicity_violations == 0
