"""Identity-preserving (modifier) evidence: string indexes, punctuation handling, gold extraction, evaluation."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abc_induction.cavi import EStepOutput  # noqa: E402
from abc_induction.corpus import Corpus  # noqa: E402
from abc_induction.grammar_model import GrammarModel, StructurePrior  # noqa: E402
from abc_induction.modifier_eval import evaluate_modifiers, gold_modifiers  # noqa: E402
from abc_induction.modifier_evidence import ModifierAnalyzer, modifier_rule_candidates  # noqa: E402
from abc_induction.spans import enumerate_spans  # noqa: E402
from abc_induction.synthetic import load_grammar  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

SENTS = [
    "John runs .", "John runs quickly .", "John runs quickly .",
    "Mary sleeps .", "apparently Mary sleeps .",
    "John likes Mary .", "John really likes Mary .",
    "dog sleeps .", "the dog sleeps .",
    "John runs !",  # punctuation variant of sentence 1
]


def _toy():
    corpus = Corpus.from_token_lists([s.split() for s in SENTS], max_sentence_length=10**9)
    inv = enumerate_spans(corpus, 8, 1)
    m = GrammarModel(alpha_rule=0.5, alpha_lex=0.5, chunks_enabled=False, prior=StructurePrior())
    S, NP, VP, Adv = (m.new_class(n) for n in ("S", "NP", "VP", "Adv"))
    m.add_rule(S, NP, VP, 10.0)
    m.add_rule(VP, VP, Adv, 5.0)
    m.root = {S: 10.0}
    out = EStepOutput()
    v = corpus.vocab
    # posterior constituents for two sentences: "John runs quickly" twice (sids 1, 2) and "John runs" (0)
    for sid in (1, 2):
        out.node_posteriors[sid] = [(0, 1, NP, 1.0), (1, 2, VP, 1.0), (2, 3, Adv, 1.0), (1, 3, VP, 1.0), (0, 3, S, 1.0)]
        out.rule_apps[sid] = [((S, NP, VP), 0, 1, 3, 1.0), ((VP, VP, Adv), 1, 2, 3, 1.0)]
    out.node_posteriors[0] = [(0, 1, NP, 1.0), (1, 2, VP, 1.0), (0, 2, S, 1.0)]
    out.rule_apps[0] = [((S, NP, VP), 0, 1, 2, 1.0)]
    for sid in range(3, len(SENTS)):
        out.node_posteriors[sid] = []
        out.rule_apps[sid] = []
    return corpus, inv, m, out, v


def test_root_string_pairs_and_punctuation() -> None:
    corpus, inv, m, out, v = _toy()
    an = ModifierAnalyzer(corpus, inv, {"max_modifier_length": 2, "max_insert_span_length": 2})
    tab = an.collect(out, m, m.names)
    rows = {(r["source"], r["candidate"], r["direction"]): r for r in tab.rows}
    right = rows[("ROOT_STRING", "quickly", "RIGHT")]
    assert right["support"] == 2.0  # two "John runs quickly ." sentences
    assert right["host_class_name"] == "ROOT" and right["candidate_class_name"] == "Adv"
    assert ("ROOT_STRING", "apparently", "LEFT") in rows
    assert ("ROOT_STRING", "really", "INTERNAL") in rows
    assert ("ROOT_STRING", "the", "LEFT") in rows  # the bare-noun control is evidence too (sentencehood, not category)
    # sentence-final punctuation is stripped: "John runs ." and "John runs !" are the same string, "." / "!" never candidates
    assert not any(r["candidate"] in (".", "!") for r in tab.rows)
    assert tab.stats["root_string"]["RIGHT"] >= 1
    # posterior rule evidence from the endomorphic rule VP + Adv -> VP
    pr = rows[("POSTERIOR_RULE", "quickly", "RIGHT")]
    assert pr["host_class_name"] == "VP" and abs(pr["support"] - 2.0) < 1e-9 and pr["mod"] == 1.0
    # class-generalised evidence inside the sentence: E(runs) = E(runs quickly) = VP
    cg = rows[("CLASS_GENERALIZED", "quickly", "RIGHT")]
    assert cg["host_class_name"] == "VP" and cg["support"] == 2.0


def test_rule_candidates_respect_mode_and_existing_rules() -> None:
    corpus, inv, m, out, v = _toy()
    an = ModifierAnalyzer(corpus, inv, {"max_modifier_length": 2})
    tab = an.collect(out, m, m.names)
    assert modifier_rule_candidates(tab, m, "off", {}) == []
    cands = modifier_rule_candidates(tab, m, "string_only", {"proposal_min_support": 1, "max_rule_proposals": 10})
    rules = {c["rule"] for c in cands}
    by_name = {n: c for c, n in m.names.items()}
    S, VP, Adv = by_name["S"], by_name["VP"], by_name["Adv"]
    assert (S, S, Adv) in rules  # S + Adv -> S from the root evidence (host = root class)
    assert (VP, VP, Adv) not in rules  # already in the grammar


def test_gold_modifiers_and_evaluation_on_synthetic4() -> None:
    g = load_grammar(os.path.join(ROOT, "grammars", "synthetic4_modifier.yaml"))
    ds = g.generate(300, seed=3)
    gm = gold_modifiers(ds)
    assert set(gm["gold_rules"]) == {"('Adj', 'Nn', 'Nn')", "('AdvS', 'S', 'S')", "('Deg', 'Vt', 'Vt')", "('VP', 'Adv', 'VP')"}
    for w, d in gm["gold_types"].items():
        assert d["direction"] in ("LEFT", "RIGHT")
    assert all(gm["gold_types"][w]["direction"] == "RIGHT" for w in ("quickly", "slowly") if w in gm["gold_types"])
    assert "the" not in gm["gold_types"] and "Mary" not in gm["gold_types"]
    # evaluation plumbing on an empty table
    corpus = Corpus.from_token_lists(ds.token_lists(), max_sentence_length=10**9)
    inv = enumerate_spans(corpus, 8, 2)
    m = GrammarModel(prior=StructurePrior())
    an = ModifierAnalyzer(corpus, inv, {})
    tab = an.collect(EStepOutput(), m, {}, sources=("ROOT_STRING",))
    res = evaluate_modifiers(tab, gm, {})
    r = res["by_source"]["ROOT_STRING"]["support>=2"]
    assert 0.0 <= r["precision"] <= 1.0 and r["recall"] > 0.0  # the A. / AB. pairs alone find gold modifiers
    assert r["RIGHT"]["precision"] >= r["LEFT"]["precision"]  # the Det control only pollutes the LEFT side
    assert res["by_source"]["POSTERIOR_RULE"]["support>=1"]["n_pred"] == 0
