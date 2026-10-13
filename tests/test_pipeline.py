"""Fast unit / smoke tests (run with ``python -m pytest tests -q``)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abc_induction.clustering import ClassState, agglomerate  # noqa: E402
from abc_induction.composition import collect_composition  # noqa: E402
from abc_induction.contexts import extract_contexts  # noqa: E402
from abc_induction.corpus import BOS, EOS, Corpus, tokenize  # noqa: E402
from abc_induction.evaluation import _pairwise, build_gold, hungarian_alignment  # noqa: E402
from abc_induction.experiment import DEFAULT_CONFIG, deep_update, run_experiment  # noqa: E402
from abc_induction.mdl import Grammar, MDLCosts, viterbi_sentence  # noqa: E402
from abc_induction.sparse_index import bhattacharyya, js_similarity, weighted_jaccard  # noqa: E402
from abc_induction.spans import enumerate_spans  # noqa: E402
from abc_induction.synthetic import load_grammar  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_tokenize_punctuation_modes() -> None:
    assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert tokenize("Hello, world!", punctuation_mode="remove") == ["Hello", "world"]
    assert tokenize("A b", lowercase=True) == ["a", "b"]


def test_span_enumeration_counts() -> None:
    corpus = Corpus.from_token_lists([["a", "b", "c"], ["a", "b", "d"]])
    inv = enumerate_spans(corpus, max_span_length=3, min_span_frequency=2)
    types = {inv.type_str(t) for t in range(inv.n_types)}
    assert types == {"a", "b", "a b"}
    assert inv.n_candidate_spans_total == 12
    assert inv.lookup(0, 0, 2) >= 0 and inv.type_str(inv.occ_tid[inv.lookup(0, 0, 2)]) == "a b"
    assert inv.lookup(0, 2, 3) == -1  # "c" occurs once


def test_context_features_use_bos_eos_and_scales() -> None:
    corpus = Corpus.from_token_lists([["x", "y"], ["x", "y"]])
    inv = enumerate_spans(corpus, 2, 2)
    cd = extract_contexts(corpus, inv, scales=(1, 2))
    tid = inv.type_index[(corpus.vocab["x"],)]
    oid = inv.type_occ[tid][0]
    feats = [cd.vocab.render(int(c), corpus) for c in cd.occ_ctx[oid]]
    assert feats[0] == f"[k=1] {BOS} _ y"
    assert feats[1] == f"[k=2] {BOS} {BOS} _ y {EOS}"
    assert cd.type_ctx.shape == (inv.n_types, cd.n_contexts)


def test_similarity_functions_bounds() -> None:
    ia, da = np.array([0, 1, 2]), np.array([2.0, 2.0, 2.0])
    ib, db = np.array([1, 2, 3]), np.array([2.0, 2.0, 2.0])
    assert abs(weighted_jaccard(ia, da, ib, db) - 4 / 8) < 1e-9
    assert 0.0 < js_similarity(ia, da, ib, db) < 1.0
    assert abs(bhattacharyya(ia, da, ia, da) - 1.0) < 1e-9
    assert abs(js_similarity(ia, da, ia, da) - 1.0) < 1e-9


def test_generator_gold_is_consistent() -> None:
    g = load_grammar(os.path.join(ROOT, "grammars", "synthetic2_ambiguity.yaml"))
    ds = g.generate(200, seed=1)
    assert len(ds.sentences) == 200
    for s in ds.sentences:
        n = len(s.tokens)
        assert 1 <= n <= 15
        for sp in s.spans:
            assert 0 <= sp.start < sp.end <= n
            assert sp.cls in ds.classes
        # the whole sentence is a gold S span
        assert any(sp.start == 0 and sp.end == n and sp.cls == "S" for sp in s.spans)
    assert sum(ds.relation_counts.values()) > 0
    # ambiguous words occur with both categories
    seen = {w: set() for w in g.ambiguity}
    for s in ds.sentences:
        for sp in s.spans:
            if sp.is_leaf and s.tokens[sp.start] in seen:
                seen[s.tokens[sp.start]].add(sp.cls)
    assert all(len(v) == 2 for v in seen.values())


def test_clustering_merges_substitutable_words() -> None:
    sents = []
    for i in range(60):
        for subj in ("john", "mary", "bill"):
            for verb in ("runs", "sleeps"):
                sents.append([subj, verb])
    corpus = Corpus.from_token_lists(sents)
    inv = enumerate_spans(corpus, 2, 2)
    cd = extract_contexts(corpus, inv, scales=(1,))
    state = agglomerate(cd, inv, {"merge_threshold": 0.7, "min_shared_contexts": 1, "max_context_df": 100, "attach": {"enabled": False}})
    cls = {inv.type_str(t): int(state.type_class[t]) for t in range(inv.n_types)}
    assert cls["john"] == cls["mary"] == cls["bill"]
    assert cls["runs"] == cls["sleeps"]
    assert cls["john"] != cls["runs"]
    assert state.history and state.history[0].evidence_contexts


def test_viterbi_prefers_rule_derivation_and_predicts_rare_spans() -> None:
    g = Grammar(MDLCosts())
    NP, VP, S = 1, 2, 3
    g.classes = {NP, VP, S}
    g.rules = {(S, NP, VP)}
    g.lex = {(NP, 10), (VP, 11)}
    g.counts = {(S, ("r", NP, VP)): 50.0, (NP, ("l", 10)): 50.0, (VP, ("l", 11)): 50.0}
    g.root_counts = {S: 50.0}
    g.refresh()
    spans = [(0, 1, 0, 10, NP), (1, 2, 1, 11, VP), (0, 2, 2, 12, S)]
    p = viterbi_sentence(2, spans, g)
    assert p.chunks == 1 and p.unk_tokens == 0
    assert any(nd.cls == S and nd.split == 1 for nd in p.nodes)
    # the sentence span is not a candidate: without prediction -> 2 chunks, with prediction -> S
    spans2 = [(0, 1, 0, 10, NP), (1, 2, 1, 11, VP)]
    assert viterbi_sentence(2, spans2, g).chunks == 2
    p2 = viterbi_sentence(2, spans2, g, predict=True)
    assert p2.chunks == 1 and p2.nodes[0].cls == S and p2.nodes[0].oid == -1


def test_pairwise_metrics_and_alignment() -> None:
    m = _pairwise(["a", "a", "b", "b"], [1, 1, 2, 2])
    assert m["f1"] == 1.0 and m["ari"] == 1.0 and m["purity"] == 1.0
    m2 = _pairwise(["a", "a", "b", "b"], [1, 1, 1, 1])
    assert m2["recall"] == 1.0 and m2["precision"] < 1.0
    assert hungarian_alignment([(1, "a"), (1, "a"), (2, "b")]) == {1: "a", 2: "b"}


def test_end_to_end_small_run() -> None:
    import tempfile

    out = os.path.join(tempfile.mkdtemp(prefix="abc_smoke_"), "smoke")
    cfg = deep_update(DEFAULT_CONFIG, {
        "name": "smoke",
        "dataset": {"grammar": os.path.join(ROOT, "grammars", "synthetic1_clean.yaml"), "n_sentences": 400},
        "output_dir": out,
        "mdl": {"max_passes": 2},
    })
    res = run_experiment(cfg)
    assert os.path.exists(os.path.join(out, "report.md"))
    assert os.path.exists(os.path.join(out, "metrics.json"))
    m = res.metrics
    assert m["constituents_derivation_level"]["all"]["f1"] > 0.5
    assert m["class_recovery_type_level"]["purity"] > 0.7
    ev = collect_composition(res.inv, res.occ_class)
    assert ev.n_occurrences_multi > 0
    assert isinstance(res.state, ClassState)
    gold = build_gold(res.dataset, res.corpus, res.inv)
    assert gold.gold_spans_total > 0
