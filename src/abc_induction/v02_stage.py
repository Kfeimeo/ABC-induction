"""Driver of the v0.2 probabilistic stage (v0.1 result -> GrammarModel -> CAVI -> structure learning -> posteriors)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cavi import Potentials, VariationalInference
from .contexts import ContextData
from .corpus import Corpus
from .grammar_model import GrammarModel, StructurePrior
from .probabilistic import (
    PosteriorDecoding,
    context_potentials,
    decode_posteriors,
    init_from_v01,
    punctuation_potentials,
    random_init,
    sentence_arrays,
    span_type_arrays,
)
from .spans import SpanInventory
from .structure_learning import StructureLearner

log = logging.getLogger(__name__)


@dataclass
class ProbabilisticResult:
    model: GrammarModel
    vi: VariationalInference
    decoding: PosteriorDecoding
    trace: list[dict[str, Any]]
    moves: list[dict[str, Any]]
    learner: StructureLearner | None
    objective: dict[str, float]
    initial_objective: dict[str, float]
    timings: dict[str, float]
    init_summary: dict[str, Any]
    potentials_info: dict[str, Any]
    posterior_stats: dict[str, Any] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)


def run_probabilistic_stage(
    cfg: dict[str, Any],
    corpus: Corpus,
    inv: SpanInventory,
    cd: ContextData,
    accepted: set[int],
    reports: dict[int, Any],
    mdl_model: Any | None,
    type_classes: dict[int, list[int]],
    occ_class: np.ndarray,
) -> ProbabilisticResult:
    """Initialise from v0.1, run CAVI, structure learning and posterior decoding."""
    pcfg = cfg.get("probabilistic") or {}
    sl_cfg = dict(cfg.get("structure_learning") or {})
    ad_cfg = cfg.get("adaptive_context") or {}
    lc_cfg = cfg.get("lexical_context") or {}
    pu_cfg = cfg.get("punctuation") or {}
    ch_cfg = cfg.get("chart") or {}
    prior = StructurePrior.from_cfg(cfg.get("structure_prior"))
    timings: dict[str, float] = {}
    t = time.time()
    if pcfg.get("init", "v01") == "random":
        model = random_init(corpus, inv, int(pcfg.get("random_k", 8)), int(cfg.get("seed", 42)), pcfg, prior)
    else:
        model = init_from_v01(inv, accepted, reports, mdl_model, type_classes, occ_class, pcfg, prior)
    init_summary = model.summary()
    sents = sentence_arrays(corpus)
    st = span_type_arrays(inv)
    timings["init"] = time.time() - t
    # ---- potentials
    t = time.time()
    leaf_scores: dict[int, list[dict[int, float]]] = {}
    if lc_cfg.get("enabled", True) or lc_cfg.get("proposals", True):
        leaf_scores = context_potentials(inv, cd, model, lc_cfg)
    leaf = leaf_scores if lc_cfg.get("enabled", True) else {}
    span = punctuation_potentials(corpus, pu_cfg) if pu_cfg.get("mode", "punctuation_as_token") == "structural_prior" else {}
    pots = Potentials(leaf, span, float(lc_cfg.get("floor", -6.0)))
    timings["potentials"] = time.time() - t
    potentials_info = {
        "lexical_context_in_elbo": bool(leaf),
        "n_sentences_with_leaf_potentials": len(leaf),
        "n_sentences_with_span_potentials": len(span),
        "punctuation_mode": pu_cfg.get("mode", "punctuation_as_token"),
    }
    backend = ch_cfg.get("backend", "numba")
    if backend == "numba":
        try:
            import numba  # noqa: F401
        except Exception:
            log.warning("numba not available: falling back to the python chart backend")
            backend = "python"
    vi = VariationalInference(model, sents, pcfg, pots, st, backend=backend, chart_cfg=ch_cfg)
    # ---- CAVI
    t = time.time()
    sl_cfg.setdefault("max_cavi_iterations", pcfg.get("max_cavi_iterations", 50))
    sl_cfg.setdefault("cavi_tolerance", pcfg.get("cavi_tolerance", 1e-5))
    sl_cfg.setdefault("seed", cfg.get("seed", 42))
    learner = StructureLearner(vi, sents, inv, sl_cfg, ad_cfg, leaf_scores if lc_cfg.get("proposals", True) else None, model.names)
    F0 = learner.full_cavi("cavi_init")
    initial_objective = dict(learner.F_parts)
    timings["cavi_init"] = time.time() - t
    log.info("v0.2 CAVI: objective %.2f (ELBO %.2f, prior %.2f), %d classes, %d rules", F0, learner.F_parts["elbo"], learner.F_parts["structure_prior"], model.n_classes, len(model.active_rules()))
    moves: list[dict[str, Any]] = []
    t = time.time()
    if sl_cfg.get("enabled", True):
        moves = learner.run()
        learner.full_cavi("cavi_final")
    timings["structure_learning"] = time.time() - t
    # ---- decoding
    t = time.time()
    assert learner.ec is not None and learner.out is not None
    r = cfg.get("relations") or {}
    dec = decode_posteriors(vi, inv, learner.out, learner.ec, pcfg, float(r.get("min_support", 2)), int(r.get("min_lexical_support", 2)))
    timings["decoding"] = time.time() - t
    names = {c: model.names.get(c, f"C{c}") for c in model.classes}
    stats = posterior_stats(model, learner, dec, inv)
    return ProbabilisticResult(model, vi, dec, list(vi.trace), moves, learner, dict(learner.F_parts), initial_objective, timings, init_summary, potentials_info, stats, names)


def posterior_stats(model: GrammarModel, learner: StructureLearner, dec: PosteriorDecoding, inv: SpanInventory) -> dict[str, Any]:
    """``posterior_stats.json``: per-class debug views and corpus-level posterior summaries."""
    out_e = learner.out
    assert out_e is not None
    names = model.names
    corpus = inv.corpus
    classes = []
    occ_by_class: dict[int, list[tuple[float, str]]] = {c: [] for c in model.classes}
    ctx_by_class: dict[int, dict[str, float]] = {c: {} for c in model.classes}
    for sid, nodes in out_e.node_posteriors.items():
        toks = corpus.sentences[sid].tokens
        n = len(toks)
        for i, j, c, p in nodes:
            if p < 0.5 or c not in occ_by_class:
                continue
            left = toks[i - 1] if i > 0 else "<BOS>"
            right = toks[j] if j < n else "<EOS>"
            key = f"{left} _ {right}"
            d = ctx_by_class[c]
            d[key] = d.get(key, 0.0) + p
            lst = occ_by_class[c]
            if len(lst) < 400:
                lst.append((p, " ".join(toks[:i]) + " [" + " ".join(toks[i:j]) + "] " + " ".join(toks[j:])))
    abstract = learner.abstract_context_report(names)
    for c in sorted(model.classes, key=lambda c: -dec.class_mass.get(c, 0.0)):
        types = sorted(dec.class_members.get(c, []), key=lambda t: -dec.type_mixture.get(t, {}).get(c, 0.0) * inv.type_freq[t])
        comp_in = sorted(((k, n) for k, n in model.rules.items() if k[0] == c and n > 0.5), key=lambda kv: -kv[1])[:8]
        comp_out = sorted(((k, n) for k, n in model.rules.items() if (k[1] == c or k[2] == c) and n > 0.5), key=lambda kv: -kv[1])[:8]
        lex = sorted(((w, n) for (cc, w), n in model.lex.items() if cc == c and n > 0.5), key=lambda kv: -kv[1])[:15]
        classes.append({
            "class": c,
            "name": names.get(c),
            "posterior_mass": round(dec.class_mass.get(c, 0.0), 2),
            "root_mass": round(model.root.get(c, 0.0), 2),
            "lineage": model.parent.get(c),
            "top_lexical_entries": [f"{corpus.itos[w] if w >= 0 else '<unk>'} ({n:.1f})" for w, n in lex],
            "top_span_types": [f"{inv.type_str(t)} ({dec.type_mixture[t].get(c, 0):.2f} x{int(inv.type_freq[t])})" for t in types[:15]],
            "top_occurrences": [s for _, s in sorted(occ_by_class.get(c, []), key=lambda t: -t[0])[:6]],
            "top_lexical_contexts": [f"{k} ({v:.1f})" for k, v in sorted(ctx_by_class.get(c, {}).items(), key=lambda kv: -kv[1])[:8]],
            "top_composition_in": [f"{names.get(a, a)} + {names.get(b, b)} -> {names.get(cc, cc)} ({n:.1f})" for (cc, a, b), n in comp_in],
            "top_composition_out": [f"{names.get(a, a)} + {names.get(b, b)} -> {names.get(cc, cc)} ({n:.1f})" for (cc, a, b), n in comp_out],
            "top_abstract_constituent_contexts": abstract.get(names.get(c, str(c)), []),
        })
    private = []
    for key, n in model.rules.items():
        real = model.realisations.get(key, {})
        if n >= model.prior.private_min_count and real:
            t, v = max(real.items(), key=lambda kv: kv[1])
            share = v / n
            if share >= 0.3:
                private.append({"rule": f"{names.get(key[1])} + {names.get(key[2])} -> {names.get(key[0])}", "expected_count": round(n, 1), "dominant_type": inv.type_str(t), "dominant_share": round(share, 3), "penalty": round(model.prior.private_penalty(n, v), 3)})
    private.sort(key=lambda d: -d["dominant_share"])
    return {
        "objective": {k: round(v, 4) for k, v in learner.F_parts.items()},
        "n_classes": model.n_classes,
        "n_rules": len(model.active_rules()),
        "n_lexical_entries": model.n_lex,
        "fully_parsed_fraction": round(dec.fully_parsed, 4),
        "mean_chunks_per_sentence": round(dec.mean_chunks, 4),
        "unknown_token_mass": dec.unknown_tokens,
        "mean_constituency_posterior_of_inventory_spans": round(float(dec.occ_const.mean()), 4) if len(dec.occ_const) else None,
        "n_ambiguous_types": len(dec.ambiguous_types),
        "highest_entropy_span_types": [
            {"span_type": inv.type_str(t), "entropy_nats": round(e, 4), "n_occurrences": n, "mixture": {names.get(c, str(c)): round(v, 3) for c, v in sorted(dec.type_mixture[t].items(), key=lambda kv: -kv[1])[:4]}}
            for t, e, n in dec.span_entropy[:25]
        ],
        "occurrence_ambiguity": list(dec.ambiguous_types.values())[:50],
        "private_rules": private[:20],
        "pruning": learner.ec.pruning if learner.ec is not None else {},
        "monotonicity_violations": learner.monotonicity_violations,
        "classes": classes,
    }
