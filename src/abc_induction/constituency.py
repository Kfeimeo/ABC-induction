"""Constituency evidence: from provisional classes to accepted constituents.

Context similarity alone does not prove constituency.  Every provisional class
``A`` receives

    Score_const(A) = (alpha S_context + beta S_composition + gamma S_productivity)
                     / (alpha + beta + gamma)  -  lambda S_complexity

* ``S_context``      : support from distinct contexts / occurrences / members.
* ``S_composition``  : fraction of multi-token occurrences that can be built
                       from two classified parts through a supported relation.
* ``S_productivity`` : how many distinct lexical realisations support the
                       relations producing ``A`` (saturating).
* ``S_complexity``   : singleton / fixed-phrase penalty.

Classes below ``min_score`` are rejected.  When MDL pruning is enabled, the
surviving classes and relations are additionally subjected to greedy backward
elimination under the description-length criterion of :mod:`abc_induction.mdl`:
a class (or a relation) is removed whenever the corpus can be encoded in fewer
bits without it.  Span types that are never used in any Viterbi derivation are
dropped from the accepted inventory as non-constituents.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .clustering import ClassState
from .composition import CompositionEvidence, collect_composition, productivity_score
from .contexts import ContextData
from .mdl import MDLCosts, MDLModel
from .spans import SpanInventory

log = logging.getLogger(__name__)


@dataclass
class ClassReport:
    """Human-readable constituency record of one provisional class."""

    cid: int
    members: list[int]
    n_members: int
    n_lexical_members: int
    n_occurrences: int
    n_distinct_contexts: int
    context_entropy: float
    s_context: float
    s_composition: float
    s_productivity: float
    s_complexity: float
    score: float
    status: str = "provisional"
    reason: str = ""
    mdl_delta: float | None = None
    mdl_delta_grammar: float | None = None
    mdl_delta_data: float | None = None
    usage: int = 0
    pruned_members: list[int] = field(default_factory=list)
    top_contexts: list[tuple[str, float]] = field(default_factory=list)

    def to_json(self, inv: SpanInventory) -> dict[str, Any]:
        return {
            "class": self.cid,
            "status": self.status,
            "reason": self.reason,
            "score": round(self.score, 4),
            "components": {
                "s_context": round(self.s_context, 4),
                "s_composition": round(self.s_composition, 4),
                "s_productivity": round(self.s_productivity, 4),
                "s_complexity": round(self.s_complexity, 4),
            },
            "n_members": self.n_members,
            "n_lexical_members": self.n_lexical_members,
            "n_occurrences": self.n_occurrences,
            "n_distinct_contexts": self.n_distinct_contexts,
            "context_entropy_bits": round(self.context_entropy, 3),
            "mdl_delta_bits": None if self.mdl_delta is None else round(self.mdl_delta, 2),
            "mdl_delta_grammar": None if self.mdl_delta_grammar is None else round(self.mdl_delta_grammar, 2),
            "mdl_delta_data": None if self.mdl_delta_data is None else round(self.mdl_delta_data, 2),
            "derivation_usage": self.usage,
            "members": [inv.type_str(t) for t in self.members[:40]],
            "pruned_members": [inv.type_str(t) for t in self.pruned_members[:40]],
            "top_contexts": [{"context": c, "p": round(p, 4)} for c, p in self.top_contexts],
        }


def _nl(v: float, vmax: float) -> float:
    return math.log1p(max(v, 0.0)) / math.log1p(max(vmax, 1.0)) if vmax > 0 else 0.0


def score_classes(
    state: ClassState,
    inv: SpanInventory,
    cd: ContextData,
    evidence: CompositionEvidence,
    occ_class: np.ndarray,
    cfg: dict[str, Any],
) -> dict[int, ClassReport]:
    """Compute the constituency score components for every provisional class."""
    alpha = float(cfg.get("alpha", 1.0))
    beta = float(cfg.get("beta", 1.0))
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("lambda_complexity", 1.0))
    singleton_pen = float(cfg.get("singleton_penalty", 0.5))
    fixed_phrase_pen = float(cfg.get("fixed_phrase_penalty", 0.5))
    min_rel_support = int(cfg.get("min_relation_support", 2))
    prod_scale = float(cfg.get("productivity_scale", 10.0))
    n_top_ctx = int(cfg.get("top_contexts", 8))
    ids, cmat = state.class_matrix(cd.type_ctx)
    row_of = {c: r for r, c in enumerate(ids)}
    lengths = inv.occ_end - inv.occ_start
    # per class occurrence stats
    n_occ = {c: int(sum(inv.type_freq[t] for t in state.members[c])) for c in ids}
    n_lex = {c: sum(1 for t in state.members[c] if inv.type_length(t) == 1) for c in ids}
    n_ctx = {c: int(cmat.indptr[row_of[c] + 1] - cmat.indptr[row_of[c]]) for c in ids}
    # supported relations by result class
    rel_by_result: dict[int, list[tuple[tuple[int, int, int], int, int]]] = {}
    for key, n in evidence.counts.items():
        if n >= min_rel_support:
            rel_by_result.setdefault(key[2], []).append((key, n, len(evidence.lexical_pairs[key])))
    # composition coverage: multi-token occurrences explained by a supported split
    explained = {c: 0 for c in ids}
    multi = {c: 0 for c in ids}
    for oid in np.nonzero(lengths >= 2)[0].tolist():
        c = int(occ_class[oid])
        if c < 0:
            continue
        multi[c] = multi.get(c, 0) + 1
        for k, a, b in evidence.occ_splits.get(oid, []):
            if evidence.counts.get((a, b, c), 0) >= min_rel_support:
                explained[c] = explained.get(c, 0) + 1
                break
    max_occ = max(n_occ.values(), default=1)
    max_mem = max((len(state.members[c]) for c in ids), default=1)
    max_ctx = max(n_ctx.values(), default=1)
    reports: dict[int, ClassReport] = {}
    for c in ids:
        members = state.members[c]
        r = row_of[c]
        a_, b_ = cmat.indptr[r], cmat.indptr[r + 1]
        data = cmat.data[a_:b_].astype(np.float64)
        p = data / data.sum() if data.sum() > 0 else data
        ent = float(-(p * np.log2(p + 1e-300)).sum()) if len(p) else 0.0
        order = np.argsort(-p)[:n_top_ctx]
        top_ctx = [(cd.vocab.render(int(cmat.indices[a_ + o]), inv.corpus), float(p[o])) for o in order]
        s_ctx = (_nl(n_ctx[c], max_ctx) + _nl(n_occ[c], max_occ) + _nl(len(members), max_mem)) / 3.0
        if multi.get(c, 0) == 0:
            s_comp = 1.0  # purely lexical class: nothing to compose
        else:
            s_comp = explained.get(c, 0) / multi[c]
        rels = rel_by_result.get(c, [])
        if rels:
            s_prod = max(productivity_score(lex, prod_scale) for _, _, lex in rels)
        elif multi.get(c, 0) == 0:
            s_prod = productivity_score(n_lex[c], prod_scale)
        else:
            s_prod = 0.0
        s_cplx = 0.0
        if len(members) == 1:
            # a single-token singleton is a lexical class (every word needs one);
            # a multi-token singleton is a candidate fixed phrase
            if inv.type_length(members[0]) >= 2:
                s_cplx += singleton_pen + fixed_phrase_pen
            else:
                s_cplx += float(cfg.get("lexical_singleton_penalty", 0.0))
        denom = alpha + beta + gamma
        score = ((alpha * s_ctx + beta * s_comp + gamma * s_prod) / denom if denom > 0 else 0.0) - lam * s_cplx
        reports[c] = ClassReport(
            cid=c,
            members=list(members),
            n_members=len(members),
            n_lexical_members=n_lex[c],
            n_occurrences=n_occ[c],
            n_distinct_contexts=n_ctx[c],
            context_entropy=ent,
            s_context=s_ctx,
            s_composition=s_comp,
            s_productivity=s_prod,
            s_complexity=s_cplx,
            score=score,
            top_contexts=top_ctx,
        )
    return reports


@dataclass
class ConstituencyResult:
    accepted: set[int]
    reports: dict[int, ClassReport]
    occ_class: np.ndarray
    evidence: CompositionEvidence
    mdl: MDLModel | None
    removed_relations: list[dict[str, Any]]
    constituent_mask: np.ndarray  # derivation-level constituents among candidate occurrences
    constituent_positions: set[tuple[int, int, int]] = field(default_factory=set)  # all derivation nodes (incl. predicted spans)


def _weak_component(rep: ClassReport) -> str:
    comps = {"context": rep.s_context, "composition": rep.s_composition, "productivity": rep.s_productivity}
    weakest = min(comps, key=comps.get)
    return f"weakest evidence: {weakest} ({comps[weakest]:.2f}); complexity penalty {rep.s_complexity:.2f}"


def select_constituents(
    state: ClassState,
    inv: SpanInventory,
    cd: ContextData,
    cfg: dict[str, Any],
    mdl_cfg: dict[str, Any] | None,
) -> ConstituencyResult:
    """Score, threshold and (optionally) MDL-prune the provisional classes."""
    occ_class = state.type_class[inv.occ_tid].astype(np.int64)
    evidence = collect_composition(inv, occ_class)
    reports = score_classes(state, inv, cd, evidence, occ_class, cfg)
    min_score = float(cfg.get("min_score", 0.3))
    accepted: set[int] = set()
    for c, rep in reports.items():
        if rep.score >= min_score:
            accepted.add(c)
            rep.status = "accepted"
        else:
            rep.status = "rejected_score"
            rep.reason = f"score {rep.score:.3f} < min_score {min_score}; " + _weak_component(rep)
    log.info("score threshold: %d/%d classes accepted", len(accepted), len(reports))
    removed_relations: list[dict[str, Any]] = []
    mdl_model: MDLModel | None = None
    mdl_cfg = mdl_cfg or {}
    predicted_positions: set[tuple[int, int, int]] = set()
    if mdl_cfg.get("enabled", False):
        mdl_model = _mdl_prune(state, inv, cfg, mdl_cfg, accepted, reports, evidence, occ_class, removed_relations)
        occ_class = np.where(np.isin(occ_class, list(accepted)), occ_class, -1)
        usage = mdl_model.usage_counts()
        subsampled = len(mdl_model.sids) < len(inv.corpus)
        # ---- label every candidate occurrence by the class of its derivation node
        # (the induced grammar's own analysis, including spans whose redundant
        # singleton class was pruned and which are now derived from their parts)
        node_class = np.full(inv.n_occurrences, -1, dtype=np.int64)
        for sid, p in mdl_model.parses.items():
            for nd in p.nodes:
                if nd.oid >= 0:
                    node_class[nd.oid] = nd.cls
                predicted_positions.add((sid, nd.start, nd.end))
        if subsampled:
            # only the parsed sentences carry derivation labels; keep assignment elsewhere
            parsed = np.zeros(inv.n_occurrences, dtype=bool)
            for sid in mdl_model.sids:
                parsed[inv.sent_occ[sid]] = True
            occ_class = np.where(parsed, node_class, occ_class)
            mask = occ_class >= 0
        else:
            occ_class = node_class
            mask = occ_class >= 0
        # ---- membership = types whose used occurrences carry the class (majority);
        # members never used in any derivation are reported as pruned
        type_counts: dict[int, dict[int, int]] = {}
        for oid in np.nonzero(occ_class >= 0)[0].tolist():
            t = int(inv.occ_tid[oid])
            d = type_counts.setdefault(t, {})
            c = int(occ_class[oid])
            d[c] = d.get(c, 0) + 1
        original_members = {c: list(reports[c].members) for c in accepted}
        for c in accepted:
            reports[c].members = []
            reports[c].usage = usage.get(c, 0)
        for t, d in type_counts.items():
            best_c = max(d.items(), key=lambda kv: kv[1])[0]
            if best_c in accepted:
                reports[best_c].members.append(t)
        for c in list(accepted):
            rep = reports[c]
            used = set(rep.members)
            rep.pruned_members = [t for t in original_members[c] if t not in used and not type_counts.get(t)]
            if not rep.members and not subsampled:
                accepted.discard(c)
                rep.status = "rejected_mdl"
                rep.reason = "no span is ever analysed as this class in a Viterbi derivation"
        mdl_model.occ_class = occ_class
        evidence = collect_composition(inv, occ_class)
    else:
        occ_class = np.where(np.isin(occ_class, list(accepted)), occ_class, -1)
        evidence = collect_composition(inv, occ_class)
        mask = occ_class >= 0
    if not predicted_positions:
        for o in np.nonzero(mask)[0].tolist():
            predicted_positions.add((int(inv.occ_sid[o]), int(inv.occ_start[o]), int(inv.occ_end[o])))
    return ConstituencyResult(accepted, reports, occ_class, evidence, mdl_model, removed_relations, mask, predicted_positions)


def _mdl_prune(
    state: ClassState,
    inv: SpanInventory,
    cfg: dict[str, Any],
    mdl_cfg: dict[str, Any],
    accepted: set[int],
    reports: dict[int, ClassReport],
    evidence: CompositionEvidence,
    occ_class: np.ndarray,
    removed_relations: list[dict[str, Any]],
) -> MDLModel:
    """Greedy backward elimination of relations and classes under the DL criterion."""
    costs = MDLCosts.from_cfg(mdl_cfg.get("costs"))
    min_rel = int(cfg.get("min_relation_support", 2))
    rng = np.random.default_rng(int(mdl_cfg.get("seed", 0)))
    n_sent = len(inv.corpus)
    max_sent = mdl_cfg.get("max_sentences")
    sids = None
    if max_sent and n_sent > int(max_sent):
        sids = sorted(rng.choice(n_sent, size=int(max_sent), replace=False).tolist())
    occ_for_mdl = np.where(np.isin(occ_class, list(accepted)), occ_class, -1)
    model = MDLModel(inv, occ_for_mdl, costs, sentence_ids=sids, min_rule_support=float(min_rel))
    # evidence keys are (A, B, C); the grammar's top-down tables use (C, A, B)
    # side productivity is a *reporting* filter for the final relation list; inside
    # the MDL grammar it removes rules that rare singleton classes need and hurts
    # (verified on Synthetic-1), so it is off by default here
    min_side = int(cfg.get("min_side_realisations", 1))
    rule_support = {}
    for (a, b, c), n in evidence.counts.items():
        if n < min_rel:
            continue
        nl, nr = evidence.side_counts((a, b, c))
        if nl < min_side or nr < min_side:
            continue
        rule_support[(c, a, b)] = float(n)
    model.initialise(accepted, rule_support)
    total0 = model.fit(em_iters=int(mdl_cfg.get("em_iters", 1)))
    log.info("MDL before pruning: %.1f bits (%s)", total0, model.summary())
    max_passes = int(mdl_cfg.get("max_passes", 6))
    margin = float(mdl_cfg.get("margin_bits", 0.0))
    max_eval = int(mdl_cfg.get("max_sentences_per_eval", 400))
    predictive = bool(mdl_cfg.get("predictive", True))
    model.beam = int(mdl_cfg.get("beam", 3))
    for pss in range(max_passes):
        n_removed = 0
        if predictive and pss >= 1 and not model.predict:
            # after the first (cheap) pass switch to the predictive chart:
            # rare spans may now be derived by the grammar from classified parts
            model.predict = True
            model.fit(em_iters=int(mdl_cfg.get("em_iters", 1)))
            log.info("MDL switched to predictive chart: %s", model.summary())
        # ---- relations first (least used first); only once the chart is predictive,
        # otherwise rules whose spans are mostly rare look useless and die too early
        rule_usage = model.rule_usage()
        rules = sorted(model.grammar.active_rules(), key=lambda k: (rule_usage.get(k, 0), k)) if (model.predict or not predictive) else []
        for key in rules:
            if key not in model.grammar.rules:
                continue
            delta, dg, dd, n_aff = model.rule_removal_delta(key)
            if delta < -margin:
                model.remove_rule(key)
                removed_relations.append(
                    {
                        "relation": list(key),
                        "label": f"{key[1]} + {key[2]} -> {key[0]}",
                        "usage": rule_usage.get(key, 0),
                        "delta_bits": round(delta, 2),
                        "delta_grammar": round(dg, 2),
                        "delta_data": round(dd, 2),
                        "why_rejected": "the corpus is encoded in fewer bits without this relation",
                    }
                )
                n_removed += 1
        # ---- classes (least used / lowest score first)
        usage = model.usage_counts()
        order = sorted(model.grammar.classes, key=lambda c: (usage.get(c, 0), reports[c].score))
        # every token needs a category: a class holding the only lexical entry of
        # some token can never be removed profitably (c_unk per occurrence), so
        # its removal test is skipped (big saving on natural text)
        lex_owner: dict[int, set[int]] = {}
        for (c_, t_) in model.grammar.active_lex():
            lex_owner.setdefault(t_, set()).add(c_)
        sole_owner = {next(iter(cs)) for cs in lex_owner.values() if len(cs) == 1}
        for c in order:
            if c not in model.grammar.classes:
                continue
            if c in sole_owner and bool(mdl_cfg.get("skip_sole_lexical_classes", True)):
                continue
            delta, dg, dd, n_aff = model.removal_delta(c, max_sentences=max_eval, rng=rng)
            rep = reports[c]
            rep.mdl_delta, rep.mdl_delta_grammar, rep.mdl_delta_data = delta, dg, dd
            if delta < -margin:
                model.remove_class(c)
                accepted.discard(c)
                rep.status = "rejected_mdl"
                rep.reason = (
                    f"removing it shortens the description by {-delta:.1f} bits "
                    f"(grammar {dg:+.1f}, data {dd:+.1f} over {n_aff} sentences; used in {usage.get(c, 0)} derivation nodes)"
                )
                n_removed += 1
        log.info("MDL pass %d: removed %d items, now %.1f bits, %d classes", pss, n_removed, model.total_bits(), len(model.grammar.classes))
        if n_removed == 0 and (model.predict or not predictive):
            break
        # re-estimate the tables from the pruned grammar before the next pass
        model.fit(em_iters=1)
    if predictive and not model.predict:
        model.predict = True
        model.fit(em_iters=int(mdl_cfg.get("em_iters", 1)))
    if int(mdl_cfg.get("em_iters_after", 1)) > 0:
        model.fit(em_iters=int(mdl_cfg.get("em_iters_after", 1)))
    log.info("MDL after pruning: %s", model.summary())
    return model
