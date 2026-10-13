"""v0.2 probabilistic stage: v0.1 result -> GrammarModel -> CAVI -> structure learning -> posteriors.

    v0.1 (E, R, lexicon, Viterbi counts)   --initialise-->   GrammarModel (warm start)
    lexical contexts (v0.1 scorer)          --potentials-->   lambda_o(C)   (leaf virtual evidence)
    punctuation                             --prior------->   psi(i, j)     (span virtual evidence)
    CAVI  ->  class merge / split, rule & lexicon proposals (structure_learning)
    posterior decoding -> occurrence classes, constituency, type mixtures, relations

Nothing of v0.1 is modified; this module only consumes its result objects.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .cavi import EStepOutput, Potentials, VariationalInference
from .composition import CompositionRelation
from .contexts import ContextData
from .corpus import BOS_ID, EOS_ID, Corpus
from .grammar_model import UNK, ExpectedCounts, GrammarModel, StructurePrior
from .spans import SpanInventory
from .sparse_index import aggregate_rows

log = logging.getLogger(__name__)

SENTENCE_FINAL = {".", "?", "!"}
PAIRED = {"(": ")", "[": "]", '"': '"'}
INTERNAL = {",", ";", ":"}


# --------------------------------------------------------------------------- sentence data
def sentence_arrays(corpus: Corpus) -> list[np.ndarray]:
    return [np.asarray(t, dtype=np.int64) for t in corpus.token_ids]


def span_type_arrays(inv: SpanInventory) -> list[np.ndarray]:
    """Per sentence ``(n+1, n+1)`` span-type ids (-1 for spans not in the inventory)."""
    out = []
    for sid, ids in enumerate(inv.corpus.token_ids):
        n = len(ids)
        st = np.full((n + 1, n + 1), -1, dtype=np.int32)
        for o in inv.sent_occ[sid].tolist():
            st[inv.occ_start[o], inv.occ_end[o]] = inv.occ_tid[o]
        out.append(st)
    return out


# --------------------------------------------------------------------------- initialisation
def init_from_v01(
    inv: SpanInventory,
    accepted: set[int],
    reports: dict[int, Any],
    mdl_model: Any | None,
    type_classes: dict[int, list[int]],
    occ_class: np.ndarray,
    cfg: dict[str, Any],
    prior: StructurePrior,
) -> GrammarModel:
    """Warm start from the v0.1 grammar (classes, rules, lexicon and Viterbi usage counts).

    ``k_max`` caps the number of classes: classes are ranked by derivation
    usage and the surplus classes are dropped (their tokens fall back to the
    UNK expansion or to the entries of other classes).  Multi-token "phrase
    leaves" of v0.1 are not carried over: every multi-token span is derived
    by binary rules in v0.2 (continuous binary composition only).
    """
    model = GrammarModel.from_cfg(cfg, prior)
    k_max = int(cfg.get("k_max", 0) or 0)
    usage = {}
    if mdl_model is not None:
        usage = mdl_model.usage_counts()
    classes = sorted(accepted, key=lambda c: (-usage.get(c, 0), -reports[c].n_occurrences if c in reports else 0))
    if k_max and len(classes) > k_max:
        log.info("init: capping %d v0.1 classes to k_max=%d (by derivation usage)", len(classes), k_max)
        classes = classes[:k_max]
    keep = set(classes)
    for c in classes:
        model.add_class(c, f"C{c}")
    tok_of_type = {t: inv.types[t].tokens[0] for t in range(inv.n_types) if inv.type_length(t) == 1}
    min_count = float(cfg.get("init_min_rule_count", 0.0))
    if mdl_model is not None:
        g = mdl_model.grammar
        for (c, a, b) in g.rules:
            if c in keep and a in keep and b in keep:
                n = g.counts.get((c, ("r", a, b)), 0.0)
                if n >= min_count:
                    model.add_rule(c, a, b, float(n))
        for (c, t) in g.lex:
            if c in keep and t in tok_of_type:
                model.add_lex(c, tok_of_type[t], float(g.counts.get((c, ("l", t)), 0.0)))
        for c, n in g.root_counts.items():
            if c in keep and n > 0:
                model.root[c] = float(n)
    else:
        # no MDL grammar (ablation): rules from composition evidence over the v0.1 assignment
        from .composition import collect_composition

        ev = collect_composition(inv, occ_class)
        for (a, b, c), n in ev.counts.items():
            if c in keep and a in keep and b in keep and n >= max(2.0, min_count):
                model.add_rule(c, a, b, float(n))
    # lexical entries from the final v0.1 type membership (covers legacy ambiguity splits)
    for t, cls in type_classes.items():
        if t not in tok_of_type:
            continue
        for c in cls:
            if c in keep and (c, tok_of_type[t]) not in model.lex:
                model.add_lex(c, tok_of_type[t], float(inv.type_freq[t]) / max(1, len(cls)))
    # every class must have at least one expansion; roots default to classes seen as sentence spans
    if not model.root:
        for c in classes[:1]:
            model.root[c] = 1.0
    for c in list(model.classes):
        if not model.expansions(c):
            model.remove_class(c)
    log.info("init from v0.1: %d classes, %d rules, %d lexical entries", model.n_classes, model.n_rules, model.n_lex)
    return model


def random_init(corpus: Corpus, inv: SpanInventory, k: int, seed: int, cfg: dict[str, Any], prior: StructurePrior, entries_per_token: int = 2, rule_density: float = 0.3) -> GrammarModel:
    """Random initialisation (robustness ablation only): ``k`` classes, random lexical
    entries for every frequent token and a random sparse rule set."""
    rng = np.random.default_rng(seed)
    model = GrammarModel.from_cfg(cfg, prior)
    classes = [model.new_class() for _ in range(k)]
    for t in range(inv.n_types):
        if inv.type_length(t) != 1:
            continue
        w = inv.types[t].tokens[0]
        for c in rng.choice(classes, size=min(entries_per_token, k), replace=False):
            model.add_lex(int(c), w, float(rng.uniform(0.5, 2.0)))
    for c in classes:
        for a in classes:
            for b in classes:
                if rng.random() < rule_density:
                    model.add_rule(c, a, b, float(rng.uniform(0.5, 2.0)))
    for c in classes:
        model.root[c] = float(rng.uniform(0.5, 2.0))
    return model


# --------------------------------------------------------------------------- lexical context potentials
def context_potentials(
    inv: SpanInventory,
    cd: ContextData,
    model: GrammarModel,
    cfg: dict[str, Any],
) -> dict[int, list[dict[int, float]]]:
    """``log lambda_o(C)`` for every token position from the v0.1 context scorer.

    For class ``C`` the context profile is the (posterior-weighted) sum of the
    context rows of its lexical members; the profile of the type of ``o``
    itself is left out (leave-one-out, as in v0.1's ambiguity detector).  The
    per-occurrence score is ``score(o, C) = sum_slots log P(ctx | C)`` with
    background backoff, and ``lambda_o(C) proportional to exp(score / T)``,
    normalised over the lexical classes and floored at ``floor`` (log).
    Only the ``top_k`` classes per occurrence are stored (sparse).
    """
    max_scale = int(cfg.get("max_scale", 2))
    T = float(cfg.get("temperature", 1.0))
    eta = float(cfg.get("backoff_eta", 1.0))
    top_k = int(cfg.get("top_k", 5))
    floor = float(cfg.get("floor", -6.0))
    col_mask = (cd.ctx_scale <= max_scale).astype(np.float64)
    slot_mask = np.asarray(cd.slot_scale) <= max_scale
    tok_type = {inv.types[t].tokens[0]: t for t in range(inv.n_types) if inv.type_length(t) == 1}
    classes = sorted(c for c in model.classes if any(w != UNK for (cc, w) in model.lex if cc == c))
    if not classes:
        return {}
    cidx = {c: i for i, c in enumerate(classes)}
    # class x type membership weights from the lexical expected counts
    rows, cols, vals = [], [], []
    for (c, w), n in model.lex.items():
        if w == UNK or c not in cidx or w not in tok_type:
            continue
        t = tok_type[w]
        tot = sum(nn for (cc, ww), nn in model.lex.items() if ww == w and ww != UNK)
        share = (n + 1e-3) / (tot + 1e-3 * max(1, sum(1 for (cc, ww) in model.lex if ww == w)))
        rows.append(cidx[c])
        cols.append(t)
        vals.append(share)
    memb = sp.csr_matrix((vals, (rows, cols)), shape=(len(classes), inv.n_types))
    xw = sp.csr_matrix(cd.type_ctx.astype(np.float64) @ sp.diags(col_mask))
    xw.eliminate_zeros()
    cmat = sp.csr_matrix(memb @ xw)
    class_total = np.asarray(cmat.sum(axis=1)).ravel()
    cmat_csc = cmat.tocsc()  # column slicing per occurrence must be O(column nnz), not O(nnz)
    xw_csc = xw.tocsc()
    colsum = np.asarray(xw.sum(axis=0)).ravel()
    bg = (colsum + 1.0) / (colsum.sum() + col_mask.sum())
    out: dict[int, list[dict[int, float]]] = {}
    memb_t = memb.T.tocsr()  # type -> class shares
    for sid in range(len(inv.corpus)):
        n = len(inv.corpus.token_ids[sid])
        entries: list[dict[int, float]] = [dict() for _ in range(n)]
        any_pos = False
        for i in range(n):
            w = int(inv.corpus.token_ids[sid][i])
            t = tok_type.get(w)
            if t is None:
                continue
            oid = inv.lookup(sid, i, i + 1)
            if oid < 0:
                continue
            ctx_ids = cd.occ_ctx[oid][slot_mask]
            sub = cmat_csc[:, ctx_ids].toarray()  # (classes, slots)
            x_row = xw_csc[t, ctx_ids].toarray().ravel() if False else np.asarray(xw_csc[:, ctx_ids][t].todense()).ravel()
            totals = class_total.copy()
            a, b = memb_t.indptr[t], memb_t.indptr[t + 1]
            x_tot = float(xw[t].sum())
            for k, share in zip(memb_t.indices[a:b], memb_t.data[a:b]):
                sub[k] = np.maximum(sub[k] - share * x_row, 0.0)
                totals[k] = max(totals[k] - share * x_tot, 0.0)
            lp = np.log((sub + eta * bg[None, ctx_ids]) / (totals[:, None] + eta))
            score = lp.sum(axis=1) / T
            score = score - score.max()
            lam = score - np.log(np.exp(score).sum())
            order = np.argsort(-lam)[:top_k]
            entries[i] = {classes[k]: float(max(lam[k], floor)) for k in order}
            any_pos = True
        if any_pos:
            out[sid] = entries
    return out


# --------------------------------------------------------------------------- punctuation prior
def punctuation_potentials(corpus: Corpus, cfg: dict[str, Any]) -> dict[int, np.ndarray]:
    """``log psi(i, j)`` span potentials from punctuation (mode ``structural_prior``).

    * sentence-final ``. ? !`` at position ``n-1``: the body ``[0, n-1)`` and the
      span ``[0, n)`` get ``+w_final``; a sentence-final mark is a strong root
      boundary, so spans that end at ``n-1`` without starting at 0 are not
      penalised (they are inside the body).
    * paired ``( ) [ ] " "``: the inside ``[a+1, b)`` and the enclosing ``[a, b+1)``
      get ``+w_paired``.
    * internal ``, ; :`` at position ``p``: spans that strictly contain the mark
      (``i < p`` and ``p + 1 < j``) get ``-w_comma`` per contained mark; the
      whole sentence and the sentence body are exempt.

    Potentials are normalised to ``<= 0`` per sentence (virtual evidence must
    be a probability), so bonuses become penalties on the other spans.
    """
    w_final = float(cfg.get("sentence_final_weight", 2.0))
    w_pair = float(cfg.get("paired_weight", 2.0))
    w_comma = float(cfg.get("comma_weight", 1.0))
    out: dict[int, np.ndarray] = {}
    for sid, sent in enumerate(corpus.sentences):
        toks = sent.tokens
        n = len(toks)
        if n < 2:
            continue
        phi = np.zeros((n + 1, n + 1))
        touched = False
        finals = [p for p, t in enumerate(toks) if t in SENTENCE_FINAL]
        body_end = n
        if toks[-1] in SENTENCE_FINAL and n >= 2:
            body_end = n - 1
            phi[0, n - 1] += w_final
            phi[0, n] += w_final
            touched = True
        stack: list[tuple[str, int]] = []
        for p, t in enumerate(toks):
            if t == '"':
                if stack and stack[-1][0] == '"':
                    a = stack.pop()[1]
                    if p - a >= 2:
                        phi[a + 1, p] += w_pair
                        phi[a, p + 1] += w_pair
                        touched = True
                else:
                    stack.append(('"', p))
            elif t in PAIRED:
                stack.append((t, p))
            elif t in (")", "]"):
                for q in range(len(stack) - 1, -1, -1):
                    if PAIRED.get(stack[q][0]) == t:
                        a = stack[q][1]
                        del stack[q:]
                        if p - a >= 2:
                            phi[a + 1, p] += w_pair
                            phi[a, p + 1] += w_pair
                            touched = True
                        break
        for p, t in enumerate(toks):
            if t in INTERNAL and 0 < p < n - 1:
                touched = True
                for i in range(0, p):
                    for j in range(p + 2, n + 1):
                        if (i == 0 and j in (n, body_end)):
                            continue
                        phi[i, j] -= w_comma
        if not touched:
            continue
        m = phi.max()
        if m > 0:
            phi -= m  # virtual evidence <= 1
        out[sid] = phi
    return out


# --------------------------------------------------------------------------- posterior decoding
@dataclass
class PosteriorDecoding:
    """Final posterior views of the corpus under the model."""

    occ_class: np.ndarray  # inventory occurrence -> argmax class (-1 if P(Q) < threshold)
    occ_const: np.ndarray  # inventory occurrence -> P(Q_ij = 1 | s)
    type_classes: dict[int, list[int]]  # span type -> [dominant class] or [c1, c2] when ambiguous
    type_mixture: dict[int, dict[int, float]]  # span type -> class -> posterior share (inventory types)
    constituent_positions: set[tuple[int, int, int]]
    viterbi_positions: set[tuple[int, int, int]]
    relations: list[CompositionRelation]
    ambiguous_types: dict[int, dict[str, Any]]
    class_mass: dict[int, float]
    class_members: dict[int, list[int]]  # class -> span types (by dominant class)
    fully_parsed: float
    mean_chunks: float
    unknown_tokens: int
    span_entropy: list[tuple[int, float, int]]  # (tid, entropy, n_occ) highest first
    sentence_log_z: dict[int, float] = field(default_factory=dict)


def decode_posteriors(
    vi: VariationalInference,
    inv: SpanInventory,
    out: EStepOutput,
    ec: ExpectedCounts,
    cfg: dict[str, Any],
    min_support: float = 2.0,
    min_lexical: int = 2,
    viterbi_positions: bool = True,
) -> PosteriorDecoding:
    model = vi.model
    thr = float(cfg.get("constituent_threshold", 0.5))
    amb_thr = float(cfg.get("ambiguity_mass", 0.2))
    amb_min_occ = int(cfg.get("ambiguity_min_occurrences", 5))
    amb_max_len = int(cfg.get("ambiguity_max_span_length", 1))  # occurrence ambiguity is reported for lexical types; longer spans keep their dominant class
    n_occ = inv.n_occurrences
    occ_class = np.full(n_occ, -1, dtype=np.int64)
    occ_const = np.zeros(n_occ)
    occ_best = np.zeros(n_occ)
    positions: set[tuple[int, int, int]] = set()
    const_mass: dict[tuple[int, int, int], float] = {}
    type_mass: dict[int, dict[int, float]] = {}
    for sid, nodes in out.node_posteriors.items():
        agg: dict[tuple[int, int], float] = {}
        for i, j, c, p in nodes:
            agg[(i, j)] = agg.get((i, j), 0.0) + p
            oid = inv.lookup(sid, i, j)
            if oid >= 0:
                if p > occ_best[oid]:
                    occ_best[oid] = p
                    occ_class[oid] = c
                t = int(inv.occ_tid[oid])
                d = type_mass.setdefault(t, {})
                d[c] = d.get(c, 0.0) + p
        for (i, j), p in agg.items():
            if p >= thr:
                positions.add((sid, i, j))
            oid = inv.lookup(sid, i, j)
            if oid >= 0:
                occ_const[oid] = p
    occ_class = np.where(occ_const >= thr, occ_class, -1)
    type_classes: dict[int, list[int]] = {}
    type_mixture: dict[int, dict[int, float]] = {}
    ambiguous: dict[int, dict[str, Any]] = {}
    members: dict[int, list[int]] = {}
    entropies: list[tuple[int, float, int]] = []
    for t, d in type_mass.items():
        tot = sum(d.values())
        n_t = int(inv.type_freq[t])
        if tot / max(1, n_t) < thr - 1e-6:
            continue  # the type is mostly not a constituent (inclusive at 1/2 like the gold criterion)
        mix = {c: v / tot for c, v in d.items()}
        type_mixture[t] = mix
        order = sorted(mix.items(), key=lambda kv: -kv[1])
        dom = order[0][0]
        cls = [dom]
        p = np.array([v for _, v in order])
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        entropies.append((t, ent, n_t))
        if len(order) >= 2 and order[1][1] >= amb_thr and n_t >= amb_min_occ and inv.type_length(t) <= amb_max_len:
            cls = [dom, order[1][0]]
            ambiguous[t] = {"span_type": inv.type_str(t), "n_occurrences": n_t, "mixture": {model.names.get(c, str(c)): round(v, 4) for c, v in order[:4]}, "entropy": round(ent, 4)}
        type_classes[t] = cls
        members.setdefault(dom, []).append(t)
    entropies.sort(key=lambda x: -x[1])
    # relations: active rules with enough expected support
    rels: list[CompositionRelation] = []
    for (c, a, b), n in model.rules.items():
        if n < min_support:
            continue
        real = model.realisations.get((c, a, b), {})
        lexical = len(real)
        if lexical < min_lexical:
            continue
        examples = []
        for t, v in sorted(real.items(), key=lambda kv: -kv[1])[:3]:
            examples.append((inv.type_str(t), "", ""))
        rels.append(CompositionRelation(a, b, c, int(round(n)), lexical, model.rule_conditional((c, a, b)), examples, lexical, lexical))
    rels.sort(key=lambda r: (-r.support, r.key))
    vit: set[tuple[int, int, int]] = set()
    n_single_chunk = 0
    if viterbi_positions:
        g = model.compile()
        if vi.backend == "numba":
            from . import chart_numba

            vocab_size = int(max((int(s.max()) if len(s) else 0) for s in vi.sentences) + 1)
            arrs = chart_numba.CompiledArrays(g, vocab_size)
            parse = lambda sid: chart_numba.viterbi_parse(vi, g, sid, arrs)
        else:
            parse = lambda sid: vi.viterbi_parse(sid, g)
        for sid in range(len(vi.sentences)):
            d = parse(sid)
            if d is None:
                continue
            for i, j, c, k, r in d.nodes:
                vit.add((sid, i, j))
            if len(d.chunks) == 1:
                n_single_chunk += 1
    class_mass = {c: 0.0 for c in model.classes}
    for (c, w), n in model.lex.items():
        class_mass[c] = class_mass.get(c, 0.0) + n
    for (c, a, b), n in model.rules.items():
        class_mass[c] = class_mass.get(c, 0.0) + n
    n_sent = max(1.0, ec.n_sentences)
    unknown = int(round(sum(n for (c, w), n in model.lex.items() if w == UNK)))
    # fraction of sentences whose MAP derivation is a single root chunk (posterior mean chunks reported separately)
    fully = n_single_chunk / max(1, len(vi.sentences)) if viterbi_positions else float("nan")
    return PosteriorDecoding(occ_class, occ_const, type_classes, type_mixture, positions, vit, rels, ambiguous, class_mass, members, fully, ec.n_chunks / n_sent, unknown, entropies, dict(out.log_z))


# --------------------------------------------------------------------------- outputs
def write_trace(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def write_moves(moves: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for m in moves:
            fh.write(json.dumps(m, default=str) + "\n")
