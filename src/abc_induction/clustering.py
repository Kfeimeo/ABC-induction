"""Provisional equivalence-class induction by agglomerative merging.

Every span type starts as a singleton class.  In each round the similarity
between all candidate class pairs (pairs sharing at least one informative
context, generated through the inverted index) is computed on the aggregated
sparse context counts, and the most confident non-conflicting pairs above the
threshold are merged.  All merges are recorded as :class:`MergeEvent` with the
shared contexts that justified them, so that every class has a lineage.

Similarities (per scale k): weighted Jaccard (Baseline A), 1 - Jensen-Shannon
(Baseline B) and the Bhattacharyya coefficient (vectorised, default).

Multi-scale combination with sparsity (attenuation) correction
--------------------------------------------------------------
Large-scale contexts are nearly unique in finite corpora, so a plain mixture
over scales drags every similarity down.  We therefore compare the observed
cross-scale similarity with the similarity that two *samples of the same
distribution* would reach given the observed sparsity, estimated by a random
split-half self-similarity ``self_k(x)``:

    sim(x, y) = sum_k w_k sim_k(x, y) / sum_k w_k sqrt(self_k(x) self_k(y))

Scales at which a class is too sparse to be similar even to itself contribute
little to numerator and denominator alike; dense scales dominate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .contexts import ContextData
from .sparse_index import (
    SIMILARITIES,
    aggregate_rows,
    candidate_pairs,
    document_frequency,
    normalize_rows,
    row_arrays,
)
from .spans import SpanInventory

log = logging.getLogger(__name__)


@dataclass
class MergeEvent:
    """One merge (or attach) step with the evidence that justified it."""

    step: int
    class_a: int
    class_b: int
    new_class: int
    score: float
    kind: str
    size_a: int
    size_b: int
    members_a: list[str]
    members_b: list[str]
    evidence_contexts: list[tuple[str, float, float]]  # (context, P_a(c), P_b(c))
    shared_contexts: int = 0
    raw_similarity: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "class_a": self.class_a,
            "class_b": self.class_b,
            "new_class": self.new_class,
            "score": round(self.score, 4),
            "raw_similarity": round(self.raw_similarity, 4),
            "size_a": self.size_a,
            "size_b": self.size_b,
            "members_a": self.members_a,
            "members_b": self.members_b,
            "shared_contexts": self.shared_contexts,
            "why_merged": [
                {"context": c, "p_a": round(pa, 4), "p_b": round(pb, 4)} for c, pa, pb in self.evidence_contexts
            ],
        }


@dataclass
class ClassState:
    """A partition of span types into classes with merge lineage.

    Class ids are never reused: merging ``a`` and ``b`` creates a fresh id so
    that cached pair similarities for untouched classes stay valid and the
    lineage can be replayed.
    """

    n_types: int
    members: dict[int, list[int]] = field(default_factory=dict)
    type_class: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    parent: dict[int, tuple[int, int]] = field(default_factory=dict)
    history: list[MergeEvent] = field(default_factory=list)
    next_id: int = 0

    def __post_init__(self) -> None:
        if not self.members:
            self.members = {t: [t] for t in range(self.n_types)}
            self.type_class = np.arange(self.n_types, dtype=np.int64)
            self.next_id = self.n_types

    def active_ids(self) -> list[int]:
        return sorted(self.members)

    def merge(self, a: int, b: int) -> int:
        new = self.next_id
        self.next_id += 1
        self.members[new] = self.members.pop(a) + self.members.pop(b)
        self.type_class[self.members[new]] = new
        self.parent[new] = (a, b)
        return new

    def class_matrix(self, x: sp.csr_matrix) -> tuple[list[int], sp.csr_matrix]:
        ids = self.active_ids()
        return ids, aggregate_rows(x, [self.members[i] for i in ids])

    def lineage(self, cid: int) -> list[int]:
        """All ancestor class ids (depth-first)."""
        out = []
        stack = [cid]
        while stack:
            c = stack.pop()
            out.append(c)
            if c in self.parent:
                stack.extend(self.parent[c])
        return out


# ------------------------------------------------------------------ similarity engine
class SimilarityEngine:
    """Multi-scale, sparsity-corrected pair similarities over class matrices."""

    def __init__(self, cd: ContextData, inv: SpanInventory, cfg: dict[str, Any], seed: int = 0) -> None:
        self.cd = cd
        self.inv = inv
        self.sim_name = cfg.get("similarity", "bhattacharyya")
        self.max_df = int(cfg.get("max_context_df", 1000))
        self.min_shared = int(cfg.get("min_shared_contexts", 2))
        self.correction = bool(cfg.get("sparsity_correction", True))
        self.self_floor = float(cfg.get("self_floor", 0.05))
        self.min_raw = float(cfg.get("min_raw_similarity", 0.0))
        scales = sorted(set(int(s) for s in cd.slot_scale.tolist()))
        sw = cfg.get("scale_weights") or {}
        self.scales = scales
        self.weights = np.array([float(sw.get(k, sw.get(str(k), 1.0))) for k in scales])
        sided_w = float(cfg.get("sided_weight", 1.0))
        colw = np.ones(cd.n_contexts)
        if sided_w != 1.0:
            colw[cd.ctx_kind != 0] = sided_w
        x = sp.csr_matrix(cd.type_ctx.astype(np.float64) @ sp.diags(colw))
        x.eliminate_zeros()
        self.x = x
        # random split halves at occurrence level (for the self-similarity)
        rng = np.random.default_rng(seed)
        half = rng.integers(0, 2, size=inv.n_occurrences)
        n_slots = cd.occ_ctx.shape[1]
        self.x_half = []
        for h in (0, 1):
            oids = np.nonzero(half == h)[0]
            rows = np.repeat(inv.occ_tid[oids].astype(np.int64), n_slots)
            cols = cd.occ_ctx[oids].reshape(-1)
            m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(inv.n_types, cd.n_contexts))
            m.sum_duplicates()
            self.x_half.append(sp.csr_matrix(m @ sp.diags(colw)))
        df = document_frequency(x)
        self.informative = df >= 2
        self.keep_cols = (df <= self.max_df) & self.informative  # sparse candidate generation
        high = np.nonzero(df > self.max_df)[0]
        max_high = int(cfg.get("max_high_df_columns", 256))
        if len(high) > max_high:  # keep the most frequent stop-contexts only
            high = high[np.argsort(-df[high])[:max_high]]
        self.high_cols = [high[cd.ctx_scale[high] == k] for k in scales]
        self.high_top = int(cfg.get("high_df_candidates", 200))
        self.scale_cols = [np.nonzero((cd.ctx_scale == k) & self.keep_cols)[0] for k in scales]
        self.all_cols = [np.nonzero((cd.ctx_scale == k) & self.informative)[0] for k in scales]
        self.cache: dict[tuple[int, int], tuple[float, float]] = {}

    # ------------------------------------------------------------ helpers
    def _self_similarity(self, groups: list[list[int]]) -> np.ndarray:
        """``self_k(class)`` for every class and scale -> array (n_classes, n_scales)."""
        h1 = aggregate_rows(self.x_half[0], groups)
        h2 = aggregate_rows(self.x_half[1], groups)
        out = np.zeros((len(groups), len(self.scales)))
        for ki, cols in enumerate(self.all_cols):
            a = normalize_rows(sp.csr_matrix(h1[:, cols]))
            b = normalize_rows(sp.csr_matrix(h2[:, cols]))
            a.data = np.sqrt(a.data)
            b.data = np.sqrt(b.data)
            out[:, ki] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        return np.maximum(out, self.self_floor)

    def pair_similarities(self, state: ClassState, ids: list[int], threshold: float) -> tuple[list[tuple[float, int, int, int, float]], sp.csr_matrix]:
        """Return ``(score, row_a, row_b, shared, raw)`` for pairs above the threshold,
        plus the aggregated class matrix ``c`` (rows aligned with ``ids``)."""
        groups = [state.members[i] for i in ids]
        c = aggregate_rows(self.x, groups)
        selfsim = self._self_similarity(groups) if self.correction else np.ones((len(ids), len(self.scales)))
        n = len(ids)
        if self.sim_name == "bhattacharyya":
            num = sp.csr_matrix((n, n))
            shared = sp.csr_matrix((n, n))
            high_sqrt = []  # dense (n x n_high_k) sqrt-probabilities for stop-contexts
            extra_pairs: set[tuple[int, int]] = set()
            for ki, cols in enumerate(self.scale_cols):
                allc = self.all_cols[ki]
                p_all = normalize_rows(sp.csr_matrix(c[:, allc]))  # distribution over all informative contexts of scale k
                # map back: low (sparse product) part and high (dense) part
                pos = {int(col): i for i, col in enumerate(allc)}
                low_idx = np.array([pos[int(col)] for col in cols], dtype=np.int64)
                high_idx = np.array([pos[int(col)] for col in self.high_cols[ki] if int(col) in pos], dtype=np.int64)
                p = sp.csr_matrix(p_all[:, low_idx]) if len(low_idx) else sp.csr_matrix((n, 0))
                b = p.copy()
                b.data = np.ones_like(b.data)
                p.data = np.sqrt(p.data)
                num = num + self.weights[ki] * sp.triu(p @ p.T, k=1)
                shared = shared + sp.triu(b @ b.T, k=1)
                h = p_all[:, high_idx].toarray() if len(high_idx) else np.zeros((n, 0))
                high_sqrt.append(np.sqrt(h))
                # capped candidate generation from stop-contexts: top rows per column
                for col in range(h.shape[1]):
                    rows = np.nonzero(h[:, col] > 0)[0]
                    if len(rows) > self.high_top:
                        rows = rows[np.argsort(-h[rows, col])[: self.high_top]]
                    rows = np.sort(rows)
                    for i_ in range(len(rows)):
                        for j_ in range(i_ + 1, len(rows)):
                            extra_pairs.add((int(rows[i_]), int(rows[j_])))
            num = num.tocsr()
            shared = shared.tocsr()
            coo = num.tocoo()
            pairs = set(zip(coo.row.tolist(), coo.col.tolist())) | extra_pairs
            if not pairs:
                return [], c
            ri = np.array([a for a, _ in pairs], dtype=np.int64)
            rj = np.array([b for _, b in pairs], dtype=np.int64)
            v = np.asarray(num[ri, rj]).ravel()
            sh = np.asarray(shared[ri, rj]).ravel()
            for ki, hs in enumerate(high_sqrt):
                if hs.shape[1]:
                    v = v + self.weights[ki] * (hs[ri] * hs[rj]).sum(axis=1)
                    sh = sh + ((hs[ri] > 0) & (hs[rj] > 0)).sum(axis=1)
            sq = np.sqrt(selfsim)
            den = (self.weights[None, :] * sq[ri] * sq[rj]).sum(axis=1)
            raw = v / float(self.weights.sum())
            score = np.minimum(1.0, np.where(den > 0, v / np.maximum(den, 1e-12), 0.0))
            ok = (score >= threshold) & (raw >= self.min_raw) & (sh >= self.min_shared)
            return [(float(score[i]), int(ri[i]), int(rj[i]), int(sh[i]), float(raw[i])) for i in np.nonzero(ok)[0]], c
        fn = SIMILARITIES[self.sim_name]
        ri, rj, sh_all = candidate_pairs(c, self.max_df, self.min_shared)
        per_scale = [sp.csr_matrix(c[:, cols]) for cols in self.all_cols]
        for m in per_scale:
            m.sort_indices()
        sq = np.sqrt(selfsim)
        out = []
        for r, cc, sh in zip(ri.tolist(), rj.tolist(), sh_all.tolist()):
            key = (ids[r], ids[cc])
            cached = self.cache.get(key)
            if cached is None:
                num = 0.0
                for ki, m in enumerate(per_scale):
                    ia, da = row_arrays(m, r)
                    ib, db = row_arrays(m, cc)
                    if len(ia) == 0 or len(ib) == 0:
                        continue
                    num += self.weights[ki] * fn(ia, da, ib, db)
                den = float((self.weights * sq[r] * sq[cc]).sum())
                raw = num / float(self.weights.sum())
                score = min(1.0, num / den) if den > 0 else 0.0
                cached = (score, raw)
                self.cache[key] = cached
            score, raw = cached
            if score >= threshold and raw >= self.min_raw:
                out.append((score, r, cc, int(sh), raw))
        return out, c


def _top_shared(p: sp.csr_matrix, ra: int, rb: int, k: int, cd: ContextData, corpus) -> tuple[list[tuple[str, float, float]], int]:
    ia, da = row_arrays(p, ra)
    ib, db = row_arrays(p, rb)
    common, ca, cb = np.intersect1d(ia, ib, return_indices=True)
    if len(common) == 0:
        return [], 0
    score = np.minimum(da[ca], db[cb])
    order = np.argsort(-score)[:k]
    return [
        (cd.vocab.render(int(common[o]), corpus), float(da[ca][o]), float(db[cb][o])) for o in order
    ], int(len(common))


def agglomerate(
    cd: ContextData,
    inv: SpanInventory,
    cfg: dict[str, Any],
    state: ClassState | None = None,
    seed: int = 0,
) -> ClassState:
    """Run batched agglomerative merging until no pair exceeds the threshold.

    Config keys (``cfg``): ``similarity``, ``merge_threshold``,
    ``min_shared_contexts``, ``max_context_df``, ``scale_weights``,
    ``sparsity_correction``, ``self_floor``, ``min_raw_similarity``,
    ``max_rounds``, ``evidence_contexts`` and ``attach``.
    """
    threshold = float(cfg.get("merge_threshold", 0.7))
    max_rounds = int(cfg.get("max_rounds", 200))
    n_evidence = int(cfg.get("evidence_contexts", 8))
    corpus = inv.corpus
    engine = SimilarityEngine(cd, inv, cfg, seed)
    state = state or ClassState(inv.n_types)
    step = len(state.history)
    for rnd in range(max_rounds):
        ids = state.active_ids()
        pairs, c = engine.pair_similarities(state, ids, threshold)
        pairs.sort(key=lambda t: (-t[0], -t[4]))
        used: set[int] = set()
        p = normalize_rows(c)
        n_merged = 0
        for score, ra, rb, sh, raw in pairs:
            a, b = ids[ra], ids[rb]
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            evidence, n_sh = _top_shared(p, ra, rb, n_evidence, cd, corpus)
            ma = [inv.type_str(t) for t in state.members[a][:6]]
            mb = [inv.type_str(t) for t in state.members[b][:6]]
            sa, sb = len(state.members[a]), len(state.members[b])
            new = state.merge(a, b)
            state.history.append(MergeEvent(step, a, b, new, float(score), "merge", sa, sb, ma, mb, evidence, n_sh, raw))
            step += 1
            n_merged += 1
        log.info("merge round %d: %d candidate pairs, %d merges, %d classes", rnd, len(pairs), n_merged, len(state.members))
        if n_merged == 0:
            break
    attach_cfg = cfg.get("attach") or {}
    if attach_cfg.get("enabled", False):
        _attach_singletons(cd, inv, state, engine, attach_cfg, n_evidence, step)
    return state


def _attach_singletons(
    cd: ContextData,
    inv: SpanInventory,
    state: ClassState,
    engine: SimilarityEngine,
    cfg: dict[str, Any],
    n_evidence: int,
    step: int,
) -> None:
    """Attach rare singleton types to the class covering their context mass.

    For a singleton ``x`` and a class ``K`` (with >= 2 members) coverage is
    ``sum_c P_x(c) [N_Kc > 0]`` over informative contexts.  ``x`` is attached
    to the best class when coverage >= ``threshold`` and the margin to the
    second best class is >= ``margin``.
    """
    thr = float(cfg.get("threshold", 0.8))
    margin = float(cfg.get("margin", 0.2))
    min_occ = int(cfg.get("min_occurrences", 2))
    corpus = inv.corpus
    for it in range(int(cfg.get("iterations", 2))):
        ids, c = state.class_matrix(engine.x)
        p = normalize_rows(c)
        q = sp.csr_matrix(p @ sp.diags(engine.informative.astype(np.float64)))
        q.eliminate_zeros()
        sizes = np.array([len(state.members[i]) for i in ids])
        big_rows = np.nonzero(sizes >= 2)[0]
        single_rows = [r for r in range(len(ids)) if sizes[r] == 1 and inv.type_freq[state.members[ids[r]][0]] >= min_occ]
        if len(big_rows) == 0 or not single_rows:
            return
        b = q[big_rows].copy()
        b.data = np.ones_like(b.data)
        cov = sp.csr_matrix(q[single_rows] @ b.T)
        n_attached = 0
        used: set[int] = set()
        for k, r in enumerate(single_rows):
            row = cov.getrow(k).toarray().ravel()
            if row.size == 0:
                continue
            order = np.argsort(-row)
            best = float(row[order[0]])
            second = float(row[order[1]]) if len(order) > 1 else 0.0
            if best < thr or best - second < margin:
                continue
            target = ids[int(big_rows[order[0]])]
            single = ids[r]
            if target in used:
                continue
            used.add(target)
            evidence, n_sh = _top_shared(p, r, int(big_rows[order[0]]), n_evidence, cd, corpus)
            ma = [inv.type_str(t) for t in state.members[single][:6]]
            mb = [inv.type_str(t) for t in state.members[target][:6]]
            sb = len(state.members[target])
            new = state.merge(single, target)
            state.history.append(MergeEvent(step, single, target, new, best, "attach", 1, sb, ma, mb, evidence, n_sh, best))
            step += 1
            n_attached += 1
        log.info("attach pass %d: %d singletons attached", it, n_attached)
        if n_attached == 0:
            return
