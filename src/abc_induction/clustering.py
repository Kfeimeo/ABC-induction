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
        self.block_budget = float(cfg.get("pair_block_budget", 1.5e7))  # candidate-pair entries materialised at once
        self.block_rows = int(cfg.get("pair_block_rows", 4096))
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

    def _bhattacharyya_pairs(self, c: sp.csr_matrix, selfsim: np.ndarray, n: int, threshold: float) -> list[tuple[float, int, int, int, float]]:
        """Vectorised multi-scale Bhattacharyya scores of all candidate pairs, computed in row blocks.

        Candidates are the row pairs that share a kept (df <= max_df) context at any scale, plus
        capped pairs from the stop-contexts.  The sparse products ``sqrt(p) sqrt(p)^T`` are never
        materialised for all rows at once: rows are processed in blocks whose estimated number of
        candidate partners stays under ``pair_block_budget`` entries, each block's pairs are
        scored and filtered, and only the pairs above the thresholds survive.  The scores are the
        same as the all-at-once computation (same per-pair sums in the same order); only the peak
        memory changes (73k types / 3M contexts: ~14 GB before, < 1 GB now).
        """
        ps: list[sp.csr_matrix] = []  # sqrt-probabilities on the kept (low-df) columns, per scale
        bs: list[sp.csr_matrix] = []  # their binary structure
        high_sqrt: list[np.ndarray] = []  # dense (n x n_high_k) sqrt-probabilities of the stop-contexts
        extra_i: list[np.ndarray] = []
        extra_j: list[np.ndarray] = []
        for ki, cols in enumerate(self.scale_cols):
            allc = self.all_cols[ki]
            p_all = normalize_rows(sp.csr_matrix(c[:, allc]))  # distribution over all informative contexts of scale k
            pos = {int(col): i for i, col in enumerate(allc)}
            low_idx = np.array([pos[int(col)] for col in cols], dtype=np.int64)
            high_idx = np.array([pos[int(col)] for col in self.high_cols[ki] if int(col) in pos], dtype=np.int64)
            p = sp.csr_matrix(p_all[:, low_idx]) if len(low_idx) else sp.csr_matrix((n, 0))
            p.sort_indices()
            b = p.copy()
            b.data = np.ones_like(b.data)
            p.data = np.sqrt(p.data)
            ps.append(p)
            bs.append(b)
            h = p_all[:, high_idx].toarray() if len(high_idx) else np.zeros((n, 0))
            high_sqrt.append(np.sqrt(h))
            # capped candidate generation from stop-contexts: top rows per column, as index arrays
            for col in range(h.shape[1]):
                rows = np.nonzero(h[:, col] > 0)[0]
                if len(rows) > self.high_top:
                    rows = rows[np.argsort(-h[rows, col])[: self.high_top]]
                rows = np.sort(rows)
                if len(rows) >= 2:
                    a_, b_ = np.triu_indices(len(rows), k=1)
                    extra_i.append(rows[a_])
                    extra_j.append(rows[b_])
        if extra_i:
            key = np.unique(np.concatenate(extra_i).astype(np.int64) * n + np.concatenate(extra_j).astype(np.int64))
            ex_i, ex_j = key // n, key % n
        else:
            ex_i = ex_j = np.zeros(0, dtype=np.int64)
        ex_start = np.searchsorted(ex_i, np.arange(n + 1))  # ex_i is sorted: extra pairs of row i are ex_start[i]:ex_start[i+1]
        pts = [p.T.tocsr() for p in ps]
        bts = [b.T.tocsr() for b in bs]
        # estimated candidate partners per row = sum over its kept contexts of their document frequency
        est = np.zeros(n)
        for b in bs:
            if b.shape[1]:
                df = np.asarray(b.sum(axis=0)).ravel()
                est += np.asarray(b @ df).ravel()
        budget = float(self.block_budget)
        sq = np.sqrt(selfsim)
        wsum = float(self.weights.sum())
        out: list[tuple[float, int, int, int, float]] = []
        start = 0
        while start < n:
            stop = start + 1
            acc = est[start]
            while stop < n and acc + est[stop] <= budget and stop - start < self.block_rows:
                acc += est[stop]
                stop += 1
            rows = np.arange(start, stop)
            num_blk = None
            sh_blk = None
            for ki in range(len(ps)):
                if ps[ki].shape[1] == 0:
                    continue
                prod = sp.csr_matrix(ps[ki][rows] @ pts[ki])
                sprod = sp.csr_matrix(bs[ki][rows] @ bts[ki])
                num_blk = self.weights[ki] * prod if num_blk is None else num_blk + self.weights[ki] * prod
                sh_blk = sprod if sh_blk is None else sh_blk + sprod
            if num_blk is None:
                num_blk = sp.csr_matrix((len(rows), n))
                sh_blk = sp.csr_matrix((len(rows), n))
            coo = num_blk.tocoo()
            gi = coo.row.astype(np.int64) + start
            gj = coo.col.astype(np.int64)
            keep = gj > gi
            pi, pj = gi[keep], gj[keep]
            ea, eb = ex_start[start], ex_start[stop]
            if eb > ea:
                key = np.unique(np.concatenate([pi * n + pj, ex_i[ea:eb] * n + ex_j[ea:eb]]))
                pi, pj = key // n, key % n
            if len(pi) == 0:
                start = stop
                continue
            li = pi - start
            v = np.asarray(num_blk[li, pj]).ravel()
            sh = np.asarray(sh_blk[li, pj]).ravel()
            for ki, hs in enumerate(high_sqrt):
                if hs.shape[1]:
                    v = v + self.weights[ki] * (hs[pi] * hs[pj]).sum(axis=1)
                    sh = sh + ((hs[pi] > 0) & (hs[pj] > 0)).sum(axis=1)
            den = (self.weights[None, :] * sq[pi] * sq[pj]).sum(axis=1)
            raw = v / wsum
            score = np.minimum(1.0, np.where(den > 0, v / np.maximum(den, 1e-12), 0.0))
            ok = (score >= threshold) & (raw >= self.min_raw) & (sh >= self.min_shared)
            for i in np.nonzero(ok)[0]:
                out.append((float(score[i]), int(pi[i]), int(pj[i]), int(sh[i]), float(raw[i])))
            start = stop
        return out

    def pair_similarities(self, state: ClassState, ids: list[int], threshold: float) -> tuple[list[tuple[float, int, int, int, float]], sp.csr_matrix]:
        """Return ``(score, row_a, row_b, shared, raw)`` for pairs above the threshold,
        plus the aggregated class matrix ``c`` (rows aligned with ``ids``)."""
        groups = [state.members[i] for i in ids]
        c = aggregate_rows(self.x, groups)
        selfsim = self._self_similarity(groups) if self.correction else np.ones((len(ids), len(self.scales)))
        n = len(ids)
        if self.sim_name == "bhattacharyya":
            return self._bhattacharyya_pairs(c, selfsim, n, threshold), c
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
        # explicit, reproducible order: score, raw similarity, shared contexts, then row indices.
        # (ties are common - scores saturate at 1.0 - and used to be broken by the iteration order of a
        # Python set of candidate pairs, i.e. by hash order)
        pairs.sort(key=lambda t: (-t[0], -t[4], -t[3], t[1], t[2]))
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
