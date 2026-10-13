"""Occurrence-level category ambiguity discovery (at most 2 senses per type).

For a span type ``x`` with occurrences ``O(x)`` we compare

    H1 : z_i = K1              (one latent class explains every occurrence)
    H2 : z_i in {A, B}         (two latent classes, both used by *other*
                                expressions)

A split proposal assigns every occurrence to the accepted class whose context
profile (excluding ``x`` itself) gives it the highest likelihood (``guided``),
or clusters the occurrences with 2-means and aligns each cluster to a class
(``kmeans``).  A split is accepted only if

1. the per-occurrence context log-likelihood gain over H1 is positive
   (within-cluster context consistency improves),
2. each cluster aligns to a class used by other expressions (coverage of the
   cluster's context mass by the class support >= ``min_alignment``),
3. the description length improves after the split penalty:
   ``Delta_split = DL(H1) - DL(H2) - lambda_split > 0`` (composition
   consistency is part of DL because DL is computed from Viterbi derivations),
4. the split also beats the current assignment.

New branches that would be used only by ``x`` itself are impossible by
construction (candidate classes need >= ``min_other_members`` members), and
topic-like context diversity is not rewarded because both clusters must be
explained by *existing* syntactic classes and must shorten the grammar.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .clustering import ClassState
from .constituency import ClassReport
from .contexts import ContextData
from .mdl import MDLModel, SentenceParse
from .sparse_index import aggregate_rows
from .spans import SpanInventory

log = logging.getLogger(__name__)


@dataclass
class SplitReport:
    tid: int
    span: str
    n_occ: int
    current_class: int
    single_class: int
    class_a: int
    class_b: int
    n_a: int
    n_b: int
    ll_gain_per_occ: float
    mi_gain_bits: float
    align_a: float
    align_b: float
    dl_current: float
    dl_single: float
    dl_double: float
    split_gain: float
    lambda_split: float
    accepted: bool
    reason: str
    contexts_a: list[str] = field(default_factory=list)
    contexts_b: list[str] = field(default_factory=list)
    composition_a: list[str] = field(default_factory=list)
    composition_b: list[str] = field(default_factory=list)
    kind: str = "split"

    def to_json(self) -> dict[str, Any]:
        return {
            "span_type": self.span,
            "kind": self.kind,
            "accepted": self.accepted,
            "reason": self.reason,
            "n_occurrences": self.n_occ,
            "current_class": self.current_class,
            "single_class": self.single_class,
            "sense_a": {"class": self.class_a, "occurrences": self.n_a, "alignment": round(self.align_a, 3), "supporting_contexts": self.contexts_a, "composition": self.composition_a},
            "sense_b": {"class": self.class_b, "occurrences": self.n_b, "alignment": round(self.align_b, 3), "supporting_contexts": self.contexts_b, "composition": self.composition_b},
            "context_ll_gain_per_occurrence": round(self.ll_gain_per_occ, 4),
            "context_mi_gain_bits": round(self.mi_gain_bits, 4),
            "dl_current": round(self.dl_current, 2),
            "dl_single": round(self.dl_single, 2),
            "dl_double": round(self.dl_double, 2),
            "split_gain": round(self.split_gain, 2),
            "lambda_split": self.lambda_split,
        }


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class AmbiguityDetector:
    """Runs the split test for every eligible span type and applies accepted splits."""

    def __init__(
        self,
        inv: SpanInventory,
        cd: ContextData,
        state: ClassState,
        reports: dict[int, ClassReport],
        accepted: set[int],
        occ_class: np.ndarray,
        model: MDLModel | None,
        cfg: dict[str, Any],
    ) -> None:
        self.inv = inv
        self.cd = cd
        self.state = state
        self.reports = reports
        self.accepted = accepted
        self.occ_class = occ_class
        self.model = model
        self.cfg = cfg
        self.max_scale = int(cfg.get("max_scale", 2))
        self.eta = float(cfg.get("backoff_eta", 1.0))
        self.lambda_split = float(cfg.get("lambda_split", 20.0))
        self.min_occ = int(cfg.get("min_occurrences", 8))
        self.min_cluster_occ = int(cfg.get("min_cluster_occurrences", 3))
        self.min_cluster_frac = float(cfg.get("min_cluster_fraction", 0.1))
        self.min_other = int(cfg.get("min_other_members", 2))
        self.min_align = float(cfg.get("min_alignment", 0.5))
        self.min_ll_gain = float(cfg.get("min_ll_gain", 0.0))
        self.max_len = int(cfg.get("max_span_length", 1))
        self.proposal = cfg.get("proposal", "guided")
        self.reattach = bool(cfg.get("reattach_singletons", True))
        self.private_share = float(cfg.get("private_rule_share", 0.5))
        self.n_ctx_show = int(cfg.get("show_contexts", 5))
        self.type_classes: dict[int, list[int]] = {}
        for c in accepted:
            for t in reports[c].members:
                self.type_classes.setdefault(t, []).append(c)
        self.events: list[SplitReport] = []
        self._prepare_profiles()

    # ------------------------------------------------------------ profiles
    def _prepare_profiles(self) -> None:
        cd = self.cd
        self.col_mask = cd.ctx_scale <= self.max_scale
        self.slot_mask = np.asarray(cd.slot_scale) <= self.max_scale
        self.classes = sorted(c for c in self.accepted if len(self.reports[c].members) >= self.min_other)
        groups = [self.reports[c].members for c in self.classes]
        cmat = aggregate_rows(cd.type_ctx, groups) if groups else sp.csr_matrix((0, cd.n_contexts))
        self.cmat = sp.csr_matrix(cmat @ sp.diags(self.col_mask.astype(np.float64)))
        self.cmat.eliminate_zeros()
        self.class_total = np.asarray(self.cmat.sum(axis=1)).ravel()
        colsum = np.asarray(cd.type_ctx.sum(axis=0)).ravel() * self.col_mask
        self.bg = (colsum + 1.0) / (colsum.sum() + self.col_mask.sum())
        self.xw = sp.csr_matrix(cd.type_ctx @ sp.diags(self.col_mask.astype(np.float64)))

    def _loglik(self, tid: int, oids: np.ndarray) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
        """Return ``(LL[n_occ, n_classes], ctx_ids, occurrence x context matrix)``."""
        occ_ctx = self.cd.occ_ctx[oids][:, self.slot_mask]
        ctx_ids, inverse = np.unique(occ_ctx.reshape(-1), return_inverse=True)
        n_occ = len(oids)
        rows = np.repeat(np.arange(n_occ), occ_ctx.shape[1])
        o_mat = sp.csr_matrix((np.ones(len(rows)), (rows, inverse.reshape(-1))), shape=(n_occ, len(ctx_ids)))
        sub = self.cmat[:, ctx_ids].toarray()
        x_row = self.xw[tid][:, ctx_ids].toarray().ravel()
        totals = self.class_total.copy()
        for c in self.type_classes.get(tid, []):
            if c in self.classes:
                k = self.classes.index(c)
                sub[k] = np.maximum(sub[k] - x_row, 0.0)
                totals[k] = max(totals[k] - float(self.xw[tid].sum()), 0.0)
        bg = self.bg[ctx_ids]
        lp = np.log((sub + self.eta * bg[None, :]) / (totals[:, None] + self.eta))
        ll = np.asarray(o_mat @ lp.T)
        return ll, ctx_ids, o_mat

    def _coverage(self, tid: int, o_mat: sp.csr_matrix, ctx_ids: np.ndarray, rows: np.ndarray, k: int) -> float:
        """Fraction of the cluster's context mass covered by class k's support (x excluded)."""
        mass = np.asarray(o_mat[rows].sum(axis=0)).ravel()
        sub = self.cmat[k][:, ctx_ids].toarray().ravel()
        if self.classes[k] in self.type_classes.get(tid, []):
            sub = np.maximum(sub - self.xw[tid][:, ctx_ids].toarray().ravel(), 0.0)
        return float((mass * (sub > 0)).sum() / max(1.0, mass.sum()))

    def _top_contexts(self, o_mat: sp.csr_matrix, ctx_ids: np.ndarray, rows: np.ndarray) -> list[str]:
        mass = np.asarray(o_mat[rows].sum(axis=0)).ravel()
        order = np.argsort(-mass)[: self.n_ctx_show]
        return [f"{self.cd.vocab.render(int(ctx_ids[o]), self.inv.corpus)} (x{int(mass[o])})" for o in order if mass[o] > 0]

    def _private_rules(self, oids: np.ndarray) -> set[tuple[int, int, int]]:
        """Rules whose current usage is dominated (>= private_rule_share) by the
        occurrences of the tested expression: they are crutches created by a
        wrong single assignment and are unavailable to every hypothesis."""
        if self.model is None:
            return set()
        inv = self.inv
        xs = set(oids.tolist())
        mine: Counter = Counter()
        for sid in self._sids(oids):
            p = self.model.parses.get(sid)
            if p is None:
                continue
            for nd in p.nodes:
                if nd.split < 0:
                    continue
                lo = inv.lookup(sid, nd.start, nd.split)
                ro = inv.lookup(sid, nd.split, nd.end)
                if lo in xs or ro in xs:
                    mine[(nd.cls, nd.left_cls, nd.right_cls)] += 1
        total = self.model.rule_usage()
        return {k for k, n in mine.items() if total.get(k, 0) > 0 and n / total[k] >= self.private_share}

    def _lex_entries(self, tid: int, classes: list[int]) -> list[tuple[int, int]]:
        return [(c, tid) for c in classes] if self.inv.type_length(tid) == 1 else []

    def _sids(self, oids: np.ndarray) -> list[int]:
        sids = sorted({int(self.inv.occ_sid[o]) for o in oids.tolist()})
        if self.model is not None and len(self.model.sids) < len(self.inv.corpus):
            keep = set(self.model.sids)
            sids = [s for s in sids if s in keep]
        return sids

    # ------------------------------------------------------------ main
    def run(self) -> list[SplitReport]:
        inv = self.inv
        if not self.classes:
            log.info("ambiguity: no candidate classes")
            return []
        cand = [t for t in range(inv.n_types) if inv.type_freq[t] >= self.min_occ and inv.type_length(t) <= self.max_len]
        log.info("ambiguity: testing %d span types against %d classes", len(cand), len(self.classes))
        for tid in cand:
            rep = self._test(tid)
            if rep is not None:
                self.events.append(rep)
        n_acc = sum(1 for e in self.events if e.accepted and e.kind == "split")
        log.info("ambiguity: %d splits accepted, %d tests recorded", n_acc, len(self.events))
        return self.events

    def _test(self, tid: int) -> SplitReport | None:
        inv = self.inv
        oids = inv.type_occ[tid]
        ll, ctx_ids, o_mat = self._loglik(tid, oids)
        n = len(oids)
        cur = self.type_classes.get(tid, [])
        if len(cur) > 1:
            return None
        current = cur[0] if cur else -1
        k1 = int(np.argmax(ll.sum(axis=0)))
        if self.proposal == "kmeans":
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize

            if n < 2 * self.min_cluster_occ:
                return None
            lab = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(normalize(o_mat))
            ka = int(np.argmax(ll[lab == 0].sum(axis=0)))
            kb = int(np.argmax(ll[lab == 1].sum(axis=0)))
            if ka == kb:
                return self._maybe_reattach(tid, oids, k1, current, ctx_ids, o_mat)
            z = np.where(lab == 0, ka, kb)
        else:
            z = ll.argmax(axis=1)
            top = Counter(z.tolist()).most_common(2)
            if len(top) < 2:
                return self._maybe_reattach(tid, oids, k1, current, ctx_ids, o_mat)
            ka, kb = top[0][0], top[1][0]
            z = np.where(ll[:, ka] >= ll[:, kb], ka, kb)
        rows_a = np.nonzero(z == ka)[0]
        rows_b = np.nonzero(z == kb)[0]
        n_a, n_b = len(rows_a), len(rows_b)
        if min(n_a, n_b) < max(self.min_cluster_occ, self.min_cluster_frac * n):
            return self._maybe_reattach(tid, oids, k1, current, ctx_ids, o_mat)
        ll_h1 = float(ll[:, k1].sum())
        ll_h2 = float(ll[rows_a, ka].sum() + ll[rows_b, kb].sum())
        ll_gain = (ll_h2 - ll_h1) / n
        full = np.asarray(o_mat.sum(axis=0)).ravel()
        mi = _entropy(full) - (n_a / n) * _entropy(np.asarray(o_mat[rows_a].sum(axis=0)).ravel()) - (n_b / n) * _entropy(np.asarray(o_mat[rows_b].sum(axis=0)).ravel())
        align_a = self._coverage(tid, o_mat, ctx_ids, rows_a, ka)
        align_b = self._coverage(tid, o_mat, ctx_ids, rows_b, kb)
        A, B, K1 = self.classes[ka], self.classes[kb], self.classes[k1]
        sids = self._sids(oids)
        assign1 = {int(o): K1 for o in oids}
        assign2 = {int(oids[i]): (A if z[i] == ka else B) for i in range(n)}
        parses2: dict[int, SentenceParse] = {}
        private: set[tuple[int, int, int]] = set()
        if self.model is not None:
            private = self._private_rules(oids)
            if private:
                # the current assignment re-evaluated without its private crutch rules
                cur_assign = {int(o): int(self.occ_class[o]) for o in oids if self.occ_class[o] >= 0}
                dl_cur, _ = self.model.hypothesis_bits(sids, cur_assign, (), private)
            else:
                dl_cur = self.model.total_bits()
            dl_single, _ = self.model.hypothesis_bits(sids, assign1, self._lex_entries(tid, [K1]), private)
            dl_double, parses2 = self.model.hypothesis_bits(sids, assign2, self._lex_entries(tid, [A, B]), private)
        else:
            dl_cur = dl_single = dl_double = 0.0
        split_gain = (dl_single - dl_double) - self.lambda_split
        reasons = []
        if private:
            log.info("ambiguity %s: %d private rules excluded from the hypotheses", inv.type_str(tid), len(private))
        if ll_gain <= self.min_ll_gain:
            reasons.append(f"context likelihood gain {ll_gain:.3f} <= {self.min_ll_gain}")
        if align_a < self.min_align or align_b < self.min_align:
            reasons.append(f"cluster alignment too weak (a={align_a:.2f}, b={align_b:.2f} < {self.min_align}): the branch would be used mostly by this expression only")
        if self.model is not None and split_gain <= 0:
            reasons.append(f"MDL split gain {split_gain:.1f} bits <= 0 (DL single {dl_single:.1f} vs double {dl_double:.1f}, lambda_split {self.lambda_split})")
        if self.model is not None and dl_double + self.lambda_split >= dl_cur and current >= 0:
            reasons.append(f"split does not beat the current single assignment (DL current {dl_cur:.1f})")
        accepted = not reasons
        comp_a, comp_b = ([], [])
        if accepted and self.model is not None:
            comp_a, comp_b = self._composition_diff(tid, oids, assign2, parses2, A)
        rep = SplitReport(
            tid=tid, span=inv.type_str(tid), n_occ=n, current_class=current, single_class=K1, class_a=A, class_b=B,
            n_a=n_a, n_b=n_b, ll_gain_per_occ=ll_gain, mi_gain_bits=mi, align_a=align_a, align_b=align_b,
            dl_current=dl_cur, dl_single=dl_single, dl_double=dl_double, split_gain=split_gain, lambda_split=self.lambda_split,
            accepted=accepted, reason="; ".join(reasons) if reasons else "all criteria satisfied",
            contexts_a=self._top_contexts(o_mat, ctx_ids, rows_a), contexts_b=self._top_contexts(o_mat, ctx_ids, rows_b),
            composition_a=comp_a, composition_b=comp_b,
        )
        if accepted:
            self._apply(tid, assign2, sids, A, B)
        return rep

    def _maybe_reattach(self, tid: int, oids: np.ndarray, k1: int, current: int, ctx_ids: np.ndarray, o_mat: sp.csr_matrix) -> SplitReport | None:
        """Move a singleton/pruned type to the best single class if DL improves."""
        if not self.reattach or self.model is None:
            return None
        K1 = self.classes[k1]
        if current == K1 or (current >= 0 and len(self.reports[current].members) > 1):
            return None
        sids = self._sids(oids)
        dl_cur = self.model.total_bits()
        assign = {int(o): K1 for o in oids}
        dl_single, _ = self.model.hypothesis_bits(sids, assign, self._lex_entries(tid, [K1]))
        gain = dl_cur - dl_single
        align = self._coverage(tid, o_mat, ctx_ids, np.arange(len(oids)), k1)
        accepted = gain > 0 and align >= self.min_align
        rep = SplitReport(
            tid=tid, span=self.inv.type_str(tid), n_occ=len(oids), current_class=current, single_class=K1, class_a=K1, class_b=-1,
            n_a=len(oids), n_b=0, ll_gain_per_occ=0.0, mi_gain_bits=0.0, align_a=align, align_b=0.0, dl_current=dl_cur,
            dl_single=dl_single, dl_double=dl_single, split_gain=gain, lambda_split=0.0, accepted=accepted,
            reason=("re-attached to a class used by other expressions: DL improves" if accepted else f"re-attach rejected (gain {gain:.1f}, alignment {align:.2f})"),
            contexts_a=self._top_contexts(o_mat, ctx_ids, np.arange(len(oids))), kind="reattach",
        )
        if accepted:
            self._apply(tid, assign, sids, K1, None)
        return rep

    def _apply(self, tid: int, assign: dict[int, int], sids: list[int], a: int, b: int | None) -> None:
        old = self.type_classes.get(tid, [])
        for c in old:
            if c in self.reports and tid in self.reports[c].members and c not in (a, b):
                self.reports[c].members.remove(tid)
        new_classes = [a] + ([b] if b is not None else [])
        self.type_classes[tid] = new_classes
        for c in new_classes:
            if tid not in self.reports[c].members:
                self.reports[c].members.append(tid)
        for o, c in assign.items():
            self.occ_class[o] = c
        if self.model is not None:
            self.model.apply_override(sids, assign, self._lex_entries(tid, new_classes))

    def _composition_diff(self, tid: int, oids: np.ndarray, assign: dict[int, int], parses: dict[int, SentenceParse], a: int) -> tuple[list[str], list[str]]:
        """Parent rules in which the two senses participate (from the H2 derivations)."""
        inv = self.inv
        ca: Counter = Counter()
        cb: Counter = Counter()
        xs = set(oids.tolist())
        for sid, p in parses.items():
            for nd in p.nodes:
                if nd.split < 0:
                    continue
                lo = inv.lookup(sid, nd.start, nd.split)
                ro = inv.lookup(sid, nd.split, nd.end)
                lc, rc = nd.left_cls, nd.right_cls
                if lo in xs:
                    (ca if assign[lo] == a else cb)[f"[x as left] {lc} + {rc} -> {nd.cls}"] += 1
                if ro in xs:
                    (ca if assign[ro] == a else cb)[f"[x as right] {lc} + {rc} -> {nd.cls}"] += 1
        fmt = lambda c: [f"{k} (x{n})" for k, n in c.most_common(4)]
        return fmt(ca), fmt(cb)
