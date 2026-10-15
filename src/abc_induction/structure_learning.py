"""Block-coordinate structure learning on ``F(G) = ELBO(G) + log P(G)``.

    INITIALISE (from v0.1)  ->  CAVI to convergence
    repeat until no move improves F:
        best merge proposal      -> accept iff dF > tau_merge   -> short CAVI
        best split proposal      -> accept iff dF > tau_split   -> short CAVI
        rule / lexicon proposals -> accept iff dF > tau_rule    -> CAVI
        adaptive constituent-context refinement (proposal features only)

Every move: lightweight checkpoint -> apply -> warm-started short CAVI
(``proposal_vb_iterations``) -> dF = F(G') - F(G) -> keep or roll back.  After
every accepted move the full objective is re-evaluated and must not have
decreased beyond ``objective_tolerance`` (otherwise the move is rolled back
and ``monotonicity_violations`` is incremented - the test-suite treats that as
a failure).  All moves are logged with the fields required by the spec.

Proposal evaluation on the *affected* sentences
-----------------------------------------------
``proposal_eval: affected`` (default for natural text) re-estimates only the
sentences whose posterior involves the touched classes / rules / tokens
(mass above ``affected_threshold``); the expected counts of the other
sentences are kept fixed (mapped through the move) and their ``log Z`` is
assumed unchanged.  ``proposal_eval: full`` re-estimates everything (used on
the synthetic suite, where it is cheap).

Rejection memory (``rejection_memory``, off by default)
-------------------------------------------------------
With it on, a proposal rejected by a clear margin (``dF < tau - rejection_margin``)
is remembered and not re-evaluated in later rounds until an accepted
*structural* move (merge, split, rule add / prune) touches one of its classes.
Every round otherwise re-evaluates every rejected merge / split / rule proposal
(``proposal_vb_iterations + 1`` E-steps each).  It is an approximation: a
proposal rejected in round 0 can become acceptable after lexicon moves in
*other* classes (observed on Synthetic-2 and Synthetic-1 at 10k sentences), so
it changes which moves are accepted.  Use it on large corpora where the
E-steps dominate; the synthetic suite runs with it off.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from .adaptive_context import AbstractContexts, compute_abstract_contexts, render_feature, select_refinements
from .cavi import EStepOutput, VariationalInference
from .corpus import BOS_ID, EOS_ID
from .grammar_model import UNK, ExpectedCounts, GrammarModel

log = logging.getLogger(__name__)

RuleKey = tuple[int, int, int]


# --------------------------------------------------------------------------- expected-count arithmetic
def _ec_sub(a: ExpectedCounts, b: ExpectedCounts) -> ExpectedCounts:
    out = ExpectedCounts()
    out.rule = {k: max(v - b.rule.get(k, 0.0), 0.0) for k, v in a.rule.items()}
    out.lex = {k: max(v - b.lex.get(k, 0.0), 0.0) for k, v in a.lex.items()}
    out.root = {k: max(v - b.root.get(k, 0.0), 0.0) for k, v in a.root.items()}
    out.n_sentences = max(a.n_sentences - b.n_sentences, 0.0)
    out.n_chunks = max(a.n_chunks - b.n_chunks, 0.0)
    out.log_z = a.log_z - b.log_z
    for k, d in a.rule_realisations.items():
        bd = b.rule_realisations.get(k, {})
        out.rule_realisations[k] = {t: max(v - bd.get(t, 0.0), 0.0) for t, v in d.items()}
    return out


def _ec_add(a: ExpectedCounts, b: ExpectedCounts) -> ExpectedCounts:
    out = ExpectedCounts()
    out.rule = dict(a.rule)
    for k, v in b.rule.items():
        out.rule[k] = out.rule.get(k, 0.0) + v
    out.lex = dict(a.lex)
    for k, v in b.lex.items():
        out.lex[k] = out.lex.get(k, 0.0) + v
    out.root = dict(a.root)
    for k, v in b.root.items():
        out.root[k] = out.root.get(k, 0.0) + v
    out.n_sentences = a.n_sentences + b.n_sentences
    out.n_chunks = a.n_chunks + b.n_chunks
    out.log_z = a.log_z + b.log_z
    out.n_underivable = b.n_underivable
    out.rule_realisations = {k: dict(d) for k, d in a.rule_realisations.items()}
    for k, d in b.rule_realisations.items():
        dd = out.rule_realisations.setdefault(k, {})
        for t, v in d.items():
            dd[t] = dd.get(t, 0.0) + v
    out.pruning = dict(b.pruning)
    return out


def _ec_rename(ec: ExpectedCounts, ren: dict[int, int]) -> ExpectedCounts:
    out = ExpectedCounts()
    for (c, a, b), v in ec.rule.items():
        k = (ren.get(c, c), ren.get(a, a), ren.get(b, b))
        out.rule[k] = out.rule.get(k, 0.0) + v
    for (c, w), v in ec.lex.items():
        k = (ren.get(c, c), w)
        out.lex[k] = out.lex.get(k, 0.0) + v
    for c, v in ec.root.items():
        k = ren.get(c, c)
        out.root[k] = out.root.get(k, 0.0) + v
    out.n_sentences, out.n_chunks, out.log_z = ec.n_sentences, ec.n_chunks, ec.log_z
    for (c, a, b), d in ec.rule_realisations.items():
        k = (ren.get(c, c), ren.get(a, a), ren.get(b, b))
        dd = out.rule_realisations.setdefault(k, {})
        for t, v in d.items():
            dd[t] = dd.get(t, 0.0) + v
    return out


def _ec_split(ec: ExpectedCounts, a: int, n1: int, n2: int, rule_shares: dict[RuleKey, float], lex_shares: dict[tuple[int, int], float], root_share: float) -> ExpectedCounts:
    """Distribute the (small) residual counts of class ``a`` over its halves."""
    out = ExpectedCounts()
    out.n_sentences, out.n_chunks, out.log_z = ec.n_sentences, ec.n_chunks, ec.log_z
    for key, v in ec.rule.items():
        if a not in key:
            out.rule[key] = out.rule.get(key, 0.0) + v
            continue
        s = rule_shares.get(key, 0.5)
        positions = [i for i, x in enumerate(key) if x == a]
        combos = [()]
        for _ in positions:
            combos = [c + (h,) for c in combos for h in (n1, n2)]
        for combo in combos:
            k = list(key)
            w = 1.0
            for pos, h in zip(positions, combo):
                k[pos] = h
                w *= s if h == n1 else 1.0 - s
            kk = (k[0], k[1], k[2])
            out.rule[kk] = out.rule.get(kk, 0.0) + v * w
    for (c, w), v in ec.lex.items():
        if c != a:
            out.lex[(c, w)] = out.lex.get((c, w), 0.0) + v
        else:
            s = lex_shares.get((c, w), 0.5)
            out.lex[(n1, w)] = out.lex.get((n1, w), 0.0) + v * s
            out.lex[(n2, w)] = out.lex.get((n2, w), 0.0) + v * (1 - s)
    for c, v in ec.root.items():
        if c != a:
            out.root[c] = out.root.get(c, 0.0) + v
        else:
            out.root[n1] = out.root.get(n1, 0.0) + v * root_share
            out.root[n2] = out.root.get(n2, 0.0) + v * (1 - root_share)
    for key, d in ec.rule_realisations.items():
        if a not in key:
            out.rule_realisations[key] = dict(d)
    return out


# --------------------------------------------------------------------------- moves
@dataclass
class Move:
    kind: str
    description: dict[str, Any]
    affected: list[int]
    apply: Callable[[GrammarModel], dict[str, Any]]  # returns info (new ids, potential-class mapping)
    map_counts: Callable[[ExpectedCounts, dict[str, Any]], ExpectedCounts]
    threshold_key: str
    proposal_reason: str = ""
    score: float = 0.0
    classes: frozenset[int] = frozenset()  # classes whose structure the move touches (rejection memory)


# --------------------------------------------------------------------------- signatures
@dataclass
class ClassSignatures:
    raw: dict[int, dict[Any, float]]
    comp_in: dict[int, dict[Any, float]]
    comp_out: dict[int, dict[Any, float]]
    lex: dict[int, dict[Any, float]]
    abstract: dict[int, dict[Any, float]]
    mass: dict[int, float]

    def blocks(self, c: int) -> dict[str, dict[Any, float]]:
        return {"raw": self.raw.get(c, {}), "comp_in": self.comp_in.get(c, {}), "comp_out": self.comp_out.get(c, {}), "lex": self.lex.get(c, {}), "abstract": self.abstract.get(c, {})}


def _js_divergence(a: dict[Any, float], b: dict[Any, float], min_mass: float = 1.0) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) of two unnormalised feature profiles; 0 if either is (nearly) empty."""
    za, zb = sum(a.values()), sum(b.values())
    if za < min_mass or zb < min_mass:
        return 0.0
    keys = set(a) | set(b)
    d = 0.0
    for k in keys:
        pa, pb = a.get(k, 0.0) / za, b.get(k, 0.0) / zb
        m = 0.5 * (pa + pb)
        if pa > 0:
            d += 0.5 * pa * math.log2(pa / m)
        if pb > 0:
            d += 0.5 * pb * math.log2(pb / m)
    return float(min(max(d, 0.0), 1.0))


def _cosine(a: dict[Any, float], b: dict[Any, float]) -> float:
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items()) / (na * nb)


# --------------------------------------------------------------------------- learner
class StructureLearner:
    def __init__(
        self,
        vi: VariationalInference,
        corpus_tokens: list[np.ndarray],
        inv: Any,
        cfg: dict[str, Any],
        adaptive_cfg: dict[str, Any] | None = None,
        lex_scores: dict[int, list[dict[int, float]]] | None = None,
        names: dict[int, str] | None = None,
        modifier_cfg: dict[str, Any] | None = None,
        corpus: Any | None = None,
    ) -> None:
        self.vi = vi
        self.model = vi.model
        self.tokens = corpus_tokens
        self.inv = inv
        self.cfg = cfg
        self.adaptive_cfg = adaptive_cfg or {}
        self.adaptive_enabled = bool(self.adaptive_cfg.get("enabled", False))
        self.lex_scores = lex_scores or {}
        self.moves: list[dict[str, Any]] = []
        self.monotonicity_violations = 0
        self.ec: ExpectedCounts | None = None
        self.out: EStepOutput | None = None
        self.F: float = float("-inf")
        self.F_parts: dict[str, float] = {}
        self.sig: ClassSignatures | None = None
        self.abstract: AbstractContexts | None = None
        self.flagged_pairs: list[tuple[int, int]] = []
        self.eval_mode = cfg.get("proposal_eval", "affected")
        self.affected_threshold = float(cfg.get("affected_threshold", 1e-3))
        self.short_iters = int(cfg.get("proposal_vb_iterations", 5))
        self.n_lengths = {sid: len(t) for sid, t in enumerate(corpus_tokens)}
        self.rounds_done = 0
        self.rejected: set[str] = set()  # proposals rejected since the last accepted move
        self.memory_enabled = bool(cfg.get("rejection_memory", False))
        self.rejection_margin = float(cfg.get("rejection_margin", 5.0))
        self.rejected_memo: dict[str, tuple[float, int, frozenset[int]]] = {}  # move key -> (dF, round, classes)
        self.n_memo_skipped = 0
        # identity-preserving (modifier) evidence: proposal source (mode) and lexical distinguishability
        self.modifier_cfg = {**(modifier_cfg or {})}
        self.modifier_mode = str(self.modifier_cfg.get("mode", "off"))
        self.modifier_analyzer = None
        if corpus is not None and (self.modifier_mode != "off" or self.modifier_cfg.get("distinguishability", {}).get("enabled", False)):
            from .modifier_evidence import ModifierAnalyzer

            self.modifier_analyzer = ModifierAnalyzer(corpus, inv, self.modifier_cfg)
        self.modifier_table = None
        self.lex_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ objective bookkeeping
    def full_cavi(self, label: str, max_iter: int | None = None) -> float:
        """CAVI to convergence, then a scored E-step with posterior collection."""
        max_iter = max_iter or int(self.cfg.get("max_cavi_iterations", 50))
        self.vi.run(max_iter, float(self.cfg.get("cavi_tolerance", 1e-5)), label=label)
        return self.rescore(label)

    def rescore(self, label: str = "score") -> float:
        ec, out = self.vi.e_step(collect=True)
        comps = self.vi.elbo_components(ec)
        self.ec, self.out = ec, out
        self.vi.last_ec = ec
        self.vi.last_log_z = dict(out.log_z)
        self.F = comps["objective"]
        self.F_parts = comps
        self.vi.m_step(ec)  # keep the model at the fixed point of this E-step
        self.sig = None
        self.abstract = None
        self.modifier_table = None
        return self.F

    # ------------------------------------------------------------------ signatures
    def signatures(self) -> ClassSignatures:
        if self.sig is not None:
            return self.sig
        assert self.out is not None and self.ec is not None
        raw: dict[int, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        for sid, nodes in self.out.node_posteriors.items():
            toks = self.tokens[sid]
            n = len(toks)
            for i, j, c, p in nodes:
                if p < 0.05:
                    continue
                left = int(toks[i - 1]) if i > 0 else BOS_ID
                right = int(toks[j]) if j < n else EOS_ID
                raw[c][("L", left)] += p
                raw[c][("R", right)] += p
                raw[c][("LR", left, right)] += p
        comp_in: dict[int, dict[Any, float]] = defaultdict(dict)
        comp_out: dict[int, dict[Any, float]] = defaultdict(dict)
        for (c, a, b), n in self.ec.rule.items():
            if n <= 0:
                continue
            comp_in[c][("in", a, b)] = comp_in[c].get(("in", a, b), 0.0) + n
            comp_out[a][("outL", c, b)] = comp_out[a].get(("outL", c, b), 0.0) + n
            comp_out[b][("outR", c, a)] = comp_out[b].get(("outR", c, a), 0.0) + n
        lex: dict[int, dict[Any, float]] = defaultdict(dict)
        for (c, w), n in self.model.lex.items():
            if n > 0 and w != UNK:
                lex[c][("w", w)] = n
        mass: dict[int, float] = defaultdict(float)
        for (c, w), n in self.model.lex.items():
            mass[c] += n
        for (c, a, b), n in self.model.rules.items():
            mass[c] += n
        abstract: dict[int, dict[Any, float]] = {}
        if self.adaptive_enabled:
            flagged = {c for pr in self.flagged_pairs for c in pr}
            self.abstract = compute_abstract_contexts(self.out, self.n_lengths, self.adaptive_cfg, flagged)
            if self.flagged_pairs:
                sel = select_refinements(self.abstract, self.flagged_pairs, int(self.adaptive_cfg.get("top_features", 20)))
            else:
                sel = {}
            for c, feats in self.abstract.class_features.items():
                keep = {f: w for f, w in feats.items() if f[1] == "11"}
                for f, _ in sel.get(c, []):
                    if f in feats:
                        keep[f] = feats[f]
                abstract[c] = keep
        self.sig = ClassSignatures({c: dict(d) for c, d in raw.items()}, dict(comp_in), dict(comp_out), dict(lex), abstract, dict(mass))
        return self.sig

    def similarity(self, a: int, b: int, sig: ClassSignatures) -> tuple[float, dict[str, float]]:
        wts = self.cfg.get("merge_similarity_weights") or {"raw": 1.0, "comp_in": 1.0, "comp_out": 1.0, "lex": 0.5, "abstract": 1.0}
        ba, bb = sig.blocks(a), sig.blocks(b)
        parts = {}
        tot_w = 0.0
        acc = 0.0
        for k, w in wts.items():
            if not ba.get(k) and not bb.get(k):
                continue
            s = _cosine(ba.get(k, {}), bb.get(k, {}))
            parts[k] = round(s, 4)
            acc += w * s
            tot_w += w
        return (acc / tot_w if tot_w > 0 else 0.0), parts

    # ------------------------------------------------------------------ affected sentences
    def _sentences_with_classes(self, classes: set[int]) -> list[int]:
        assert self.out is not None
        out = []
        for sid, nodes in self.out.node_posteriors.items():
            if any(c in classes and p >= self.affected_threshold for _, _, c, p in nodes):
                out.append(sid)
        return out

    def _sentences_with_rules(self, rules: set[RuleKey]) -> list[int]:
        assert self.out is not None
        out = []
        for sid, apps in self.out.rule_apps.items():
            if any(r in rules for r, _, _, _, _ in apps):
                out.append(sid)
        return out

    def _sentences_with_token(self, w: int) -> list[int]:
        return [sid for sid, t in enumerate(self.tokens) if (t == w).any()]

    # ------------------------------------------------------------------ merge proposals
    def propose_merges(self) -> list[Move]:
        sig = self.signatures()
        classes = [c for c in self.model.classes if sig.mass.get(c, 0.0) >= float(self.cfg.get("min_class_mass", 2.0))]
        inv_index: dict[Any, list[int]] = defaultdict(list)
        top_f = int(self.cfg.get("signature_top_features", 15))
        for c in classes:
            for block in ("raw", "comp_in", "comp_out", "abstract"):
                feats = sig.blocks(c)[block]
                for f, _ in sorted(feats.items(), key=lambda kv: -kv[1])[:top_f]:
                    inv_index[(block, f)].append(c)
        pairs: set[tuple[int, int]] = set()
        for key, cs in inv_index.items():
            if len(cs) > int(self.cfg.get("max_classes_per_feature", 50)):
                continue
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs.add((min(cs[i], cs[j]), max(cs[i], cs[j])))
        scored = []
        for a, b in pairs:
            s, parts = self.similarity(a, b, sig)
            scored.append((s, a, b, parts))
        scored.sort(key=lambda t: -t[0])
        min_sim = float(self.cfg.get("merge_min_similarity", 0.3))
        moves = []
        # flag raw-similar but composition-different pairs for adaptive refinement
        flagged = []
        for s, a, b, parts in scored:
            if parts.get("raw", 0) >= 0.6 and min(parts.get("comp_in", 1), parts.get("comp_out", 1)) < 0.3:
                flagged.append((a, b))
        self.flagged_pairs = flagged[: int(self.cfg.get("max_flagged_pairs", 10))]
        for s, a, b, parts in scored[: int(self.cfg.get("max_merge_candidates", 8))]:
            if s < min_sim:
                break
            moves.append(self._merge_move(a, b, s, parts))
        return moves

    def _merge_move(self, a: int, b: int, score: float, parts: dict[str, float]) -> Move:
        names = self.model.names
        affected = self._sentences_with_classes({a, b})
        desc = {"a": a, "b": b, "a_name": names.get(a), "b_name": names.get(b), "similarity": round(score, 4), "similarity_parts": parts, "n_affected_sentences": len(affected)}

        def apply(model: GrammarModel) -> dict[str, Any]:
            new = model.merge_classes(a, b)
            return {"new": new, "potential_sources": {new: self.vi.class_map.get(a, [a]) + self.vi.class_map.get(b, [b])}, "rename": {a: new, b: new}}

        def map_counts(ec: ExpectedCounts, info: dict[str, Any]) -> ExpectedCounts:
            return _ec_rename(ec, info["rename"])

        reason = "most similar blocks: " + ", ".join(f"{k}={v}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1])[:3])
        return Move("merge", desc, affected, apply, map_counts, "merge_threshold", reason, score, frozenset({a, b}))

    # ------------------------------------------------------------------ split proposals
    def propose_splits(self) -> list[Move]:
        assert self.out is not None
        min_occ = int(self.cfg.get("split_min_occurrences", 20))
        sig = self.signatures()
        occ_by_class: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
        for sid, nodes in self.out.node_posteriors.items():
            for i, j, c, p in nodes:
                if p >= 0.5:
                    occ_by_class[c].append((sid, i, j, p))
        # parent / child rule context per node from rule applications
        parent: dict[tuple[int, int, int], dict[Any, float]] = defaultdict(dict)
        child: dict[tuple[int, int, int], dict[Any, float]] = defaultdict(dict)
        for sid, apps in self.out.rule_apps.items():
            for r, i, k, j, p in apps:
                parent[(sid, i, k)][("parL", r)] = parent[(sid, i, k)].get(("parL", r), 0.0) + p
                parent[(sid, k, j)][("parR", r)] = parent[(sid, k, j)].get(("parR", r), 0.0) + p
                child[(sid, i, j)][("chi", r)] = child[(sid, i, j)].get(("chi", r), 0.0) + p
        candidates = []
        fw = self.cfg.get("split_feature_weights") or {}
        w_ctx, w_rule, w_abs = float(fw.get("context", 1.0)), float(fw.get("composition", 3.0)), float(fw.get("abstract", 2.0))
        for c, occs in occ_by_class.items():
            if len(occs) < min_occ:
                continue
            feats: list[dict[Any, float]] = []
            weights = []
            for sid, i, j, p in occs:
                toks = self.tokens[sid]
                n = len(toks)
                f: dict[Any, float] = {}
                f[("L", int(toks[i - 1]) if i > 0 else BOS_ID)] = w_ctx
                f[("R", int(toks[j]) if j < n else EOS_ID)] = w_ctx
                for k, v in parent.get((sid, i, j), {}).items():
                    f[k] = f.get(k, 0.0) + w_rule * v
                for k, v in child.get((sid, i, j), {}).items():
                    f[k] = f.get(k, 0.0) + w_rule * v
                if self.abstract is not None:
                    for k, v in self.abstract.occ_features.get((sid, i, j), {}).items():
                        f[k] = f.get(k, 0.0) + w_abs * v
                feats.append(f)
                weights.append(p)
            labels, quality, balance = _weighted_two_means(feats, np.asarray(weights), int(self.cfg.get("seed", 0)))
            if labels is None:
                continue
            # stable bimodality signal: how different are the composition signatures of the two halves?
            sig = [defaultdict(float), defaultdict(float)]
            for f, lab, p in zip(feats, labels, weights):
                for k, v in f.items():
                    if k[0] in ("parL", "parR", "chi", "abs"):
                        sig[lab][k] += v * p
            separation = 1.0 - _cosine(sig[0], sig[1])
            score = separation * balance
            candidates.append((score, c, occs, labels, separation, balance))
        candidates.sort(key=lambda t: -t[0])
        moves = []
        min_score = float(self.cfg.get("split_min_separation", 0.05))
        for score, c, occs, labels, quality, balance in candidates[: int(self.cfg.get("max_split_candidates", 5))]:
            if score < min_score:
                break
            moves.append(self._split_move(c, occs, labels, score, quality, balance, parent, child))
        return moves

    def _split_move(self, a: int, occs, labels: np.ndarray, score: float, quality: float, balance: float, parent, child) -> Move:
        # shares: fraction of A-occurrence mass in cluster 0 per rule (as child / as parent), lexical entry, root
        rule_mass: dict[RuleKey, list[float]] = defaultdict(lambda: [0.0, 0.0])
        lex_mass: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        root_mass = [0.0, 0.0]
        side_examples: dict[int, dict[str, float]] = {0: defaultdict(float), 1: defaultdict(float)}
        for (sid, i, j, p), lab in zip(occs, labels):
            toks = self.tokens[sid]
            for key, v in parent.get((sid, i, j), {}).items():
                rule_mass[key[1]][lab] += v
                side_examples[lab][f"{'left' if key[0] == 'parL' else 'right'} child of {self._rule_str(key[1])}"] += v
            for key, v in child.get((sid, i, j), {}).items():
                rule_mass[key[1]][lab] += v
                side_examples[lab][f"result of {self._rule_str(key[1])}"] += v
            if j - i == 1:
                lex_mass[(a, int(toks[i]))][lab] += p
            if i == 0 and j == len(toks):
                root_mass[lab] += p
            left = self.inv.corpus.itos[int(toks[i - 1])] if i > 0 else "<BOS>"
            right = self.inv.corpus.itos[int(toks[j])] if j < len(toks) else "<EOS>"
            side_examples[lab][f"ctx {left} _ {right}"] += p
        rule_shares = {k: (v[0] + 0.5) / (v[0] + v[1] + 1.0) for k, v in rule_mass.items()}
        lex_shares = {k: (v[0] + 0.5) / (v[0] + v[1] + 1.0) for k, v in lex_mass.items()}
        root_share = (root_mass[0] + 0.5) / (root_mass[0] + root_mass[1] + 1.0)
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        affected = self._sentences_with_classes({a})
        top = lambda d: [f"{k} ({v:.1f})" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:6]]
        desc = {"a": a, "a_name": self.model.names.get(a), "composition_separation": round(quality, 4), "balance": round(balance, 4), "n_occurrences": [n0, n1], "n_affected_sentences": len(affected),
                "partition_a1": top(side_examples[0]), "partition_a2": top(side_examples[1])}

        def apply(model: GrammarModel) -> dict[str, Any]:
            n1_, n2_ = model.split_class(a, rule_shares, lex_shares, root_share)
            src = self.vi.class_map.get(a, [a])
            return {"new": (n1_, n2_), "potential_sources": {n1_: src, n2_: src}, "split": (a, n1_, n2_)}

        def map_counts(ec: ExpectedCounts, info: dict[str, Any]) -> ExpectedCounts:
            a_, n1_, n2_ = info["split"]
            return _ec_split(ec, a_, n1_, n2_, rule_shares, lex_shares, root_share)

        return Move("split", desc, affected, apply, map_counts, "split_threshold", f"weighted 2-means: composition-signature separation {quality:.3f} x balance {balance:.3f}", score, frozenset({a}))

    def _rule_str(self, key: RuleKey) -> str:
        c, a, b = key
        n = self.model.names
        return f"{n.get(a, a)} + {n.get(b, b)} -> {n.get(c, c)}"

    # ------------------------------------------------------------------ rule / lexicon proposals
    def propose_rule_moves(self) -> list[Move]:
        assert self.ec is not None and self.out is not None
        moves: list[Move] = []
        prune_count = float(self.cfg.get("rule_prune_count", 0.5))
        weak = {k for k in self.model.active_rules() if self.model.rules.get(k, 0.0) < prune_count}
        if weak:
            affected = self._sentences_with_rules(weak)
            weak_l = sorted(weak)

            def apply_prune(model: GrammarModel, weak_l=weak_l) -> dict[str, Any]:
                for k in weak_l:
                    model.remove_rule(k)
                return {"removed": weak_l}

            def map_prune(ec: ExpectedCounts, info: dict[str, Any]) -> ExpectedCounts:
                out = _ec_rename(ec, {})
                for k in info["removed"]:
                    out.rule.pop(k, None)
                    out.rule_realisations.pop(k, None)
                return out

            moves.append(Move("rule_prune", {"rules": [self._rule_str(k) for k in weak_l[:20]], "n_rules": len(weak_l), "n_affected_sentences": len(affected)}, affected, apply_prune, map_prune, "rule_threshold", f"{len(weak_l)} rules with E[N] < {prune_count}", 0.0, frozenset(c for k in weak_l for c in k)))
        weak_lex = [k for k, n in self.model.lex.items() if k[1] != UNK and n < prune_count]
        if weak_lex:
            affected = sorted({sid for (c, w) in weak_lex[:200] for sid in self._sentences_with_token(w)})

            def apply_lp(model: GrammarModel, weak_lex=weak_lex) -> dict[str, Any]:
                for k in weak_lex:
                    model.remove_lex(k)
                return {"removed": weak_lex}

            def map_lp(ec: ExpectedCounts, info: dict[str, Any]) -> ExpectedCounts:
                out = _ec_rename(ec, {})
                for k in info["removed"]:
                    out.lex.pop(k, None)
                return out

            moves.append(Move("lex_prune", {"entries": [f"{self.model.names.get(c, c)}: {self.inv.corpus.itos[w]}" for c, w in weak_lex[:20]], "n_entries": len(weak_lex), "n_affected_sentences": len(affected)}, affected, apply_lp, map_lp, "rule_threshold", f"{len(weak_lex)} lexical entries with E[N] < {prune_count}", 0.0, frozenset(c for c, _ in weak_lex)))
        return moves

    def propose_lex_adds(self) -> list[Move]:
        return self._propose_lex_adds()

    def propose_rule_adds(self) -> list[Move]:
        return self._propose_modifier_rules() + self._propose_rule_adds()

    # ------------------------------------------------------------------ identity-preserving evidence
    def modifier_evidence(self):
        """Evidence table for the current posterior (cached until the next re-score)."""
        if self.modifier_analyzer is None or self.out is None:
            return None
        if self.modifier_table is None:
            self.modifier_table = self.modifier_analyzer.collect(self.out, self.model, self.model.names)
        return self.modifier_table

    def _propose_modifier_rules(self) -> list[Move]:
        """``rule_add(C, C, M)`` / ``rule_add(C, M, C)`` for endomorphic rules the evidence asks for but the grammar lacks.

        Which evidence counts is the ``modifier.mode`` (string_only: ROOT_STRING + CONSTITUENT_STRING;
        string_plus_class: + CLASS_GENERALIZED).  Acceptance is still ``dF > tau``.
        """
        if self.modifier_mode in ("off", "posterior_rule_only"):
            return []
        from .modifier_evidence import modifier_rule_candidates

        table = self.modifier_evidence()
        if table is None:
            return []
        moves: list[Move] = []
        for cand in modifier_rule_candidates(table, self.model, self.modifier_mode, self.modifier_cfg):
            key = cand["rule"]
            c, a, b = key
            if key in self.model.rules:
                continue
            affected = self._sentences_with_classes({a, b})
            n = float(cand["support"])

            def apply_add(model: GrammarModel, key=key, n=n) -> dict[str, Any]:
                model.add_rule(*key, count=n)
                return {"added": key}

            direction = "RIGHT" if key[1] == c else "LEFT"
            desc = {"rule": self._rule_str(key), "modifier_direction": direction, "modifier_sources": cand["sources"], "modifier_support": round(n, 2), "modifier_candidates": cand["candidates"], "modifier_productivity": cand["productivity"], "n_affected_sentences": len(affected)}
            moves.append(Move("rule_add", desc, affected, apply_add, lambda ec, info: _ec_rename(ec, {}), "rule_threshold", f"identity-preserving evidence ({'/'.join(cand['sources'])}): {direction} modifier candidates {cand['candidates'][:3]} of {self.model.names.get(c)} (support {n:.0f}, {cand['productivity']} distinct hosts)", n, frozenset(key)))
        return moves

    def _propose_rule_adds(self) -> list[Move]:
        """Adjacent high-posterior constituents (A, B) whose parent span is never a constituent:
        propose A + B -> C for C in {A, B, most frequent root class}."""
        assert self.out is not None
        pair_count: dict[tuple[int, int], float] = defaultdict(float)
        pair_sids: dict[tuple[int, int], set[int]] = defaultdict(set)
        existing = set(self.model.rules)
        for sid, nodes in self.out.node_posteriors.items():
            spans: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
            for i, j, c, p in nodes:
                spans[(i, j)][c] = p
            const = {k: sum(d.values()) for k, d in spans.items()}
            for (i, k), dl in spans.items():
                if const[(i, k)] < 0.5:
                    continue
                for (k2, j), dr in spans.items():
                    if k2 != k or const[(k2, j)] < 0.5:
                        continue
                    if const.get((i, j), 0.0) >= 0.5:
                        continue  # the parent span already is a constituent
                    for a, pa in dl.items():
                        for b, pb in dr.items():
                            w = pa * pb
                            if w >= 0.25:
                                pair_count[(a, b)] += w
                                pair_sids[(a, b)].add(sid)
        root_class = max(self.model.root.items(), key=lambda kv: kv[1])[0] if self.model.root else None
        cands = sorted(pair_count.items(), key=lambda kv: -kv[1])[: int(self.cfg.get("max_rule_add_pairs", 5))]
        moves: list[Move] = []
        min_count = float(self.cfg.get("rule_add_min_count", 5.0))
        for (a, b), n in cands:
            if n < min_count:
                break
            results = [a, b] + ([root_class] if root_class not in (a, b, None) else [])
            for c in results:
                key = (c, a, b)
                if key in existing:
                    continue
                affected = sorted(pair_sids[(a, b)])

                def apply_add(model: GrammarModel, key=key, n=n) -> dict[str, Any]:
                    model.add_rule(*key, count=n)
                    return {"added": key}

                moves.append(Move("rule_add", {"rule": self._rule_str(key), "adjacency_count": round(n, 2), "n_affected_sentences": len(affected)}, affected, apply_add, lambda ec, info: _ec_rename(ec, {}), "rule_threshold", f"adjacent constituents {self.model.names.get(a)} {self.model.names.get(b)} seen {n:.1f}x without a parent", n, frozenset(key)))
        return moves

    def _propose_lex_adds(self) -> list[Move]:
        """Lexical entries (C, w) demanded by the lexical-context scorer but absent from the lexicon."""
        if not self.lex_scores:
            return []
        demand: dict[tuple[int, int], float] = defaultdict(float)
        n_occ: dict[int, int] = defaultdict(int)
        src_of: dict[int, list[int]] = {c: self.vi.class_map.get(c, [c]) for c in self.model.classes}
        inv_src: dict[int, list[int]] = defaultdict(list)
        for c, srcs in src_of.items():
            for s in srcs:
                inv_src[s].append(c)
        existing = set(self.model.lex)
        for sid, entries in self.lex_scores.items():
            toks = self.tokens[sid]
            for i, d in enumerate(entries):
                if not d:
                    continue
                w = int(toks[i])
                n_occ[w] += 1
                for s, v in d.items():
                    for c in inv_src.get(s, []):
                        if (c, w) not in existing:
                            demand[(c, w)] += math.exp(v)
        min_share = float(self.cfg.get("lex_add_min_share", 0.15))
        cands = [((c, w), v) for (c, w), v in demand.items() if n_occ[w] >= int(self.cfg.get("lex_add_min_occurrences", 8)) and v / n_occ[w] >= min_share]
        cands.sort(key=lambda kv: -kv[1])
        moves: list[Move] = []
        for (c, w), v in cands[: int(self.cfg.get("max_lex_add_candidates", 6))]:
            affected = self._sentences_with_token(w)

            def apply_lex(model: GrammarModel, c=c, w=w, v=v) -> dict[str, Any]:
                model.add_lex(c, w, count=v)
                return {"added": (c, w)}

            moves.append(Move("lex_add", {"class": c, "class_name": self.model.names.get(c), "token": self.inv.corpus.itos[w], "context_demand": round(v, 2), "n_occurrences": n_occ[w], "n_affected_sentences": len(affected)}, affected, apply_lex, lambda ec, info: _ec_rename(ec, {}), "rule_threshold", f"context scorer demands {self.model.names.get(c)} for {self.inv.corpus.itos[w]} ({v:.1f}/{n_occ[w]} occurrences)", v, frozenset({c})))
        return moves

    # ------------------------------------------------------------------ evaluation of one move
    @staticmethod
    def move_key(move: Move) -> str:
        d = move.description
        return f"{move.kind}:{d.get('a')}:{d.get('b')}:{d.get('rule')}:{d.get('class')}:{d.get('token')}:{d.get('n_rules')}:{d.get('n_entries')}"

    STRUCTURAL_KINDS = ("merge", "split", "rule_add", "rule_prune")

    def memo_skip(self, move: Move) -> bool:
        """True if ``move`` was rejected by a clear margin earlier and nothing structural touched its classes since."""
        if not self.memory_enabled:
            return False
        return self.move_key(move) in self.rejected_memo

    def _memo_record(self, move: Move, delta: float, tau: float) -> None:
        if self.memory_enabled and delta < tau - self.rejection_margin:
            self.rejected_memo[self.move_key(move)] = (delta, self.rounds_done, move.classes)

    def _memo_invalidate(self, move: Move) -> None:
        """An accepted structural move frees the memo entries that share a class with it."""
        if move.kind not in self.STRUCTURAL_KINDS or not self.rejected_memo:
            return
        self.rejected_memo = {k: v for k, v in self.rejected_memo.items() if not (v[2] & move.classes)}

    def evaluate(self, move: Move) -> dict[str, Any]:
        assert self.ec is not None and self.out is not None
        model = self.model
        vi = self.vi
        tau = float(self.cfg.get(move.threshold_key, 0.0))
        self.rejected.add(self.move_key(move))
        t0 = time.time()
        snap = model.checkpoint()
        class_map_snap = {k: list(v) for k, v in vi.class_map.items()}
        full = self.eval_mode == "full" or not move.affected
        affected = list(range(len(self.tokens))) if full else move.affected
        old_logz = sum(self.out.log_z.get(s, 0.0) for s in affected if self.out.log_z.get(s, 0.0) > float("-inf"))
        n_underivable_old = sum(1 for s in affected if self.out.log_z.get(s, 0.0) == float("-inf"))
        old_kl = model.kl_to_prior()
        old_prior, old_prior_parts = model.structure_log_prior()
        old_part = old_logz - old_kl + old_prior
        old_elbo = old_logz - old_kl
        if full:
            ec_rest = ExpectedCounts()
        else:
            ec_aff_old, _ = vi.e_step(affected)
            ec_rest = _ec_sub(self.ec, ec_aff_old)
        info = move.apply(model)
        for new_c, srcs in info.get("potential_sources", {}).items():
            vi.class_map[new_c] = list(srcs)
        ec_rest = move.map_counts(ec_rest, info)
        for it in range(self.short_iters):
            ec_aff, _ = vi.e_step(affected)
            vi.m_step(_ec_add(ec_rest, ec_aff))
        dist_cfg = self.modifier_cfg.get("distinguishability") or {}
        want_dist = move.kind == "lex_add" and bool(dist_cfg.get("enabled", False))
        ec_aff, out_aff = vi.e_step(affected, collect=want_dist)
        new_kl = model.kl_to_prior()
        new_prior, new_prior_parts = model.structure_log_prior()
        new_logz = ec_aff.log_z
        new_part = new_logz - new_kl + new_prior
        delta = new_part - old_part
        penalty = 0.0
        dist: dict[str, Any] = {}
        if want_dist:
            dist = self.lexical_distinguishability(info["added"], out_aff, affected, dist_cfg)
            penalty = dist["duplicate_penalty"]
        # a move may not make any sentence underivable; sentences that become derivable are credited by log Z
        accepted = (delta - penalty) > tau and ec_aff.n_underivable <= n_underivable_old
        rec = {
            "kind": move.kind,
            "round": self.rounds_done,
            **move.description,
            "proposal_reason": move.proposal_reason,
            "old_objective": round(old_part, 4),
            "new_objective": round(new_part, 4),
            "delta": round(delta, 4),
            "delta_ELBO": round((new_logz - new_kl) - old_elbo, 4),
            "delta_structure_prior": round(new_prior - old_prior, 4),
            "delta_total_objective": round(delta, 4),
            "delta_prior_components": {k: round(new_prior_parts[k] - old_prior_parts.get(k, 0.0), 4) for k in new_prior_parts},
            "threshold": tau,
            "evaluated_on": "all sentences" if full else f"{len(affected)} affected sentences",
            "accepted": bool(accepted),
            "seconds": round(time.time() - t0, 2),
        }
        if move.kind == "lex_add":
            rec_lex = {
                "word": move.description.get("token"),
                "old_classes": dist.get("old_classes", []),
                "new_class": move.description.get("class_name"),
                "delta_elbo": rec["delta_ELBO"],
                "delta_prior": rec["delta_structure_prior"],
                "composition_distance": dist.get("composition_distance"),
                "modifier_distance": dist.get("modifier_distance"),
                "derivability_gain": dist.get("derivability_gain"),
                "structural_distance": dist.get("structural_distance"),
                "combine": dist.get("combine"),
                "duplicate_penalty": round(penalty, 4),
                "final_delta": round(delta - penalty, 4),
                "accepted": bool(accepted),
                "rejected_by_penalty": bool(delta > tau and not accepted and ec_aff.n_underivable <= n_underivable_old),
                "threshold": tau,
                "lambda_dup_to_reject": round((delta - tau) / math.exp(-float(dist.get("structural_distance", 1.0)) / float(dist_cfg.get("tau", 0.1))), 2) if dist and delta > tau else None,
                "profiles": dist.get("profiles"),
                "round": self.rounds_done,
            }
            self.lex_records.append(rec_lex)
            rec["duplicate_penalty"] = round(penalty, 4)
            rec["structural_distance"] = dist.get("structural_distance")
        if accepted:
            self.rejected = set()
            self._memo_invalidate(move)
            # commit: the full expectations are rest + affected
            ec_new = _ec_add(ec_rest, ec_aff)
            vi.m_step(ec_new)
            self.ec = ec_new
            for sid, lz in out_aff.log_z.items():
                self.out.log_z[sid] = lz
            self.sig = None
            self.abstract = None
            self.modifier_table = None
        else:
            model.restore(snap)
            vi.class_map = class_map_snap
            self._memo_record(move, delta, tau)
        self.moves.append(rec)
        log.info("%s %s: dF=%.3f (ELBO %+.3f, prior %+.3f) -> %s", move.kind, move.description.get("a_name") or move.description.get("rule") or move.description.get("token") or "", delta, rec["delta_ELBO"], rec["delta_structure_prior"], "ACCEPT" if accepted else "reject")
        return rec

    def lexical_distinguishability(self, added: tuple[int, int], out_aff: EStepOutput, affected: list[int], cfg: dict[str, Any]) -> dict[str, Any]:
        """Structural distinguishability of a new lexical entry ``(B, w)`` from the existing entries ``(A, w)``.

        From the rule applications of the trial posterior, every occurrence of ``w`` is profiled by
        the composition it takes part in (role, result class, sibling class) and by its
        modifier role (LEFT / RIGHT modifier of host X, host of a LEFT / RIGHT modifier X, or a
        category-changing composition).  ``D_comp`` / ``D_modifier`` are the Jensen-Shannon
        divergences between the profiles of ``w`` under ``B`` and under its closest existing class
        ``A``; ``D_struct = lambda_c D_comp + lambda_m D_modifier`` and the duplicate penalty is
        ``lambda_dup * exp(-D_struct / tau)``.  Entries with no mass on one side are
        indistinguishable (D = 0).
        """
        b, w = added
        old_classes = sorted(c for (c, ww) in self.model.lex if ww == w and c != b)
        comp: dict[int, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        modp: dict[int, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        for sid in affected:
            apps = out_aff.rule_apps.get(sid)
            if not apps:
                continue
            toks = self.tokens[sid]
            for (res, a, bb), i, k, j, p in apps:
                if k - i == 1 and int(toks[i]) == w:  # w is the left child, class a
                    comp[a][("L", res, bb)] += p
                    modp[a][("mod_L", bb) if res == bb and res != a else (("host_R", bb) if res == a and res != bb else ("other_L", res, bb))] += p
                if j - k == 1 and int(toks[k]) == w:  # w is the right child, class bb
                    comp[bb][("R", res, a)] += p
                    modp[bb][("mod_R", a) if res == a and res != bb else (("host_L", a) if res == bb and res != a else ("other_R", res, a))] += p
        min_mass = float(cfg.get("min_mass", 1.0))
        lam_c, lam_m = float(cfg.get("lambda_comp", 1.0)), float(cfg.get("lambda_modifier", 1.0))
        d_comp, d_mod, closest = None, None, None
        for a in old_classes:
            dc = _js_divergence(comp.get(a, {}), comp.get(b, {}), min_mass)
            dm = _js_divergence(modp.get(a, {}), modp.get(b, {}), min_mass)
            if d_comp is None or lam_c * dc + lam_m * dm < lam_c * d_comp + lam_m * d_mod:
                d_comp, d_mod, closest = dc, dm, a
        if d_comp is None:  # first entry of the word: nothing to duplicate
            d_comp, d_mod = 1.0, 1.0
        # derivability gain: share of the new entry's leaf mass in sentences that the *old* grammar could not
        # derive as one root (posterior root mass < 1/2).  A new entry whose occurrences were all already
        # derivable only re-routes existing derivations through other rules: a duplicate analysis in the
        # sense that matters for the grammar, whatever its composition profile looks like.
        root_old = self._root_mass()
        gain, mass_b = 0.0, 0.0
        for sid in affected:
            nodes = out_aff.node_posteriors.get(sid)
            if not nodes:
                continue
            toks = self.tokens[sid]
            for i, j, c, p in nodes:
                if j - i == 1 and c == b and int(toks[i]) == w:
                    mass_b += p
                    if root_old.get(sid, 0.0) < 0.5:
                        gain += p
        d_cover = gain / mass_b if mass_b > 0 else 0.0
        combine = str(cfg.get("combine", "gated"))
        if combine == "sum":  # the first-version linear form
            d_struct = lam_c * d_comp + lam_m * d_mod
        elif combine == "cover":
            d_struct = d_cover
        else:  # gated: profile distance only counts where the entry adds derivations
            d_struct = d_cover * (lam_c * d_comp + lam_m * d_mod) / max(1e-9, lam_c + lam_m)
        penalty = float(cfg.get("lambda_dup", 10.0)) * math.exp(-d_struct / float(cfg.get("tau", 0.1)))
        names = self.model.names

        def top(d: dict[Any, float], n: int = 5) -> dict[str, float]:
            return {" ".join(str(names.get(x, x)) if isinstance(x, int) else str(x) for x in k): round(v, 2) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]}

        return {
            "old_classes": [names.get(a, a) for a in old_classes],
            "closest_old_class": names.get(closest, closest) if closest is not None else None,
            "composition_distance": round(d_comp, 4),
            "modifier_distance": round(d_mod, 4),
            "derivability_gain": round(d_cover, 4),
            "new_entry_mass": round(mass_b, 2),
            "structural_distance": round(d_struct, 4),
            "combine": combine,
            "duplicate_penalty": penalty,
            "profiles": {"new": {"composition": top(comp.get(b, {})), "modifier": top(modp.get(b, {})), "mass": round(sum(comp.get(b, {}).values()), 2)},
                         "closest_old": {"composition": top(comp.get(closest, {})), "modifier": top(modp.get(closest, {})), "mass": round(sum(comp.get(closest, {}).values()), 2)} if closest is not None else None},
        }

    def _root_mass(self) -> dict[int, float]:
        """Per sentence: posterior mass of a single-root analysis under the current (pre-move) posterior."""
        if getattr(self, "_root_mass_cache", None) is None or self._root_mass_version != id(self.out):
            rm: dict[int, float] = {}
            if self.out is not None:
                for sid, nodes in self.out.node_posteriors.items():
                    n = self.n_lengths[sid]
                    rm[sid] = sum(p for i, j, c, p in nodes if i == 0 and j == n)
            self._root_mass_cache = rm
            self._root_mass_version = id(self.out)
        return self._root_mass_cache

    def _commit_check(self, before: float, label: str) -> bool:
        """After accepted moves: full CAVI, full re-score; F must not drop beyond tolerance."""
        snap = self.model.checkpoint()
        class_map_snap = {k: list(v) for k, v in self.vi.class_map.items()}
        self.full_cavi(label, int(self.cfg.get("post_move_cavi_iterations", 10)))
        tol = float(self.cfg.get("objective_tolerance", 1e-6)) * max(1.0, abs(before))
        if self.F < before - tol:
            self.monotonicity_violations += 1
            log.warning("objective decreased after %s (%.3f -> %.3f): rolling back", label, before, self.F)
            self.moves.append({"kind": "rollback", "round": self.rounds_done, "reason": f"full objective decreased after {label}", "old_objective": before, "new_objective": self.F, "accepted": False})
            self.model.restore(snap)
            self.vi.class_map = class_map_snap
            return False
        return True

    # ------------------------------------------------------------------ main loop
    def run(self) -> list[dict[str, Any]]:
        cfg = self.cfg
        max_rounds = int(cfg.get("max_rounds", 6))
        max_per_stage = int(cfg.get("max_accepts_per_stage", 2))
        if self.ec is None:
            self.full_cavi("cavi_init")
        for rnd in range(max_rounds):
            self.rounds_done = rnd
            any_accept = False
            stages = []
            if cfg.get("merge_enabled", True):
                stages.append(("merge", self.propose_merges))
            if cfg.get("split_enabled", True):
                stages.append(("split", self.propose_splits))
            if cfg.get("rule_proposals_enabled", True):
                stages.append(("prune", self.propose_rule_moves))
                stages.append(("lexicon", self.propose_lex_adds))
                stages.append(("rule_add", self.propose_rule_adds))
            for stage, proposer in stages:
                n_acc = 0
                tried = 0
                while n_acc < max_per_stage:
                    moves = proposer()
                    if not moves:
                        break
                    accepted_one = False
                    moves = [mv for mv in moves if self.move_key(mv) not in self.rejected]
                    n_before = len(moves)
                    moves = [mv for mv in moves if not self.memo_skip(mv)]
                    self.n_memo_skipped += n_before - len(moves)
                    if not moves:
                        break
                    for mv in moves:
                        tried += 1
                        rec = self.evaluate(mv)
                        if rec["accepted"]:
                            before_F = self.F
                            if self._commit_check(before_F, f"post_{mv.kind}"):
                                n_acc += 1
                                any_accept = True
                                accepted_one = True
                            else:
                                self.rescore("after_rollback")
                            break
                        if tried >= int(cfg.get("max_proposals_per_stage", 12)):
                            break
                    if not accepted_one or tried >= int(cfg.get("max_proposals_per_stage", 12)):
                        break
            log.info("structure round %d: %s (F=%.3f, classes %d, rules %d; %d remembered rejections skipped so far)", rnd, "moves accepted" if any_accept else "no improving move", self.F, self.model.n_classes, len(self.model.active_rules()), self.n_memo_skipped)
            if not any_accept:
                break
        self.rounds_done += 1
        return self.moves

    # ------------------------------------------------------------------ reporting helpers
    def abstract_context_report(self, names: dict[int, str], top: int = 6) -> dict[str, list[str]]:
        if self.abstract is None:
            return {}
        out = {}
        for c, feats in self.abstract.class_features.items():
            items = sorted(feats.items(), key=lambda kv: -kv[1])[:top]
            out[names.get(c, str(c))] = [f"{render_feature(f, names)} ({w:.1f})" for f, w in items]
        return out


# --------------------------------------------------------------------------- weighted 2-means
def _weighted_two_means(feats: list[dict[Any, float]], weights: np.ndarray, seed: int = 0) -> tuple[np.ndarray | None, float, float]:
    """Weighted 2-means on L2-normalised sparse feature rows.

    Returns ``(labels, quality, balance)`` with ``quality = 1 - SS_within / SS_total``
    (fraction of weighted variance explained) and ``balance = min cluster share * 2``.
    """
    from sklearn.cluster import KMeans

    feat_ids: dict[Any, int] = {}
    rows, cols, vals = [], [], []
    for r, f in enumerate(feats):
        for k, v in f.items():
            j = feat_ids.setdefault(k, len(feat_ids))
            rows.append(r)
            cols.append(j)
            vals.append(v)
    if len(feat_ids) < 2 or len(feats) < 4:
        return None, 0.0, 0.0
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(feats), len(feat_ids)))
    # idf-like down-weighting of ubiquitous features, then L2 rows
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((1.0 + len(feats)) / (1.0 + df)) + 0.1
    X = sp.csr_matrix(X @ sp.diags(idf))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = sp.csr_matrix(sp.diags(1.0 / norms) @ X)
    km = KMeans(n_clusters=2, n_init=5, random_state=seed)
    # k = 2 on a few thousand sparse rows is a tiny problem: OpenMP worker threads only add
    # contention (50x slower when several experiments share the machine), so run it single-threaded
    with threadpool_limits(limits=1):
        labels = km.fit_predict(X, sample_weight=weights)
    w = weights / weights.sum()
    mean = np.asarray(X.multiply(w[:, None]).sum(axis=0)).ravel()
    # weighted sums of squared distances without per-row sparse slicing:
    # sum_r w_r ||x_r - m||^2 = sum_r w_r (||x_r||^2 - 2 x_r.m + ||m||^2)
    sq = np.asarray(X.multiply(X).sum(axis=1)).ravel()

    def wss(idx: np.ndarray, centre: np.ndarray) -> float:
        xm = X[idx] @ centre
        return float((w[idx] * (sq[idx] - 2.0 * xm + float(centre @ centre))).sum())

    ss_total = wss(np.arange(X.shape[0]), mean)
    ss_within = 0.0
    for lab in (0, 1):
        idx = np.nonzero(labels == lab)[0]
        if len(idx) == 0:
            continue
        wl = w[idx] / w[idx].sum()
        cen = np.asarray(X[idx].multiply(wl[:, None]).sum(axis=0)).ravel()
        ss_within += wss(idx, cen)
    quality = 1.0 - ss_within / ss_total if ss_total > 0 else 0.0
    share = float(min(w[labels == 0].sum(), w[labels == 1].sum()))
    return labels, quality, 2.0 * share
