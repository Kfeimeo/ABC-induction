"""Identity-preserving (endomorphic) constituent evidence from raw text.

The question of this module is whether *category-preserving insertion /
deletion* is a usable structural signal:  a span ``M`` is a *modifier
candidate* of a host ``C`` when ``C + M`` (or ``M + C``) behaves like ``C``.
Nothing here claims linguistic modifier status; the evidence is recorded
separately per source so that string-anchored and class-generalised evidence
can be compared:

``ROOT_STRING``
    sentence-level exact-string evidence.  ``A.`` and ``AB.`` both occur as
    complete sentences (``A`` the identical token sequence) -> ``B`` gets
    RIGHT evidence with host *sentence*;  ``A.`` / ``XA.`` -> ``X`` LEFT;
    ``AB.`` / ``AXB.`` -> ``X`` INTERNAL (attachment level unknown).
    Independent of the induced classes: the most trustworthy source.
``CONSTITUENT_STRING``
    the same deletion test on posterior constituents below the root:
    ``A`` and ``AB`` (identical ``A`` string) are both constituents with
    ``P(Q) > tau_q`` somewhere in the corpus.  The host string is unchanged;
    only the constituency decision comes from the model.
``CLASS_GENERALIZED``
    ``E(A') = E(A'B) = C`` inside one sentence for *any* host ``A'`` of class
    ``C`` (posterior class of both spans above ``tau_q``); INTERNAL:
    ``E(AB) = E(AXB) = C`` with the reduced string a constituent of class
    ``C`` elsewhere.  High coverage, but inherits every false merge.
``POSTERIOR_RULE``
    the Inside-Outside expected counts of the active endomorphic rules
    ``C + M -> C`` / ``M + C -> C`` (rule applications above the output
    threshold).  Probabilistically the cleanest statistic, but only for rules
    the grammar already has.

Per (candidate, host, direction, source) the table keeps ``support``
(evidence occurrences), ``posterior_mass`` (sum of the posterior weight of
the evidence; for the string sources the constituency / sentence weight is 1),
``distinct_host_types`` (productivity), ``distinct_host_occurrences`` and,
for class-level sources, ``mod`` = ``E[N_{C,M->C}] / sum_Y E[N_{C,M->Y}]``.
Marginals are never multiplied: class-generalised evidence is thresholded
and its mass is the smaller of the two node posteriors.

Punctuation tokens are sentence-boundary observations, never candidates.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .corpus import Corpus
from .grammar_model import GrammarModel
from .spans import SpanInventory

log = logging.getLogger(__name__)

PUNCT = {".", "!", "?", ",", ";", ":", '"', "'", "-", "(", ")"}
SOURCES = ("ROOT_STRING", "CONSTITUENT_STRING", "CLASS_GENERALIZED", "POSTERIOR_RULE")
DIRECTIONS = ("LEFT", "RIGHT", "INTERNAL")
ROOT = "ROOT"

DEFAULT_CFG: dict[str, Any] = {
    "mode": "off",  # off | string_only | string_plus_class | posterior_rule_only  (which sources feed rule proposals)
    "diagnostics": True,  # collect and write the evidence tables (never touches the objective)
    "max_modifier_length": 3,  # longest removed prefix / suffix (tokens)
    "max_insert_span_length": 4,  # longest removed internal span
    "constituent_threshold": 0.8,  # tau_q for CONSTITUENT_STRING / CLASS_GENERALIZED
    "min_support": 2,  # candidate reported when support >= min_support
    "min_mod": 0.5,  # class-level candidates also need mod >= min_mod
    "max_rule_proposals": 6,
    "proposal_min_support": 3,
}


# --------------------------------------------------------------------------- records
@dataclass
class _Acc:
    support: float = 0.0
    mass: float = 0.0
    hosts: set = field(default_factory=set)  # host strings (productivity)
    host_occ: set = field(default_factory=set)  # (sid, i, j) of the host occurrence
    cand_occ: set = field(default_factory=set)  # (sid, i, j) of the candidate occurrence
    host_classes: dict = field(default_factory=lambda: defaultdict(float))
    cand_classes: dict = field(default_factory=lambda: defaultdict(float))
    examples: list = field(default_factory=list)
    agree: float = 0.0  # CONSTITUENT_STRING: mass where E(A) == E(AB)


@dataclass
class EvidenceTable:
    rows: list[dict[str, Any]]
    pairs: list[dict[str, Any]]  # example string pairs (A., AB.) etc.
    by_candidate: dict[str, dict[str, Any]]
    stats: dict[str, Any]

    def candidates(self, source: str | None = None, min_support: float = 2.0, min_mod: float | None = None, direction: str | None = None) -> list[dict[str, Any]]:
        out = []
        for r in self.rows:
            if source is not None and r["source"] != source:
                continue
            if direction is not None and r["direction"] != direction:
                continue
            if r["support"] < min_support:
                continue
            if min_mod is not None and r.get("mod") is not None and r["mod"] < min_mod:
                continue
            out.append(r)
        return out


# --------------------------------------------------------------------------- analyzer
class ModifierAnalyzer:
    """Collects identity-preserving evidence for one corpus; the string indexes are built once."""

    def __init__(self, corpus: Corpus, inv: SpanInventory, cfg: dict[str, Any] | None = None) -> None:
        self.corpus = corpus
        self.inv = inv
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.punct_ids = {i for i, w in enumerate(corpus.itos) if w in PUNCT}
        self.max_mod = int(self.cfg["max_modifier_length"])
        self.max_ins = int(self.cfg["max_insert_span_length"])
        self.tau_q = float(self.cfg["constituent_threshold"])
        # sentence string (punctuation stripped) -> sids ; plus the stripped token tuples
        self.stripped: list[tuple[int, ...]] = []
        self.sent_index: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for sid, ids in enumerate(corpus.token_ids):
            toks = tuple(int(t) for t in ids)
            core = self._strip(toks)
            self.stripped.append(core)
            if core:
                self.sent_index[core].append(sid)

    # ------------------------------------------------------------------ helpers
    def _strip(self, toks: tuple[int, ...]) -> tuple[int, ...]:
        """Remove sentence-final punctuation (one or more tokens)."""
        n = len(toks)
        while n > 0 and toks[n - 1] in self.punct_ids:
            n -= 1
        return toks[:n]

    def _is_punct_span(self, toks: Iterable[int]) -> bool:
        toks = list(toks)
        return bool(toks) and all(t in self.punct_ids for t in toks)

    def _s(self, toks: Iterable[int]) -> str:
        return self.corpus.render(toks)

    # ------------------------------------------------------------------ main entry
    def collect(self, out: Any, model: GrammarModel, names: dict[int, str] | None = None, sources: Iterable[str] = SOURCES) -> EvidenceTable:
        """Evidence table from the posterior ``out`` (an ``EStepOutput`` with node posteriors and rule applications)."""
        names = names or model.names
        sources = set(sources)
        node: dict[tuple[int, int, int], dict[int, float]] = {}
        qmass: dict[tuple[int, int, int], float] = {}
        for sid, nodes in out.node_posteriors.items():
            for i, j, c, p in nodes:
                d = node.setdefault((sid, i, j), {})
                d[c] = d.get(c, 0.0) + p
                qmass[(sid, i, j)] = qmass.get((sid, i, j), 0.0) + p
        self._node, self._q = node, qmass
        acc: dict[tuple, _Acc] = defaultdict(_Acc)
        pairs: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        if "ROOT_STRING" in sources:
            stats["root_string"] = self._root_string(acc, pairs)
        if "CONSTITUENT_STRING" in sources or "CLASS_GENERALIZED" in sources:
            cindex = self._constituent_index()
            if "CONSTITUENT_STRING" in sources:
                stats["constituent_string"] = self._constituent_string(acc, cindex, pairs)
            if "CLASS_GENERALIZED" in sources:
                stats["class_generalized"] = self._class_generalized(acc, cindex)
        if "POSTERIOR_RULE" in sources:
            stats["posterior_rule"] = self._posterior_rule(acc, out, model)
        rows = self._rows(acc, model, names)
        by_cand = self._combine(rows)
        stats["n_rows"] = len(rows)
        stats["n_candidates"] = len(by_cand)
        return EvidenceTable(rows, pairs[:400], by_cand, stats)

    # ------------------------------------------------------------------ A. root string
    def _root_string(self, acc: dict, pairs: list) -> dict[str, int]:
        n_pairs = {"RIGHT": 0, "LEFT": 0, "INTERNAL": 0}
        seen_pairs: set[tuple] = set()
        for s, sids in self.sent_index.items():
            n = len(s)
            if n < 2:
                continue
            w = float(len(sids))
            sid0 = sids[0]
            # right insertion: A. / AB.
            for k in range(1, min(self.max_mod, n - 1) + 1):
                a, b = s[:-k], s[-k:]
                if a in self.sent_index and not self._is_punct_span(b):
                    self._add(acc, ("ROOT_STRING", b, ROOT, "RIGHT"), w, w, a, (sids[0], 0, n - k), [(sd, n - k, n) for sd in sids], host_class=None, example=(self._s(a), self._s(s)))
                    n_pairs["RIGHT"] += 1
                    if (a, s) not in seen_pairs and len(pairs) < 2000:
                        seen_pairs.add((a, s))
                        pairs.append({"kind": "root_right", "short": self._s(a), "long": self._s(s), "candidate": self._s(b), "n_short": len(self.sent_index[a]), "n_long": len(sids)})
            # left insertion: A. / XA.
            for k in range(1, min(self.max_mod, n - 1) + 1):
                x, a = s[:k], s[k:]
                if a in self.sent_index and not self._is_punct_span(x):
                    self._add(acc, ("ROOT_STRING", x, ROOT, "LEFT"), w, w, a, (sids[0], k, n), [(sd, 0, k) for sd in sids], host_class=None, example=(self._s(a), self._s(s)))
                    n_pairs["LEFT"] += 1
                    if (a, s) not in seen_pairs and len(pairs) < 2000:
                        seen_pairs.add((a, s))
                        pairs.append({"kind": "root_left", "short": self._s(a), "long": self._s(s), "candidate": self._s(x), "n_short": len(self.sent_index[a]), "n_long": len(sids)})
            # internal insertion: AB. / AXB.
            for i in range(1, n - 1):
                for l in range(1, self.max_ins + 1):
                    j = i + l
                    if j >= n:
                        break
                    x = s[i:j]
                    red = s[:i] + s[j:]
                    if red in self.sent_index and not self._is_punct_span(x):
                        self._add(acc, ("ROOT_STRING", x, ROOT, "INTERNAL"), w, w, red, (sids[0], 0, n), [(sd, i, j) for sd in sids], host_class=None, example=(self._s(red), self._s(s)))
                        n_pairs["INTERNAL"] += 1
                        if (red, s) not in seen_pairs and len(pairs) < 2000:
                            seen_pairs.add((red, s))
                            pairs.append({"kind": "root_internal", "short": self._s(red), "long": self._s(s), "candidate": self._s(x), "left_context": self._s(s[:i]), "right_context": self._s(s[j:]), "n_short": len(self.sent_index[red]), "n_long": len(sids)})
            sid0 = sid0  # noqa: F841 (kept for readability)
        return n_pairs

    # ------------------------------------------------------------------ constituent index
    def _constituent_index(self) -> dict[tuple[int, ...], list[tuple[int, int, int, int, float]]]:
        """string -> [(sid, i, j, argmax class, P(Q))] for posterior constituents with P(Q) >= tau_q (sub-sentential)."""
        idx: dict[tuple[int, ...], list[tuple[int, int, int, int, float]]] = defaultdict(list)
        for (sid, i, j), q in self._q.items():
            if q < self.tau_q:
                continue
            toks = tuple(int(t) for t in self.corpus.token_ids[sid][i:j])
            if self._is_punct_span(toks):
                continue
            d = self._node[(sid, i, j)]
            c = max(d.items(), key=lambda kv: kv[1])[0]
            idx[toks].append((sid, i, j, c, q))
        return idx

    # ------------------------------------------------------------------ A'. constituent string
    def _constituent_string(self, acc: dict, cindex: dict, pairs: list) -> dict[str, int]:
        n_pairs = {"RIGHT": 0, "LEFT": 0, "INTERNAL": 0}
        n_changing = 0  # (A, AB) constituent pairs whose posterior class differs: string-identical but not category-preserving
        for ab, occs in cindex.items():
            n = len(ab)
            if n < 2:
                continue
            # skip full-sentence occurrences (those are ROOT_STRING evidence)
            sub = [(sid, i, j, c, q) for sid, i, j, c, q in occs if not (i == 0 and j == len(self.stripped[sid]))]
            if not sub:
                continue
            for k in range(1, min(self.max_mod, n - 1) + 1):
                for direction, host, cand, cpos in (("RIGHT", ab[:-k], ab[-k:], lambda i, j: (i + n - k, j)), ("LEFT", ab[k:], ab[:k], lambda i, j: (i, i + k))):
                    hocc = cindex.get(host)
                    if not hocc or self._is_punct_span(cand):
                        continue
                    host_cls = defaultdict(float)
                    for _, _, _, c, q in hocc:
                        host_cls[c] += q
                    hc = max(host_cls.items(), key=lambda kv: kv[1])[0]
                    keep = [(sid, i, j, c, q) for sid, i, j, c, q in sub if c == hc]  # E(AB) == E(A): category preserved
                    n_changing += len(sub) - len(keep)
                    if not keep:
                        continue
                    self._add(acc, ("CONSTITUENT_STRING", cand, hc, direction), float(len(keep)), float(sum(q for *_, q in keep)), host, (hocc[0][0], hocc[0][1], hocc[0][2]), [(sid,) + cpos(i, j) for sid, i, j, _, _ in keep], host_class=hc, example=(self._s(host), self._s(ab)), agree=float(len(keep)) / len(sub))
                    n_pairs[direction] += 1
                    if len(pairs) < 4000:
                        pairs.append({"kind": f"constituent_{direction.lower()}", "short": self._s(host), "long": self._s(ab), "candidate": self._s(cand), "n_short": len(hocc), "n_long": len(sub), "n_class_preserving": len(keep), "host_class": hc})
            for i in range(1, n - 1):
                for l in range(1, self.max_ins + 1):
                    j = i + l
                    if j >= n:
                        break
                    x = ab[i:j]
                    red = ab[:i] + ab[j:]
                    hocc = cindex.get(red)
                    if not hocc or self._is_punct_span(x):
                        continue
                    host_cls = defaultdict(float)
                    for _, _, _, c, q in hocc:
                        host_cls[c] += q
                    hc = max(host_cls.items(), key=lambda kv: kv[1])[0]
                    keep = [(sid, i0, j0, c, q) for sid, i0, j0, c, q in sub if c == hc]
                    n_changing += len(sub) - len(keep)
                    if not keep:
                        continue
                    self._add(acc, ("CONSTITUENT_STRING", x, hc, "INTERNAL"), float(len(keep)), float(sum(q for *_, q in keep)), red, (hocc[0][0], hocc[0][1], hocc[0][2]), [(sid, i0 + i, i0 + j) for sid, i0, _, _, _ in keep], host_class=hc, example=(self._s(red), self._s(ab)), agree=float(len(keep)) / len(sub))
                    n_pairs["INTERNAL"] += 1
        n_pairs["class_changing_occurrences"] = n_changing
        return n_pairs

    # ------------------------------------------------------------------ B. class generalised
    def _class_generalized(self, acc: dict, cindex: dict) -> dict[str, int]:
        n_ev = {"RIGHT": 0, "LEFT": 0, "INTERNAL": 0}
        by_sent: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
        for (sid, i, j), d in self._node.items():
            if self._q[(sid, i, j)] < self.tau_q:
                continue
            c, p = max(d.items(), key=lambda kv: kv[1])
            if p < self.tau_q:
                continue
            by_sent[sid].append((i, j, c, p))
        class_strings: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for toks, occs in cindex.items():
            for _, _, _, c, _ in occs:
                class_strings[c].add(toks)
        for sid, spans in by_sent.items():
            toks = self.corpus.token_ids[sid]
            at: dict[tuple[int, int], tuple[int, float]] = {(i, j): (c, p) for i, j, c, p in spans}
            for i, j, c, p in spans:
                # right: (i, k) of class c inside (i, j) -> candidate (k, j)
                for k in range(i + 1, j):
                    inner = at.get((i, k))
                    if inner and inner[0] == c and j - k <= self.max_mod:
                        cand = tuple(int(t) for t in toks[k:j])
                        if self._is_punct_span(cand):
                            continue
                        self._add(acc, ("CLASS_GENERALIZED", cand, c, "RIGHT"), 1.0, min(p, inner[1]), tuple(int(t) for t in toks[i:k]), (sid, i, k), [(sid, k, j)], host_class=c, example=(self._s(toks[i:k]), self._s(toks[i:j])))
                        n_ev["RIGHT"] += 1
                    inner = at.get((k, j))
                    if inner and inner[0] == c and k - i <= self.max_mod:
                        cand = tuple(int(t) for t in toks[i:k])
                        if self._is_punct_span(cand):
                            continue
                        self._add(acc, ("CLASS_GENERALIZED", cand, c, "LEFT"), 1.0, min(p, inner[1]), tuple(int(t) for t in toks[k:j]), (sid, k, j), [(sid, i, k)], host_class=c, example=(self._s(toks[k:j]), self._s(toks[i:j])))
                        n_ev["LEFT"] += 1
                # internal: E(AB) = E(AXB) = c with AB a constituent string of class c elsewhere
                strings_c = class_strings.get(c)
                if strings_c and j - i >= 3:
                    full = tuple(int(t) for t in toks[i:j])
                    for a in range(1, j - i - 1):
                        for l in range(1, self.max_ins + 1):
                            b = a + l
                            if b >= j - i:
                                break
                            red = full[:a] + full[b:]
                            if red in strings_c and not self._is_punct_span(full[a:b]):
                                self._add(acc, ("CLASS_GENERALIZED", full[a:b], c, "INTERNAL"), 1.0, p, red, (sid, i, j), [(sid, i + a, i + b)], host_class=c, example=(self._s(red), self._s(full)))
                                n_ev["INTERNAL"] += 1
        return n_ev

    # ------------------------------------------------------------------ C. posterior rule
    def _posterior_rule(self, acc: dict, out: Any, model: GrammarModel) -> dict[str, Any]:
        endo = {k for k in model.rules if k[0] == k[1] or k[0] == k[2]}
        n_apps = 0
        for sid, apps in out.rule_apps.items():
            toks = self.corpus.token_ids[sid]
            for r, i, k, j, p in apps:
                if r not in endo:
                    continue
                c, a, b = r
                if c == a and c != b:  # C + M -> C : right modifier M = (k, j)
                    cand = tuple(int(t) for t in toks[k:j])
                    if self._is_punct_span(cand):
                        continue
                    self._add(acc, ("POSTERIOR_RULE", cand, c, "RIGHT"), p, p, tuple(int(t) for t in toks[i:k]), (sid, i, k), [(sid, k, j)], host_class=c, example=(self._s(toks[i:k]), self._s(toks[i:j])), cand_classes={b: p})
                    n_apps += 1
                elif c == b and c != a:  # M + C -> C
                    cand = tuple(int(t) for t in toks[i:k])
                    if self._is_punct_span(cand):
                        continue
                    self._add(acc, ("POSTERIOR_RULE", cand, c, "LEFT"), p, p, tuple(int(t) for t in toks[k:j]), (sid, k, j), [(sid, i, k)], host_class=c, example=(self._s(toks[k:j]), self._s(toks[i:j])), cand_classes={a: p})
                    n_apps += 1
                # c == a == b (C + C -> C): both children preserve the category; recorded as RIGHT and LEFT
                elif c == a == b:
                    for direction, cand, host, cpos, hpos in (("RIGHT", toks[k:j], toks[i:k], (sid, k, j), (sid, i, k)), ("LEFT", toks[i:k], toks[k:j], (sid, i, k), (sid, k, j))):
                        cand = tuple(int(t) for t in cand)
                        if self._is_punct_span(cand):
                            continue
                        self._add(acc, ("POSTERIOR_RULE", cand, c, direction), p, p, tuple(int(t) for t in host), hpos, [cpos], host_class=c, example=(self._s(host), self._s(toks[i:j])), cand_classes={c: p})
                    n_apps += 1
        return {"n_endomorphic_rules": len(endo), "n_applications": n_apps}

    # ------------------------------------------------------------------ accumulation
    @staticmethod
    def _add(acc: dict, key: tuple, support: float, mass: float, host: tuple, host_occ: tuple, cand_occs: list, host_class: int | None, example: tuple, agree: float = 0.0, cand_classes: dict | None = None) -> None:
        a = acc[key]
        a.support += support
        a.mass += mass
        a.hosts.add(host)
        a.host_occ.add(host_occ)
        a.cand_occ.update(cand_occs)
        if host_class is not None:
            a.host_classes[host_class] += mass
        if cand_classes:
            for c, v in cand_classes.items():
                a.cand_classes[c] += v
        a.agree += agree
        if len(a.examples) < 4:
            a.examples.append(f"{example[0]}  ->  {example[1]}")

    def _rows(self, acc: dict, model: GrammarModel, names: dict[int, str]) -> list[dict[str, Any]]:
        # candidate class: dominant posterior class of the candidate occurrences
        rows = []
        rule_out: dict[tuple[int, int, str], float] = defaultdict(float)  # (host, cand class, direction) -> sum_Y E[N]
        for (c, a, b), n in model.rules.items():
            rule_out[(a, b, "RIGHT")] += n  # a + b -> Y : b is a right attachment to host a
            rule_out[(b, a, "LEFT")] += n  # a + b -> Y : a is a left attachment to host b
        for (source, cand, host, direction), a in acc.items():
            cand_cls: dict[int, float] = defaultdict(float)
            if a.cand_classes:
                cand_cls.update(a.cand_classes)
            else:
                for occ in a.cand_occ:
                    d = self._node.get(occ)
                    if d:
                        for c, p in d.items():
                            cand_cls[c] += p
            cc = max(cand_cls.items(), key=lambda kv: kv[1])[0] if cand_cls else None
            cq = (sum(cand_cls.values()) / max(1, len(a.cand_occ))) if cand_cls else 0.0
            host_class = None if host == ROOT else int(host)
            mod = None
            if host_class is not None and cc is not None and direction in ("LEFT", "RIGHT"):
                key = (host_class, cc, direction)
                endo_key = (host_class, host_class, cc) if direction == "RIGHT" else (host_class, cc, host_class)
                tot = rule_out.get(key, 0.0)
                mod = (model.rules.get(endo_key, 0.0) / tot) if tot > 0 else None
            rows.append({
                "candidate": self._s(cand),
                "candidate_tokens": list(cand),
                "candidate_class": cc,
                "candidate_class_name": names.get(cc, str(cc)) if cc is not None else None,
                "candidate_constituency": round(cq, 4),
                "host_class": host_class,
                "host_class_name": ROOT if host_class is None else names.get(host_class, str(host_class)),
                "direction": direction,
                "source": source,
                "support": round(a.support, 3),
                "posterior_mass": round(a.mass, 3),
                "distinct_host_types": len(a.hosts),
                "distinct_host_occurrences": len(a.host_occ),
                "distinct_candidate_occurrences": len(a.cand_occ),
                "string_exact_support": round(a.support, 3) if source in ("ROOT_STRING", "CONSTITUENT_STRING") else 0.0,
                "class_generalized_support": round(a.support, 3) if source == "CLASS_GENERALIZED" else 0.0,
                "posterior_rule_support": round(a.support, 3) if source == "POSTERIOR_RULE" else 0.0,
                "insertion_support": round(a.support, 3) if direction == "INTERNAL" else 0.0,
                "class_preserving_share": round(a.agree / max(1.0, len(a.hosts)), 4) if source == "CONSTITUENT_STRING" else None,
                "mod": round(mod, 4) if mod is not None else None,
                "productivity": len(a.hosts),
                "examples": a.examples,
                "_cand_occ": sorted(a.cand_occ),  # occurrence positions (not written to JSON)
            })
        rows.sort(key=lambda r: (-r["support"], r["candidate"]))
        return rows

    @staticmethod
    def _combine(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Per candidate string: support per source, dominant direction, hosts."""
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            d = out.setdefault(r["candidate"], {"candidate": r["candidate"], "support": {s: 0.0 for s in SOURCES}, "direction_support": {x: 0.0 for x in DIRECTIONS}, "hosts": defaultdict(float), "candidate_class": None, "productivity": 0, "mod": None})
            d["support"][r["source"]] += r["support"]
            d["direction_support"][r["direction"]] += r["support"]
            d["hosts"][f"{r['host_class_name']}:{r['direction']}:{r['source']}"] += r["support"]
            d["productivity"] = max(d["productivity"], r["productivity"])
            if r["candidate_class"] is not None and d["candidate_class"] is None:
                d["candidate_class"] = r["candidate_class_name"]
            if r.get("mod") is not None:
                d["mod"] = max(d["mod"] or 0.0, r["mod"])
        for d in out.values():
            d["hosts"] = dict(sorted(d["hosts"].items(), key=lambda kv: -kv[1])[:8])
            d["direction"] = max(d["direction_support"].items(), key=lambda kv: kv[1])[0]
            d["total_support"] = sum(d["support"].values())
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["total_support"]))


# --------------------------------------------------------------------------- proposals
def modifier_rule_candidates(table: EvidenceTable, model: GrammarModel, mode: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Endomorphic rules suggested by the evidence (for ``rule_add`` proposals; acceptance stays with dF).

    ``string_only``        ROOT_STRING + CONSTITUENT_STRING
    ``string_plus_class``  + CLASS_GENERALIZED
    ``posterior_rule_only`` nothing new (the rules already exist); kept as a control
    """
    if mode in ("off", "posterior_rule_only"):
        return []
    sources = ["ROOT_STRING", "CONSTITUENT_STRING"] + (["CLASS_GENERALIZED"] if mode == "string_plus_class" else [])
    min_support = float(cfg.get("proposal_min_support", 3))
    root_classes = sorted(model.root.items(), key=lambda kv: -kv[1])
    root_class = root_classes[0][0] if root_classes else None
    scored: dict[tuple[int, int, int], dict[str, Any]] = {}
    for r in table.rows:
        if r["source"] not in sources or r["direction"] == "INTERNAL" or r["support"] < min_support:
            continue
        m = r["candidate_class"]
        if m is None or r["candidate_constituency"] < 0.5:
            continue
        host = r["host_class"] if r["host_class"] is not None else root_class
        if host is None or host == m:
            continue
        key = (host, host, m) if r["direction"] == "RIGHT" else (host, m, host)
        if key in model.rules:
            continue
        d = scored.setdefault(key, {"rule": key, "support": 0.0, "sources": set(), "candidates": set(), "productivity": 0})
        d["support"] += r["support"]
        d["sources"].add(r["source"])
        d["candidates"].add(r["candidate"])
        d["productivity"] = max(d["productivity"], r["productivity"])
    out = sorted(scored.values(), key=lambda d: (-("ROOT_STRING" in d["sources"]), -d["support"]))
    for d in out:
        d["sources"] = sorted(d["sources"])
        d["candidates"] = sorted(d["candidates"])[:6]
    return out[: int(cfg.get("max_rule_proposals", 6))]


def table_to_json(table: EvidenceTable) -> dict[str, Any]:
    return {"stats": table.stats, "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in table.rows]}
