"""Gold evaluation of identity-preserving (modifier) evidence on the synthetic grammars.

Gold modifier structure is read off the gold derivations: a rule ``A + B -> C``
with ``C == A`` makes its right child an endomorphic (RIGHT) modifier of host
``A``; ``C == B`` makes the left child a LEFT modifier of host ``B``.  Nothing
else counts.  Candidate strings are judged at the type level (a string is a
gold modifier when at least half of its gold-span occurrences are modifier
children), which is the level at which the evidence tables report them.

Reported per source (ROOT_STRING, CONSTITUENT_STRING, CLASS_GENERALIZED,
POSTERIOR_RULE) and COMBINED:

* candidate precision / recall / F1 at several support thresholds,
* LEFT / RIGHT direction accuracy on true positives,
* host-class accuracy (class-level sources, after the run's class alignment;
  ROOT_STRING always names the sentence, which is reported but not an error),
* INTERNAL (insertion) precision / recall and the gold attachment level of the
  insertion pairs (host inside the left context, inside the right context, or
  the whole reduced string).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .modifier_evidence import ROOT, SOURCES, EvidenceTable
from .synthetic import SyntheticDataset


def gold_modifiers(ds: SyntheticDataset) -> dict[str, Any]:
    """Gold modifier occurrences and type-level labels from the gold derivations."""
    occ: dict[tuple[int, int, int], dict[str, Any]] = {}  # (sid, i, j) -> {direction, host, host_span, rule}
    per_string_mod: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    per_string_span: Counter = Counter()
    internal_strings: set[tuple[str, ...]] = set()
    for sid, gs in enumerate(ds.sentences):
        spans = {(g.start, g.end): g.cls for g in gs.spans if not g.noisy}
        n = len(gs.tokens)
        for (i, j), cls in spans.items():
            per_string_span[tuple(gs.tokens[i:j])] += 1
            if j - i < 2:
                continue
            for k in range(i + 1, j):
                a, b = spans.get((i, k)), spans.get((k, j))
                if a is None or b is None:
                    continue
                if cls == a and cls != b:
                    occ[(sid, k, j)] = {"direction": "RIGHT", "host": a, "host_span": (i, k), "rule": (a, b, cls)}
                    per_string_mod[tuple(gs.tokens[k:j])][("RIGHT", a)] += 1
                    if k > 0 and j < n:
                        internal_strings.add(tuple(gs.tokens[k:j]))
                elif cls == b and cls != a:
                    occ[(sid, i, k)] = {"direction": "LEFT", "host": b, "host_span": (k, j), "rule": (a, b, cls)}
                    per_string_mod[tuple(gs.tokens[i:k])][("LEFT", b)] += 1
                    if i > 0 and k < n:
                        internal_strings.add(tuple(gs.tokens[i:k]))
                break
    types: dict[str, dict[str, Any]] = {}
    for s, cnt in per_string_mod.items():
        n_mod = sum(cnt.values())
        n_span = per_string_span.get(s, n_mod)
        (direction, host), _ = cnt.most_common(1)[0]
        types[" ".join(s)] = {"frac": n_mod / max(1, n_span), "direction": direction, "host": host, "n_modifier_occurrences": n_mod, "n_gold_span_occurrences": n_span, "internal": s in internal_strings}
    gold_types = {s: d for s, d in types.items() if d["frac"] >= 0.5}
    rules = sorted({str(d["rule"]) for d in occ.values()})
    return {"occurrences": occ, "types": types, "gold_types": gold_types, "n_gold_types": len(gold_types), "gold_rules": rules}


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def evaluate_modifiers(table: EvidenceTable, gold: dict[str, Any], class_to_gold: dict[int, str], min_mod: float = 0.5, support_levels: tuple[float, ...] = (1, 2, 3, 5)) -> dict[str, Any]:
    gold_types = gold["gold_types"]
    gold_set = set(gold_types)
    gold_internal = {s for s, d in gold_types.items() if d["internal"]}
    out: dict[str, Any] = {"n_gold_types": len(gold_set), "n_gold_internal_types": len(gold_internal), "gold_rules": gold["gold_rules"], "by_source": {}}
    views: dict[str, list[dict[str, Any]]] = {s: [r for r in table.rows if r["source"] == s] for s in SOURCES}
    views["COMBINED"] = list(table.rows)
    occ_gold = gold["occurrences"]
    for name, rows in views.items():
        res: dict[str, Any] = {"n_rows": len(rows)}
        mm = min_mod if name in ("CLASS_GENERALIZED", "POSTERIOR_RULE") else None
        for lvl in support_levels:
            r = _eval_rows(rows, gold_types, gold_set, gold_internal, class_to_gold, float(lvl), mm)
            r["constituent_candidates_only"] = {k: v for k, v in _eval_rows([x for x in rows if x["candidate_constituency"] >= 0.5], gold_types, gold_set, gold_internal, class_to_gold, float(lvl), mm).items() if k in ("precision", "recall", "f1", "n_pred", "direction_accuracy", "host_accuracy")}
            for d in ("LEFT", "RIGHT"):
                sub = [x for x in rows if x["direction"] == d]
                gold_d = {c for c, g in gold_types.items() if g["direction"] == d}
                rd = _eval_rows(sub, gold_types, gold_d, set(), class_to_gold, float(lvl), mm)
                r[d] = {k: rd[k] for k in ("precision", "recall", "f1", "n_pred", "n_gold")}
            r["occurrence_precision"] = _occurrence_precision(rows, occ_gold, float(lvl), mm)
            res[f"support>={lvl:g}"] = r
        out["by_source"][name] = res
    # attachment level of the root insertion pairs
    out["insertion_attachment"] = _attachment(table, gold["occurrences"])
    return out


def _eval_rows(rows: list[dict[str, Any]], gold_types: dict[str, Any], gold_set: set[str], gold_internal: set[str], class_to_gold: dict[int, str], min_support: float, min_mod: float | None) -> dict[str, Any]:
    pred_dir: dict[str, Counter] = defaultdict(Counter)  # candidate -> direction support (LEFT/RIGHT)
    pred_host: dict[str, Counter] = defaultdict(Counter)
    pred_internal: dict[str, float] = defaultdict(float)
    root_only: set[str] = set()
    for r in rows:
        if r["support"] < min_support:
            continue
        if min_mod is not None and r.get("mod") is not None and r["mod"] < min_mod:
            continue
        c = r["candidate"]
        if r["direction"] == "INTERNAL":
            pred_internal[c] += r["support"]
            continue
        pred_dir[c][r["direction"]] += r["support"]
        if r["host_class"] is None:
            pred_host[c][ROOT] += r["support"]
            root_only.add(c)
        else:
            pred_host[c][class_to_gold.get(r["host_class"], f"C{r['host_class']}")] += r["support"]
    pred = set(pred_dir)
    tp = pred & gold_set
    m = _prf(len(tp), len(pred), len(gold_set))
    # direction accuracy on true positives
    dir_ok = sum(1 for c in tp if pred_dir[c].most_common(1)[0][0] == gold_types[c]["direction"])
    m["direction_accuracy"] = round(dir_ok / len(tp), 4) if tp else None
    # host accuracy (class-level hosts only; ROOT hosts counted separately)
    cls_tp = [c for c in tp if pred_host[c] and pred_host[c].most_common(1)[0][0] != ROOT]
    host_ok = sum(1 for c in cls_tp if pred_host[c].most_common(1)[0][0] == gold_types[c]["host"])
    m["host_accuracy"] = round(host_ok / len(cls_tp), 4) if cls_tp else None
    m["n_host_evaluated"] = len(cls_tp)
    m["n_root_hosted"] = sum(1 for c in tp if c in root_only)
    # insertion
    pi = set(pred_internal)
    tpi = pi & gold_set
    m["insertion"] = _prf(len(tpi), len(pi), len(gold_internal))
    m["false_positives"] = sorted(pred - gold_set, key=lambda c: -sum(pred_dir[c].values()))[:12]
    m["missed"] = sorted(gold_set - pred, key=lambda c: -gold_types[c]["n_modifier_occurrences"])[:12]
    m["false_insertions"] = sorted(pi - gold_set, key=lambda c: -pred_internal[c])[:10]
    return m


def _occurrence_precision(rows: list[dict[str, Any]], occ_gold: dict[tuple[int, int, int], dict[str, Any]], min_support: float, min_mod: float | None) -> dict[str, Any]:
    """Share of evidence occurrences (candidate positions) that are gold modifier occurrences, per direction."""
    tot: Counter = Counter()
    ok: Counter = Counter()
    dir_ok: Counter = Counter()
    for r in rows:
        if r["support"] < min_support or (min_mod is not None and r.get("mod") is not None and r["mod"] < min_mod):
            continue
        for key in r.get("_cand_occ", []):
            d = r["direction"]
            tot[d] += 1
            g = occ_gold.get(tuple(key))
            if g is not None:
                ok[d] += 1
                if d == "INTERNAL" or g["direction"] == d:
                    dir_ok[d] += 1
    out = {}
    for d in ("LEFT", "RIGHT", "INTERNAL"):
        if tot[d]:
            out[d] = {"precision": round(ok[d] / tot[d], 4), "direction_consistent": round(dir_ok[d] / tot[d], 4), "n_occurrences": tot[d]}
    n = sum(tot.values())
    out["ALL"] = {"precision": round(sum(ok.values()) / n, 4) if n else None, "n_occurrences": n}
    return out


def _attachment(table: EvidenceTable, occ: dict[tuple[int, int, int], dict[str, Any]]) -> dict[str, Any]:
    """For ROOT_STRING INTERNAL evidence occurrences: is the candidate a gold modifier, and where does its gold host sit?"""
    counts: Counter = Counter()
    n_total = 0
    for r in table.rows:
        if r["direction"] != "INTERNAL" or r["source"] != "ROOT_STRING":
            continue
        for key in r.get("_cand_occ", []):
            n_total += 1
            d = occ.get(tuple(key))
            if d is None:
                counts["not_a_gold_modifier"] += 1
                continue
            sid, i, j = key
            hi, hj = d["host_span"]
            if hj == i:
                counts["host_in_left_context"] += 1
            elif hi == j:
                counts["host_in_right_context"] += 1
            else:
                counts["host_spans_insertion"] += 1
    return {"n_internal_root_occurrences": n_total, "attachment": dict(counts)}
