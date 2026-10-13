"""Evaluation against synthetic gold annotation.

* equivalence-class recovery (ARI, pairwise P/R/F1, purity, #classes),
* constituent discovery (span-level P/R/F1, by length),
* composition recovery after Hungarian class alignment,
* ambiguity detection (type-level P/R/F1, occurrence accuracy, false/missed
  splits),
* human-readable diagnostics: false merges, missed merges, false constituents.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .clustering import ClassState, MergeEvent
from .composition import CompositionRelation
from .corpus import Corpus
from .spans import SpanInventory
from .synthetic import SyntheticDataset


@dataclass
class GoldInfo:
    """Gold annotation projected onto the candidate span inventory."""

    occ_gold: list[str | None]
    occ_is_const: np.ndarray
    type_gold_class: dict[int, str]
    type_const_frac: dict[int, float]
    gold_spans_by_len: Counter
    gold_spans_total: int
    gold_spans_in_inventory: int
    gold_spans_in_inventory_by_len: Counter
    ambiguous_tids: dict[int, dict[str, float]]
    gold_relations: dict[tuple[str, str, str], int]
    class_names: list[str]
    gold_const_positions: set[tuple[int, int, int]] = field(default_factory=set)


def build_gold(ds: SyntheticDataset, corpus: Corpus, inv: SpanInventory, min_relation_support: int = 2) -> GoldInfo:
    assert len(corpus) == len(ds.sentences), "corpus and dataset must align 1:1"
    occ_gold: list[str | None] = [None] * inv.n_occurrences
    occ_is_const = np.zeros(inv.n_occurrences, dtype=bool)
    by_len: Counter = Counter()
    in_inv_by_len: Counter = Counter()
    total = 0
    in_inv = 0
    positions: set[tuple[int, int, int]] = set()
    for sid, gs in enumerate(ds.sentences):
        for g in gs.spans:
            if g.noisy:
                continue
            total += 1
            by_len[g.end - g.start] += 1
            positions.add((sid, g.start, g.end))
            oid = inv.lookup(sid, g.start, g.end)
            if oid >= 0:
                in_inv += 1
                in_inv_by_len[g.end - g.start] += 1
                occ_gold[oid] = g.cls
                occ_is_const[oid] = True
    type_gold: dict[int, str] = {}
    type_frac: dict[int, float] = {}
    for tid in range(inv.n_types):
        occs = inv.type_occ[tid]
        labels = [occ_gold[o] for o in occs.tolist() if occ_gold[o] is not None]
        type_frac[tid] = len(labels) / max(1, len(occs))
        if labels:
            type_gold[tid] = Counter(labels).most_common(1)[0][0]
    amb: dict[int, dict[str, float]] = {}
    for w, dist in ds.ambiguous_words.items():
        tid = corpus.vocab.get(w)
        if tid is None:
            continue
        key = (tid,)
        if key in inv.type_index:
            amb[inv.type_index[key]] = dist
    rels = {k: v for k, v in ds.relation_counts.items() if v >= min_relation_support}
    return GoldInfo(
        occ_gold, occ_is_const, type_gold, type_frac, by_len, total, in_inv, in_inv_by_len, amb, rels, list(ds.classes), positions
    )


# ------------------------------------------------------------------ helpers
def _prf(tp: float, n_pred: float, n_gold: float) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "n_pred": int(n_pred), "n_gold": int(n_gold), "tp": int(tp)}


def _pairwise(gold: list[Any], pred: list[Any]) -> dict[str, float]:
    cont: Counter = Counter(zip(gold, pred))
    gold_c: Counter = Counter(gold)
    pred_c: Counter = Counter(pred)
    c2 = lambda n: n * (n - 1) / 2.0
    tp = sum(c2(n) for n in cont.values())
    pp = sum(c2(n) for n in pred_c.values())
    gp = sum(c2(n) for n in gold_c.values())
    out = _prf(tp, pp, gp)
    # purity
    by_pred: dict[Any, Counter] = defaultdict(Counter)
    for (g, p), n in cont.items():
        by_pred[p][g] += n
    out["purity"] = round(sum(max(c.values()) for c in by_pred.values()) / max(1, len(gold)), 4)
    g_ids = {g: i for i, g in enumerate(gold_c)}
    p_ids = {p: i for i, p in enumerate(pred_c)}
    out["ari"] = round(float(adjusted_rand_score([g_ids[g] for g in gold], [p_ids[p] for p in pred])), 4)
    out["n_items"] = len(gold)
    out["n_pred_classes"] = len(pred_c)
    out["n_gold_classes"] = len(gold_c)
    return out


def hungarian_alignment(pairs: list[tuple[int, str]]) -> dict[int, str]:
    """One-to-one alignment of inferred class ids to gold class names
    maximising the shared item count."""
    cont: Counter = Counter(pairs)
    inf_ids = sorted({a for a, _ in cont})
    gold_ids = sorted({b for _, b in cont})
    if not inf_ids or not gold_ids:
        return {}
    m = np.zeros((len(inf_ids), len(gold_ids)))
    for (a, b), n in cont.items():
        m[inf_ids.index(a), gold_ids.index(b)] = n
    ri, ci = linear_sum_assignment(-m)
    return {inf_ids[i]: gold_ids[j] for i, j in zip(ri, ci) if m[i, j] > 0}


# ------------------------------------------------------------------ metrics
def evaluate(
    gold: GoldInfo,
    inv: SpanInventory,
    state: ClassState,
    type_classes: dict[int, list[int]],
    occ_class: np.ndarray,
    constituent_mask: np.ndarray,
    relations: list[CompositionRelation],
    split_tids: dict[int, dict[str, Any]],
    min_span_frequency: int,
    constituent_positions: set[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Compute all gold-based metrics.

    Args:
        type_classes: final accepted class ids per span type (empty = pruned).
        occ_class: final occurrence-level class ids (-1 = not a constituent).
        constituent_mask: derivation-level constituent occurrences.
        relations: accepted composition relations.
        split_tids: span types with accepted ambiguity splits.
    """
    metrics: dict[str, Any] = {}
    # ---------------------------------------------------- class recovery (type level)
    universe = [t for t in range(inv.n_types) if gold.type_const_frac.get(t, 0) >= 0.5 and t in gold.type_gold_class]
    g_lab = [gold.type_gold_class[t] for t in universe]
    p_lab: list[Any] = []
    for t in universe:
        cls = type_classes.get(t, [])
        p_lab.append(cls[0] if len(cls) == 1 else (("multi", tuple(cls)) if cls else ("single", t)))
    type_level = _pairwise(g_lab, p_lab)
    type_level["n_pruned_types"] = sum(1 for t in universe if not type_classes.get(t))
    metrics["class_recovery_type_level"] = type_level
    # ---------------------------------------------------- class recovery (occurrence level)
    occ_ids = [o for o in range(inv.n_occurrences) if gold.occ_gold[o] is not None]
    g_occ = [gold.occ_gold[o] for o in occ_ids]
    p_occ = [int(occ_class[o]) if occ_class[o] >= 0 else ("none", o) for o in occ_ids]
    metrics["class_recovery_occurrence_level"] = _pairwise(g_occ, p_occ)
    # ---------------------------------------------------- alignment
    align_pairs = [(int(occ_class[o]), gold.occ_gold[o]) for o in occ_ids if occ_class[o] >= 0]
    mapping = hungarian_alignment(align_pairs)
    metrics["class_alignment"] = {str(k): v for k, v in mapping.items()}
    # occurrence category accuracy under the alignment
    correct = sum(1 for o in occ_ids if occ_class[o] >= 0 and mapping.get(int(occ_class[o])) == gold.occ_gold[o])
    metrics["occurrence_category_accuracy"] = round(correct / max(1, len(occ_ids)), 4)
    # ---------------------------------------------------- constituents
    def const_metrics(pred_mask: np.ndarray) -> dict[str, Any]:
        pred = np.nonzero(pred_mask)[0]
        lengths = (inv.occ_end - inv.occ_start)
        tp_by_len: Counter = Counter()
        pred_by_len: Counter = Counter()
        for o in pred.tolist():
            L = int(lengths[o])
            pred_by_len[L] += 1
            if gold.occ_is_const[o]:
                tp_by_len[L] += 1
        tp = sum(tp_by_len.values())
        out = {
            "all": _prf(tp, len(pred), gold.gold_spans_total),
            "all_recall_frequent_only": round(tp / max(1, gold.gold_spans_in_inventory), 4),
            "len_ge_2": _prf(
                sum(v for L, v in tp_by_len.items() if L >= 2),
                sum(v for L, v in pred_by_len.items() if L >= 2),
                sum(v for L, v in gold.gold_spans_by_len.items() if L >= 2),
            ),
            "by_length": {},
        }
        for L in sorted(set(gold.gold_spans_by_len) | set(pred_by_len)):
            d = _prf(tp_by_len[L], pred_by_len[L], gold.gold_spans_by_len[L])
            d["recall_frequent_only"] = round(tp_by_len[L] / max(1, gold.gold_spans_in_inventory_by_len[L]), 4)
            out["by_length"][str(L)] = d
        return out

    def const_metrics_positions(pred: set[tuple[int, int, int]]) -> dict[str, Any]:
        tp_by_len: Counter = Counter()
        pred_by_len: Counter = Counter()
        for (sid, a, b) in pred:
            L = b - a
            pred_by_len[L] += 1
            if (sid, a, b) in gold.gold_const_positions:
                tp_by_len[L] += 1
        tp = sum(tp_by_len.values())
        out = {
            "all": _prf(tp, len(pred), gold.gold_spans_total),
            "all_recall_frequent_only": round(tp / max(1, gold.gold_spans_in_inventory), 4),
            "len_ge_2": _prf(
                sum(v for L, v in tp_by_len.items() if L >= 2),
                sum(v for L, v in pred_by_len.items() if L >= 2),
                sum(v for L, v in gold.gold_spans_by_len.items() if L >= 2),
            ),
            "by_length": {},
        }
        for L in sorted(set(gold.gold_spans_by_len) | set(pred_by_len)):
            d = _prf(tp_by_len[L], pred_by_len[L], gold.gold_spans_by_len[L])
            d["recall_frequent_only"] = round(tp_by_len[L] / max(1, gold.gold_spans_in_inventory_by_len[L]), 4)
            out["by_length"][str(L)] = d
        return out

    if constituent_positions is not None:
        metrics["constituents_derivation_level"] = const_metrics_positions(constituent_positions)
    else:
        metrics["constituents_derivation_level"] = const_metrics(constituent_mask)
    metrics["constituents_class_level"] = const_metrics(occ_class >= 0)
    # ---------------------------------------------------- composition
    gold_rels = set(gold.gold_relations)
    inf_rels = [(r.left_class, r.right_class, r.result_class) for r in relations]
    mapped = []
    for a, b, c in inf_rels:
        ma, mb, mc = mapping.get(a), mapping.get(b), mapping.get(c)
        mapped.append((ma, mb, mc) if None not in (ma, mb, mc) else None)
    tp_rel = sum(1 for m in mapped if m in gold_rels)
    found = {m for m in mapped if m in gold_rels}
    comp = _prf(tp_rel, len(inf_rels), len(gold_rels))
    comp["recovered"] = sorted(" + ".join(m[:2]) + " -> " + m[2] for m in found)
    comp["missed"] = sorted(" + ".join(m[:2]) + " -> " + m[2] for m in gold_rels - found)
    comp["false_relations"] = [
        {"inferred": f"{a} + {b} -> {c}", "mapped": (" + ".join(m[:2]) + " -> " + m[2]) if m else "unaligned", "support": r.support, "lexical": r.distinct_lexical_support}
        for (a, b, c), m, r in zip(inf_rels, mapped, relations)
        if m not in gold_rels
    ][:25]
    metrics["composition_recovery"] = comp
    # ---------------------------------------------------- ambiguity
    gold_amb = set(gold.ambiguous_tids)
    pred_amb = set(split_tids)
    amb = _prf(len(gold_amb & pred_amb), len(pred_amb), len(gold_amb))
    amb["false_splits"] = [inv.type_str(t) for t in sorted(pred_amb - gold_amb)]
    amb["missed_splits"] = [inv.type_str(t) for t in sorted(gold_amb - pred_amb)]
    amb["n_false_splits"] = len(pred_amb - gold_amb)
    amb["n_missed_splits"] = len(gold_amb - pred_amb)
    occ_amb = [o for t in gold_amb for o in inv.type_occ[t].tolist() if gold.occ_gold[o] is not None]
    if occ_amb:
        corr = sum(1 for o in occ_amb if occ_class[o] >= 0 and mapping.get(int(occ_class[o])) == gold.occ_gold[o])
        amb["occurrence_assignment_accuracy"] = round(corr / len(occ_amb), 4)
        # majority baseline: assign every occurrence of the word to its majority sense
        maj = 0
        for t in gold_amb:
            labs = Counter(gold.occ_gold[o] for o in inv.type_occ[t].tolist() if gold.occ_gold[o] is not None)
            maj += labs.most_common(1)[0][1] if labs else 0
        amb["occurrence_assignment_majority_baseline"] = round(maj / len(occ_amb), 4)
        per_word = {}
        for t in gold_amb:
            occs = [o for o in inv.type_occ[t].tolist() if gold.occ_gold[o] is not None]
            c = sum(1 for o in occs if occ_class[o] >= 0 and mapping.get(int(occ_class[o])) == gold.occ_gold[o])
            per_word[inv.type_str(t)] = {"accuracy": round(c / max(1, len(occs)), 4), "n": len(occs), "gold_split": dict(Counter(gold.occ_gold[o] for o in occs)), "split_accepted": t in pred_amb}
        amb["per_word"] = per_word
    metrics["ambiguity_detection"] = amb
    return metrics


# ------------------------------------------------------------------ diagnostics
def diagnostics(
    gold: GoldInfo,
    inv: SpanInventory,
    state: ClassState,
    type_classes: dict[int, list[int]],
    reports: dict[int, Any],
    accepted: set[int],
    history: list[MergeEvent],
    mapping: dict[int, str],
    limit: int = 15,
) -> dict[str, Any]:
    """Human-readable failure analysis."""
    def majority(members: list[int]) -> tuple[str | None, float]:
        labs = Counter(gold.type_gold_class.get(t, "<non-const>") for t in members)
        if not labs:
            return None, 0.0
        lab, n = labs.most_common(1)[0]
        return lab, n / len(members)

    # false merges: merge events joining classes with different gold majorities
    members_at: dict[int, list[int]] = {t: [t] for t in range(inv.n_types)}
    false_merges = []
    for ev in history:
        ma = members_at.get(ev.class_a, [])
        mb = members_at.get(ev.class_b, [])
        members_at[ev.new_class] = ma + mb
        la, fa = majority(ma)
        lb, fb = majority(mb)
        if la is not None and lb is not None and la != lb and la != "<non-const>" and lb != "<non-const>":
            false_merges.append(
                {
                    "step": ev.step,
                    "kind": ev.kind,
                    "score": round(ev.score, 3),
                    "gold_a": la,
                    "gold_b": lb,
                    "size_a": len(ma),
                    "size_b": len(mb),
                    "members_a": [inv.type_str(t) for t in ma[:5]],
                    "members_b": [inv.type_str(t) for t in mb[:5]],
                    "why_merged": [c for c, _, _ in ev.evidence_contexts[:5]],
                }
            )
    false_merges.sort(key=lambda d: -(d["size_a"] + d["size_b"]))
    # missed merges: gold classes fragmented over several inferred classes
    frag: dict[str, Counter] = defaultdict(Counter)
    for t, g in gold.type_gold_class.items():
        if gold.type_const_frac.get(t, 0) < 0.5:
            continue
        cls = type_classes.get(t, [])
        frag[g][cls[0] if len(cls) == 1 else ("pruned" if not cls else "multi")] += 1
    missed = []
    for g, c in frag.items():
        if len(c) > 1:
            missed.append(
                {
                    "gold_class": g,
                    "fragments": [
                        {"inferred_class": str(k), "n_types": n, "examples": [inv.type_str(t) for t in gold.type_gold_class if gold.type_gold_class[t] == g and ((type_classes.get(t) or ["pruned"])[0] == k or (k == "pruned" and not type_classes.get(t)))][:4]}
                        for k, n in c.most_common(6)
                    ],
                }
            )
    # false constituents: accepted types that are not gold constituents
    false_const = []
    missed_const = []
    for t in range(inv.n_types):
        cls = type_classes.get(t, [])
        frac = gold.type_const_frac.get(t, 0.0)
        if cls and frac < 0.5:
            false_const.append({"span": inv.type_str(t), "freq": int(inv.type_freq[t]), "gold_const_frac": round(frac, 2), "class": cls[0], "class_gold": mapping.get(cls[0])})
        if not cls and frac >= 0.5 and inv.type_length(t) >= 1:
            rep_status = None
            c0 = int(state.type_class[t])
            if c0 in reports:
                if reports[c0].status == "accepted":
                    rep_status = "class accepted but this member was never used in a Viterbi derivation (pruned as non-constituent member)"
                else:
                    rep_status = reports[c0].status + ": " + reports[c0].reason
            missed_const.append({"span": inv.type_str(t), "freq": int(inv.type_freq[t]), "gold_class": gold.type_gold_class.get(t), "why": rep_status})
    false_const.sort(key=lambda d: -d["freq"])
    missed_const.sort(key=lambda d: -d["freq"])
    return {
        "false_merges": false_merges[:limit],
        "n_false_merges": len(false_merges),
        "missed_merges": missed[:limit],
        "false_constituents": false_const[:limit],
        "n_false_constituent_types": len(false_const),
        "missed_constituents": missed_const[:limit],
        "n_missed_constituent_types": len(missed_const),
    }
