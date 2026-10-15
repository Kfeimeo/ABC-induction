"""Intrinsic evaluation for corpora without gold annotation (TinyStories).

No syntax-recovery accuracy is claimed here.  We measure

* class stability across random disjoint corpus subsets and across seeds
  (pairwise F1 / ARI between two runs' class assignments on shared span types,
  plus agreement on which span types are accepted as constituents),
* compositional productivity (distinct lexical realisations per relation),
* MDL / compression (bits per sentence, grammar bits),
* held-out context prediction (log-likelihood of the k=1 context of held-out
  occurrences under the class model vs. the span-type model vs. background),
* ambiguity split consistency across subsets.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .corpus import BOS_ID, EOS_ID
from .evaluation import _pairwise
from .experiment import RunResult


def type_labels(res: RunResult) -> dict[str, int]:
    """span string -> single accepted class id (ambiguous types excluded)."""
    out = {}
    for tid, classes in res.type_classes.items():
        if len(classes) == 1:
            out[res.inv.type_str(tid)] = classes[0]
    return out


def compare_runs(a: RunResult, b: RunResult) -> dict[str, Any]:
    """Agreement between two runs on the span types they share."""
    la, lb = type_labels(a), type_labels(b)
    types_a = {a.inv.type_str(t) for t in range(a.inv.n_types)}
    types_b = {b.inv.type_str(t) for t in range(b.inv.n_types)}
    shared_types = types_a & types_b
    acc_a = {s for s in shared_types if s in la}
    acc_b = {s for s in shared_types if s in lb}
    both = sorted(acc_a & acc_b)
    out: dict[str, Any] = {
        "n_shared_span_types": len(shared_types),
        "constituent_agreement_jaccard": round(len(acc_a & acc_b) / max(1, len(acc_a | acc_b)), 4),
        "n_types_classified_in_both": len(both),
    }
    if both:
        m = _pairwise([la[s] for s in both], [lb[s] for s in both])
        out["class_pairwise_f1"] = m["f1"]
        out["class_ari"] = m["ari"]
        out["n_classes_a"] = m["n_gold_classes"]
        out["n_classes_b"] = m["n_pred_classes"]
    sa = {s.span for s in a.splits if s.accepted and s.kind == "split"}
    sb = {s.span for s in b.splits if s.accepted and s.kind == "split"}
    out["splits_a"] = sorted(sa)
    out["splits_b"] = sorted(sb)
    out["split_agreement_jaccard"] = round(len(sa & sb) / max(1, len(sa | sb)), 4)
    return out


def productivity_summary(res: RunResult) -> dict[str, Any]:
    lex = [r.distinct_lexical_support for r in res.relations]
    sup = [r.support for r in res.relations]
    if not lex:
        return {"n_relations": 0}
    return {
        "n_relations": len(lex),
        "lexical_support_mean": round(float(np.mean(lex)), 2),
        "lexical_support_median": float(np.median(lex)),
        "lexical_support_max": int(max(lex)),
        "relations_with_lexical_support_ge_10": int(sum(1 for x in lex if x >= 10)),
        "support_total": int(sum(sup)),
        "top_relations": [r.to_json(res.class_names)["label"] + f" (support {r.support}, lexical {r.distinct_lexical_support})" for r in res.relations[:15]],
    }


def heldout_context_prediction(res: RunResult, heldout: list[list[str]], max_span_length: int = 4, eta: float = 1.0) -> dict[str, Any]:
    """Log-likelihood of the k=1 context of held-out occurrences.

    For every held-out span whose type exists in the training inventory and has
    a class, the (left, right) k=1 context is scored under
    ``P(ctx | class)`` (contexts pooled over the class), ``P(ctx | type)`` and the
    background ``P(ctx)``, each backed off to the background with weight ``eta``.
    Higher is better; the class model wins when classes generalise to contexts
    the individual type was never seen in.
    """
    inv, cd, corpus = res.inv, res.cd, res.corpus
    k1_cols = np.nonzero((cd.ctx_scale == 1) & (cd.ctx_kind == 0))[0]
    col_pos = {int(c): i for i, c in enumerate(k1_cols)}
    x = cd.type_ctx[:, k1_cols].tocsr()
    bg = np.asarray(x.sum(axis=0)).ravel()
    bg_p = (bg + 1.0) / (bg.sum() + len(bg))
    type_tot = np.asarray(x.sum(axis=1)).ravel()
    # class profiles
    classes = sorted(res.const.accepted)
    members: dict[int, list[int]] = {c: [] for c in classes}
    for tid, cls in res.type_classes.items():
        for c in cls:
            if c in members:
                members[c].append(tid)
    cls_rows = {}
    for c, mem in members.items():
        if mem:
            cls_rows[c] = np.asarray(x[mem].sum(axis=0)).ravel()
    ll = {"type": [], "class": [], "background": []}
    rare_ll = {"type": [], "class": [], "background": []}
    n_seen = 0
    n_unseen_type_seen_class = 0
    vocab = corpus.vocab
    for toks in heldout:
        ids = [vocab.get(t) for t in toks]
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_span_length) + 1):
                tt = tuple(ids[i:j])
                if any(t is None for t in tt):
                    continue
                tid = inv.type_index.get(tt)
                if tid is None or tid not in res.type_classes:
                    continue
                left = (ids[i - 1],) if i > 0 else (BOS_ID,)
                right = (ids[j],) if j < n else (EOS_ID,)
                if None in left or None in right:
                    continue
                key = (1, left, right)
                cid = cd.vocab.key_to_id.get(key)
                pos = col_pos.get(cid) if cid is not None else None
                n_seen += 1
                pb = bg_p[pos] if pos is not None else 1.0 / (bg.sum() + len(bg))
                row_t = x[tid].toarray().ravel()
                nt = row_t[pos] if pos is not None else 0.0
                pt = (nt + eta * pb) / (type_tot[tid] + eta)
                cls = res.type_classes[tid][0]
                row_c = cls_rows.get(cls)
                if row_c is None:
                    continue
                nc = row_c[pos] if pos is not None else 0.0
                pc = (nc + eta * pb) / (row_c.sum() + eta)
                if nt == 0 and nc > 0:
                    n_unseen_type_seen_class += 1
                for name, p in (("type", pt), ("class", pc), ("background", pb)):
                    ll[name].append(math.log2(p))
                    if inv.type_freq[tid] < 10:
                        rare_ll[name].append(math.log2(p))
    out = {"n_heldout_occurrences": n_seen, "fraction_context_unseen_for_type_but_seen_for_class": round(n_unseen_type_seen_class / max(1, n_seen), 4)}
    for name in ("type", "class", "background"):
        out[f"mean_log2_likelihood_{name}"] = round(float(np.mean(ll[name])), 3) if ll[name] else None
        out[f"mean_log2_likelihood_{name}_rare_types"] = round(float(np.mean(rare_ll[name])), 3) if rare_ll[name] else None
    out["n_rare_type_occurrences"] = len(rare_ll["type"])
    return out


def top_classes_for_inspection(res: RunResult, n_classes: int = 15, n_members: int = 15) -> list[dict[str, Any]]:
    reps = sorted((r for r in res.const.reports.values() if r.status == "accepted"), key=lambda r: -r.n_occurrences)
    out = []
    for r in reps[:n_classes]:
        lex = [res.inv.type_str(t) for t in r.members if res.inv.type_length(t) == 1][:n_members]
        multi = [res.inv.type_str(t) for t in r.members if res.inv.type_length(t) >= 2][:n_members]
        out.append({"class": r.cid, "n_members": len(r.members), "n_occurrences": r.n_occurrences, "usage": r.usage, "lexical_members": lex, "multi_token_members": multi, "top_contexts": [c for c, _ in r.top_contexts[:5]]})
    return out


def split_summary(res: RunResult, n: int = 20) -> list[dict[str, Any]]:
    acc = [s for s in res.splits if s.accepted and s.kind == "split"]
    acc.sort(key=lambda s: -s.split_gain)
    return [
        {"span": s.span, "n": s.n_occ, "a": (res.class_names.get(s.class_a, s.class_a), s.n_a, s.contexts_a[:3]), "b": (res.class_names.get(s.class_b, s.class_b), s.n_b, s.contexts_b[:3]), "split_gain": round(s.split_gain, 1), "ll_gain": round(s.ll_gain_per_occ, 3)}
        for s in acc[:n]
    ]


# --------------------------------------------------------------------------- v0.2 views
def type_labels_v02(res: RunResult) -> dict[str, int]:
    """span string -> dominant v0.2 class (posterior decoding)."""
    if res.v02 is None:
        return {}
    return {res.inv.type_str(t): cls[0] for t, cls in res.v02.decoding.type_classes.items() if cls}


def compare_runs_v02(a: RunResult, b: RunResult, min_rule_support: float = 3.0) -> dict[str, Any]:
    """Agreement of two v0.2 runs on shared span types: class pairwise F1 / ARI, constituent-set
    agreement, and rule agreement after aligning the two class inventories (Hungarian on the
    shared classified types)."""
    from .evaluation import hungarian_alignment

    la, lb = type_labels_v02(a), type_labels_v02(b)
    shared = {a.inv.type_str(t) for t in range(a.inv.n_types)} & {b.inv.type_str(t) for t in range(b.inv.n_types)}
    acc_a = {s for s in shared if s in la}
    acc_b = {s for s in shared if s in lb}
    both = sorted(acc_a & acc_b)
    out: dict[str, Any] = {"n_shared_span_types": len(shared), "constituent_agreement_jaccard": round(len(acc_a & acc_b) / max(1, len(acc_a | acc_b)), 4), "n_types_classified_in_both": len(both)}
    if not both:
        return out
    m = _pairwise([la[s] for s in both], [lb[s] for s in both])
    out["class_pairwise_f1"] = m["f1"]
    out["class_ari"] = m["ari"]
    out["n_classes_a"] = m["n_gold_classes"]
    out["n_classes_b"] = m["n_pred_classes"]
    mapping = hungarian_alignment([(la[s], lb[s]) for s in both])
    ra = {(r.left_class, r.right_class, r.result_class) for r in a.v02.decoding.relations if r.support >= min_rule_support}
    rb = {(r.left_class, r.right_class, r.result_class) for r in b.v02.decoding.relations if r.support >= min_rule_support}
    ra_mapped = {(mapping.get(x), mapping.get(y), mapping.get(z)) for x, y, z in ra}
    ra_mapped = {k for k in ra_mapped if None not in k}
    out["rule_agreement_jaccard"] = round(len(ra_mapped & rb) / max(1, len(ra_mapped | rb)), 4)
    out["n_rules_a"] = len(ra)
    out["n_rules_b"] = len(rb)
    out["n_rules_a_alignable"] = len(ra_mapped)
    aa = {a.inv.type_str(t) for t in a.v02.decoding.ambiguous_types}
    ab = {b.inv.type_str(t) for t in b.v02.decoding.ambiguous_types}
    out["ambiguous_types_a"] = sorted(aa)[:15]
    out["ambiguous_types_b"] = sorted(ab)[:15]
    out["ambiguity_agreement_jaccard"] = round(len(aa & ab) / max(1, len(aa | ab)), 4)
    return out


def _modifier_sets(res: RunResult, source: str | None, min_support: float, mapping: dict[int, int] | None = None, constituent_only: bool = True) -> tuple[set[str], dict[str, str], dict[str, Any]]:
    """Candidate strings (constituent candidates with support >= min_support) with their dominant
    direction and host class (mapped through ``mapping`` when given)."""
    tab = res.v02.modifier_table if res.v02 is not None else None
    if tab is None:
        return set(), {}, {}
    cands: dict[str, dict[str, float]] = {}
    hosts: dict[str, dict[Any, float]] = {}
    for r in tab.rows:
        if source is not None and r["source"] != source:
            continue
        if r["direction"] == "INTERNAL" or r["support"] < min_support or (constituent_only and r["candidate_constituency"] < 0.5):
            continue
        c = r["candidate"]
        cands.setdefault(c, {})[r["direction"]] = cands.setdefault(c, {}).get(r["direction"], 0.0) + r["support"]
        h = r["host_class"]
        if h is not None and mapping is not None:
            h = mapping.get(h)
        hosts.setdefault(c, {})[h if h is not None else "ROOT"] = hosts.setdefault(c, {}).get(h if h is not None else "ROOT", 0.0) + r["support"]
    direction = {c: max(d.items(), key=lambda kv: kv[1])[0] for c, d in cands.items()}
    host = {c: max(d.items(), key=lambda kv: kv[1])[0] for c, d in hosts.items()}
    return set(cands), direction, host


def compare_modifiers_v02(a: RunResult, b: RunResult, min_support: float = 3.0) -> dict[str, Any]:
    """Agreement of two runs on identity-preserving (modifier) candidates, per evidence source, next
    to the general rule agreement: candidate-set Jaccard, direction agreement and host-class agreement
    (after the same Hungarian class alignment as ``compare_runs_v02``)."""
    from .evaluation import hungarian_alignment

    la, lb = type_labels_v02(a), type_labels_v02(b)
    both = sorted(s for s in la if s in lb)
    mapping = hungarian_alignment([(la[s], lb[s]) for s in both]) if both else {}
    out: dict[str, Any] = {}
    for src in ("ROOT_STRING", "CONSTITUENT_STRING", "CLASS_GENERALIZED", "POSTERIOR_RULE", None):
        sa, da, ha = _modifier_sets(a, src, min_support, mapping)
        sb, db, hb = _modifier_sets(b, src, min_support, None)
        shared = sa & sb
        name = src or "COMBINED"
        out[name] = {
            "n_a": len(sa), "n_b": len(sb), "n_shared": len(shared),
            "candidate_jaccard": round(len(shared) / max(1, len(sa | sb)), 4),
            "direction_agreement": round(sum(1 for c in shared if da[c] == db[c]) / len(shared), 4) if shared else None,
            "host_agreement": round(sum(1 for c in shared if ha.get(c) == hb.get(c)) / len(shared), 4) if shared else None,
            "shared_examples": sorted(shared)[:12],
        }
    return out


def modifier_inspection(res: RunResult, n: int = 12, min_support: float = 3.0) -> dict[str, list[dict[str, Any]]]:
    """Top candidates per source for human inspection (natural text: no gold, no linguistic claim)."""
    tab = res.v02.modifier_table if res.v02 is not None else None
    if tab is None:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for src, direction in (("ROOT_STRING", "RIGHT"), ("ROOT_STRING", "LEFT"), ("ROOT_STRING", "INTERNAL"), ("CONSTITUENT_STRING", None), ("CLASS_GENERALIZED", None), ("POSTERIOR_RULE", None)):
        rows = [r for r in tab.rows if r["source"] == src and (direction is None or r["direction"] == direction) and r["support"] >= min_support]
        rows.sort(key=lambda r: (-r["productivity"], -r["support"]))
        out[f"{src}{'/' + direction if direction else ''}"] = [{"candidate": r["candidate"], "direction": r["direction"], "host_class": r["host_class_name"], "candidate_class": r["candidate_class_name"], "examples": r["examples"][:2], "posterior_support": r["posterior_mass"], "support": r["support"], "distinct_hosts": r["distinct_host_types"], "productivity": r["productivity"], "mod": r["mod"], "source": r["source"]} for r in rows[:n]]
    return out


def coverage_v02(res: RunResult) -> dict[str, Any]:
    if res.v02 is None:
        return {}
    v02 = res.v02
    n = max(1, len(res.corpus))
    ps = v02.posterior_stats
    return {
        "fully_parsed_fraction": ps.get("fully_parsed_fraction"),
        "mean_chunks_per_sentence": ps.get("mean_chunks_per_sentence"),
        "unknown_token_mass": ps.get("unknown_token_mass"),
        "active_classes": ps.get("n_classes"),
        "active_rules": ps.get("n_rules"),
        "lexical_entries": ps.get("n_lexical_entries"),
        "elbo_bits_per_sentence": round(-v02.objective.get("elbo", 0.0) / n / 0.6931, 3),
        "objective": round(v02.objective.get("objective", 0.0), 2),
        "pruning": ps.get("pruning"),
        "n_accepted_moves": sum(1 for m in v02.moves if m.get("accepted")),
        "n_proposed_moves": len(v02.moves),
        "accepted_moves": [f"{m['kind']} {m.get('a_name') or m.get('rule') or m.get('token') or ''}{(' + ' + str(m.get('b_name'))) if m.get('b_name') else ''}" for m in v02.moves if m.get("accepted")][:12],
        "runtime_sec": {k: round(v, 1) for k, v in v02.timings.items()},
        "n_ambiguous_types": ps.get("n_ambiguous_types"),
        "highest_entropy_span_types": ps.get("highest_entropy_span_types", [])[:12],
    }


def top_classes_v02(res: RunResult, n_classes: int = 15) -> list[dict[str, Any]]:
    if res.v02 is None:
        return []
    return [{"name": c["name"], "mass": c["posterior_mass"], "lexical": c["top_lexical_entries"][:12], "spans": c["top_span_types"][:6], "contexts": c["top_lexical_contexts"][:3], "in": c["top_composition_in"][:3], "abstract": c["top_abstract_constituent_contexts"][:3]} for c in res.v02.posterior_stats.get("classes", [])[:n_classes]]
