"""Evaluation of the v0.2 posterior against synthetic gold, with the v0.1 metric definitions.

``evaluate_v02`` feeds the posterior decoding (dominant type classes, argmax
occurrence classes, posterior constituents, posterior relations, ambiguous
types) into :func:`evaluation.evaluate` so that v0.1 and v0.2 numbers are
directly comparable.  ``failure_modes_v02`` quantifies the v0.1 failure modes
on the v0.2 result:

* NP/Nn fragmentation      - gold classes spread over several inferred classes
* bare-S over-refinement   - an extra inferred class aligned to S (or unaligned root class)
* Adj -> Det contamination - impurity of inferred classes (second gold label share)
* private crutch rules     - rules dominated by one lexical realisation
* filler-induced rules     - rules whose expected realisations are mostly filler tokens
* long-context sparsity    - share of occurrences whose class is decided by context potentials vs. grammar
* ambiguity pollution      - purity of the classes that host gold-ambiguous words
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .evaluation import GoldInfo, evaluate


def evaluate_v02(gold: GoldInfo, inv, state, v02, min_relation_support: int, min_span_frequency: int) -> dict[str, Any]:
    dec = v02.decoding
    split_tids = {t: d for t, d in dec.ambiguous_types.items()}
    m = evaluate(gold, inv, state, dec.type_classes, dec.occ_class, dec.occ_const >= 0.5, dec.relations, split_tids, min_span_frequency, dec.constituent_positions)
    # Viterbi-decoded constituents as a second view (MAP derivation instead of posterior > 0.5)
    tp = sum(1 for pos in dec.viterbi_positions if pos in gold.gold_const_positions)
    n_pred = len(dec.viterbi_positions)
    p = tp / n_pred if n_pred else 0.0
    r = tp / gold.gold_spans_total if gold.gold_spans_total else 0.0
    m["constituents_viterbi"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "n_pred": n_pred, "n_gold": gold.gold_spans_total}
    # occurrence-level ambiguity on gold-ambiguous words: posterior mixture vs gold split
    amb = m["ambiguity_detection"]
    per_word = {}
    for t, dist in gold.ambiguous_tids.items():
        mix = dec.type_mixture.get(t, {})
        mapping = {int(k): v for k, v in m["class_alignment"].items()}
        mapped = defaultdict(float)
        for c, v in mix.items():
            mapped[mapping.get(c, f"C{c}")] += v
        per_word[inv.type_str(t)] = {"gold": dist, "posterior_mixture": {str(k): round(v, 3) for k, v in sorted(mapped.items(), key=lambda kv: -kv[1])[:4]}}
    amb["posterior_mixtures"] = per_word
    return m


def failure_modes_v02(gold: GoldInfo, inv, v02, metrics: dict[str, Any], ds) -> dict[str, Any]:
    dec = v02.decoding
    model = v02.model
    names = model.names
    mapping = {int(k): v for k, v in metrics.get("class_alignment", {}).items()}
    # ---- fragmentation / contamination from the type-level assignment
    gold_of_type = {t: g for t, g in gold.type_gold_class.items() if gold.type_const_frac.get(t, 0) >= 0.5}
    by_gold: dict[str, Counter] = defaultdict(Counter)
    by_inf: dict[int, Counter] = defaultdict(Counter)
    for t, cls in dec.type_classes.items():
        g = gold_of_type.get(t)
        if g is None:
            continue
        w = int(inv.type_freq[t])
        by_gold[g][cls[0]] += w
        by_inf[cls[0]][g] += w
    fragmentation = {}
    for g, cnt in by_gold.items():
        tot = sum(cnt.values())
        parts = [(names.get(c, str(c)), round(n / tot, 3)) for c, n in cnt.most_common() if n / tot >= 0.05]
        fragmentation[g] = {"n_fragments_ge_5pct": len(parts), "fragments": parts}
    contamination = {}
    for c, cnt in by_inf.items():
        tot = sum(cnt.values())
        top = cnt.most_common(2)
        second = top[1] if len(top) > 1 else (None, 0)
        contamination[names.get(c, str(c))] = {"dominant": top[0][0], "purity": round(top[0][1] / tot, 3), "second": second[0], "second_share": round(second[1] / tot, 3), "n_types_weighted": tot}
    # ---- bare-S over-refinement: inferred root classes beyond the one aligned to S
    root_classes = [c for c, n in model.root.items() if n >= 1.0]
    extra_roots = [names.get(c, str(c)) for c in root_classes if mapping.get(c) != "S"]
    # ---- private crutch rules
    private = [p for p in v02.posterior_stats.get("private_rules", []) if p["dominant_share"] >= model.prior.private_tau]
    # ---- filler-induced rules
    fillers = set((ds.config.get("noise") or {}).get("filler_tokens", [])) if ds is not None else set()
    filler_rules = []
    if fillers:
        filler_tids = {t for t in range(inv.n_types) if any(inv.corpus.itos[w] in fillers for w in inv.types[t].tokens)}
        for key, n in model.rules.items():
            real = model.realisations.get(key, {})
            tot = sum(real.values())
            if tot <= 0 or n < 2:
                continue
            fshare = sum(v for t, v in real.items() if t in filler_tids) / tot
            if fshare >= 0.3:
                filler_rules.append({"rule": f"{names.get(key[1])} + {names.get(key[2])} -> {names.get(key[0])}", "expected_count": round(n, 1), "filler_share": round(fshare, 3)})
    filler_classes = {}
    if fillers:
        for t, cls in dec.type_classes.items():
            if inv.type_length(t) == 1 and inv.corpus.itos[inv.types[t].tokens[0]] in fillers:
                filler_classes[inv.type_str(t)] = f"{names.get(cls[0])}~{mapping.get(cls[0])}"
    # ---- ambiguity pollution: purity of the classes hosting gold-ambiguous words
    pollution = {}
    for t in gold.ambiguous_tids:
        for c in dec.type_classes.get(t, []):
            pollution[f"{inv.type_str(t)} in {names.get(c)}~{mapping.get(c)}"] = contamination.get(names.get(c, str(c)), {}).get("purity")
    # ---- context sparsity: how often the lexical potentials disagree with the grammar posterior
    disagreement = None
    pots = v02.vi.potentials.leaf
    if pots:
        n_tot = 0
        n_dis = 0
        for sid, nodes in (v02.learner.out.node_posteriors.items() if v02.learner and v02.learner.out else []):
            entries = pots.get(sid)
            if entries is None:
                continue
            for i, j, c, p in nodes:
                if j - i != 1 or p < 0.5:
                    continue
                d = entries[i]
                if not d:
                    continue
                best = max(d.items(), key=lambda kv: kv[1])[0]
                src = v02.vi.class_map.get(c, [c])
                n_tot += 1
                if best not in src:
                    n_dis += 1
        disagreement = round(n_dis / n_tot, 4) if n_tot else None
    return {
        "fragmentation": fragmentation,
        "contamination": contamination,
        "extra_root_classes": extra_roots,
        "private_crutch_rules": private,
        "filler_induced_rules": filler_rules,
        "filler_token_classes": filler_classes,
        "ambiguity_pollution_class_purity": pollution,
        "context_vs_grammar_disagreement": disagreement,
    }
