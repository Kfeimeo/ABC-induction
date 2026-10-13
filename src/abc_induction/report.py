"""Human-readable ``report.md`` for one run (debug cases included)."""
from __future__ import annotations

import os
from typing import Any

from .experiment import RunResult


def _tbl(rows: list[list[Any]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _prf_row(name: str, d: dict[str, Any]) -> list[Any]:
    return [name, d.get("precision"), d.get("recall"), d.get("f1"), d.get("n_pred"), d.get("n_gold")]


def build_report(res: RunResult) -> str:
    cfg = res.cfg
    m = res.metrics
    inv = res.inv
    names = res.class_names
    L: list[str] = []
    L.append(f"# Run report: {cfg.get('name')}\n")
    # ------------------------------------------------------------ dataset
    d = cfg["dataset"]
    L.append("## Dataset\n")
    if d.get("kind", "synthetic") == "synthetic":
        L.append(f"- synthetic grammar: `{d['grammar']}` ({res.dataset.name if res.dataset else ''}), {len(res.corpus)} sentences, seed {cfg.get('seed')}")
        if res.dataset is not None:
            L.append(f"- gold classes: {', '.join(res.dataset.classes)}")
            L.append(f"- gold relations (attested): " + ", ".join(f"{a}+{b}->{c} ({n})" for (a, b, c), n in sorted(res.dataset.relation_counts.items(), key=lambda kv: -kv[1])))
            if res.dataset.ambiguous_words:
                L.append(f"- gold ambiguous words: " + ", ".join(f"{w} {dist}" for w, dist in res.dataset.ambiguous_words.items()))
            noise = res.dataset.config.get("noise")
            if noise:
                L.append(f"- noise: `{noise}`")
    else:
        L.append(f"- text corpus `{d.get('path')}`: {len(res.corpus)} sentences, {res.corpus.n_tokens} tokens, vocab {len(res.corpus.itos)}")
    L.append("")
    # ------------------------------------------------------------ hyperparameters
    L.append("## Hyperparameters\n")
    L.append("```yaml")
    import yaml

    L.append(yaml.safe_dump({k: cfg[k] for k in ("preprocess", "contexts", "clustering", "constituency", "mdl", "ambiguity", "relations") if k in cfg}, sort_keys=False).strip())
    L.append("```\n")
    if res.v02 is not None:
        L.append("v0.2 (probabilistic stage):\n")
        L.append("```yaml")
        L.append(yaml.safe_dump({k: cfg[k] for k in ("probabilistic", "structure_prior", "structure_learning", "lexical_context", "adaptive_context", "punctuation", "chart") if k in cfg}, sort_keys=False).strip())
        L.append("```\n")
        L.extend(v02_section(res))
    # ------------------------------------------------------------ metrics
    eff = m.get("efficiency", {})
    L.append("## Efficiency\n")
    L.append(_tbl([[k, v] for k, v in eff.items() if k != "runtime_sec"] + [[f"runtime {k} (s)", v] for k, v in eff.get("runtime_sec", {}).items()], ["quantity", "value"]))
    L.append("")
    if "mdl" in m:
        L.append("## MDL / compression\n")
        L.append(_tbl([[k, v] for k, v in m["mdl"].items()], ["quantity", "value"]))
        L.append("")
    if "class_recovery_type_level" in m:
        L.append("## Recovery metrics\n")
        tl, ol = m["class_recovery_type_level"], m["class_recovery_occurrence_level"]
        L.append("### Equivalence classes\n")
        L.append(_tbl([
            ["type level", tl["ari"], tl["precision"], tl["recall"], tl["f1"], tl["purity"], tl["n_pred_classes"], tl["n_gold_classes"]],
            ["occurrence level", ol["ari"], ol["precision"], ol["recall"], ol["f1"], ol["purity"], ol["n_pred_classes"], ol["n_gold_classes"]],
        ], ["level", "ARI", "pair P", "pair R", "pair F1", "purity", "#inferred", "#gold"]))
        L.append(f"\n- occurrence category accuracy (after Hungarian alignment): {m['occurrence_category_accuracy']}")
        L.append(f"- class alignment: " + ", ".join(f"C{k}->{v}" for k, v in m["class_alignment"].items()))
        L.append(f"- gold-constituent span types left unmerged/pruned: {tl['n_pruned_types']}\n")
        L.append("### Constituents\n")
        rows = []
        for lvl in ("constituents_derivation_level", "constituents_class_level"):
            cm = m[lvl]
            rows.append(_prf_row(lvl.replace("constituents_", "") + " / all", cm["all"]))
            rows.append(_prf_row(lvl.replace("constituents_", "") + " / len>=2", cm["len_ge_2"]))
        L.append(_tbl(rows, ["setting", "P", "R", "F1", "#pred", "#gold"]))
        L.append(f"\n(recall over gold spans whose type is frequent enough to be a candidate: derivation-level {m['constituents_derivation_level']['all_recall_frequent_only']}, class-level {m['constituents_class_level']['all_recall_frequent_only']})\n")
        L.append("By length (derivation level):\n")
        L.append(_tbl([[k, v["precision"], v["recall"], v["recall_frequent_only"], v["f1"], v["n_pred"], v["n_gold"]] for k, v in m["constituents_derivation_level"]["by_length"].items()], ["len", "P", "R", "R(frequent)", "F1", "#pred", "#gold"]))
        L.append("")
        cr = m["composition_recovery"]
        L.append("### Composition relations\n")
        L.append(_tbl([_prf_row("R(A,B,C)", cr)], ["setting", "P", "R", "F1", "#pred", "#gold"]))
        L.append(f"\n- recovered: {', '.join(cr['recovered']) or '-'}")
        L.append(f"- missed: {', '.join(cr['missed']) or '-'}")
        if cr["false_relations"]:
            L.append("- false relations (top): " + "; ".join(f"{x['inferred']} => {x['mapped']} (support {x['support']}, lex {x['lexical']})" for x in cr["false_relations"][:8]))
        L.append("")
        am = m["ambiguity_detection"]
        L.append("### Ambiguity detection\n")
        L.append(_tbl([_prf_row("ambiguous span types", am)], ["setting", "P", "R", "F1", "#pred", "#gold"]))
        if "occurrence_assignment_accuracy" in am:
            L.append(f"\n- occurrence assignment accuracy on gold-ambiguous words: {am['occurrence_assignment_accuracy']} (majority-sense baseline {am['occurrence_assignment_majority_baseline']})")
            for w, pw in am.get("per_word", {}).items():
                L.append(f"  - `{w}`: accuracy {pw['accuracy']} over {pw['n']} occurrences, gold split {pw['gold_split']}, split accepted: {pw['split_accepted']}")
        L.append(f"- false splits ({am['n_false_splits']}): {', '.join(am['false_splits'][:15]) or '-'}")
        L.append(f"- missed splits ({am['n_missed_splits']}): {', '.join(am['missed_splits']) or '-'}\n")
    sa = m.get("structural_ambiguity", {})
    L.append(f"Structural ambiguity recorded (not resolved): {sa.get('n_occurrences_with_multiple_splits', 0)} occurrences have >= 2 licensed splits (hyperpaths) out of {sa.get('n_occurrences_with_split', 0)} composed occurrences.\n")
    # ------------------------------------------------------------ classes
    L.append("## Top inferred classes (debug view)\n")
    reps = [r for r in res.const.reports.values() if r.status == "accepted"]
    reps.sort(key=lambda r: -r.n_occurrences)
    rel_by_class: dict[int, list[str]] = {}
    for r in res.relations:
        lab = f"{names.get(r.left_class, r.left_class)} + {names.get(r.right_class, r.right_class)} -> {names.get(r.result_class, r.result_class)}  (support {r.support}, lexical {r.distinct_lexical_support}, P={r.probability:.2f})"
        for c in (r.left_class, r.right_class, r.result_class):
            rel_by_class.setdefault(c, []).append(lab)
    for r in reps[:12]:
        L.append(f"### {names.get(r.cid, 'C%d' % r.cid)}  ({r.n_members} members, {r.n_occurrences} occurrences)\n")
        L.append("```")
        L.append("members:")
        for t in r.members[:12]:
            L.append(f"    {inv.type_str(t)}")
        if r.n_members > 12:
            L.append(f"    ... ({r.n_members - 12} more)")
        if r.pruned_members:
            L.append("members pruned as non-constituents (never used in a derivation):")
            for t in r.pruned_members[:6]:
                L.append(f"    {inv.type_str(t)}")
        L.append("top shared contexts:")
        for c, p in r.top_contexts[:6]:
            L.append(f"    {c}   (p={p:.3f})")
        L.append(f"constituency score: {r.score:.3f}  = context {r.s_context:.2f}, composition {r.s_composition:.2f}, productivity {r.s_productivity:.2f}, complexity penalty {r.s_complexity:.2f}")
        if r.mdl_delta is not None:
            L.append(f"MDL: removing this class would change DL by {r.mdl_delta:+.1f} bits (grammar {r.mdl_delta_grammar:+.1f}, data {r.mdl_delta_data:+.1f}); used in {r.usage} derivation nodes")
        L.append("composition participation:")
        for lab in sorted(set(rel_by_class.get(r.cid, [])))[:8]:
            L.append(f"    {lab}")
        if not rel_by_class.get(r.cid):
            L.append("    (none)")
        L.append("```\n")
    # ------------------------------------------------------------ merges
    L.append("## Merge decisions: why merged?\n")
    hist = sorted(res.state.history, key=lambda e: -(e.size_a + e.size_b))[:8]
    for ev in hist:
        L.append(f"- step {ev.step} ({ev.kind}, score {ev.score:.3f}): {ev.members_a[:4]} (+{max(0, ev.size_a - 4)}) ∪ {ev.members_b[:4]} (+{max(0, ev.size_b - 4)}) -> C{ev.new_class}; shared contexts {ev.shared_contexts}; evidence: " + "; ".join(f"`{c}` ({pa:.2f}/{pb:.2f})" for c, pa, pb in ev.evidence_contexts[:4]))
    L.append("")
    # ------------------------------------------------------------ prunes
    L.append("## Pruned classes: why rejected as constituent?\n")
    rej = [r for r in res.const.reports.values() if r.status != "accepted"]
    rej.sort(key=lambda r: -r.n_occurrences)
    for r in rej[:12]:
        mem = [inv.type_str(t) for t in (r.members or r.pruned_members)[:5]]
        L.append(f"- C{r.cid} ({r.n_members} members, {r.n_occurrences} occ) {mem}: **{r.status}** — {r.reason}")
    if res.const.removed_relations:
        L.append("\nRelations removed by MDL (top):\n")
        for d in sorted(res.const.removed_relations, key=lambda d: -d["usage"])[:10]:
            L.append(f"- {d['label']} (used {d['usage']}x): ΔDL {d['delta_bits']} bits (grammar {d['delta_grammar']}, data {d['delta_data']})")
    L.append("")
    # ------------------------------------------------------------ splits
    L.append("## Ambiguity splits: why split?\n")
    acc = [s for s in res.splits if s.accepted and s.kind == "split"]
    for s in acc[:12]:
        L.append(f"### SpanType `{s.span}` ({s.n_occ} occurrences) — ACCEPTED\n")
        L.append("```")
        L.append(f"sense A: {names.get(s.class_a, 'C%d' % s.class_a)}  {s.n_a} occurrences  (alignment {s.align_a:.2f})")
        for c in s.contexts_a:
            L.append(f"    {c}")
        L.append(f"sense B: {names.get(s.class_b, 'C%d' % s.class_b)}  {s.n_b} occurrences  (alignment {s.align_b:.2f})")
        for c in s.contexts_b:
            L.append(f"    {c}")
        L.append(f"context LL gain/occ: {s.ll_gain_per_occ:.3f}   context MI gain: {s.mi_gain_bits:.3f} bits")
        L.append(f"DL current {s.dl_current:.1f}  single {s.dl_single:.1f}  double {s.dl_double:.1f}  -> split_gain {s.split_gain:.1f} (lambda_split {s.lambda_split})")
        L.append("composition differences:")
        for c in s.composition_a:
            L.append(f"    A: {c}")
        for c in s.composition_b:
            L.append(f"    B: {c}")
        L.append("```\n")
    rejected = [s for s in res.splits if not s.accepted and s.kind == "split"]
    rejected.sort(key=lambda s: -s.split_gain)
    if rejected:
        L.append("Rejected split proposals (closest to acceptance first):\n")
        for s in rejected[:10]:
            L.append(f"- `{s.span}` ({s.n_occ} occ): {names.get(s.class_a, s.class_a)} {s.n_a} vs {names.get(s.class_b, s.class_b)} {s.n_b}; LL gain {s.ll_gain_per_occ:.2f}; split_gain {s.split_gain:.1f} — {s.reason}")
    re_at = [s for s in res.splits if s.kind == "reattach" and s.accepted]
    if re_at:
        L.append(f"\nSingleton types re-attached to an existing class by MDL: {len(re_at)} (e.g. " + ", ".join(f"`{s.span}`->{names.get(s.class_a, s.class_a)}" for s in re_at[:8]) + ")")
    L.append("")
    # ------------------------------------------------------------ relations
    L.append("## Recovered composition relations A + B -> C\n")
    L.append(_tbl([[names.get(r.left_class, r.left_class), names.get(r.right_class, r.right_class), names.get(r.result_class, r.result_class), r.support, r.distinct_lexical_support, f"{r.probability:.2f}", "; ".join(f"{u} | {v}" for u, v, _ in r.examples[:2])] for r in res.relations[:25]], ["A", "B", "C", "support", "lexical", "P(C|A,B)", "examples"]))
    L.append("")
    # ------------------------------------------------------------ failures
    dg = res.diagnostics
    if dg:
        L.append("## Failure analysis\n")
        L.append(f"### False merges ({dg['n_false_merges']})\n")
        for f in dg["false_merges"][:8]:
            L.append(f"- step {f['step']} ({f['kind']}, score {f['score']}): {f['gold_a']} {f['members_a'][:3]} (+{f['size_a']}) with {f['gold_b']} {f['members_b'][:3]} (+{f['size_b']}); why merged: {f['why_merged'][:3]}")
        L.append(f"\n### Missed merges (gold classes fragmented)\n")
        for f in dg["missed_merges"][:8]:
            L.append(f"- {f['gold_class']}: " + "; ".join(f"{fr['inferred_class']} ({fr['n_types']} types, e.g. {fr['examples'][:3]})" for fr in f["fragments"][:5]))
        L.append(f"\n### False constituents ({dg['n_false_constituent_types']} types)\n")
        for f in dg["false_constituents"][:10]:
            L.append(f"- `{f['span']}` (freq {f['freq']}, gold constituent in {f['gold_const_frac']:.0%}) accepted in C{f['class']}~{f['class_gold']}")
        L.append(f"\n### Missed constituents ({dg['n_missed_constituent_types']} gold-constituent types not accepted)\n")
        for f in dg["missed_constituents"][:10]:
            L.append(f"- `{f['span']}` (freq {f['freq']}, gold {f['gold_class']}): {f['why']}")
        L.append("")
    return "\n".join(L)


def _cmp_rows(m1: dict[str, Any], m2: dict[str, Any]) -> list[list[Any]]:
    def g(d, *path):
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return None
            d = d[p]
        return d

    specs = [
        ("class pair F1 (type)", ("class_recovery_type_level", "f1")),
        ("class ARI (type)", ("class_recovery_type_level", "ari")),
        ("class purity (type)", ("class_recovery_type_level", "purity")),
        ("#inferred classes", ("class_recovery_type_level", "n_pred_classes")),
        ("occurrence class F1", ("class_recovery_occurrence_level", "f1")),
        ("occurrence category accuracy", ("occurrence_category_accuracy",)),
        ("constituent F1 (all)", ("constituents_derivation_level", "all", "f1")),
        ("constituent F1 (len>=2)", ("constituents_derivation_level", "len_ge_2", "f1")),
        ("constituent P (len>=2)", ("constituents_derivation_level", "len_ge_2", "precision")),
        ("composition P", ("composition_recovery", "precision")),
        ("composition R", ("composition_recovery", "recall")),
        ("composition F1", ("composition_recovery", "f1")),
        ("ambiguity type F1", ("ambiguity_detection", "f1")),
        ("ambiguity occurrence accuracy", ("ambiguity_detection", "occurrence_assignment_accuracy")),
    ]
    return [[name, g(m1, *path), g(m2, *path)] for name, path in specs]


def v02_section(res: RunResult) -> list[str]:
    """Report section of the probabilistic stage."""
    v02 = res.v02
    assert v02 is not None
    m2 = res.metrics_v02
    names = v02.class_names or v02.model.names
    inv = res.inv
    L: list[str] = []
    L.append("## v0.2 probabilistic stage\n")
    obj0, obj1 = v02.initial_objective, v02.objective
    L.append(_tbl([
        ["objective F = ELBO + log P(G)", round(obj0.get("objective", 0), 2), round(obj1.get("objective", 0), 2)],
        ["ELBO", round(obj0.get("elbo", 0), 2), round(obj1.get("elbo", 0), 2)],
        ["sum log Z~", round(obj0.get("log_z", 0), 2), round(obj1.get("log_z", 0), 2)],
        ["KL(q(Theta) || p(Theta))", round(obj0.get("kl", 0), 2), round(obj1.get("kl", 0), 2)],
        ["log P(G)", round(obj0.get("structure_prior", 0), 2), round(obj1.get("structure_prior", 0), 2)],
        ["ELBO per sentence (bits)", round(-obj0.get("elbo", 0) / max(1, len(res.corpus)) / 0.6931, 3), round(-obj1.get("elbo", 0) / max(1, len(res.corpus)) / 0.6931, 3)],
        ["active classes", v02.init_summary.get("n_classes"), v02.model.n_classes],
        ["active rules", v02.init_summary.get("n_rules"), len(v02.model.active_rules())],
        ["lexical entries", v02.init_summary.get("n_lexical_entries"), v02.model.n_lex],
    ], ["quantity", "after init + CAVI", "final"]))
    ps = m2.get("posterior", {})
    L.append(f"\n- fully parsed sentences (single root): {ps.get('fully_parsed_fraction')}; mean chunks/sentence {ps.get('mean_chunks_per_sentence')}; unknown-token mass {ps.get('unknown_token_mass')}; ambiguous span types {ps.get('n_ambiguous_types')}; pruning {ps.get('pruning')}")
    L.append(f"- potentials: {v02.potentials_info}; structure moves {m2.get('n_structure_moves')} proposed / {m2.get('n_accepted_moves')} accepted; monotonicity violations {m2.get('monotonicity_violations')}; runtime {m2.get('runtime_sec')}\n")
    if "class_recovery_type_level" in m2:
        L.append("### v0.1 vs v0.2 (same corpus, same gold)\n")
        L.append(_tbl(_cmp_rows(res.metrics, m2), ["metric", "v0.1", "v0.2"]))
        cr = m2["composition_recovery"]
        L.append(f"\n- v0.2 relations recovered: {', '.join(cr['recovered']) or '-'}; missed: {', '.join(cr['missed']) or '-'}")
        if cr["false_relations"]:
            L.append("- v0.2 false relations: " + "; ".join(f"{x['inferred']} => {x['mapped']} (support {x['support']})" for x in cr["false_relations"][:8]))
        L.append(f"- v0.2 class alignment: " + ", ".join(f"{names.get(int(k), k)}->{v}" for k, v in m2["class_alignment"].items()))
        am = m2["ambiguity_detection"]
        L.append(f"- v0.2 occurrence ambiguity: {am.get('n_pred')} types flagged (second class mass >= threshold); false {am.get('false_splits', [])[:10]}; missed {am.get('missed_splits', [])}")
        for w, d in am.get("posterior_mixtures", {}).items():
            L.append(f"  - `{w}`: gold {d['gold']} -> posterior mixture {d['posterior_mixture']}")
        cv = m2.get("constituents_viterbi", {})
        L.append(f"- constituents from the MAP derivation instead of posterior > 0.5: P {cv.get('precision')} R {cv.get('recall')} F1 {cv.get('f1')}\n")
    # ---- classes
    L.append("### v0.2 classes (posterior view)\n")
    for c in v02.posterior_stats.get("classes", [])[:12]:
        L.append(f"#### {names.get(c['class'], c['name'])}  (posterior mass {c['posterior_mass']}, root mass {c['root_mass']}, lineage {c['lineage']})\n")
        L.append("```")
        L.append("lexical entries: " + ", ".join(c["top_lexical_entries"][:12]))
        L.append("top span types: " + ", ".join(c["top_span_types"][:10]))
        for o in c["top_occurrences"][:3]:
            L.append("    " + o)
        L.append("top lexical contexts: " + ", ".join(c["top_lexical_contexts"][:6]))
        L.append("composition in : " + "; ".join(c["top_composition_in"][:5]))
        L.append("composition out: " + "; ".join(c["top_composition_out"][:5]))
        if c["top_abstract_constituent_contexts"]:
            L.append("abstract constituent contexts: " + "; ".join(c["top_abstract_constituent_contexts"][:5]))
        L.append("```\n")
    # ---- moves
    L.append("### Structure moves (class merge / class split / rule and lexicon proposals)\n")
    L.append("Terminology: *occurrence ambiguity* = one span type with posterior mass on several classes; *class split* = A -> A1, A2 changes the inventory; *structural ambiguity* = several derivations of one span, marginalised by Inside-Outside.\n")
    for mv in v02.moves[:60]:
        kind = mv.get("kind")
        head = {"merge": f"merge {mv.get('a_name')} + {mv.get('b_name')}", "split": f"split {mv.get('a_name')}", "rule_add": f"rule add {mv.get('rule')}", "rule_prune": f"rule prune x{mv.get('n_rules')}", "lex_add": f"lexicon add {mv.get('class_name')}: {mv.get('token')}", "lex_prune": f"lexicon prune x{mv.get('n_entries')}", "rollback": "rollback"}.get(kind, kind)
        L.append(f"- **{head}** -> {'ACCEPTED' if mv.get('accepted') else 'rejected'}; dF {mv.get('delta')} (dELBO {mv.get('delta_ELBO')}, dprior {mv.get('delta_structure_prior')}, threshold {mv.get('threshold')}); why proposed: {mv.get('proposal_reason')}; evaluated on {mv.get('evaluated_on')}")
        if kind == "split":
            L.append(f"  - partition A1 ({mv.get('n_occurrences', [0, 0])[0]} occ): {mv.get('partition_a1', [])[:4]}")
            L.append(f"  - partition A2 ({mv.get('n_occurrences', [0, 0])[1]} occ): {mv.get('partition_a2', [])[:4]}")
        if kind == "merge":
            L.append(f"  - similarity {mv.get('similarity')} by block {mv.get('similarity_parts')}")
    L.append("")
    # ---- ambiguity / entropy
    L.append("### Highest posterior entropy span types (occurrence ambiguity candidates, not linguistic claims)\n")
    for e in v02.posterior_stats.get("highest_entropy_span_types", [])[:12]:
        L.append(f"- `{e['span_type']}` ({e['n_occurrences']} occ): entropy {e['entropy_nats']} nats, mixture {e['mixture']}")
    pr = v02.posterior_stats.get("private_rules", [])
    if pr:
        L.append("\nRules with a dominant lexical realisation (private-rule prior applies):\n")
        for x in pr[:8]:
            L.append(f"- {x['rule']}: E[N] {x['expected_count']}, dominant `{x['dominant_type']}` share {x['dominant_share']}, penalty {x['penalty']}")
    L.append("")
    dg = res.diagnostics_v02
    if dg:
        L.append("### v0.1 failure modes re-checked on v0.2\n")
        L.append(f"- fragmentation (gold class -> inferred fragments >= 5%): " + "; ".join(f"{g}: {d['n_fragments_ge_5pct']} {d['fragments'][:4]}" for g, d in dg.get("fragmentation", {}).items()))
        L.append(f"- contamination (inferred class purity): " + "; ".join(f"{c}: {d['purity']} ({d['dominant']}, second {d['second']} {d['second_share']})" for c, d in list(dg.get("contamination", {}).items())[:12]))
        L.append(f"- extra root classes beyond S: {dg.get('extra_root_classes')}")
        L.append(f"- private crutch rules: {[(p['rule'], p['dominant_share']) for p in dg.get('private_crutch_rules', [])[:6]]}")
        L.append(f"- filler-induced rules: {[(p['rule'], p['filler_share']) for p in dg.get('filler_induced_rules', [])[:6]]}; filler token classes: {dg.get('filler_token_classes')}")
        L.append(f"- ambiguity pollution (purity of classes hosting gold-ambiguous words): {dg.get('ambiguity_pollution_class_purity')}")
        L.append(f"- context-vs-grammar disagreement on leaf classes: {dg.get('context_vs_grammar_disagreement')}\n")
    return L


def write_report(res: RunResult, out_dir: str) -> str:
    text = build_report(res)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
